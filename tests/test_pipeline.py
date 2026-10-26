import asyncio
import json
import os
import time
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from libclaim import pipeline as pipeline_module
from libclaim.packet import assemble as assemble_module
from libclaim.live.session import Scheduling
from libclaim.live.tools import Exclusion, ExclusionLog, Statement, StatementKind, StatementLog
from libclaim.llm import QuotaExhausted
from libclaim.models import Still
from libclaim.packet.models import ClaimPacket
from libclaim.packet.totals import compute_totals
from libclaim.pipeline import LiveHooks, Services, SweepRun, attach_statements, excluded_by, load_stills, run_pipeline
from libclaim.server import KIND_STILL, create_app, pack_binary
from libclaim.settings import Settings
from libclaim.stages import STAGES, BookSlot, DeviceInfo, load_pipeline_config
from libclaim.sweep import SweepDir
from libclaim.vision import masks
from libclaim.vision import reading as reading_module
from libclaim.vision.boxes import Box
from libclaim.vision.resolve import Resolution
from libclaim.vision.spines import Orientation
from tests import sweep12 as fx

BOOK_COUNT = len(fx.BOOKS)
UNREADABLE = {"blank_a", "blank_b", "blank_c", "ulysses"}
APPRAISAL = {"The Hobbit", "The God of Small Things", "The Great Gatsby"}
PRICED = {"Sapiens", "Everything I Never Told You"}
TOKEN = "pipeline-token-0123456789abcdef"
VOLATILE = {"costs", "latency_s"}
WAIT_S = 30
MAX_MESSAGES = 200


@pytest.fixture(autouse=True)
def offline_models(monkeypatch):
    monkeypatch.setattr(masks, "find_spec", lambda name: None)
    monkeypatch.setattr(reading_module, "ocr_text", fx.fake_ocr)


@pytest.fixture
def online(monkeypatch):
    monkeypatch.delenv("LIBCLAIM_OFFLINE")


@pytest.fixture
def sweep(tmp_path):
    return fx.copy_sweep(tmp_path / "run")


@dataclass
class Recorder:
    events: list[dict] = field(default_factory=list)
    injected: list[tuple[dict, Scheduling]] = field(default_factory=list)
    recaptures: list[str] = field(default_factory=list)

    async def emit(self, event):
        self.events.append(event)

    async def inject(self, event, scheduling):
        self.injected.append((event, scheduling))

    async def recapture(self, region):
        self.recaptures.append(region)

    def hooks(self):
        return LiveHooks(emit=self.emit, inject=self.inject, recapture=self.recapture)

    def of(self, kind):
        return [event for event in self.events if event["type"] == kind]


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def services(sweep, tmp_path, web, llm=None, local=None, clock=time.monotonic):
    return Services(
        settings=fx.SETTINGS,
        llm=llm or fx.fake_llm(sweep, tmp_path),
        catalogs=fx.catalogs_for(sweep, web),
        http=fx.price_http_for(sweep, web),
        local_boxes=local or fx.local_with_miss(sweep),
        depth_fn=fx.no_depth,
        clock=clock,
    )


def fast_config(**changes):
    return load_pipeline_config().model_copy(update={"deadline_poll_s": 0.01, **changes})


@pytest.fixture
async def first(sweep, tmp_path, online):
    web, recorder = fx.Web(), Recorder()
    client = fx.FakeVision()
    llm = fx.fake_llm(sweep, tmp_path, client)
    packet = await run_pipeline(sweep, services=services(sweep, tmp_path, web, llm), hooks=recorder.hooks())
    return packet, web, recorder, client


def has_dims(book):
    return bool(book.spine_height_cm or book.dims_estimated_from_prior)


def by_title(packet):
    return {book.title: book for book in packet.books if book.title}


def stable(packet):
    return packet.model_dump(mode="json", exclude=VOLATILE)


async def test_fixture_sweep_gives_a_valid_packet_with_traceable_lines(first, sweep):
    packet, web, recorder, client = first
    saved = ClaimPacket.model_validate_json((sweep.root / "claim_packet.json").read_text(encoding="utf-8"))
    assert stable(saved) == stable(packet)
    assert (sweep.root / "report.html").is_file()
    assert packet.totals == compute_totals(packet.books, packet.items)
    assert packet.totals.book_count == BOOK_COUNT
    refs = [b.frame_ref for b in packet.books] + [i.frame_ref for i in packet.items] + packet.room.frame_refs
    refs += [entry.frame_ref for entry in packet.review_queue if entry.frame_ref]
    assert refs and all(not ref.startswith(("/", "\\")) and (sweep.root / ref).is_file() for ref in refs)
    assert packet.sweep.id == "run" and packet.sweep.currency == "INR" and packet.sweep.duration_s == (len(fx.frames()) - 1) * fx.STEP_MS / 1000
    assert web.requests


async def test_book_lines_follow_the_evidence(first):
    packet = first[0]
    titled = by_title(packet)
    assert {title for title, book in titled.items() if book.status == "needs_appraisal"} == APPRAISAL
    assert all(titled[title].replacement_cost.amount is not None and titled[title].replacement_cost.url for title in PRICED)
    assert titled["Sapiens"].isbn == "9780099590088" and titled["Sapiens"].used_value.source_type
    assert all(book.replacement_cost.amount is None for book in packet.books if book.status == "needs_appraisal")
    assert titled["Sapiens"].scale_method == "aruco_homography"
    assert titled["Dune"].scale_method == "book_format_prior"
    assert titled["Sapiens"].spine_height_cm > titled["Sapiens"].spine_thickness_cm > 0
    dune = titled["Dune"]
    assert dune.spine_height_cm is None and dune.spine_thickness_cm is None
    estimate = dune.dims_estimated_from_prior
    assert estimate.height_cm > estimate.thickness_cm > 0 and estimate.book_format
    assert titled["Sapiens"].dims_estimated_from_prior is None
    assert {book.shelf for book in packet.books} == fx.SHELF_LABELS


async def test_unreadable_spines_are_unidentified_with_dims(first):
    packet = first[0]
    unread = [book for book in packet.books if book.status == "unidentified"]
    assert len(unread) == len(UNREADABLE)
    assert all(not b.title and not b.isbn and has_dims(b) for b in unread)


async def test_every_stage_records_latency_and_cost(first):
    packet = first[0]
    assert set(STAGES) | {"after_sweep"} | {f"{stage}_busy" for stage in STAGES[1:6]} <= set(packet.latency_s)
    assert all(packet.latency_s[f"{stage}_busy"] >= 0 for stage in STAGES[1:6])
    assert packet.costs.tokens > 0 and packet.costs.usd_at_paid_rates > 0
    assert {"spines", "reading", "items"} <= set(packet.costs.by_stage_usd)
    assert packet.costs.billed_usd is None and "unknown" in packet.costs.billing


@pytest.mark.parametrize(("free", "billed", "words"), [(True, 0.0, "free tier"), (False, None, "paid rates")])
async def test_billing_follows_the_free_tier_setting(sweep, tmp_path, online, free, billed, words):
    settings = fx.SETTINGS.model_copy(update={"free_tier": free})
    found = services(sweep, tmp_path, fx.Web())
    found.settings = settings
    packet = await run_pipeline(sweep, services=found)
    assert words in packet.costs.billing
    expected = packet.costs.usd_at_paid_rates if billed is None else billed
    assert packet.costs.billed_usd == pytest.approx(expected)


async def test_live_hooks_get_progress_inventory_recapture_and_readback(first):
    packet, _, recorder, client = first
    assert {event["stage"] for event in recorder.of("pipeline.progress")} >= set(STAGES[2:])
    assert {event["item"]["id"] for event in recorder.of("inventory.upsert")} >= {b.id for b in packet.books}
    ready = recorder.of("packet_ready")[-1]
    assert ready["json"] == "/runs/run/claim_packet.json" and ready["report"] == "/runs/run/report.html"
    readback, scheduling = recorder.injected[-1]
    assert scheduling == Scheduling.WHEN_IDLE and readback["readback"] == ready["summary"]
    assert f"I counted {BOOK_COUNT} books" in ready["summary"]
    assert recorder.recaptures == []
    assert (fx.TIEBREAK, "TileSpines") in client.calls
    assert any("count_mismatch" in entry.reason and "re-queried" in entry.reason for entry in packet.review_queue)


async def test_room_without_depth_falls_back_and_items_follow_statements(first):
    packet = first[0]
    assert packet.room.scale_method == "aruco_only" and packet.room.floor_area_m2 is None
    assert packet.room.shelved_wall_area_m2 > 0
    assert any(entry.ref_id == "room" and "[ml] extra" in entry.reason for entry in packet.review_queue)
    categories = sorted(item.category for item in packet.items)
    assert categories == sorted(thing.category for thing in fx.THINGS)
    shelf = next(item for item in packet.items if item.category == "furniture")
    assert shelf.status == "range" and shelf.replacement_cost.low is not None and shelf.replacement_cost.url
    art = next(item for item in packet.items if item.category == "artwork")
    assert any(entry.ref_id == art.id and "too_few_listings" in entry.reason for entry in packet.review_queue)
    assert packet.second_country.country == "GB" and packet.second_country.books


async def test_replay_from_price_makes_zero_network_calls(first, sweep, tmp_path):
    packet = first[0]
    dead = fx.Web(fail=True)
    replay = await run_pipeline(sweep, "price", services=services(sweep, tmp_path, dead))
    assert dead.requests == []
    assert stable(replay) == stable(packet)


async def test_replays_are_deterministic(first, sweep, tmp_path):
    packet = first[0]
    again = await run_pipeline(sweep, "tracks", services=services(sweep, tmp_path, fx.Web()))
    once_more = await run_pipeline(sweep, "tracks", services=services(sweep, tmp_path, fx.Web()))
    assert stable(again) == stable(once_more) == stable(packet)


@dataclass
class Poisoned(fx.FakeVision):
    async def generate(self, model, parts, schema):
        self.calls.append((model, schema.__name__))
        raise AssertionError("no model call is allowed in a replay")


@pytest.mark.parametrize("stage", ["spines", "tracks", "reading", "items"])
async def test_moved_run_replays_from_cache_without_any_network_or_model_call(first, sweep, tmp_path, stage):
    packet = first[0]
    moved = SweepDir(tmp_path / "moved")
    sweep.root.rename(moved.root)
    dead, poisoned = fx.Web(fail=True), Poisoned()
    llm = fx.fake_llm(moved, tmp_path, poisoned)
    replay = await run_pipeline(moved, stage, services=services(moved, tmp_path, dead, llm, local=fx.local_with_miss(moved)))
    assert dead.requests == [] and poisoned.calls == []
    assert stable(replay) == stable(packet) | {"sweep": stable(replay)["sweep"]}
    assert stable(replay)["sweep"]["id"] == "moved"


def test_cli_replays_from_price_offline(first, sweep, monkeypatch, capsys):
    packet = first[0]
    monkeypatch.setattr(pipeline_module, "load_settings", lambda: fx.SETTINGS)
    pipeline_module.main([str(sweep.root), "--from", "price"])
    assert os.environ["LIBCLAIM_OFFLINE"] == "1"
    out = capsys.readouterr().out
    assert "claim_packet.json" in out and "I counted" in out
    saved = ClaimPacket.model_validate_json((sweep.root / "claim_packet.json").read_text(encoding="utf-8"))
    assert stable(saved) == stable(packet)


async def test_unknown_stage_is_rejected(sweep, tmp_path):
    with pytest.raises(ValueError):
        await run_pipeline(sweep, "nonsense", services=services(sweep, tmp_path, fx.Web(fail=True)))


@dataclass
class Stalling(fx.FakeVision):
    stalled: int = 0

    async def generate(self, model, parts, schema):
        if schema.__name__ == "SpineReads":
            self.stalled += 1
            await asyncio.Event().wait()
        return await super().generate(model, parts, schema)


async def test_forced_stall_finalises_at_the_deadline_with_review(sweep, tmp_path, online):
    clock, client, recorder = Clock(), Stalling(), Recorder()
    llm = fx.fake_llm(sweep, tmp_path, client)
    run = SweepRun(sweep, services(sweep, tmp_path, fx.Web(), llm, clock=clock), recorder.hooks(), fast_config())
    for still in load_stills(sweep):
        run.add_still(still)
    finishing = asyncio.create_task(run.finish())
    async with asyncio.timeout(WAIT_S):
        while client.stalled < len(fx.STRIPS):
            await asyncio.sleep(0.01)
    clock.now += load_pipeline_config().deadline_s + 1
    packet = await asyncio.wait_for(finishing, WAIT_S)
    assert packet.totals.book_count == BOOK_COUNT
    assert packet.totals == compute_totals(packet.books, packet.items)
    assert all(book.status == "unidentified" and has_dims(book) for book in packet.books)
    unfinished = [e for e in packet.review_queue if "did not finish" in e.reason]
    assert {e.ref_id for e in unfinished if e.reason.startswith("reading")} == {b.id for b in packet.books}
    assert any(e.ref_id == "room" and e.severity == "high" for e in unfinished)
    assert recorder.of("packet_ready") and (sweep.root / "claim_packet.json").is_file()


@dataclass
class SpineStall(fx.FakeVision):
    stalled: int = 0

    async def generate(self, model, parts, schema):
        if schema.__name__ == "TileSpines":
            self.stalled += 1
            await asyncio.Event().wait()
        return await super().generate(model, parts, schema)


async def test_spine_stall_hits_the_vision_cutoff_and_still_counts_books(sweep, tmp_path, online):
    clock, client = Clock(), SpineStall()
    llm = fx.fake_llm(sweep, tmp_path, client)
    run = SweepRun(sweep, services(sweep, tmp_path, fx.Web(), llm, local=fx.fake_local, clock=clock), None,
                   fast_config())
    for still in load_stills(sweep):
        run.add_still(still)
    finishing = asyncio.create_task(run.finish())
    async with asyncio.timeout(WAIT_S):
        while client.stalled < len(fx.STRIPS) * fx.ROW_FRAMES:
            await asyncio.sleep(0.01)
    clock.now += load_pipeline_config().vision_cutoff_s + 1
    packet = await asyncio.wait_for(finishing, WAIT_S)
    assert packet.totals.book_count == BOOK_COUNT
    assert {b.title for b in packet.books if b.status == "identified"} >= PRICED
    assert any("spine detection failed (vision_cutoff)" in entry.reason for entry in packet.review_queue)


def boom(*args, **kwargs):
    raise RuntimeError("injected")


@pytest.mark.parametrize(
    ("module", "target", "stage"),
    [
        (pipeline_module, "measure_rows", "scale"),
        (pipeline_module, "spine_crops", "reading"),
        (pipeline_module, "room_from_depth", "room"),
        (pipeline_module, "attach_statements", "appraisal"),
        (pipeline_module, "price_book", "pricing"),
        (assemble_module, "item_lines", "items"),
        (assemble_module, "room_of", "room"),
    ],
)
async def test_a_failing_stage_still_yields_a_valid_packet(sweep, tmp_path, online, monkeypatch, module, target, stage):
    monkeypatch.setattr(module, target, boom)
    packet = await run_pipeline(sweep, services=services(sweep, tmp_path, fx.Web()))
    assert packet.totals.book_count == BOOK_COUNT
    assert packet.totals == compute_totals(packet.books, packet.items)
    failed = [e for e in packet.review_queue if e.severity == "high" and "RuntimeError" in e.reason]
    assert failed and all(e.reason.startswith(stage) for e in failed)
    assert (sweep.root / "claim_packet.json").is_file()


async def test_statements_that_match_nothing_go_to_review(sweep, tmp_path, online):
    lonely = [
        Statement(subject="the poster", kind=StatementKind.art_print, value="print", t_ms=0, still_seq_from=1,
                  still_seq_to=2),
        Statement(subject="that one", kind=StatementKind.signed, value="signed", t_ms=0, still_seq_from=9,
                  still_seq_to=10),
    ]
    sweep.write_stage("statements", StatementLog(items=[*fx.STATEMENTS, *lonely]))
    packet = await run_pipeline(sweep, services=services(sweep, tmp_path, fx.Web()))
    reasons = [entry.reason for entry in packet.review_queue]
    assert any("'the poster' is a print but no detected artwork" in reason for reason in reasons)
    assert any("'that one'" in reason and "no book was in view" in reason for reason in reasons)


async def test_streaming_reports_segments_before_finish_and_alerts_while_fresh(sweep, tmp_path, online):
    recorder = Recorder()
    run = SweepRun(sweep, services(sweep, tmp_path, fx.Web()), recorder.hooks(), fast_config())
    for still in load_stills(sweep):
        run.add_still(still)
        async with asyncio.timeout(WAIT_S):
            while not all(task.done() for task in run.detections):
                await asyncio.sleep(0.01)
    async with asyncio.timeout(WAIT_S):
        while not any(event["stage"] == "resolution" for event in recorder.of("pipeline.progress")):
            await asyncio.sleep(0.01)
    assert not recorder.of("packet_ready")
    seconds = (fx.REQUERY_FRAME - 1) * fx.STEP_MS // 1000
    assert recorder.recaptures == [load_pipeline_config().recapture_region.format(seconds=seconds)]
    packet = await asyncio.wait_for(run.finish(), WAIT_S)
    assert packet.totals.book_count == BOOK_COUNT and recorder.of("packet_ready")


async def test_appraisal_status_survives_when_pricing_fails(sweep, tmp_path, online, monkeypatch):
    monkeypatch.setattr(pipeline_module, "price_book", boom)
    packet = await run_pipeline(sweep, services=services(sweep, tmp_path, fx.Web()))
    assert {title for title, book in by_title(packet).items() if book.status == "needs_appraisal"} == APPRAISAL


def test_device_platform_comes_from_the_user_agent():
    agent = "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) Safari/604.1"
    assert DeviceInfo(user_agent=agent).platform == "iPhone; CPU iPhone OS 18_0 like Mac OS X"
    assert DeviceInfo(user_agent="curl/8").platform == "curl/8"


class Storm:
    def __init__(self):
        self.calls = 0

    async def structured(self, models, parts, schema, stage=None):
        self.calls += 1
        raise QuotaExhausted("rate limited on every model")


async def test_vision_quota_storm_finalises_with_review_entries(sweep, tmp_path, online):
    storm = Storm()
    packet = await run_pipeline(sweep, services=services(sweep, tmp_path, fx.Web(), storm, local=fx.fake_local))
    assert storm.calls == 1
    assert packet.totals.book_count == BOOK_COUNT
    assert all(book.status == "unidentified" and has_dims(book) for book in packet.books)
    reasons = [entry.reason for entry in packet.review_queue]
    assert any("spine detection failed (QuotaExhausted)" in reason for reason in reasons)
    assert any("item detection failed (QuotaExhausted)" in reason for reason in reasons)
    assert packet.totals == compute_totals(packet.books, packet.items)


def slot(book_id, seen, shelf="S1"):
    still = Still(seq=seen[0], t_ms=0, path="", w=1, h=1, blur=1.0, glare=0.0, motion_dps=0.0)
    return BookSlot(id=book_id, segment=0, track=0, shelf=shelf, position=1, orientation=Orientation.vertical,
                    frame=still, box=Box(x1=0, y1=0, x2=1, y2=1), seen=seen)


def test_statement_attaches_to_the_named_book_or_else_the_whole_view():
    books = [slot("a", [1, 2]), slot("b", [2]), slot("c", [9])]
    titles = {"a": Resolution(status="identified", title="Dune"), "b": Resolution(status="identified", title="Emma")}
    named = Statement(subject="dune", kind=StatementKind.signed, value="signed", t_ms=0, still_seq_from=2, still_seq_to=2)
    vague = Statement(subject="this one", kind=StatementKind.first_edition, value="", t_ms=0, still_seq_from=1,
                      still_seq_to=2)
    art = Statement(subject="dune", kind=StatementKind.original_art, value="", t_ms=0, still_seq_from=1, still_seq_to=9)
    nowhere = Statement(subject="the red one", kind=StatementKind.signed, value="", t_ms=0, still_seq_from=5,
                        still_seq_to=6)
    attached, unattached = attach_statements(books, titles, [named, vague, art, nowhere], 80)
    assert attached == {"a": [named, vague], "b": [vague], "c": []}
    assert len(unattached) == 1 and "'the red one'" in unattached[0]


def test_exclusion_by_shelf_id_or_by_time_window():
    stills = [Still(seq=1, t_ms=10_000, path="", w=1, h=1, blur=1.0, glare=0.0, motion_dps=0.0)]
    by_shelf = Exclusion(description="not mine", shelf_id="U1-1", t_ms=0)
    by_time = Exclusion(description="skip that shelf", t_ms=12_000)
    later = Exclusion(description="later", t_ms=30_000)
    assert excluded_by({"u1-1"}, [], [by_shelf], 4000) == "not mine"
    assert excluded_by({"s2"}, stills, [later, by_time], 4000) == "skip that shelf"
    assert excluded_by({"s2"}, stills, [later], 4000) == ""


async def test_excluded_shelf_is_kept_unpriced_with_a_review_entry(first, sweep, tmp_path):
    sweep.write_stage("exclusions", ExclusionLog(items=[Exclusion(description="not mine", shelf_id="u1-1", t_ms=0)]))
    packet = await run_pipeline(sweep, "appraisal", services=services(sweep, tmp_path, fx.Web()))
    shelf = [book for book in packet.books if book.shelf == "U1-1"]
    assert shelf and all(book.replacement_cost.amount is None for book in shelf)
    excluded = {e.ref_id for e in packet.review_queue if e.reason == "excluded by the user: not mine"}
    assert excluded == {book.id for book in shelf}


class Bridge:
    def __init__(self, hooks):
        self.hooks = hooks
        self.session = object()
        self.injected = []
        self.closed = asyncio.Event()

    async def run(self):
        await self.closed.wait()

    async def send_audio(self, data):
        return None

    async def send_frame(self, data):
        return None

    def end_sweep(self):
        return None

    async def inject(self, event, scheduling):
        self.injected.append(event)

    async def close(self):
        self.closed.set()


def test_server_sweep_streams_stills_into_the_pipeline_and_produces_a_packet(tmp_path, online):
    runs = tmp_path / "runs"
    bridges = []

    def bridge_factory(hooks):
        bridges.append(Bridge(hooks))
        return bridges[-1]

    def pipeline_factory(sweep, hooks):
        return SweepRun(sweep, services(sweep, tmp_path, fx.Web(), local=fx.fake_local), hooks)

    app = create_app(Settings(access_token=TOKEN), bridge_factory, pipeline_factory, runs_dir=runs,
                     web_dist=tmp_path / "none")
    stills = load_stills(SweepDir(fx.ROOT))
    with TestClient(app) as client, client.websocket_connect("/ws/sweep") as ws:
        ws.send_text(json.dumps({"type": "hello", "token": TOKEN, "t_ms": 1}))
        sweep_id = ws.receive_json()["sweep_id"]
        for still in stills:
            header = {"seq": still.seq, "t_ms": still.t_ms, "w": still.w, "h": still.h}
            ws.send_bytes(pack_binary(KIND_STILL, header, open(still.path, "rb").read()))
        ws.send_text(json.dumps({"type": "control", "action": "end_sweep"}))
        ready = next(m for m in (ws.receive_json() for _ in range(MAX_MESSAGES)) if m["type"] == "packet_ready")
    packet = ClaimPacket.model_validate_json((runs / sweep_id / "claim_packet.json").read_text(encoding="utf-8"))
    assert ready["json"] == f"/runs/{sweep_id}/claim_packet.json"
    assert packet.totals.book_count == BOOK_COUNT
    assert all((runs / sweep_id / book.frame_ref).is_file() for book in packet.books)
    assert any("readback" in event for event in bridges[0].injected)


async def test_every_line_has_an_existing_crop_and_a_box_inside_its_frame(first, sweep):
    packet = first[0]
    lines = [*packet.books, *packet.items]
    assert lines
    for line in lines:
        width, height = line.frame_size
        x1, y1, x2, y2 = line.box
        assert 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
        assert line.crop_ref.startswith("frames/crops/") and (sweep.root / line.crop_ref).is_file()
        assert (sweep.root / line.frame_ref).is_file()
    crops = {line.id: line.crop_ref for line in lines}
    assert all(entry.crop_ref == crops.get(entry.ref_id, "") for entry in packet.review_queue)


async def test_books_without_a_marker_get_one_review_entry_per_shelf(first):
    packet = first[0]
    prior = [book for book in packet.books if book.scale_method == "book_format_prior"]
    entries = [entry for entry in packet.review_queue if "no measured size" in entry.reason]
    assert prior and {entry.ref_id for entry in entries} == {book.shelf for book in prior}
    assert len(entries) == len({book.shelf for book in prior})
    assert all("no scale marker was visible" in entry.reason for entry in entries)
    assert not any("ARUCO" in entry.reason or "_CM" in entry.reason for entry in packet.review_queue)
    assert not any(entry.reason.startswith("book_format_prior") for entry in packet.review_queue)


async def test_totals_separate_measured_and_estimated_shelf_runs(first):
    packet = first[0]
    measured = sum(book.spine_thickness_cm for book in packet.books if book.spine_thickness_cm is not None)
    estimated = sum(book.dims_estimated_from_prior.thickness_cm for book in packet.books if book.dims_estimated_from_prior)
    assert packet.totals.shelf_run_m == pytest.approx(measured / 100, abs=1e-3)
    assert packet.totals.shelf_run_estimated_m == pytest.approx(estimated / 100, abs=1e-3)


async def test_owner_can_fill_the_contract_dims_from_the_prior(first, sweep, tmp_path, online):
    config = fast_config(dims=load_pipeline_config().dims.model_copy(update={"fill_from_prior": True}))
    packet = await run_pipeline(sweep, "packet", services=services(sweep, tmp_path, fx.Web()), config=config)
    dune = by_title(packet)["Dune"]
    assert dune.spine_height_cm == dune.dims_estimated_from_prior.height_cm
    assert dune.spine_thickness_cm == dune.dims_estimated_from_prior.thickness_cm
    assert packet.totals.shelf_run_estimated_m is None
