import argparse
import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from google.genai import errors
from pydantic import BaseModel
from rapidfuzz import fuzz
from rapidfuzz.utils import default_process

from libclaim.budget import Budget, BudgetExceeded
from libclaim.capture.quality import CaptureConfig, load_capture_config, score
from libclaim.catalog.http import CatalogOffline
from libclaim.live.bridge import load_live_config
from libclaim.live.session import Scheduling
from libclaim.live.tools import (
    EXCLUSIONS_STAGE,
    LOCALE_STAGE,
    STATEMENTS_STAGE,
    Exclusion,
    ExclusionLog,
    LocaleChoice,
    Statement,
    StatementKind,
    StatementLog,
)
from libclaim.llm import LiveCallBlocked, Llm, LlmReplyInvalid, QuotaExhausted, load_llm_config
from libclaim.models import Still
from libclaim.packet.assemble import book_lines, build_packet, frame_ref, slots, summary_text
from libclaim.packet.models import ClaimPacket, Item, SweepInfo
from libclaim.packet.report import REPORT_NAME, render_report
from libclaim.pricing.books import BookPrices, Market, price_book
from libclaim.pricing.config import ItemSources, load_item_sources
from libclaim.pricing.http import PriceHttp
from libclaim.pricing.items import ItemPrice, price_item, second_country
from libclaim.pricing.sources.bookchor import Bookchor
from libclaim.pricing.sources.bookswagon import Bookswagon
from libclaim.pricing.sources.fx import Fx
from libclaim.pricing.sources.listings import Listings, matches
from libclaim.pricing.sources.penguin_uk import PenguinUk
from libclaim.pricing.sources.pustakkosh import Pustakkosh
from libclaim.resolve_config import load_resolve_config
from libclaim.room.depth import DepthSet, metric_depth, room_from_depth, scaled_depths
from libclaim.settings import Locale, Settings, load_locale, load_settings
from libclaim.stages import (
    POST_STAGES,
    SEGMENT_STAGES,
    STAGES,
    STILLS,
    VISION_CUTOFF,
    AppraisalStage,
    BookAppraisal,
    BookRead,
    BookSlot,
    FrameSpines,
    ItemsStage,
    KeyframesStage,
    PipelineConfig,
    PricedItem,
    PricingStage,
    ReadingStage,
    ResolutionStage,
    Results,
    ScaleStage,
    SegmentPlan,
    SegmentTracks,
    SpinesStage,
    StillLog,
    TracksStage,
    book_key,
    item_key,
    load_pipeline_config,
)
from libclaim.room.geometry import RoomGeometry
from libclaim.sweep import SweepDir
from libclaim.vision.boxes import Box, TrackConfig, load_vision_config, read_image
from libclaim.vision.gate import VisionGate
from libclaim.vision.items import ItemDet, detect_items
from libclaim.vision.keyframes import KeyframeStream, load_keyframe_config
from libclaim.vision.reading import Reading, SpineCrop, crop_spine, read_spines
from libclaim.vision.resolve import Catalogs, Resolution, read_text, resolve
from libclaim.vision.resolve import SpineDims as ResolveDims
from libclaim.vision.rfdetr import rfdetr_boxes
from libclaim.vision.rows import (
    anchor_boxes,
    best_ref,
    continues,
    is_row,
    marker_unit,
    measure_rows,
    plane_boxes,
    shelf_units,
    split_rows,
)
from libclaim.vision.scale import SpineDims, load_measure_config, unit_scale
from libclaim.vision.spines import Orientation, Source, SpineDet, detect_spines, local_orientation, reconcile
from libclaim.vision.tracks import BookTrack, Frame, merge_tracks

logger = logging.getLogger(__name__)

VISION_FAILURES = (QuotaExhausted, LlmReplyInvalid, errors.APIError, BudgetExceeded, LiveCallBlocked)
SOURCES = {"bookchor": Bookchor, "pustakkosh": Pustakkosh, "bookswagon": Bookswagon, "penguin_uk": PenguinUk}
ART_KINDS = {StatementKind.original_art, StatementKind.art_print}
NO_LISTING = "no listing category in the item sources matches this item"
OFFLINE_LISTING = "listing pages are not cached and the pipeline is offline"
FRAME_SEPARATOR = "_"
STAGE_TYPES: dict[str, type[BaseModel]] = {
    STILLS: StillLog,
    "keyframes": KeyframesStage,
    "spines": SpinesStage,
    "tracks": TracksStage,
    "scale": ScaleStage,
    "reading": ReadingStage,
    "resolution": ResolutionStage,
    "room": RoomGeometry,
    "items": ItemsStage,
    "appraisal": AppraisalStage,
    "pricing": PricingStage,
}
OPTIONAL_STAGES = {"room", "items"}

Emit = Callable[[dict[str, Any]], Awaitable[None]]


async def ignore(event: dict[str, Any]) -> None:
    return None


@dataclass
class LiveHooks:
    emit: Emit = ignore
    inject: Callable[[dict[str, Any], Scheduling], Awaitable[None]] | None = None
    recapture: Callable[[str], Awaitable[None]] | None = None


@dataclass
class Services:
    settings: Settings
    llm: Any
    catalogs: Catalogs
    http: PriceHttp
    local_boxes: Callable[[np.ndarray], list[Box] | None] = rfdetr_boxes
    depth_fn: Callable[[Still], np.ndarray] = metric_depth
    clock: Callable[[], float] = time.monotonic

    async def aclose(self) -> None:
        await self.http.aclose()
        await self.catalogs.open_library.http.aclose()


def services_for(sweep: SweepDir, settings: Settings, budget: Budget | None = None) -> Services:
    cache = sweep.root / load_resolve_config().catalog.cache_dir
    return Services(
        settings=settings,
        llm=Llm(budget=budget or Budget(), ledger=sweep.ledger),
        catalogs=Catalogs.for_sweep(sweep.root),
        http=PriceHttp(cache),
    )


def still_from_file(path: Path, cfg: CaptureConfig) -> Still:
    seq, t_ms = path.stem.split(FRAME_SEPARATOR)
    image = read_image(str(path))
    quality = score(image, cfg)
    height, width = image.shape[:2]
    return Still(seq=int(seq), t_ms=int(t_ms), path=str(path), w=width, h=height, blur=quality.blur,
                 glare=quality.glare, motion_dps=0.0)


def load_stills(sweep: SweepDir) -> list[Still]:
    if (sweep.stages / f"{STILLS}.json").is_file():
        logged = sweep.read_stage(STILLS, StillLog).stills
        return [still.model_copy(update={"path": str(sweep.frames / Path(still.path).name)}) for still in logged]
    cfg = load_capture_config()
    stills = [still_from_file(path, cfg) for path in sorted(sweep.frames.glob("*.jpg"))]
    return sorted(stills, key=lambda still: (still.t_ms, still.seq))


def read_or(sweep: SweepDir, name: str, cls: type[BaseModel], default: BaseModel | None) -> Any:
    return sweep.read_stage(name, cls) if (sweep.stages / f"{name}.json").is_file() else default


def det_of(track: BookTrack, frames: list[FrameSpines], stills: list[Still]) -> SpineDet:
    ref = best_ref(track, stills)
    dets = frames[ref.frame].dets
    return dets[[det.box_px for det in dets].index(ref.box)]


def build_segment(plan: SegmentPlan, carried: list[FrameSpines], frames: list[FrameSpines],
                  cfg: TrackConfig) -> SegmentTracks:
    frames = carried + frames
    stills = [frame.still for frame in frames]
    merged = merge_tracks([Frame(still=f.still, boxes=[d.box_px for d in f.dets]) for f in frames])
    tracks = [track for track in merged if min(ref.frame for ref in track.refs) >= len(carried)]
    if not tracks:
        return SegmentTracks(index=plan.index, stills=stills, tracks=[], rows=[], flat=[], books=[])
    orientation = [det_of(track, frames, stills).orientation for track in tracks]
    rows, flat = split_rows(anchor_boxes(tracks, stills, cfg), [o == Orientation.vertical for o in orientation])
    books = []
    for number, row in enumerate(rows, 1):
        shelf = plan.label if len(rows) == 1 else f"{plan.label}.{number}"
        for position, index in enumerate(row, 1):
            track = tracks[index]
            books.append(
                BookSlot(
                    id=book_key(plan.index, len(books)),
                    segment=plan.index,
                    track=index,
                    shelf=shelf,
                    position=position,
                    orientation=orientation[index],
                    frame=track.best_frame,
                    box=track.best_box,
                    seen=sorted({stills[ref.frame].seq for ref in track.refs}),
                )
            )
    return SegmentTracks(index=plan.index, stills=stills, tracks=tracks, rows=rows, flat=flat, books=books)


def spine_crops(books: list[BookSlot]) -> list[SpineCrop]:
    images: dict[str, np.ndarray] = {}
    crops = []
    for index, book in enumerate(books):
        if book.frame.path not in images:
            images[book.frame.path] = read_image(book.frame.path)
        det = SpineDet(box_px=book.box, orientation=book.orientation, source=Source.gemini)
        crops.append(crop_spine(images[book.frame.path], det, index))
    return crops


def subject_matches(statement: Statement, resolution: Resolution | None, threshold: float) -> bool:
    if resolution is None or not resolution.title:
        return False
    return fuzz.token_set_ratio(statement.subject, resolution.title, processor=default_process) >= threshold


def attach_statements(books: list[BookSlot], resolutions: dict[str, Resolution], statements: list[Statement],
                      threshold: float) -> dict[str, list[Statement]]:
    attached: dict[str, list[Statement]] = {book.id: [] for book in books}
    for statement in statements:
        if statement.kind in ART_KINDS or statement.still_seq_from is None or statement.still_seq_to is None:
            continue
        window = [b for b in books if any(statement.still_seq_from <= seq <= statement.still_seq_to for seq in b.seen)]
        named = [b for b in window if subject_matches(statement, resolutions.get(b.id), threshold)]
        for book in named or window:
            attached[book.id].append(statement)
    return attached


def excluded_by(labels: set[str], stills: list[Still], exclusions: list[Exclusion], window_ms: int) -> str:
    for exclusion in exclusions:
        if exclusion.shelf_id and exclusion.shelf_id.casefold() in labels:
            return exclusion.description
        if any(exclusion.t_ms - window_ms <= still.t_ms <= exclusion.t_ms for still in stills):
            return exclusion.description
    return ""


def listing_category(det: ItemDet, sources: ItemSources | None) -> str:
    text = f"{det.description} {det.brand_model}"
    categories = sources.categories.items() if sources else []
    return next((name for name, spec in categories if matches(text, spec)), "")


def item_sources(country: str) -> ItemSources | None:
    try:
        return load_item_sources(country)
    except FileNotFoundError:
        return None


class SweepRun:
    def __init__(self, sweep: SweepDir, services: Services, hooks: LiveHooks | None = None,
                 config: PipelineConfig | None = None, close_services: bool = False):
        self.sweep = sweep
        self.services = services
        self.hooks = hooks or LiveHooks()
        self.cfg = config or load_pipeline_config()
        self.close_services = close_services
        self.results = Results()
        self.llm = VisionGate(services.llm, load_llm_config().retry_delay_cap_seconds, services.clock)
        self.measure_cfg = load_measure_config(services.settings)
        self.tracks_cfg = load_vision_config().tracks
        self.stream = KeyframeStream(load_keyframe_config())
        self.queue: asyncio.Queue[asyncio.Task | None] = asyncio.Queue()
        self.consumer: asyncio.Task | None = None
        self.detections: list[asyncio.Task] = []
        self.segment_tasks: list[asyncio.Task] = []
        self.open: list[FrameSpines] = []
        self.carry: tuple[SegmentPlan, list[FrameSpines]] | None = None
        self.tails: list[tuple[SegmentPlan, list[FrameSpines]]] = []
        self.requeries = 0
        self.cut = False
        self.calls: set[asyncio.Future] = set()
        self.timings: dict[str, float] = defaultdict(float)
        self.done: dict[str, int] = defaultdict(int)
        self.computed: set[str] = set()
        self.deadline_at = float("inf")
        self.depth_task: asyncio.Future | None = None

    def add_still(self, still: Still) -> None:
        self.results.stills.stills.append(still)
        self.computed |= {STILLS, "keyframes"}
        started = time.perf_counter()
        keyframes = self.stream.add(still)
        self.timings["keyframes"] += time.perf_counter() - started
        for keyframe in keyframes:
            self.add_keyframe(keyframe)

    def add_keyframe(self, still: Still) -> None:
        self.results.keyframes.keyframes.append(still)
        self.queue_detection(still)

    def queue_detection(self, still: Still) -> None:
        self.computed.add("spines")
        if self.consumer is None:
            self.consumer = asyncio.create_task(self.consume())
        task = asyncio.create_task(self.timed("spines", self.detect(still)))
        self.detections.append(task)
        self.queue.put_nowait(task)

    async def timed(self, stage: str, step: Coroutine) -> Any:
        started = time.perf_counter()
        try:
            return await step
        finally:
            self.timings[stage] += time.perf_counter() - started

    async def consume(self) -> None:
        while (task := await self.queue.get()) is not None:
            try:
                await self.place(await task)
            except Exception as error:
                logger.error("frame placement failed: %s", type(error).__name__, exc_info=error)
        await self.close_segment()

    async def follows(self, frames: list[FrameSpines], frame: FrameSpines) -> bool:
        return await asyncio.to_thread(continues, [f.still for f in frames], frame.still, self.tracks_cfg)

    async def place(self, frame: FrameSpines) -> None:
        self.results.spines.frames.append(frame)
        if not frame.rows:
            return
        if self.open and len(self.open) < self.cfg.segment_max_frames and await self.follows(self.open, frame):
            self.open.append(frame)
            return
        await self.close_segment()
        self.open = [frame]
        self.carry = await self.find_carry(frame)

    async def find_carry(self, frame: FrameSpines) -> tuple[SegmentPlan, list[FrameSpines]] | None:
        for plan, tail in list(reversed(self.tails))[: self.cfg.revisit_tails]:
            if await self.follows(tail, frame):
                return plan, tail
        return None

    async def close_segment(self) -> None:
        if not self.open:
            return
        unit = await asyncio.to_thread(marker_unit, [f.still for f in self.open], self.measure_cfg.scale)
        self.start_segment(self.plan_open(unit), SEGMENT_STAGES[0])

    def plan_open(self, unit: int | None) -> SegmentPlan:
        frames, self.open = self.open, []
        source, carried = self.carry or (None, [])
        self.carry = None
        index = len(self.results.spines.segments)
        unit = unit if unit is not None or source is None else source.unit
        label = source.label if source is not None else self.label(unit, index)
        plan = SegmentPlan(index=index, label=label, unit=unit, seqs=[f.still.seq for f in frames],
                           carry=[f.still.seq for f in carried])
        self.results.spines.segments.append(plan)
        self.tails.append((plan, (carried + frames)[-self.tracks_cfg.anchor_lookback :]))
        return plan

    def label(self, unit: int | None, index: int) -> str:
        if unit is None:
            return f"S{index + 1}"
        return f"U{unit}-{1 + len({plan.label for plan in self.results.spines.segments if plan.unit == unit})}"

    def start_segment(self, plan: SegmentPlan, start: str) -> None:
        self.segment_tasks.append(asyncio.create_task(self.process_segment(plan, start)))

    def is_row(self, boxes: list[Box], still: Still) -> bool:
        return is_row(boxes, still, self.cfg.row_min_spines, self.cfg.row_min_spine_height_frac)

    async def detect(self, still: Still) -> FrameSpines:
        image = await asyncio.to_thread(read_image, still.path)
        local = await asyncio.to_thread(self.services.local_boxes, image)
        if local is not None and not self.is_row(local, still):
            return FrameSpines(still=still, rows=False, local=local)
        if self.cut:
            return self.local_only(still, local, VISION_CUTOFF)
        call = asyncio.ensure_future(self.detect_with(still, local, self.services.settings.vision_models))
        self.calls.add(call)
        call.add_done_callback(self.calls.discard)
        try:
            frame = await call
        except asyncio.CancelledError:
            if not self.cut or asyncio.current_task().cancelling():
                raise
            return self.local_only(still, local, VISION_CUTOFF)
        if frame.flagged and not self.cut:
            await self.notify(self.hooks.recapture, f"frame {still.seq}")
            frame = await self.requery(frame)
        return frame

    async def detect_with(self, still: Still, local: list[Box] | None, models: list[str]) -> FrameSpines:
        try:
            gemini = await detect_spines(still, self.llm, models)
        except VISION_FAILURES as error:
            return self.local_only(still, local, type(error).__name__)
        key = str(still.seq)
        result = reconcile({key: gemini}, {key: local})
        dets = result.rows[key]
        return FrameSpines(still=still, rows=self.is_row([d.box_px for d in dets], still), gemini=gemini,
                           local=local, dets=dets, notes=result.review_entries, flagged=bool(result.requery_rows))

    def local_only(self, still: Still, local: list[Box] | None, failure: str) -> FrameSpines:
        cfg = load_vision_config().reconcile
        dets = [SpineDet(box_px=box, orientation=local_orientation(box, cfg), source=Source.rfdetr) for box in local or []]
        return FrameSpines(still=still, rows=self.is_row(local or [], still), local=local, dets=dets, failure=failure)

    async def requery(self, frame: FrameSpines) -> FrameSpines:
        model = self.services.settings.tiebreak_model
        if not model or self.requeries >= load_vision_config().reconcile.max_requery_rows:
            return frame
        self.requeries += 1
        again = await self.detect_with(frame.still, frame.local, [model])
        if again.failure:
            return frame
        notes = frame.notes + [note for note in again.notes if note not in frame.notes]
        return again.model_copy(update={"notes": notes, "requeried": True})

    async def notify(self, hook: Callable[..., Awaitable[None]] | None, *args: Any) -> None:
        if hook is None:
            return
        try:
            await hook(*args)
        except Exception as error:
            logger.warning("live hook failed: %s", type(error).__name__)

    async def process_segment(self, plan: SegmentPlan, start: str) -> None:
        steps = {"tracks": self.track_segment, "scale": self.scale_segment, "reading": self.read_segment,
                 "resolution": self.resolve_segment}
        for stage in SEGMENT_STAGES[SEGMENT_STAGES.index(start):]:
            self.computed.add(stage)
            await self.timed(stage, steps[stage](plan))
            self.done[stage] += 1
            await self.progress(stage, self.done[stage], len(self.results.spines.segments))
        await self.prefetch(plan)

    def frames_of(self, seqs: list[int]) -> list[FrameSpines]:
        by_seq = {frame.still.seq: frame for frame in self.results.spines.frames}
        return [by_seq[seq] for seq in seqs]

    async def track_segment(self, plan: SegmentPlan) -> None:
        carried, frames = self.frames_of(plan.carry), self.frames_of(plan.seqs)
        segment = await asyncio.to_thread(build_segment, plan, carried, frames, self.tracks_cfg)
        self.results.tracks.segments[plan.index] = segment
        await self.upsert_books(segment.books)

    async def scale_segment(self, plan: SegmentPlan) -> None:
        segment = self.results.tracks.segments[plan.index]
        dims, planes = await asyncio.to_thread(self.measure_segment, plan, segment)
        self.results.scale.dims.update(dims)
        self.results.scale.planes.update(planes)

    def measure_segment(self, plan: SegmentPlan, segment: SegmentTracks) -> tuple[dict[str, SpineDims], dict[str, Box]]:
        unit = None if plan.unit is None else unit_scale(plan.unit, segment.stills, self.measure_cfg)
        by_track = measure_rows(segment.tracks, segment.stills, segment.rows, segment.flat, self.measure_cfg, unit)
        planes = plane_boxes({book.id: (book.frame, book.box) for book in segment.books}, self.measure_cfg, unit)
        return {book.id: by_track[book.track] for book in segment.books}, planes

    async def read_segment(self, plan: SegmentPlan) -> None:
        books = self.results.tracks.segments[plan.index].books
        crops = await asyncio.to_thread(spine_crops, books)
        try:
            readings = await read_spines(crops, self.llm, self.services.settings.vision_models)
        except VISION_FAILURES as error:
            readings = [Reading(track_id=c.track_id, rotation=c.rotation, failure=type(error).__name__) for c in crops]
        by_id = {reading.track_id: reading for reading in readings}
        for index, book in enumerate(books):
            crop = crops[index]
            self.results.reading.reads[book.id] = BookRead(reading=by_id[index], ocr_text=crop.ocr_text,
                                                           ocr_score=crop.ocr_score)

    async def resolve_segment(self, plan: SegmentPlan) -> None:
        books = self.results.tracks.segments[plan.index].books
        for book in books:
            read = self.results.reading.reads.get(book.id)
            if read is not None:
                self.results.resolution.resolutions[book.id] = await self.resolve_book(read, self.results.scale.dims.get(book.id))
        await self.upsert_books(books)
        await self.log_segment(plan, books)

    async def resolve_book(self, read: BookRead, dims: SpineDims | None) -> Resolution:
        spine = None if dims is None else ResolveDims(height_cm=dims.height_cm, thickness_cm=dims.thickness_cm)
        try:
            return await resolve(read.reading, (read.ocr_text, read.ocr_score), spine, self.services.catalogs)
        except CatalogOffline:
            return Resolution(status="unidentified", reason="catalogue_unavailable", vlm_text=read_text(read.reading),
                              ocr_text=read.ocr_text, dims=spine)

    async def prefetch(self, plan: SegmentPlan) -> None:
        locale = self.locale()
        market, other, fx = self.market(locale), self.market(self.other_locale(locale)), Fx(self.services.http)
        for book in self.results.tracks.segments[plan.index].books:
            resolution = self.results.resolution.resolutions.get(book.id)
            read = self.results.reading.reads.get(book.id)
            if resolution is not None and resolution.status == "identified" and read is not None:
                await price_book(resolution, market, reading=read.reading, statements=[], other=other, fx=fx)

    def upsert_event(self, book: BookSlot) -> dict[str, Any]:
        resolution = self.results.resolution.resolutions.get(book.id)
        identified = resolution is not None and resolution.status == "identified"
        return {
            "type": "inventory.upsert",
            "item": {
                "id": book.id,
                "kind": "book" if identified else "unidentified_book",
                "label": resolution.title if identified else None,
                "shelf": book.shelf,
                "status": "identified" if identified else "detected",
                "confidence": resolution.id_confidence if identified else None,
                "frame_ref": frame_ref(self.sweep, book.frame.path),
            },
        }

    async def upsert_books(self, books: list[BookSlot]) -> None:
        for book in books:
            await self.notify(self.hooks.emit, self.upsert_event(book))

    async def log_segment(self, plan: SegmentPlan, books: list[BookSlot]) -> None:
        resolutions = self.results.resolution.resolutions
        identified = sum(1 for b in books if b.id in resolutions and resolutions[b.id].status == "identified")
        event = {"logged": {"shelf": plan.label, "books": len(books), "identified": identified,
                            "unreadable": len(books) - identified}}
        await self.notify(self.hooks.inject, event, Scheduling.SILENT)

    async def progress(self, stage: str, done: int, total: int) -> None:
        await self.notify(self.hooks.emit, {"type": "pipeline.progress", "stage": stage, "done": done, "total": total})

    def statements(self) -> list[Statement]:
        return read_or(self.sweep, STATEMENTS_STAGE, StatementLog, StatementLog()).items

    def exclusions(self) -> list[Exclusion]:
        return read_or(self.sweep, EXCLUSIONS_STAGE, ExclusionLog, ExclusionLog()).items

    def locale(self) -> Locale:
        choice = read_or(self.sweep, LOCALE_STAGE, LocaleChoice, None)
        return load_locale(choice.country if choice else self.services.settings.default_country)

    def other_locale(self, locale: Locale) -> Locale:
        settings = self.services.settings
        return load_locale(settings.second_country if settings.second_country != locale.country else settings.default_country)

    def market(self, locale: Locale) -> Market:
        names = dict.fromkeys(locale.new_sources + locale.used_sources)
        return Market(locale, {name: SOURCES[name](self.services.http) for name in names if name in SOURCES})

    def room_stills(self) -> list[Still]:
        return [frame.still for frame in self.results.spines.frames if not frame.rows]

    async def depths(self) -> DepthSet:
        if self.depth_task is None:
            self.depth_task = asyncio.ensure_future(asyncio.to_thread(
                scaled_depths, self.room_stills(), self.services.depth_fn, self.services.settings.aruco_marker_size_cm
            ))
        return await self.depth_task

    async def detect_room_items(self) -> None:
        depths = await self.depths()
        try:
            dets = await detect_items(self.room_stills(), self.llm, self.services.settings.vision_models,
                                      self.statements(), depths)
        except VISION_FAILURES as error:
            self.results.items = ItemsStage(failure=type(error).__name__)
            return
        self.results.items = ItemsStage(items=dets)

    def unit_rows(self) -> dict[int, list[list[SpineDims]]]:
        rows: dict[int, list[list[SpineDims]]] = defaultdict(list)
        dims = self.results.scale.dims
        for plan in self.results.spines.segments:
            segment = self.results.tracks.segments.get(plan.index)
            if plan.unit is None or segment is None:
                continue
            ids = {book.track: book.id for book in segment.books}
            for row in segment.rows:
                rows[plan.unit].append([dims[ids[t]] for t in row if t not in segment.flat and ids[t] in dims])
        return rows

    async def measure_room(self) -> None:
        depths = await self.depths()
        shelves = shelf_units(self.unit_rows())
        self.results.room = await asyncio.to_thread(room_from_depth, self.room_stills(), shelves, depths)

    async def appraise(self) -> None:
        books = list(slots(self.results))
        attached = attach_statements(books, self.results.resolution.resolutions, self.statements(),
                                     self.cfg.statement_subject_match)
        window = load_live_config().statement_window_ms
        plans = {plan.index: plan for plan in self.results.spines.segments}
        exclusions = self.exclusions()
        appraisals = {}
        for book in books:
            segment = self.results.tracks.segments[book.segment]
            labels = {book.shelf.casefold(), plans[book.segment].label.casefold()}
            appraisals[book.id] = BookAppraisal(statements=attached[book.id],
                                                excluded=excluded_by(labels, segment.stills, exclusions, window))
        self.results.appraisal = AppraisalStage(books=appraisals)

    async def price_slot(self, book: BookSlot, market: Market, other: Market, fx: Fx) -> BookPrices | None:
        resolution = self.results.resolution.resolutions.get(book.id)
        if resolution is None:
            return None
        appraisal = self.results.appraisal.books.get(book.id, BookAppraisal())
        if appraisal.excluded:
            return BookPrices(status="identified" if resolution.status == "identified" else "unidentified")
        read = self.results.reading.reads.get(book.id)
        return await price_book(resolution, market, reading=read.reading if read else Reading(track_id=book.track),
                                statements=appraisal.statements, other=other, fx=fx)

    async def price_det(self, index: int, det: ItemDet, locale: Locale, sources: ItemSources | None,
                        listings: Listings) -> PricedItem:
        listing = listing_category(det, sources)
        item = Item(id=item_key(index), category=listing or det.category,
                    status="needs_appraisal" if det.needs_appraisal else "range")
        if not det.needs_appraisal and not listing:
            return PricedItem(price=ItemPrice(status="needs_appraisal", review=[NO_LISTING]))
        try:
            return PricedItem(listing=listing, price=await price_item(item, locale, listings=listings))
        except CatalogOffline:
            return PricedItem(listing=listing, price=ItemPrice(status="needs_appraisal", review=[OFFLINE_LISTING]))

    async def price_all(self) -> None:
        locale = self.locale()
        market, other, fx = self.market(locale), self.market(self.other_locale(locale)), Fx(self.services.http)
        pricing = self.results.pricing
        for book in slots(self.results):
            prices = await self.price_slot(book, market, other, fx)
            if prices is not None:
                pricing.books[book.id] = prices
        sources, listings = item_sources(locale.country), Listings(self.services.http)
        for index, det in enumerate(self.results.items.items if self.results.items else []):
            pricing.items[item_key(index)] = await self.price_det(index, det, locale, sources, listings)
        provisional = ClaimPacket(sweep=SweepInfo(currency=locale.currency), books=book_lines(self.results, self.sweep, self.cfg))
        pricing.second_country = await second_country(provisional, market=other, other=market, fx=fx)

    async def run_stage(self, name: str, step: Callable[[], Coroutine]) -> None:
        self.computed.add(name)
        await self.timed(name, step())
        await self.progress(name, 1, 1)

    async def wait_until(self, limit: float, tasks: list[asyncio.Future]) -> set[asyncio.Future]:
        pending = {task for task in tasks if not task.done()}
        while pending and (remaining := limit - self.services.clock()) > 0:
            _, pending = await asyncio.wait(pending, timeout=min(remaining, self.cfg.deadline_poll_s))
        return pending

    def cut_vision(self) -> None:
        self.cut = True
        for call in list(self.calls):
            call.cancel()

    async def until_deadline(self, steps: list[Awaitable]) -> None:
        tasks = [asyncio.ensure_future(step) for step in steps]
        pending = await self.wait_until(self.deadline_at, tasks)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in tasks:
            if not task.cancelled() and task.exception() is not None:
                logger.error("pipeline step failed: %s", type(task.exception()).__name__, exc_info=task.exception())

    def cancel_detections(self) -> None:
        for task in self.detections:
            task.cancel()

    async def finish(self, post: str = POST_STAGES[0]) -> ClaimPacket:
        started = time.perf_counter()
        now = self.services.clock()
        self.deadline_at = now + self.cfg.deadline_s
        for keyframe in self.stream.flush():
            self.add_keyframe(keyframe)
        if self.consumer is not None:
            self.queue.put_nowait(None)
            await self.wait_until(now + self.cfg.vision_cutoff_s, [self.consumer])
            self.cut_vision()
            await self.until_deadline([self.consumer])
        self.cancel_detections()
        if self.open:
            self.plan_open(None)
        todo = POST_STAGES[POST_STAGES.index(post):]
        items = [self.run_stage("items", self.detect_room_items)] if "items" in todo else []
        await self.until_deadline([*self.segment_tasks, *items])
        if "room" in todo:
            await self.until_deadline([self.run_stage("room", self.measure_room)])
        if "appraisal" in todo:
            await self.run_stage("appraisal", self.appraise)
        if "pricing" in todo:
            await self.until_deadline([self.run_stage("pricing", self.price_all)])
        packet = await self.publish(started)
        if self.close_services:
            await self.services.aclose()
        return packet

    def latency(self, started: float) -> dict[str, float]:
        recorded = {entry.stage: entry.seconds for entry in self.sweep.ledger.entries if entry.stage in STAGES}
        measured = {stage: self.timings[stage] for stage in STAGES if stage in self.computed}
        return {stage: round(seconds, 3) for stage, seconds in (recorded | measured).items()} | {
            "after_sweep": round(time.perf_counter() - started, 3)
        }

    def write_stages(self) -> None:
        for name, cls in STAGE_TYPES.items():
            model = getattr(self.results, name)
            if name in self.computed and model is not None:
                self.sweep.write_stage(name, model)

    async def publish(self, started: float) -> ClaimPacket:
        self.computed.add("packet")
        self.write_stages()
        packet = build_packet(self.results, self.sweep, self.locale(), self.cfg, self.latency(started))
        self.sweep.write_stage("packet", packet)
        (self.sweep.root / self.cfg.packet_name).write_text(packet.model_dump_json(indent=2), encoding="utf-8")
        render_report(packet, self.sweep)
        for stage in STAGES:
            if stage in self.computed:
                self.sweep.ledger.add(stage, packet.latency_s.get(stage, 0.0), 0, 0.0)
        await self.progress("packet", 1, 1)
        await self.announce(packet)
        return packet

    async def announce(self, packet: ClaimPacket) -> None:
        text = summary_text(packet, self.cfg)
        base = f"{self.cfg.runs_url}/{self.sweep.root.name}"
        for book in packet.books:
            await self.notify(self.hooks.emit, {"type": "inventory.upsert", "item": {
                "id": book.id, "kind": "book" if book.title else "unidentified_book", "label": book.title or None,
                "shelf": book.shelf, "status": book.status, "confidence": book.id_confidence, "frame_ref": book.frame_ref}})
        for item in packet.items:
            await self.notify(self.hooks.emit, {"type": "inventory.upsert", "item": {
                "id": item.id, "kind": "item", "label": item.description or item.category, "shelf": "",
                "status": item.status, "confidence": item.confidence, "frame_ref": item.frame_ref}})
        await self.notify(self.hooks.inject, {"readback": text, "instruction": self.cfg.readback.instruction},
                          Scheduling.WHEN_IDLE)
        await self.notify(self.hooks.emit, {"type": "packet_ready", "json": f"{base}/{self.cfg.packet_name}",
                                            "report": f"{base}/{REPORT_NAME}", "summary": text})

    def load(self, names: list[str]) -> None:
        for name in names:
            default = None if name in OPTIONAL_STAGES else STAGE_TYPES[name]()
            setattr(self.results, name, read_or(self.sweep, name, STAGE_TYPES[name], default))

    async def replay(self, from_stage: str | None = None) -> ClaimPacket:
        stage = self.cfg.stage_aliases.get(from_stage, from_stage) if from_stage else STAGES[0]
        if stage not in STAGES:
            raise ValueError(f"unknown stage {from_stage}; choose one of {STAGES}")
        stills = load_stills(self.sweep)
        self.load(STAGES[: STAGES.index(stage)])
        if stage == STAGES[0]:
            for still in stills:
                self.add_still(still)
            return await self.finish()
        self.results.stills = StillLog(stills=stills)
        if stage == "spines":
            for keyframe in self.results.keyframes.keyframes:
                self.queue_detection(keyframe)
        elif stage in SEGMENT_STAGES:
            for plan in self.results.spines.segments:
                self.start_segment(plan, stage)
        return await self.finish(stage if stage in POST_STAGES else POST_STAGES[0])



def sweep_pipeline(settings: Settings, budget: Budget) -> Callable[[SweepDir, LiveHooks], SweepRun]:
    def make(sweep: SweepDir, hooks: LiveHooks) -> SweepRun:
        return SweepRun(sweep, services_for(sweep, settings, budget), hooks, close_services=True)

    return make


async def run_pipeline(sweep: SweepDir, from_stage: str | None = None, *, services: Services | None = None,
                       hooks: LiveHooks | None = None, config: PipelineConfig | None = None) -> ClaimPacket:
    owned = services is None
    run = SweepRun(sweep, services or services_for(sweep, load_settings()), hooks, config, close_services=owned)
    return await run.replay(from_stage)


def main(argv: list[str] | None = None) -> None:
    cfg = load_pipeline_config()
    parser = argparse.ArgumentParser(prog="python -m libclaim.pipeline")
    parser.add_argument("sweep", type=Path)
    parser.add_argument("--from", dest="from_stage", choices=STAGES + sorted(cfg.stage_aliases))
    args = parser.parse_args(argv)
    packet = asyncio.run(run_pipeline(SweepDir(args.sweep), args.from_stage))
    print(args.sweep / cfg.packet_name, flush=True)
    print(summary_text(packet, cfg), flush=True)


if __name__ == "__main__":
    main()
