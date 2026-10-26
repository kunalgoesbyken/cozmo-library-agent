import logging
from collections import defaultdict
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

from libclaim.packet.models import (
    Book,
    BookReplacementCost,
    ClaimPacket,
    Costs,
    Item,
    ItemReplacementCost,
    ReviewEntry,
    Room,
    SweepInfo,
    UsedValue,
)
from libclaim.packet.review import build_review_queue
from libclaim.packet.totals import compute_totals
from libclaim.pricing.appraisal import appraisal_reasons
from libclaim.vision.reading import Reading
from libclaim.settings import Locale
from libclaim.stages import BookSlot, PipelineConfig, Results, item_key
from libclaim.sweep import Ledger, SweepDir
from libclaim.vision.boxes import Box, iou, load_vision_config
from libclaim.vision.resolve import Resolution
from libclaim.vision.scale import ScaleMethod, SpineDims, cross_check
from libclaim.vision.spines import ReviewEntry as ReconcileNote

MS_PER_S = 1000
T = TypeVar("T")
logger = logging.getLogger(__name__)
UNFINISHED = "{stage} did not finish before the packet was finalised"


def frame_ref(sweep: SweepDir, path: str) -> str:
    return (sweep.frames / Path(path).name).relative_to(sweep.root).as_posix() if path else ""


def revisits(results: Results) -> dict[str, str]:
    threshold = load_vision_config().tracks.track_iou
    placed: dict[int, list[tuple[int, str, Box]]] = defaultdict(list)
    found: dict[str, str] = {}
    for plan in results.spines.segments:
        segment = results.tracks.segments.get(plan.index)
        if plan.unit is None or segment is None:
            continue
        for book in segment.books:
            plane = results.scale.planes.get(book.id)
            if plane is None:
                continue
            earlier = next((key for index, key, box in placed[plan.unit]
                            if index != plan.index and iou(box, plane) >= threshold), None)
            if earlier is None:
                placed[plan.unit].append((plan.index, book.id, plane))
            else:
                found[book.id] = earlier
    return found


def slots(results: Results) -> Iterator[BookSlot]:
    repeated = revisits(results)
    for index in sorted(results.tracks.segments):
        yield from (book for book in results.tracks.segments[index].books if book.id not in repeated)


def positions(results: Results) -> dict[str, int]:
    counters: dict[str, int] = defaultdict(int)
    numbered = {}
    for book in slots(results):
        counters[book.shelf] += 1
        numbered[book.id] = counters[book.shelf]
    return numbered


def book_status(slot: BookSlot, results: Results, locale: Locale) -> str:
    resolution = results.resolution.resolutions.get(slot.id)
    prices = results.pricing.books.get(slot.id)
    if prices is not None:
        return prices.status
    if resolution is None or resolution.status != "identified":
        return "unidentified"
    read = results.reading.reads.get(slot.id)
    appraisal = results.appraisal.books.get(slot.id)
    reasons = appraisal_reasons(resolution, read.reading if read else Reading(track_id=slot.track),
                                appraisal.statements if appraisal else [], [], locale)
    return "needs_appraisal" if reasons else "identified"


def book_line(slot: BookSlot, position: int, results: Results, sweep: SweepDir, cfg: PipelineConfig,
              locale: Locale) -> Book:
    dims = results.scale.dims.get(slot.id)
    resolution = results.resolution.resolutions.get(slot.id)
    prices = results.pricing.books.get(slot.id)
    status = book_status(slot, results, locale)
    named = resolution if status != "unidentified" and resolution is not None else Resolution(status="unidentified")
    evidence = (resolution.source_urls if resolution else []) + (prices.evidence if prices else [])
    return Book(
        id=slot.id,
        shelf=slot.shelf,
        position=position,
        frame_ref=frame_ref(sweep, slot.frame.path),
        status=status,
        title=named.title,
        author=named.author,
        edition=named.edition,
        isbn=named.isbn,
        spine_height_cm=round(dims.height_cm, cfg.cm_decimals) if dims else None,
        spine_thickness_cm=round(dims.thickness_cm, cfg.cm_decimals) if dims else None,
        scale_method=dims.scale_method.value if dims else "",
        id_confidence=named.id_confidence if named.status == "identified" else None,
        replacement_cost=prices.replacement_cost if prices else BookReplacementCost(),
        used_value=prices.used_value if prices else UsedValue(),
        evidence=list(dict.fromkeys(evidence)),
    )


def item_lines(results: Results, sweep: SweepDir) -> list[Item]:
    lines = []
    for index, det in enumerate(results.items.items if results.items else []):
        priced = results.pricing.items.get(item_key(index))
        price = priced.price if priced and not det.needs_appraisal else None
        lines.append(
            Item(
                id=item_key(index),
                category=det.category,
                description=det.description,
                brand_model=det.brand_model,
                frame_ref=frame_ref(sweep, det.frame_path),
                dimensions_cm=det.dimensions_cm,
                dimensions_method=det.dimensions_method,
                status=price.status if price else "needs_appraisal",
                replacement_cost=price.replacement_cost if price else ItemReplacementCost(),
                evidence=[row.url for row in priced.price.evidence] if priced else [],
            )
        )
    return lines


def room_of(results: Results, sweep: SweepDir) -> Room:
    if results.room is None:
        return Room()
    refs = list(dict.fromkeys(frame_ref(sweep, path) for path in results.room.frame_refs))
    return Room.model_validate(results.room.model_dump(exclude={"review", "frame_refs"}) | {"frame_refs": refs})


def sweep_info(results: Results, sweep: SweepDir, locale: Locale, device: str) -> SweepInfo:
    times = sorted(still.t_ms for still in results.stills.stills)
    return SweepInfo(
        id=sweep.root.name,
        captured_at=datetime.fromtimestamp(times[0] / MS_PER_S, UTC).isoformat() if times else "",
        duration_s=(times[-1] - times[0]) / MS_PER_S if times else None,
        device=device,
        country=locale.country,
        currency=locale.currency,
    )


def costs_of(ledger: Ledger, free_tier: bool | None, cfg: PipelineConfig) -> Costs:
    by_stage: dict[str, float] = defaultdict(float)
    for entry in ledger.entries:
        if entry.tokens:
            by_stage[entry.stage] += entry.usd_at_paid_rates
    paid = ledger.total_usd()
    billing = cfg.billing.unknown if free_tier is None else cfg.billing.free if free_tier else cfg.billing.paid
    return Costs(
        usd_at_paid_rates=paid,
        billed_usd=None if free_tier is None else 0.0 if free_tier else paid,
        billing=billing,
        tokens=sum(entry.tokens for entry in ledger.entries),
        by_stage_usd=dict(by_stage),
    )


class Review:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.entries: list[ReviewEntry] = []

    def add(self, kind: str, ref_id: str, reason: str, line_ref: str, frame: str = "") -> None:
        entry = ReviewEntry(ref_id=ref_id, severity=self.cfg.severity[kind], reason=reason, line_ref=line_ref,
                            frame_ref=frame)
        if all((e.ref_id, e.reason) != (entry.ref_id, entry.reason) for e in self.entries):
            self.entries.append(entry)

    def unfinished(self, ref_id: str, stage: str, line_ref: str, frame: str = "") -> None:
        self.add("unfinished", ref_id, UNFINISHED.format(stage=stage), line_ref, frame)


def scale_review(review: Review, book: Book, line: str, dims: SpineDims, resolution: Resolution | None) -> None:
    kind = "scale_prior" if dims.scale_method == ScaleMethod.book_format_prior else "scale"
    for reason in dims.review:
        review.add(kind, book.id, reason, line, book.frame_ref)
    if resolution is not None and resolution.status == "identified" and resolution.physical_dimensions:
        for reason in cross_check(dims, resolution.physical_dimensions):
            review.add("cross_check", book.id, reason, line, book.frame_ref)


def book_review(review: Review, packet: ClaimPacket, results: Results) -> None:
    for index, (book, slot) in enumerate(zip(packet.books, slots(results))):
        line = f"books[{index}]"
        dims = results.scale.dims.get(slot.id)
        resolution = results.resolution.resolutions.get(slot.id)
        prices = results.pricing.books.get(slot.id)
        excluded = results.appraisal.books.get(slot.id)
        if dims is None:
            review.unfinished(book.id, "scale", line, book.frame_ref)
        else:
            scale_review(review, book, line, dims, resolution)
        if slot.id not in results.reading.reads:
            review.unfinished(book.id, "reading", line, book.frame_ref)
        if resolution is None:
            review.unfinished(book.id, "resolution", line, book.frame_ref)
        elif resolution.status != "identified":
            review.add("identification", book.id, f"identification: {resolution.reason}", line, book.frame_ref)
        if excluded is not None and excluded.excluded:
            review.add("excluded", book.id, f"excluded by the user: {excluded.excluded}", line, book.frame_ref)
        if prices is None:
            review.unfinished(book.id, "pricing", line, book.frame_ref)
        else:
            for reason in prices.review:
                review.add("pricing", book.id, reason, line, book.frame_ref)


def frame_label(results: Results, seq: int) -> str:
    plan = next((p for p in results.spines.segments if seq in p.seqs), None)
    return plan.label if plan else f"frame {seq}"


def note_reason(note: ReconcileNote, requeried: bool) -> str:
    reason = f"{note.reason}: {note.detail}"
    return f"{reason} (re-queried)" if requeried else reason


def frame_review(review: Review, results: Results, sweep: SweepDir) -> None:
    detected = {frame.still.seq for frame in results.spines.frames}
    for frame in results.spines.frames:
        ref = frame_label(results, frame.still.seq)
        line = frame_ref(sweep, frame.still.path)
        if frame.failure:
            counted = "local detector boxes used" if frame.dets else "books in this frame are not counted"
            review.add("failure", ref, f"spine detection failed ({frame.failure}); {counted}", "frames", line)
        for note in frame.notes:
            review.add("reconcile", note.row if note.row == "sweep" else ref, note_reason(note, frame.requeried),
                       "frames", "" if note.row == "sweep" else line)
    for still in results.keyframes.keyframes:
        if still.seq not in detected:
            review.unfinished(f"frame {still.seq}", "spines", "frames", frame_ref(sweep, still.path))
    for plan in results.spines.segments:
        if plan.index not in results.tracks.segments:
            review.unfinished(plan.label, "tracks", "frames")
    planned = {seq for plan in results.spines.segments for seq in plan.seqs}
    for frame in results.spines.frames:
        if frame.rows and frame.still.seq not in planned:
            review.unfinished(f"frame {frame.still.seq}", "segments", "frames", frame_ref(sweep, frame.still.path))
    for key, earlier in revisits(results).items():
        review.add("revisit", key, f"already counted as {earlier} (same place on the unit marker plane)", "tracks")


def room_review(review: Review, results: Results) -> None:
    if results.room is None:
        review.unfinished("room", "room", "room")
        return
    for reason in results.room.review:
        review.add("room", "room", reason, "room")


def item_review(review: Review, packet: ClaimPacket, results: Results) -> None:
    if results.items is None:
        review.unfinished("items", "items", "items")
        return
    if results.items.failure:
        review.add("failure", "items", f"item detection failed ({results.items.failure})", "items")
    for index, (item, det) in enumerate(zip(packet.items, results.items.items)):
        line = f"items[{index}]"
        priced = results.pricing.items.get(item.id)
        for reason in det.review + (priced.price.review if priced else []):
            review.add("items", item.id, reason, line, item.frame_ref)
        if priced is None and not det.needs_appraisal:
            review.unfinished(item.id, "pricing", line, item.frame_ref)


def statement_review(review: Review, results: Results) -> None:
    for reason in results.appraisal.review:
        review.add("statements", "books", reason, "books")
    for reason in results.items.review if results.items else []:
        review.add("statements", "items", reason, "items")


def failure_review(review: Review, results: Results) -> None:
    for failure in results.failures.items:
        review.add("failure", failure.ref, f"{failure.stage} failed ({failure.error})", failure.stage)


def pipeline_review(packet: ClaimPacket, results: Results, sweep: SweepDir, cfg: PipelineConfig) -> list[ReviewEntry]:
    review = Review(cfg)
    failure_review(review, results)
    book_review(review, packet, results)
    frame_review(review, results, sweep)
    room_review(review, results)
    item_review(review, packet, results)
    statement_review(review, results)
    return review.entries


def book_lines(results: Results, sweep: SweepDir, cfg: PipelineConfig, locale: Locale) -> list[Book]:
    numbered = positions(results)
    return [book_line(slot, numbered[slot.id], results, sweep, cfg, locale) for slot in slots(results)]


class Sections:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.failed: list[ReviewEntry] = []

    def build(self, name: str, make: Callable[[], T], empty: T) -> T:
        try:
            return make()
        except Exception as error:
            logger.error("packet section %s failed: %s", name, type(error).__name__, exc_info=error)
            reason = f"{name} failed while building the packet ({type(error).__name__})"
            self.failed.append(ReviewEntry(ref_id=name, severity=self.cfg.severity["failure"], reason=reason,
                                           line_ref=name))
            return empty


def build_packet(results: Results, sweep: SweepDir, locale: Locale, cfg: PipelineConfig,
                 latency_s: dict[str, float], free_tier: bool | None = None, device: str = "") -> ClaimPacket:
    sections = Sections(cfg)
    packet = ClaimPacket(
        sweep=sweep_info(results, sweep, locale, device),
        second_country=results.pricing.second_country,
        room=sections.build("room", lambda: room_of(results, sweep), Room()),
        books=sections.build("books", lambda: book_lines(results, sweep, cfg, locale), []),
        items=sections.build("items", lambda: item_lines(results, sweep), []),
        costs=costs_of(sweep.ledger, free_tier, cfg),
        latency_s=latency_s,
    )
    packet.totals = compute_totals(packet.books, packet.items)
    review = sections.build("review", lambda: pipeline_review(packet, results, sweep, cfg), [])
    packet.review_queue = sections.failed + build_review_queue(packet) + review
    return packet


def summary_text(packet: ClaimPacket, cfg: PipelineConfig) -> str:
    totals, room, text = packet.totals, packet.room, cfg.summary

    def money(value) -> str:
        return format(value, cfg.money_format)

    values = totals.model_dump() | {
        "currency": packet.sweep.currency,
        "books_replacement_cost": money(totals.books_replacement_cost),
        "books_used_value": money(totals.books_used_value),
        "items_low": money(totals.items_replacement_cost_low),
        "items_high": money(totals.items_replacement_cost_high),
        "item_count": len(packet.items),
        "review_count": len(packet.review_queue),
    }
    parts = [text.books.format(**values), text.book_value.format(**values), text.items.format(**values)]
    if room.floor_area_m2 is not None and room.wall_area_m2 is not None:
        areas = {"floor_area_m2": format(room.floor_area_m2, cfg.area_format),
                 "wall_area_m2": format(room.wall_area_m2, cfg.area_format)}
        parts.append(text.room.format(**areas))
    else:
        parts.append(text.no_room)
    parts.append(text.review.format(**values))
    return " ".join(parts)
