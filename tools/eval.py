import csv
import math
import sys
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, Field, ValidationError
from rapidfuzz import fuzz, utils

from libclaim.packet.models import Book, ClaimPacket, Item
from libclaim.settings import CONFIG_DIR
from libclaim.sweep import Ledger

PASS = "PASS"
FAIL = "FAIL"
NOT_EVALUATED = "NOT EVALUATED"


class AlignmentConfig(BaseModel):
    gap_cost: float
    unknown_cost: float


class TitleConfig(BaseModel):
    scorer: str
    threshold: float
    min_accuracy: float
    max_confidently_wrong: float


class ToleranceConfig(BaseModel):
    tolerance: float


class DimsConfig(BaseModel):
    tolerance: float
    min_share_within: float
    min_sample: int


class PriceConfig(BaseModel):
    tolerance: float
    min_share_within: float
    min_comparable: int
    new_conditions: list[str]
    observed_source_type: str
    source_scorer: str
    source_threshold: float


class ItemsConfig(BaseModel):
    scorer: str
    threshold: float
    min_found: float
    min_truth: int


class RoomConfig(BaseModel):
    floor_tolerance: float
    wall_tolerance: float


class TimeConfig(BaseModel):
    max_seconds: float
    elapsed_key: str
    streaming_stages: list[str]
    expected_stages: list[str]


class EvalConfig(BaseModel):
    shelf_label: str
    truthy: list[str]
    closure_tolerance_m: float
    default_interior_angle_deg: float
    alignment: AlignmentConfig
    title: TitleConfig
    count: ToleranceConfig
    dims: DimsConfig
    price: PriceConfig
    items: ItemsConfig
    room: RoomConfig
    time: TimeConfig


def load_eval_config() -> EvalConfig:
    return EvalConfig.model_validate(yaml.safe_load((CONFIG_DIR / "eval.yaml").read_text(encoding="utf-8")))


class GtBook(BaseModel):
    unit: str
    row: str
    position: int
    title: str
    author: str = ""
    legible: bool
    orientation: str = ""


class GtSpine(BaseModel):
    unit: str
    row: str
    position: int
    height_cm: float
    thickness_cm: float


class GtPrice(BaseModel):
    isbn: str
    title: str = ""
    source: str
    url: str = ""
    price: float
    currency: str
    checked_on: str = ""
    condition: str = ""
    screenshot: str = ""


class GtItem(BaseModel):
    name: str
    category: str
    w_cm: float | None = None
    h_cm: float | None = None
    d_cm: float | None = None
    brand: str = ""


class Wall(BaseModel):
    id: str = ""
    length_m: float = Field(gt=0)
    interior_angle_deg: float | None = None


class RoomOpening(BaseModel):
    kind: str = ""
    wall: str = ""
    width_m: float
    height_m: float


class ShelvingFront(BaseModel):
    unit: str = ""
    wall: str = ""
    width_m: float
    height_m: float


class RoomTruth(BaseModel):
    ceiling_height_m: float = Field(gt=0)
    walls: list[Wall] = Field(min_length=3)
    openings: list[RoomOpening] = Field(default_factory=list)
    shelving_fronts: list[ShelvingFront] = Field(default_factory=list)


@dataclass
class Section:
    title: str
    status: str
    metrics: dict = field(default_factory=dict)
    lines: list[str] = field(default_factory=list)
    reason: str = ""
    bar: bool = True


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [
            {key.strip(): (value or "").strip() for key, value in row.items() if key}
            for row in csv.DictReader(handle)
        ]


def gt_book(row: dict[str, str], cfg: EvalConfig) -> GtBook:
    book = GtBook(
        unit=row["unit"],
        row=row["row"],
        position=int(row["position"]),
        title=row["title"],
        author=row.get("author", ""),
        legible=row["legible"].casefold() in cfg.truthy,
        orientation=row.get("orientation", ""),
    )
    if book.legible and not book.title:
        raise ValueError(f"unit {book.unit} row {book.row} position {book.position} is legible but has no title")
    return book


def load_books(path: Path, cfg: EvalConfig) -> list[GtBook]:
    return [gt_book(row, cfg) for row in read_rows(path)]


def load_spines(path: Path) -> list[GtSpine]:
    return [GtSpine.model_validate({**row, "position": int(row["position"])}) for row in read_rows(path)]


def load_prices(path: Path) -> list[GtPrice]:
    return [GtPrice.model_validate(row) for row in read_rows(path)]


def load_items(path: Path) -> list[GtItem]:
    return [
        GtItem.model_validate({key: value for key, value in row.items() if value != ""})
        for row in read_rows(path)
    ]


def load_room(path: Path) -> RoomTruth:
    return RoomTruth.model_validate_json(path.read_text(encoding="utf-8"))


def floor_area_m2(room: RoomTruth, cfg: EvalConfig) -> float:
    x = y = heading = 0.0
    points = [(x, y)]
    for wall in room.walls:
        x += wall.length_m * math.cos(math.radians(heading))
        y += wall.length_m * math.sin(math.radians(heading))
        points.append((x, y))
        angle = wall.interior_angle_deg if wall.interior_angle_deg is not None else cfg.default_interior_angle_deg
        heading += 180.0 - angle
    gap = math.hypot(x, y)
    if gap > cfg.closure_tolerance_m:
        raise ValueError(f"the walls do not close: the end of the last wall is {gap:.2f} m from the start")
    twice = sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(points, points[1:]))
    return abs(twice) / 2


def gross_wall_area_m2(room: RoomTruth) -> float:
    return sum(wall.length_m for wall in room.walls) * room.ceiling_height_m


def openings_area_m2(room: RoomTruth) -> float:
    return sum(opening.width_m * opening.height_m for opening in room.openings)


def shelving_area_m2(room: RoomTruth) -> float:
    return sum(front.width_m * front.height_m for front in room.shelving_fronts)


def similarity(scorer: str, left: str, right: str) -> float:
    return getattr(fuzz, scorer)(left, right, processor=utils.default_process)


def shelf_label(cfg: EvalConfig, unit: str, row: str) -> str:
    return cfg.shelf_label.format(unit=unit, row=row)


def align(truth: list[str], packet: list[str], cfg: EvalConfig) -> list[tuple[int | None, int | None]]:
    gap = cfg.alignment.gap_cost

    def substitution(i: int, j: int) -> float:
        if not packet[j].strip() or not truth[i].strip():
            return cfg.alignment.unknown_cost
        return 1 - similarity(cfg.title.scorer, truth[i], packet[j]) / 100

    rows, cols = len(truth), len(packet)
    cost = [[0.0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        cost[i][0] = i * gap
    for j in range(1, cols + 1):
        cost[0][j] = j * gap
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            cost[i][j] = min(
                cost[i - 1][j - 1] + substitution(i - 1, j - 1),
                cost[i - 1][j] + gap,
                cost[i][j - 1] + gap,
            )
    pairs = []
    i, j = rows, cols
    while i or j:
        if i and j and math.isclose(cost[i][j], cost[i - 1][j - 1] + substitution(i - 1, j - 1)):
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and math.isclose(cost[i][j], cost[i - 1][j] + gap):
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    return pairs[::-1]


@dataclass
class BookMatch:
    gt: GtBook | None
    book: Book | None
    score: float | None = None

    def title_ok(self, cfg: EvalConfig) -> bool:
        return (
            self.book is not None
            and self.book.status == "identified"
            and self.score is not None
            and self.score >= cfg.title.threshold
        )


def match_books(truth: list[GtBook], packet: ClaimPacket, cfg: EvalConfig) -> list[BookMatch]:
    gt_rows: dict[str, list[GtBook]] = {}
    for book in sorted(truth, key=lambda b: (b.unit, b.row, b.position)):
        gt_rows.setdefault(shelf_label(cfg, book.unit, book.row), []).append(book)
    packet_rows: dict[str, list[Book]] = {}
    for book in sorted(
        packet.books, key=lambda b: (b.shelf, b.position if b.position is not None else math.inf)
    ):
        packet_rows.setdefault(book.shelf, []).append(book)
    matches = []
    for label in [*gt_rows, *(key for key in packet_rows if key not in gt_rows)]:
        gts, books = gt_rows.get(label, []), packet_rows.get(label, [])
        for i, j in align([g.title for g in gts], [b.title for b in books], cfg):
            gt = gts[i] if i is not None else None
            book = books[j] if j is not None else None
            score = (
                similarity(cfg.title.scorer, gt.title, book.title)
                if gt is not None and book is not None and gt.title.strip() and book.title.strip()
                else None
            )
            matches.append(BookMatch(gt, book, score))
    return matches


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def table(headers: list[str], rows: list[list]) -> list[str]:
    def cell(value) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(cell(v) for v in row) + " |" for row in rows),
    ]


def verdict(passed: bool) -> str:
    return PASS if passed else FAIL


@dataclass
class TruthError:
    filename: str
    reason: str


def missing(title: str, filename: str) -> Section:
    return Section(title, NOT_EVALUATED, reason=f"{filename} not found in the ground-truth directory")


def unavailable(title: str, filename: str, truth, bar: bool = True) -> Section | None:
    if truth is None:
        section = missing(title, filename)
    elif isinstance(truth, TruthError):
        section = Section(title, NOT_EVALUATED, reason=f"{filename} could not be used: {truth.reason}")
    elif isinstance(truth, list) and not truth:
        section = Section(title, NOT_EVALUATED, reason=f"{filename} has no rows")
    else:
        return None
    section.bar = bar
    return section


def relative_error(predicted: float, truth: float) -> float:
    if truth == 0:
        return 0.0 if predicted == 0 else math.inf
    return abs(predicted - truth) / abs(truth)


def error_stats(errors: list[float], tolerance: float, total: int | None = None) -> dict:
    total = len(errors) if total is None else total
    within = sum(e <= tolerance for e in errors)
    if not errors:
        return {"n": 0, "mean": None, "median": None, "p90": None, "share_within": 0.0 if total else None}
    return {
        "n": len(errors),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "p90": float(np.percentile(errors, 90)),
        "share_within": within / total,
    }


def fmt_stat(value: float | None) -> str:
    return "n/a" if value is None else pct(value)


def count_section(truth, matches: list[BookMatch], packet: ClaimPacket, cfg: EvalConfig) -> Section:
    title = "Book count"
    if (blocked := unavailable(title, "books.csv", truth)) is not None:
        return blocked
    true_count, counted = len(truth), len(packet.books)
    error = relative_error(counted, true_count)
    status = verdict(error <= cfg.count.tolerance)
    misses = sum(m.book is None for m in matches)
    insertions = sum(m.gt is None for m in matches)
    lines = table(
        ["True count", "Packet count", "Error", "Aligned misses", "Aligned insertions",
         f"Bar (within {pct(cfg.count.tolerance)})"],
        [[true_count, counted, pct(error), misses, insertions, status]],
    )
    wanted = {shelf_label(cfg, g.unit, g.row) for g in truth}
    found = {b.shelf for b in packet.books}
    if not wanted & found:
        lines += [
            "",
            "No packet shelf label matches any ground-truth shelf label, so no book could be paired. "
            f"Ground-truth labels: {', '.join(sorted(wanted))}. "
            f"Packet labels: {', '.join(sorted(found)) or 'none'}. "
            "Set shelf_label in config/eval.yaml or rename units and rows in books.csv.",
        ]
    return Section(title, status, {"true": true_count, "counted": counted, "error": error,
                                   "misses": misses, "insertions": insertions}, lines)


def title_sections(truth, matches: list[BookMatch], packet: ClaimPacket, cfg: EvalConfig) -> list[Section]:
    title = "Title identification"
    if (blocked := unavailable(title, "books.csv", truth)) is not None:
        return [blocked, replace(blocked, title="Confidently wrong books", bar=False)]
    legible = [m for m in matches if m.gt is not None and m.gt.legible]
    correct = [m for m in legible if m.title_ok(cfg)]
    identified = [b for b in packet.books if b.status == "identified"]
    wrong = [
        m
        for m in matches
        if m.book is not None
        and m.book.status == "identified"
        and (m.gt is None or (m.gt.legible and not m.title_ok(cfg)))
    ]
    accuracy = len(correct) / len(legible) if legible else None
    wrong_rate = len(wrong) / len(identified) if identified else 0.0
    if accuracy is None:
        return [
            Section(title, NOT_EVALUATED, reason="books.csv has no legible spines"),
            Section("Confidently wrong books", NOT_EVALUATED, reason="books.csv has no legible spines", bar=False),
        ]
    status = verdict(accuracy >= cfg.title.min_accuracy and wrong_rate <= cfg.title.max_confidently_wrong)
    main = Section(
        title,
        status,
        {
            "legible": len(legible),
            "correct": len(correct),
            "accuracy": accuracy,
            "identified": len(identified),
            "confidently_wrong": len(wrong),
            "wrong_rate": wrong_rate,
        },
        table(
            ["Measure", "Value", "Bar"],
            [
                ["Legible ground-truth spines", len(legible), ""],
                ["Correctly identified", f"{len(correct)} of {len(legible)}", ""],
                ["Accuracy", pct(accuracy), f">= {pct(cfg.title.min_accuracy)}"],
                ["Packet books with status identified", len(identified), ""],
                ["Confidently wrong", f"{len(wrong)} of {len(identified)}", ""],
                ["Confidently wrong rate", pct(wrong_rate), f"<= {pct(cfg.title.max_confidently_wrong)}"],
                ["Result", status, ""],
            ],
        ),
    )
    rows = [
        [
            m.book.id,
            f"{m.book.shelf} #{m.book.position}",
            m.gt.title if m.gt else "(no ground-truth book at this slot)",
            m.book.title,
            "n/a" if m.score is None else f"{m.score:.0f}",
            m.book.frame_ref,
        ]
        for m in wrong
    ]
    listing = Section(
        "Confidently wrong books",
        "REPORTED",
        {"count": len(rows)},
        table(["Book", "Slot", "Ground truth", "Packet title", "Match score", "Frame"], rows)
        if rows
        else ["No confidently wrong books."],
        bar=False,
    )
    return [main, listing]


def honesty_section(packet: ClaimPacket) -> Section:
    statuses = [b.status for b in packet.books]
    severities = [e.severity for e in packet.review_queue]
    rows = [
        ["Unidentified books", statuses.count("unidentified")],
        ["Books needing appraisal", statuses.count("needs_appraisal")],
        ["Review queue entries", len(packet.review_queue)],
        *([f"Review queue, {s}", severities.count(s)] for s in ("high", "medium", "low")),
    ]
    return Section(
        "Unidentified and review queue",
        "REPORTED",
        {"unidentified": statuses.count("unidentified"), "review": len(packet.review_queue)},
        table(["Measure", "Count"], rows),
        bar=False,
    )


def spine_key(unit: str, row: str, position: int | None, cfg: EvalConfig) -> tuple[str, int | None]:
    return shelf_label(cfg, unit, row), position


def duplicated(keys) -> list:
    return sorted(key for key, count in Counter(keys).items() if count > 1)


def slot_lookup(truth_books, matches: list[BookMatch], packet: ClaimPacket, cfg: EvalConfig):
    if isinstance(truth_books, list):
        slots: dict = {}
        for m in matches:
            if m.gt is not None:
                accepted = m.book if m.book is not None and m.score is not None and m.score >= cfg.title.threshold else None
                slots.setdefault(spine_key(m.gt.unit, m.gt.row, m.gt.position, cfg), []).append(accepted)
        return slots, duplicated(spine_key(g.unit, g.row, g.position, cfg) for g in truth_books)
    slots = {}
    for b in packet.books:
        slots.setdefault((b.shelf, b.position), []).append(b)
    return slots, duplicated((b.shelf, b.position) for b in packet.books)


def spines_section(truth, truth_books, matches: list[BookMatch], packet: ClaimPacket, cfg: EvalConfig) -> Section:
    title = "Spine dimensions"
    if (blocked := unavailable(title, "spines20.csv", truth)) is not None:
        return blocked
    if len(truth) < cfg.dims.min_sample:
        return Section(
            title,
            NOT_EVALUATED,
            reason=f"spines20.csv has {len(truth)} spines; the bar needs {cfg.dims.min_sample}",
        )
    slots, duplicate_slots = slot_lookup(truth_books, matches, packet, cfg)
    spine_keys = [spine_key(sp.unit, sp.row, sp.position, cfg) for sp in truth]
    duplicate_spines = duplicated(spine_keys)
    ambiguous = set(duplicate_slots) | set(duplicate_spines)
    errors: dict[str, list[float]] = {"height": [], "thickness": []}
    unmatched = 0
    for spine, key in zip(truth, spine_keys):
        candidates = slots.get(key, [])
        book = candidates[0] if len(candidates) == 1 and key not in ambiguous else None
        if book is None:
            unmatched += 1
            continue
        for name, predicted, actual in (
            ("height", book.spine_height_cm, spine.height_cm),
            ("thickness", book.spine_thickness_cm, spine.thickness_cm),
        ):
            if predicted is not None:
                errors[name].append(relative_error(predicted, actual))
    stats = {name: error_stats(values, cfg.dims.tolerance, len(truth)) for name, values in errors.items()}
    passed = all(
        s["n"] and s["mean"] <= cfg.dims.tolerance and s["share_within"] >= cfg.dims.min_share_within
        for s in stats.values()
    )
    status = verdict(bool(passed))
    rows = [
        [name, len(truth), s["n"], fmt_stat(s["mean"]), fmt_stat(s["median"]), fmt_stat(s["p90"]),
         f"{round(s['share_within'] * len(truth))} of {len(truth)} ({pct(s['share_within'])})"]
        for name, s in stats.items()
    ]
    lines = table(
        ["Dimension", "Measured spines", "Compared", "Mean error", "Median error", "p90 error",
         f"Within {pct(cfg.dims.tolerance)}, of all measured"],
        rows,
    )
    lines += [
        "",
        f"Spines with no usable packet book at their slot: {unmatched}; they and missing dimensions count as failures "
        f"against all {len(truth)} measured spines. Result: {status} (share within {pct(cfg.dims.tolerance)} "
        f">= {pct(cfg.dims.min_share_within)} and mean error <= {pct(cfg.dims.tolerance)}, both dimensions).",
    ]
    if ambiguous:
        lines += ["", "Duplicate slot keys, left unmatched: " + ", ".join(f"{k[0]} #{k[1]}" for k in sorted(ambiguous, key=str)) + "."]
    return Section(title, status, {**stats, "unmatched": unmatched, "duplicates": len(ambiguous)}, lines)


def isbn_key(isbn: str) -> str:
    return "".join(c for c in isbn if c.isalnum()).upper()


def price_row(truth: GtPrice, book: Book, packet: ClaimPacket, cfg: EvalConfig):
    new = truth.condition.casefold() in cfg.price.new_conditions
    if new:
        amount, source, observed = book.replacement_cost.amount, book.replacement_cost.source, True
    else:
        used = book.used_value
        amount, source = used.amount, used.source
        observed = used.source_type == cfg.price.observed_source_type
    if amount is None:
        return "no packet price", None
    reasons = []
    if not observed:
        reasons.append("derived, not an observed price")
    if (
        not source
        or similarity(cfg.price.source_scorer, source, truth.source) < cfg.price.source_threshold
    ):
        reasons.append(f"source {source or 'none'} differs from {truth.source}")
    if not packet.sweep.currency:
        reasons.append("packet currency is unknown")
    elif packet.sweep.currency.casefold() != truth.currency.casefold():
        reasons.append(f"currency {packet.sweep.currency} differs from {truth.currency}")
    return "; ".join(reasons), float(amount)


def prices_section(truth, packet: ClaimPacket, cfg: EvalConfig) -> Section:
    title = "Book prices"
    if (blocked := unavailable(title, "prices15.csv", truth)) is not None:
        return blocked
    by_isbn: dict[str, Book] = {}
    for b in packet.books:
        if b.isbn:
            by_isbn.setdefault(isbn_key(b.isbn), b)
    same, mismatched, absent = [], [], []
    for entry in truth:
        book = by_isbn.get(isbn_key(entry.isbn))
        if book is None:
            absent.append([entry.isbn, entry.title, "ISBN not in the packet"])
            continue
        reason, amount = price_row(entry, book, packet, cfg)
        if amount is None:
            absent.append([entry.isbn, entry.title, reason])
        elif reason:
            mismatched.append([entry.isbn, entry.title, entry.source, entry.price, amount, reason])
        else:
            same.append([entry.isbn, entry.title, entry.source, entry.price, amount,
                         relative_error(amount, entry.price)])
    errors = [row[-1] for row in same]
    stats = error_stats(errors, cfg.price.tolerance)
    comparable = f"{stats['n']} of {len(truth)}"
    reason = ""
    if stats["n"] < cfg.price.min_comparable:
        status = NOT_EVALUATED
        reason = f"comparable {comparable}; the bar needs at least {cfg.price.min_comparable} same-source rows"
    else:
        status = verdict(
            stats["mean"] <= cfg.price.tolerance and stats["share_within"] >= cfg.price.min_share_within
        )
    headers = ["ISBN", "Title", "Source", "Hand-checked", "Packet", "Error"]
    lines = table(
        ["Hand-checked books", "Comparable (same source type)", "Mismatched source", "Not priced", "Mean error",
         "Median error", f"Within {pct(cfg.price.tolerance)}, of comparable", "Result"],
        [[len(truth), comparable, len(mismatched), len(absent), fmt_stat(stats["mean"]),
          fmt_stat(stats["median"]), fmt_stat(stats["share_within"]), status]],
    )
    if same:
        lines += ["", "Same source type:", ""] + table(headers, [[*r[:-1], pct(r[-1])] for r in same])
    if mismatched:
        lines += ["", "Mismatched source rows, excluded from the bar:", ""] + table(
            ["ISBN", "Title", "Source", "Hand-checked", "Packet", "Why"], mismatched
        )
    if absent:
        lines += ["", "Not compared:", ""] + table(["ISBN", "Title", "Why"], absent)
    return Section(
        title,
        status,
        {**stats, "mismatched": len(mismatched), "not_priced": len(absent), "total": len(truth)},
        lines,
        reason,
    )


def items_section(truth, packet: ClaimPacket, cfg: EvalConfig) -> Section:
    title = "Non-book items"
    if (blocked := unavailable(title, "items.csv", truth)) is not None:
        return blocked

    def text(brand: str, name: str) -> str:
        return f"{brand} {name}".strip()

    candidates = sorted(
        (
            (similarity(cfg.items.scorer, text(g.brand, g.name), text(item.brand_model, item.description)), i, j)
            for i, g in enumerate(truth)
            for j, item in enumerate(packet.items)
        ),
        reverse=True,
    )
    candidates = [c for c in candidates if c[0] >= cfg.items.threshold]
    same_category = [c for c in candidates if truth[c[1]].category.casefold() == packet.items[c[2]].category.casefold()]
    assigned: dict[int, int] = {}
    used: set[int] = set()
    for _, i, j in [*same_category, *candidates]:
        if i not in assigned and j not in used:
            assigned[i] = j
            used.add(j)
    rows, found, correct = [], 0, 0
    for i, g in enumerate(truth):
        item: Item | None = packet.items[assigned[i]] if i in assigned else None
        ok = item is not None and item.category.casefold() == g.category.casefold()
        found += item is not None
        correct += ok
        rows.append([g.name, g.category, item.description or item.brand_model if item else "not found",
                     item.category if item else "", "yes" if ok else "no"])
    share = correct / len(truth)
    enough = len(truth) >= cfg.items.min_truth
    status = verdict(share >= cfg.items.min_found) if enough else NOT_EVALUATED
    reason = "" if enough else f"items.csv has {len(truth)} items; the bar needs at least {cfg.items.min_truth}"
    lines = table(
        ["Ground-truth items", "Found", "Found with correct category", "Found with wrong category", "Rate",
         f"Bar (>= {pct(cfg.items.min_found)})"],
        [[len(truth), found, correct, found - correct, pct(share), status]],
    ) + [""] + table(["Item", "True category", "Packet item", "Packet category", "Correct"], rows)
    return Section(
        title, status,
        {"total": len(truth), "found": found, "correct": correct, "wrong_category": found - correct, "rate": share},
        lines, reason,
    )


def area_row(name: str, formula: str, truth: float, predicted: float | None, tolerance: float):
    if predicted is None:
        return [name, formula, f"{truth:.2f}", "none", "n/a", f"within {pct(tolerance)}", FAIL], FAIL
    error = relative_error(predicted, truth)
    status = verdict(error <= tolerance)
    return [name, formula, f"{truth:.2f}", f"{predicted:.2f}", pct(error), f"within {pct(tolerance)}", status], status


def room_sections(truth, packet: ClaimPacket, cfg: EvalConfig) -> list[Section]:
    titles = ["Floor area", "Wall area"]
    if (blocked := unavailable(titles[0], "room.json", truth)) is not None:
        return [blocked, replace(blocked, title=titles[1])]
    try:
        floor = floor_area_m2(truth, cfg)
    except ValueError as error:
        reason = f"room.json: {error}"
        return [Section(t, NOT_EVALUATED, reason=reason) for t in titles]
    headers = ["Measure", "Formula", "Tape", "Packet", "Error", "Bar", "Result"]
    row, floor_status = area_row("Floor area (m2)", "shoelace over the wall polygon", floor,
                                 packet.room.floor_area_m2, cfg.room.floor_tolerance)
    gross = gross_wall_area_m2(truth)
    row_wall, wall_status = area_row("Gross wall area (m2)", "perimeter x ceiling height", gross,
                                     packet.room.wall_area_m2, cfg.room.wall_tolerance)
    extra = [
        ["Net wall area (m2)", "gross minus openings", f"{gross - openings_area_m2(truth):.2f}",
         "n/a" if packet.room.wall_area_net_m2 is None else f"{packet.room.wall_area_net_m2:.2f}"],
        ["Shelved wall area (m2)", "sum of shelving fronts width x height", f"{shelving_area_m2(truth):.2f}",
         "n/a" if packet.room.shelved_wall_area_m2 is None else f"{packet.room.shelved_wall_area_m2:.2f}"],
    ]
    wall_lines = (
        table(headers, [row_wall])
        + ["", "Reported without a pass bar:", ""]
        + table(["Measure", "Formula", "Tape", "Packet"], extra)
    )
    return [
        Section(titles[0], floor_status, {"true": floor, "packet": packet.room.floor_area_m2}, table(headers, [row])),
        Section(titles[1], wall_status, {"true": gross, "packet": packet.room.wall_area_m2}, wall_lines),
    ]


def stage_rows(packet: ClaimPacket, ledger: Ledger) -> dict[str, dict[str, float]]:
    stages: dict[str, dict[str, float]] = {}
    for entry in ledger.entries:
        row = stages.setdefault(entry.stage, {"seconds": 0.0, "usd": 0.0})
        row["seconds"] += entry.seconds
        row["usd"] += entry.usd_at_paid_rates
    for name, seconds in packet.latency_s.items():
        stages.setdefault(name, {"seconds": 0.0, "usd": 0.0})["seconds"] = seconds
    for name, usd in packet.costs.by_stage_usd.items():
        stages.setdefault(name, {"seconds": 0.0, "usd": 0.0})["usd"] = usd
    return stages


def time_sections(packet: ClaimPacket, ledger: Ledger, cfg: EvalConfig) -> list[Section]:
    stages = stage_rows(packet, ledger)
    elapsed = cfg.time.elapsed_key
    post_stages = [n for n in stages if n not in cfg.time.streaming_stages and n != elapsed]
    if not post_stages:
        reason = "no post-sweep stage timings in the packet latency or the stage ledger"
        return [
            Section("Time to packet", NOT_EVALUATED, reason=reason),
            Section("Cost and latency per stage", NOT_EVALUATED, reason=reason, bar=False),
        ]
    post = stages[elapsed]["seconds"] if elapsed in stages else sum(stages[n]["seconds"] for n in post_stages)
    status = verdict(post < cfg.time.max_seconds)
    absent = [n for n in cfg.time.expected_stages if n not in stages]
    total_usd = packet.costs.usd_at_paid_rates or sum(row["usd"] for row in stages.values())
    time_lines = table(
        ["Seconds after the sweep ended", "Source", "Bar", "Result"],
        [[f"{post:.1f}", elapsed if elapsed in stages else "sum of post-sweep stages",
          f"< {cfg.time.max_seconds:.0f}", status]],
    )
    if absent:
        time_lines += ["", "Stages with no timing recorded: " + ", ".join(absent) + "."]
    per_stage = Section(
        "Cost and latency per stage",
        "REPORTED",
        {"usd": total_usd},
        table(
            ["Stage", "Seconds", "USD at paid rates"],
            [
                *([name, f"{row['seconds']:.1f}", f"{row['usd']:.4f}"] for name, row in stages.items()),
                ["total (stage sum)", f"{sum(r['seconds'] for n, r in stages.items() if n != elapsed):.1f}",
                 f"{total_usd:.4f}"],
            ],
        ),
        bar=False,
    )
    return [Section("Time to packet", status, {"seconds": post, "missing_stages": absent}, time_lines), per_stage]


def parse_error(error: Exception) -> str:
    if isinstance(error, KeyError):
        return f"missing column {error}"
    if isinstance(error, ValidationError):
        first = error.errors()[0]
        return f"{'.'.join(str(p) for p in first['loc'])}: {first['msg']}"
    return str(error)


def optional(directory: Path, name: str, loader):
    path = directory / name
    if not path.is_file():
        return None
    try:
        return loader(path)
    except (ValueError, KeyError) as error:
        return TruthError(name, parse_error(error))


def usable(truth):
    return truth if isinstance(truth, list) and truth else None


def evaluate(run_dir: Path, truth_dir: Path, cfg: EvalConfig | None = None) -> list[Section]:
    cfg = cfg or load_eval_config()
    pipeline = yaml.safe_load((CONFIG_DIR / "pipeline.yaml").read_text(encoding="utf-8"))
    packet = ClaimPacket.model_validate_json(
        (run_dir / pipeline["packet_name"]).read_text(encoding="utf-8")
    )
    ledger = Ledger(run_dir / "stages" / "_ledger.json")
    books = optional(truth_dir, "books.csv", lambda p: load_books(p, cfg))
    matches = match_books(books, packet, cfg) if usable(books) else []
    return [
        count_section(books, matches, packet, cfg),
        *title_sections(books, matches, packet, cfg),
        honesty_section(packet),
        spines_section(optional(truth_dir, "spines20.csv", load_spines), usable(books), matches, packet, cfg),
        prices_section(optional(truth_dir, "prices15.csv", load_prices), packet, cfg),
        items_section(optional(truth_dir, "items.csv", load_items), packet, cfg),
        *room_sections(optional(truth_dir, "room.json", load_room), packet, cfg),
        *time_sections(packet, ledger, cfg),
    ]


def interpretation(cfg: EvalConfig) -> list[str]:
    return [
        "How the bars are read in this project (the brief gives tolerances, not how to aggregate them):",
        "",
        f"- Book count: error within {pct(cfg.count.tolerance)} of the ground-truth row count.",
        f"- Titles: correct identifications over legible ground-truth spines at least {pct(cfg.title.min_accuracy)}, "
        f"and confidently wrong over all identified books at most {pct(cfg.title.max_confidently_wrong)}.",
        f"- Spine dimensions: the share of all {cfg.dims.min_sample} hand-measured spines within "
        f"{pct(cfg.dims.tolerance)} must be at least {pct(cfg.dims.min_share_within)} and the mean error at most "
        f"{pct(cfg.dims.tolerance)}, for height and thickness. Unmatched spines and missing predictions count as "
        "failures; fewer than the required spines means not evaluated.",
        f"- Prices: over rows with the same source type, share within {pct(cfg.price.tolerance)} at least "
        f"{pct(cfg.price.min_share_within)} and mean error at most {pct(cfg.price.tolerance)}; at least "
        f"{cfg.price.min_comparable} comparable rows are required, otherwise not evaluated.",
        f"- Items: at least {pct(cfg.items.min_found)} found with the correct category, from at least "
        f"{cfg.items.min_truth} items.",
        f"- Floor within {pct(cfg.room.floor_tolerance)}, wall within {pct(cfg.room.wall_tolerance)}.",
        f"- Time: seconds after the sweep ended under {cfg.time.max_seconds:.0f}.",
    ]


def render(sections: list[Section], run_dir: Path, cfg: EvalConfig) -> str:
    bars = [s for s in sections if s.bar]
    out = [
        "# Results against ground truth",
        "",
        f"Run: `{run_dir.name}`",
        "",
        *interpretation(cfg),
        "",
        "## Pass bars",
        "",
        *table(
            ["Pass bar", "Result", "Note"],
            [[s.title, s.status, s.reason] for s in bars],
        ),
    ]
    not_evaluated = [s.title for s in bars if s.status == NOT_EVALUATED]
    if not_evaluated:
        out += ["", "Could not be evaluated: " + ", ".join(not_evaluated) + "."]
    for s in sections:
        out += ["", f"## {s.title}", ""]
        out += s.lines if s.lines else [f"Not evaluated: {s.reason}."]
    return "\n".join(out) + "\n"


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: python tools/eval.py runs/<id> ground_truth/", file=sys.stderr)
        return 2
    run_dir, truth_dir = Path(argv[1]), Path(argv[2])
    cfg = load_eval_config()
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdout.write(render(evaluate(run_dir, truth_dir, cfg), run_dir, cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
