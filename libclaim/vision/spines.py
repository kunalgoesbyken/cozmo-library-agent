from enum import StrEnum
from statistics import median

import cv2
import numpy as np
from google.genai import types
from pydantic import BaseModel

from libclaim.llm import Llm, default_llm
from libclaim.models import Still
from libclaim.settings import load_settings
from libclaim.vision.boxes import Box, ReconcileConfig, VisionConfig, containment, iou, load_vision_config, nms, read_image

BOX_COORDS = 4


class Orientation(StrEnum):
    vertical = "vertical"
    horizontal = "horizontal"


class Source(StrEnum):
    gemini = "gemini"
    rfdetr = "rfdetr"


class SpineDet(BaseModel):
    box_px: Box
    orientation: Orientation
    source: Source


class TileSpine(BaseModel):
    image_index: int
    box_2d: list[int]
    orientation: Orientation


class TileSpines(BaseModel):
    spines: list[TileSpine]


class ReviewEntry(BaseModel):
    row: str
    reason: str
    detail: str


class ReconcileResult(BaseModel):
    rows: dict[str, list[SpineDet]]
    review_entries: list[ReviewEntry]
    requery_rows: list[str]


def tile_origins(width: int, height: int, cfg: VisionConfig) -> list[Box]:
    tiles = cfg.tiles
    tile_w = width / (tiles.cols - (tiles.cols - 1) * tiles.overlap)
    tile_h = height / (tiles.rows - (tiles.rows - 1) * tiles.overlap)
    step_x = tile_w * (1 - tiles.overlap)
    step_y = tile_h * (1 - tiles.overlap)
    return [
        Box(x1=col * step_x, y1=row * step_y, x2=col * step_x + tile_w, y2=row * step_y + tile_h)
        for row in range(tiles.rows)
        for col in range(tiles.cols)
    ]


def encode_tile(img: np.ndarray, tile: Box, cfg: VisionConfig) -> types.Part:
    crop = img[round(tile.y1) : round(tile.y2), round(tile.x1) : round(tile.x2)]
    ok, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
    return types.Part.from_bytes(data=buffer.tobytes(), mime_type=cfg.jpeg_mime)


def to_px(spine: TileSpine, tile: Box, cfg: VisionConfig) -> Box | None:
    if len(spine.box_2d) != BOX_COORDS:
        return None
    ymin, xmin, ymax, xmax = (min(max(v, 0), cfg.box_scale) / cfg.box_scale for v in spine.box_2d)
    box = Box(
        x1=tile.x1 + xmin * tile.width,
        y1=tile.y1 + ymin * tile.height,
        x2=tile.x1 + xmax * tile.width,
        y2=tile.y1 + ymax * tile.height,
    )
    return box if box.area > 0 else None


async def detect_spines(keyframe: Still, llm: Llm | None = None, models: list[str] | None = None) -> list[SpineDet]:
    cfg = load_vision_config()
    img = read_image(keyframe.path)
    height, width = img.shape[:2]
    tiles = tile_origins(width, height, cfg)
    parts = [encode_tile(img, tile, cfg) for tile in tiles] + [cfg.prompt]
    answer = await (llm or default_llm()).structured(models or load_settings().vision_models, parts, TileSpines)
    dets = []
    for spine in answer.spines:
        if not 0 <= spine.image_index < len(tiles):
            continue
        box = to_px(spine, tiles[spine.image_index], cfg)
        if box is not None:
            dets.append(SpineDet(box_px=box, orientation=spine.orientation, source=Source.gemini))
    kept = nms([det.box_px for det in dets], cfg.nms_iou, cfg.nms_containment)
    return [dets[i] for i in kept]


def row_gaps(dets: list[SpineDet]) -> list[tuple[float, float]]:
    upright = sorted((d.box_px for d in dets if d.orientation == Orientation.vertical), key=lambda b: b.x1)
    if len(upright) < 2:
        return []
    typical = median(b.width for b in upright)
    gaps = []
    edge = upright[0].x2
    for box in upright[1:]:
        gaps.append((box.x1 - edge, typical))
        edge = max(edge, box.x2)
    return gaps


def count_difference(gemini: int, local: int) -> float:
    largest = max(gemini, local)
    return abs(gemini - local) / largest if largest else 0.0


def local_orientation(box: Box, cfg: ReconcileConfig) -> Orientation:
    vertical = box.width > 0 and box.height / box.width >= cfg.vertical_min_aspect
    return Orientation.vertical if vertical else Orientation.horizontal


def unmatched_local(local: list[Box], dets: list[SpineDet], cfg: ReconcileConfig) -> list[Box]:
    pairs = sorted(
        ((iou(b, d.box_px), li, di) for li, b in enumerate(local) for di, d in enumerate(dets)),
        reverse=True,
    )
    used_local: set[int] = set()
    used_gemini: set[int] = set()
    for overlap, li, di in pairs:
        if overlap < cfg.match_iou:
            break
        if li not in used_local and di not in used_gemini:
            used_local.add(li)
            used_gemini.add(di)
    return [
        b
        for li, b in enumerate(local)
        if li not in used_local and all(containment(b, d.box_px) < cfg.phantom_containment for d in dets)
    ]


def reconcile(gemini: dict[str, list[SpineDet]], rfdetr: dict[str, list[Box] | None]) -> ReconcileResult:
    cfg = load_vision_config().reconcile
    rows: dict[str, list[SpineDet]] = {}
    entries: list[ReviewEntry] = []
    severity: dict[str, float] = {}
    if any(rfdetr.get(row) is None for row in gemini):
        entries.append(ReviewEntry(row=cfg.sweep_label, reason="local_counter_unavailable", detail="no local boxes"))
    for row, dets in gemini.items():
        local = rfdetr.get(row)
        merged = list(dets)
        flagged = 0.0
        if local is not None:
            merged += [
                SpineDet(box_px=b, orientation=local_orientation(b, cfg), source=Source.rfdetr)
                for b in unmatched_local(local, dets, cfg)
            ]
            diff = count_difference(len(dets), len(local))
            if diff > cfg.count_diff_fraction:
                flagged = max(flagged, diff / cfg.count_diff_fraction)
                entries.append(
                    ReviewEntry(row=row, reason="count_mismatch", detail=f"gemini {len(dets)} rfdetr {len(local)}")
                )
        for gap, typical in row_gaps(merged):
            if gap > cfg.gap_thickness_factor * typical:
                flagged = max(flagged, gap / (cfg.gap_thickness_factor * typical))
                entries.append(ReviewEntry(row=row, reason="gap", detail=f"gap {gap:.0f}px thickness {typical:.0f}px"))
                break
        rows[row] = merged
        if flagged:
            severity[row] = flagged
    requery = sorted(severity, key=severity.get, reverse=True)[: cfg.max_requery_rows]
    return ReconcileResult(rows=rows, review_entries=entries, requery_rows=requery)
