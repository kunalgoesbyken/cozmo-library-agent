from dataclasses import dataclass
from enum import StrEnum

import cv2
import numpy as np
import yaml
from pydantic import BaseModel, Field

from libclaim.live.tools import Statement, StatementKind
from libclaim.llm import Llm, default_llm
from libclaim.models import Still
from libclaim.packet.models import Dimensions
from libclaim.room.depth import CM_PER_M, DepthSet, camera
from libclaim.settings import CONFIG_DIR, load_settings
from libclaim.vision.boxes import Box, VisionConfig, iou, load_vision_config, read_image
from libclaim.vision.spines import encode_tile, to_px
from libclaim.vision.tracks import pair_homography, project


def load_categories() -> list[str]:
    return yaml.safe_load((CONFIG_DIR / "taxonomy.yaml").read_text(encoding="utf-8"))["item_categories"]


Category = StrEnum("Category", {name: name for name in load_categories()})


class FrameItem(BaseModel):
    image_index: int
    category: Category
    description: str
    brand_model_text: str
    box_2d: list[int]
    is_art: bool
    truncated: bool


class FrameItems(BaseModel):
    items: list[FrameItem]


class ItemDet(BaseModel):
    category: str
    description: str
    brand_model: str
    frame_seq: int
    frame_path: str
    box_px: Box
    seen_in: list[int]
    is_art: bool
    truncated: bool
    needs_appraisal: bool
    dimensions_cm: Dimensions = Field(default_factory=Dimensions)
    dimensions_method: str = ""
    review: list[str] = Field(default_factory=list)


@dataclass
class Sighting:
    frame: int
    item: FrameItem
    box: Box


def whole(image: np.ndarray) -> Box:
    height, width = image.shape[:2]
    return Box(x1=0, y1=0, x2=width, y2=height)


async def sightings_in(stills: list[Still], start: int, llm: Llm, models: list[str], cfg: VisionConfig) -> list[Sighting]:
    images = [read_image(still.path) for still in stills]
    frames = [whole(image) for image in images]
    parts: list = []
    for index, (image, frame) in enumerate(zip(images, frames)):
        parts += [f"image_index: {index}", encode_tile(image, frame, cfg)]
    answer = await llm.structured(models, parts + [cfg.items.prompt], FrameItems)
    found = []
    for item in answer.items:
        if 0 <= item.image_index < len(stills):
            box = to_px(item.box_2d, frames[item.image_index], cfg)
            if box is not None:
                found.append(Sighting(start + item.image_index, item, box))
    return found


def dedupe(sightings: list[Sighting], stills: list[Still], cfg: VisionConfig) -> list[list[Sighting]]:
    grays: dict[int, np.ndarray] = {}
    steps: dict[tuple[int, int], np.ndarray | None] = {}

    def gray(frame: int) -> np.ndarray:
        if frame not in grays:
            grays[frame] = read_image(stills[frame].path, cv2.IMREAD_GRAYSCALE)
        return grays[frame]

    def step(earlier: int, later: int) -> np.ndarray | None:
        if (earlier, later) not in steps:
            steps[earlier, later] = pair_homography(gray(earlier), gray(later), cfg.tracks)
        return steps[earlier, later]

    parent = list(range(len(sightings)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    candidates = []
    for b, later in enumerate(sightings):
        for a, earlier in enumerate(sightings):
            if earlier.item.category != later.item.category:
                continue
            if not 0 < later.frame - earlier.frame <= cfg.items.dedup_lookback:
                continue
            matrix = step(earlier.frame, later.frame)
            if matrix is None:
                continue
            overlap = iou(project(later.box, matrix), earlier.box)
            if overlap >= cfg.items.dedup_iou:
                candidates.append((overlap, a, b))
    frames_of = {index: {sighting.frame} for index, sighting in enumerate(sightings)}
    matched: set[tuple[int, int]] = set()
    for _, a, b in sorted(candidates, reverse=True):
        frame_a, frame_b = sightings[a].frame, sightings[b].frame
        if (a, frame_b) in matched or (b, frame_a) in matched:
            continue
        root_a, root_b = root(a), root(b)
        if frames_of[root_a] & frames_of[root_b]:
            continue
        parent[root_b] = root_a
        frames_of[root_a] |= frames_of.pop(root_b)
        matched |= {(a, frame_b), (b, frame_a)}
    groups: dict[int, list[Sighting]] = {}
    for index, sighting in enumerate(sightings):
        groups.setdefault(root(index), []).append(sighting)
    return list(groups.values())


def size(box: Box, still: Still, depths: DepthSet | None, cfg: VisionConfig) -> tuple[Dimensions, str, list[str]]:
    depth = None if depths is None else depths.maps.get(still.seq)
    if depth is None:
        return Dimensions(), "", ["no metric depth for this frame; item size unknown"]
    sx, sy = depth.shape[1] / still.w, depth.shape[0] / still.h
    margin = (1 - cfg.items.depth_core) / 2
    x1, x2 = (round((box.x1 + f * box.width) * sx) for f in (margin, 1 - margin))
    y1, y2 = (round((box.y1 + f * box.height) * sy) for f in (margin, 1 - margin))
    core = depth[y1 : max(y2, y1 + 1), x1 : max(x2, x1 + 1)]
    core = core[np.isfinite(core) & (core > 0)]
    if not core.size:
        return Dimensions(), "", ["no valid depth inside the item box; item size unknown"]
    distance = float(np.median(core))
    k = camera(still, depth.shape)
    dims = Dimensions(
        w=round(box.width * sx * distance / k[0, 0] * CM_PER_M, 1),
        h=round(box.height * sy * distance / k[1, 1] * CM_PER_M, 1),
    )
    review = []
    if depths.confidence < cfg.items.min_depth_confidence:
        review.append(f"item size uses low-confidence depth ({depths.scale_method}, {depths.confidence:.2f})")
    return dims, depths.scale_method, review


def to_det(group: list[Sighting], stills: list[Still], depths: DepthSet | None, cfg: VisionConfig) -> ItemDet:
    ranked = sorted(group, key=lambda s: (s.item.truncated, -stills[s.frame].blur))
    best = ranked[0]
    still = stills[best.frame]
    is_art = any(s.item.is_art for s in group) or best.item.category in cfg.items.art_categories
    dims, method, review = size(best.box, still, depths, cfg)
    if best.item.truncated:
        review.append("item is cut off in every frame; size is a lower bound")
    return ItemDet(
        category=best.item.category.value,
        description=best.item.description.strip(),
        brand_model=next((s.item.brand_model_text.strip() for s in ranked if s.item.brand_model_text.strip()), ""),
        frame_seq=still.seq,
        frame_path=still.path,
        box_px=best.box,
        seen_in=sorted({stills[s.frame].seq for s in group}),
        is_art=is_art,
        truncated=best.item.truncated,
        needs_appraisal=is_art,
        dimensions_cm=dims,
        dimensions_method=method,
        review=review,
    )


def art_in_view(dets: list[ItemDet], statement: Statement) -> list[int]:
    if statement.still_seq_from is None or statement.still_seq_to is None:
        return []
    return [
        index
        for index, det in enumerate(dets)
        if det.is_art and any(statement.still_seq_from <= seq <= statement.still_seq_to for seq in det.seen_in)
    ]


def apply_statements(dets: list[ItemDet], statements: list[Statement]) -> list[ItemDet]:
    prints: set[int] = set()
    originals: set[int] = set()
    for statement in statements:
        in_view = art_in_view(dets, statement)
        if statement.kind == StatementKind.original_art:
            originals.update(in_view)
        elif statement.kind == StatementKind.art_print and len(in_view) == 1:
            prints.update(in_view)
        elif statement.kind == StatementKind.art_print:
            for index in in_view:
                dets[index].review.append("user said print but several artworks were in view")
    for index in prints - originals:
        dets[index].needs_appraisal = False
    return dets


def unmatched_statements(dets: list[ItemDet], statements: list[Statement]) -> list[str]:
    return [
        f"user said '{statement.subject}' is a print but no detected artwork was in view"
        for statement in statements
        if statement.kind == StatementKind.art_print and not art_in_view(dets, statement)
    ]


async def detect_items(room_keyframes: list[Still], llm: Llm | None = None, models: list[str] | None = None,
                       statements: list[Statement] | None = None, depths: DepthSet | None = None) -> list[ItemDet]:
    cfg = load_vision_config()
    batch = cfg.items.batch_size
    sightings: list[Sighting] = []
    for start in range(0, len(room_keyframes), batch):
        sightings += await sightings_in(
            room_keyframes[start : start + batch],
            start,
            llm or default_llm(),
            models or load_settings().vision_models,
            cfg,
        )
    groups = dedupe(sightings, room_keyframes, cfg)
    return apply_statements([to_det(group, room_keyframes, depths, cfg) for group in groups], statements or [])
