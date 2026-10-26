from collections import Counter
from statistics import mean, median

import cv2
import numpy as np

from libclaim.models import Still
from libclaim.room.geometry import ShelfUnit
from libclaim.vision.boxes import Box, ScaleConfig, TrackConfig, read_image
from libclaim.vision.scale import (
    MeasureConfig,
    ScaleMethod,
    SpineDims,
    UnitScale,
    aruco_detector,
    fallback_reason,
    frame_scale,
    measure,
    measure_row,
    middle,
    prior_placement,
)
from libclaim.vision.tracks import BookTrack, BoxRef, pair_homography, project, register

CM_PER_M = 100.0
ROW_NOTE = "books could not be measured as one shelf row; measured one by one"

Anchor = tuple[int, Box]


def gray(still: Still) -> np.ndarray:
    return read_image(still.path, cv2.IMREAD_GRAYSCALE)


def continues(previous: list[Still], still: Still, cfg: TrackConfig) -> bool:
    current = gray(still)
    recent = reversed(previous[-cfg.anchor_lookback :])
    return any(pair_homography(gray(earlier), current, cfg) is not None for earlier in recent)


def marker_unit(stills: list[Still], cfg: ScaleConfig) -> int | None:
    detector = aruco_detector(cfg)
    counts: Counter[int] = Counter()
    for still in stills:
        _, ids, _ = detector.detectMarkers(gray(still))
        if ids is not None:
            counts.update({int(marker) for marker in ids.flatten()})
    return min(counts, key=lambda marker: (-counts[marker], marker)) if counts else None


def best_ref(track: BookTrack, stills: list[Still]) -> BoxRef:
    return next(
        ref for ref in track.refs if stills[ref.frame].seq == track.best_frame.seq and ref.box == track.best_box
    )


def anchor_boxes(tracks: list[BookTrack], stills: list[Still], cfg: TrackConfig) -> list[Anchor]:
    placed = register([gray(still) for still in stills], cfg)
    anchors = []
    for track in tracks:
        ref = best_ref(track, stills)
        group, matrix = placed[ref.frame]
        anchors.append((group, project(ref.box, matrix)))
    return anchors


def same_row(a: Box, b: Box) -> bool:
    return a.y1 <= middle(b) <= a.y2 and b.y1 <= middle(a) <= b.y2


def upright_rows(anchors: list[Anchor], upright: list[int]) -> list[list[int]]:
    parent = {index: index for index in upright}

    def root(index: int) -> int:
        while parent[index] != index:
            index = parent[index]
        return index

    for a in upright:
        for b in upright:
            if a < b and anchors[a][0] == anchors[b][0] and same_row(anchors[a][1], anchors[b][1]):
                parent[root(b)] = root(a)
    groups: dict[int, list[int]] = {}
    for index in upright:
        groups.setdefault(root(index), []).append(index)
    return list(groups.values())


def holds(row: list[int], anchor: Anchor, anchors: list[Anchor]) -> bool:
    group, box = anchor
    top = min(anchors[j][1].y1 for j in row)
    bottom = max(anchors[j][1].y2 for j in row)
    return anchors[row[0]][0] == group and top <= middle(box) <= bottom


def split_rows(anchors: list[Anchor], vertical: list[bool]) -> tuple[list[list[int]], list[int]]:
    rows = upright_rows(anchors, [i for i, upright in enumerate(vertical) if upright])
    flat = [i for i, upright in enumerate(vertical) if not upright]
    for index in flat:
        home = next((row for row in rows if holds(row, anchors[index], anchors)), None)
        if home is None:
            rows.append([index])
        else:
            home.append(index)
    rows.sort(key=lambda row: (anchors[row[0]][0], mean(middle(anchors[j][1]) for j in row)))
    return [sorted(row, key=lambda j: anchors[j][1].x1) for row in rows], flat


def flat_dims(track: BookTrack, stills: list[Still], cfg: MeasureConfig, unit: UnitScale | None) -> SpineDims:
    dims = measure(track, stills, cfg, unit)
    if dims.scale_method == ScaleMethod.aruco_homography:
        return dims.model_copy(update={"height_cm": dims.thickness_cm, "thickness_cm": dims.height_cm})
    lying = track.best_box
    standing = Box(x1=0, y1=0, x2=lying.height, y2=lying.width)
    placement = prior_placement(standing, cfg, fallback_reason(cfg, unit))
    k = float(placement.matrix[0, 0])
    return dims.model_copy(
        update={"height_cm": k * standing.height, "thickness_cm": k * standing.width,
                "book_format": placement.book_format, "fallback_reason": placement.fallback_reason}
    )


def with_note(dims: SpineDims, note: str) -> SpineDims:
    return dims.model_copy(update={"review": dims.review + [note]})


def measure_upright(indices: list[int], tracks: list[BookTrack], stills: list[Still], cfg: MeasureConfig,
                    unit: UnitScale | None) -> dict[int, SpineDims]:
    if not indices:
        return {}
    chosen = [tracks[i] for i in indices]
    try:
        return dict(zip(indices, measure_row(chosen, stills, cfg, unit)))
    except ValueError:
        return {i: with_note(measure(t, stills, cfg, unit), ROW_NOTE) for i, t in zip(indices, chosen)}


def measure_rows(tracks: list[BookTrack], stills: list[Still], rows: list[list[int]], flat: list[int],
                 cfg: MeasureConfig, unit: UnitScale | None) -> dict[int, SpineDims]:
    lying = set(flat)
    dims: dict[int, SpineDims] = {}
    for row in rows:
        dims.update(measure_upright([i for i in row if i not in lying], tracks, stills, cfg, unit))
        dims.update({i: flat_dims(tracks[i], stills, cfg, unit) for i in row if i in lying})
    return dims


def is_row(boxes: list[Box], still: Still, min_spines: int, min_height_frac: float) -> bool:
    return len(boxes) >= min_spines and median(box.height for box in boxes) >= min_height_frac * still.h


def plane_boxes(frames: dict[str, tuple[Still, Box]], cfg: MeasureConfig, unit: UnitScale | None) -> dict[str, Box]:
    planes = {}
    for key, (still, box) in frames.items():
        scale = frame_scale(still, cfg, unit)
        if scale is not None:
            planes[key] = project(box, scale[0])
    return planes


def shelf_units(rows_by_unit: dict[int, list[list[SpineDims]]], paths: dict[int, str]) -> list[ShelfUnit]:
    units = []
    for marker in sorted(rows_by_unit):
        metric = [[d for d in row if d.scale_method == ScaleMethod.aruco_homography] for row in rows_by_unit[marker]]
        metric = [row for row in metric if row]
        if metric:
            units.append(
                ShelfUnit(
                    width_m=max(sum(d.thickness_cm for d in row) for row in metric) / CM_PER_M,
                    height_m=sum(max(d.height_cm for d in row) for row in metric) / CM_PER_M,
                    frame_ref=paths.get(metric[0][0].frame_seq, ""),
                )
            )
    return units
