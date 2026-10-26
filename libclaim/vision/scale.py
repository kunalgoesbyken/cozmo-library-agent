import math
from dataclasses import dataclass
from enum import StrEnum
from statistics import mean, median

import cv2
import numpy as np
import yaml
from pydantic import BaseModel, Field

from libclaim.models import Still
from libclaim.settings import CONFIG_DIR, Settings, load_settings
from libclaim.vision import masks
from libclaim.vision.boxes import Box, ScaleConfig, TrackConfig, load_vision_config, read_image
from libclaim.vision.tracks import BookTrack, pair_homography

FULL_DIMENSIONS = 3


class ScaleMethod(StrEnum):
    aruco_homography = "aruco_homography"
    book_format_prior = "book_format_prior"


class BookFormat(BaseModel):
    height_cm: float
    thickness_cm: float


class MeasureConfig(BaseModel):
    scale: ScaleConfig
    tracks: TrackConfig
    formats: dict[str, BookFormat]
    marker_size_cm: float | None


class PlaneScale(BaseModel):
    seq: int
    path: str
    matrix: list[list[float]]
    side_px: float
    distance_cm: float


class UnitScale(BaseModel):
    marker_id: int
    planes: dict[int, PlaneScale]


class SpineDims(BaseModel):
    height_cm: float
    thickness_cm: float
    scale_method: ScaleMethod
    confidence: float
    frame_seq: int
    review: list[str] = Field(default_factory=list)


@dataclass
class Placement:
    frame: int
    box: Box
    matrix: np.ndarray
    method: ScaleMethod
    confidence: float
    review: list[str]


def load_measure_config(settings: Settings | None = None) -> MeasureConfig:
    vision = load_vision_config()
    taxonomy = yaml.safe_load((CONFIG_DIR / "taxonomy.yaml").read_text(encoding="utf-8"))
    return MeasureConfig(
        scale=vision.scale,
        tracks=vision.tracks,
        formats=taxonomy["book_formats"],
        marker_size_cm=(settings or load_settings()).aruco_marker_size_cm,
    )


def aruco_detector(cfg: ScaleConfig) -> cv2.aruco.ArucoDetector:
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = getattr(cv2.aruco, cfg.corner_refinement)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.aruco_dictionary))
    return cv2.aruco.ArucoDetector(dictionary, params)


def find_marker(gray: np.ndarray, marker_id: int, cfg: ScaleConfig) -> np.ndarray | None:
    corners, ids, _ = aruco_detector(cfg).detectMarkers(gray)
    if ids is None:
        return None
    return next((c.reshape(4, 2) for c, found in zip(corners, ids.flatten()) if found == marker_id), None)


def focal_px(still: Still, cfg: ScaleConfig) -> float:
    if still.intrinsics and cfg.focal_key in still.intrinsics:
        return still.intrinsics[cfg.focal_key]
    return still.w / 2 / math.tan(math.radians(cfg.fallback_hfov_deg) / 2)


def unit_scale(marker_id: int, stills: list[Still], cfg: MeasureConfig) -> UnitScale:
    planes: dict[int, PlaneScale] = {}
    size = cfg.marker_size_cm
    if size is not None:
        world = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
        for still in stills:
            corners = find_marker(read_image(still.path, cv2.IMREAD_GRAYSCALE), marker_id, cfg.scale)
            if corners is None:
                continue
            side_px = float(np.mean(np.linalg.norm(corners - np.roll(corners, 1, axis=0), axis=1)))
            planes[still.seq] = PlaneScale(
                seq=still.seq,
                path=still.path,
                matrix=cv2.getPerspectiveTransform(corners.astype(np.float32), world).tolist(),
                side_px=side_px,
                distance_cm=focal_px(still, cfg.scale) * size / side_px,
            )
    return UnitScale(marker_id=marker_id, planes=planes)


def aruco_placement(frame: int, box: Box, plane: PlaneScale, step: np.ndarray, factor: float,
                    cfg: MeasureConfig) -> Placement:
    recess = (plane.distance_cm + cfg.scale.spine_recess_cm) / plane.distance_cm
    matrix = np.diag([recess, recess, 1.0]) @ np.array(plane.matrix) @ step
    confidence = cfg.scale.aruco_confidence * min(1.0, plane.side_px / cfg.scale.min_marker_px) * factor
    return Placement(frame, box, matrix, ScaleMethod.aruco_homography, confidence, [])


def prior_placement(frame: int, box: Box, cfg: MeasureConfig, why: str) -> Placement:
    aspect = box.height / box.width
    name, fmt = min(
        cfg.formats.items(), key=lambda item: abs(math.log(aspect * item[1].thickness_cm / item[1].height_cm))
    )
    k = fmt.height_cm / box.height
    review = [f"{ScaleMethod.book_format_prior} {name}: {why}"]
    return Placement(frame, box, np.diag([k, k, 1.0]), ScaleMethod.book_format_prior, cfg.scale.prior_confidence, review)


def fallback_reason(cfg: MeasureConfig, planes: dict[int, PlaneScale]) -> str:
    if cfg.marker_size_cm is None:
        return "ARUCO_MARKER_SIZE_CM is not set"
    if not planes:
        return "no ArUco marker of this shelf unit was visible"
    return "no frame of this book overlaps a frame with the shelf-unit marker"


def place(track: BookTrack, stills: list[Still], cfg: MeasureConfig, unit: UnitScale | None) -> Placement:
    refs = sorted(track.refs, key=lambda ref: stills[ref.frame].blur, reverse=True)
    planes = unit.planes if unit else {}
    for ref in refs:
        plane = planes.get(stills[ref.frame].seq)
        if plane is not None:
            return aruco_placement(ref.frame, ref.box, plane, np.eye(3), 1.0, cfg)
    best = refs[0]
    if planes:
        seq = stills[best.frame].seq
        gray = read_image(stills[best.frame].path, cv2.IMREAD_GRAYSCALE)
        for plane in sorted(planes.values(), key=lambda p: abs(p.seq - seq)):
            step = pair_homography(read_image(plane.path, cv2.IMREAD_GRAYSCALE), gray, cfg.tracks)
            if step is not None:
                return aruco_placement(best.frame, best.box, plane, step, cfg.scale.chained_confidence_factor, cfg)
    return prior_placement(best.frame, best.box, cfg, fallback_reason(cfg, planes))


def length_cm(matrix: np.ndarray, start: tuple[float, float], end: tuple[float, float]) -> float:
    a, b = cv2.perspectiveTransform(np.float64([start, end]).reshape(-1, 1, 2), matrix).reshape(-1, 2)
    return float(np.linalg.norm(b - a))


def middle(box: Box) -> float:
    return (box.y1 + box.y2) / 2


def clusters(edges: list[tuple[float, float]], factor: float) -> list[list[int]]:
    typical = median(right - left for left, right in edges)
    groups = [[0]]
    for index in range(1, len(edges)):
        if edges[index][0] - edges[index - 1][1] > factor * typical:
            groups.append([])
        groups[-1].append(index)
    return groups


def shared_bounds(edges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    cuts = [edges[0][0]] + [(edges[i - 1][1] + edges[i][0]) / 2 for i in range(1, len(edges))] + [edges[-1][1]]
    return list(zip(cuts, cuts[1:]))


def rescale(values: list[float], indexes: list[int], total: float) -> None:
    current = sum(values[i] for i in indexes)
    if current > 0:
        for i in indexes:
            values[i] *= total / current


def measure_row(tracks: list[BookTrack], stills: list[Still], cfg: MeasureConfig, unit: UnitScale | None = None,
                run_cm: float | None = None) -> list[SpineDims]:
    placed = [place(track, stills, cfg, unit) for track in tracks]
    thickness = [0.0] * len(placed)
    by_frame: dict[int, list[int]] = {}
    for index, placement in enumerate(placed):
        by_frame.setdefault(placement.frame, []).append(index)
    for frame, members in by_frame.items():
        members.sort(key=lambda i: placed[i].box.x1)
        boxes = [placed[i].box for i in members]
        edges = masks.spine_edges(stills[frame].path, boxes) or [(b.x1, b.x2) for b in boxes]
        for cluster in clusters(edges, cfg.scale.shared_gap_factor):
            chosen = [members[j] for j in cluster]
            bounds = shared_bounds([edges[j] for j in cluster])
            for i, (left, right) in zip(chosen, bounds):
                y = middle(placed[i].box)
                thickness[i] = length_cm(placed[i].matrix, (left, y), (right, y))
            first = placed[chosen[0]]
            if run_cm is None and first.method == ScaleMethod.aruco_homography:
                y = mean(middle(placed[i].box) for i in chosen)
                rescale(thickness, chosen, length_cm(first.matrix, (bounds[0][0], y), (bounds[-1][1], y)))
    if run_cm is not None:
        rescale(thickness, list(range(len(placed))), run_cm)
    dims = []
    for placement, value in zip(placed, thickness):
        box = placement.box
        centre = (box.x1 + box.x2) / 2
        dims.append(
            SpineDims(
                height_cm=length_cm(placement.matrix, (centre, box.y1), (centre, box.y2)),
                thickness_cm=value,
                scale_method=placement.method,
                confidence=placement.confidence,
                frame_seq=stills[placement.frame].seq,
                review=placement.review,
            )
        )
    return dims


def measure(track: BookTrack, stills: list[Still], cfg: MeasureConfig, unit: UnitScale | None = None) -> SpineDims:
    return measure_row([track], stills, cfg, unit)[0]


def parse_dimensions(text: str, units: dict[str, float]) -> list[float] | None:
    tokens = text.lower().replace("x", " ").split()
    if len(tokens) < 2 or tokens[-1] not in units:
        return None
    try:
        values = [float(token) for token in tokens[:-1]]
    except ValueError:
        return None
    return [value * units[tokens[-1]] for value in values]


def cross_check(dims: SpineDims, ol_dims: str) -> list[str]:
    cfg = load_vision_config().scale
    values = parse_dimensions(ol_dims, cfg.length_units_cm)
    if not values:
        return []
    checks = [("height", dims.height_cm, max(values))]
    if len(values) >= FULL_DIMENSIONS:
        checks.append(("thickness", dims.thickness_cm, min(values)))
    return [
        f"spine {name} {measured:.1f} cm differs from Open Library {expected:.1f} cm"
        for name, measured, expected in checks
        if abs(measured - expected) > cfg.cross_check_tolerance * expected
    ]
