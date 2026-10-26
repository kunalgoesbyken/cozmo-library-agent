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
    plane_distance_cm: float


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


def find_marker(detector: cv2.aruco.ArucoDetector, gray: np.ndarray, marker_id: int) -> np.ndarray | None:
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None
    return next((c.reshape(4, 2) for c, found in zip(corners, ids.flatten()) if found == marker_id), None)


def camera_matrix(still: Still, cfg: ScaleConfig) -> np.ndarray:
    if still.intrinsics and cfg.focal_key in still.intrinsics:
        focal = still.intrinsics[cfg.focal_key]
    else:
        focal = max(still.w, still.h) / 2 / math.tan(math.radians(cfg.fallback_long_side_fov_deg) / 2)
    return np.array([[focal, 0, still.w / 2], [0, focal, still.h / 2], [0, 0, 1]])


def plane_distance_cm(corners: np.ndarray, size: float, still: Still, cfg: ScaleConfig) -> float:
    square = np.float64([[0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0]])
    _, rvec, tvec = cv2.solvePnP(square, corners.astype(np.float64), camera_matrix(still, cfg), None,
                                 flags=cv2.SOLVEPNP_IPPE)
    rotation, _ = cv2.Rodrigues(rvec)
    return float(abs(rotation[:, 2] @ tvec.ravel()))


def unit_scale(marker_id: int, stills: list[Still], cfg: MeasureConfig) -> UnitScale:
    planes: dict[int, PlaneScale] = {}
    size = cfg.marker_size_cm
    if size is not None:
        detector = aruco_detector(cfg.scale)
        world = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
        for still in stills:
            corners = find_marker(detector, read_image(still.path, cv2.IMREAD_GRAYSCALE), marker_id)
            if corners is None:
                continue
            distance = plane_distance_cm(corners, size, still, cfg.scale)
            recess = (distance + cfg.scale.spine_recess_cm) / distance
            homography = cv2.getPerspectiveTransform(corners.astype(np.float32), world)
            planes[still.seq] = PlaneScale(
                seq=still.seq,
                path=still.path,
                matrix=(np.diag([recess, recess, 1.0]) @ homography).tolist(),
                side_px=float(np.mean(np.linalg.norm(corners - np.roll(corners, 1, axis=0), axis=1))),
                plane_distance_cm=distance,
            )
    return UnitScale(marker_id=marker_id, planes=planes)


def plane_confidence(plane: PlaneScale, factor: float, cfg: ScaleConfig) -> float:
    return cfg.aruco_confidence * min(1.0, plane.side_px / cfg.min_marker_px) * factor


def frame_scale(still: Still, cfg: MeasureConfig, unit: UnitScale | None) -> tuple[np.ndarray, float] | None:
    planes = unit.planes if unit else {}
    plane = planes.get(still.seq)
    if plane is not None:
        return np.array(plane.matrix), plane_confidence(plane, 1.0, cfg.scale)
    if not planes:
        return None
    gray = read_image(still.path, cv2.IMREAD_GRAYSCALE)
    for plane in sorted(planes.values(), key=lambda p: abs(p.seq - still.seq)):
        step = pair_homography(read_image(plane.path, cv2.IMREAD_GRAYSCALE), gray, cfg.tracks)
        if step is not None:
            return np.array(plane.matrix) @ step, plane_confidence(plane, cfg.scale.chained_confidence_factor, cfg.scale)
    return None


def fallback_reason(cfg: MeasureConfig, unit: UnitScale | None) -> str:
    if cfg.marker_size_cm is None:
        return "ARUCO_MARKER_SIZE_CM is not set"
    if unit is None or not unit.planes:
        return "no ArUco marker of this shelf unit was visible"
    return "the measuring frame does not overlap a frame with the shelf-unit marker"


def prior_placement(box: Box, cfg: MeasureConfig, why: str) -> Placement:
    aspect = box.height / box.width
    name, fmt = min(
        cfg.formats.items(), key=lambda item: abs(math.log(aspect * item[1].thickness_cm / item[1].height_cm))
    )
    k = fmt.height_cm / box.height
    review = [f"{ScaleMethod.book_format_prior} {name}: {why}"]
    return Placement(box, np.diag([k, k, 1.0]), ScaleMethod.book_format_prior, cfg.scale.prior_confidence, review)


def tight_boxes(path: str, boxes: list[Box], cfg: ScaleConfig) -> list[Box]:
    found = masks.spine_boxes(path, boxes)
    if found is not None:
        return found
    trim = cfg.box_height_trim / 2
    return [b.model_copy(update={"y1": b.y1 + trim * b.height, "y2": b.y2 - trim * b.height}) for b in boxes]


def length_cm(matrix: np.ndarray, start: tuple[float, float], end: tuple[float, float]) -> float:
    a, b = cv2.perspectiveTransform(np.float64([start, end]).reshape(-1, 1, 2), matrix).reshape(-1, 2)
    return float(np.linalg.norm(b - a))


def middle(box: Box) -> float:
    return (box.y1 + box.y2) / 2


def clusters(boxes: list[Box], factor: float) -> list[list[int]]:
    typical = median(b.width for b in boxes)
    groups = [[0]]
    for index in range(1, len(boxes)):
        if boxes[index].x1 - boxes[index - 1].x2 > factor * typical:
            groups.append([])
        groups[-1].append(index)
    return groups


def shared_bounds(boxes: list[Box]) -> list[tuple[float, float]]:
    cuts = [boxes[0].x1] + [(boxes[i - 1].x2 + boxes[i].x1) / 2 for i in range(1, len(boxes))] + [boxes[-1].x2]
    return list(zip(cuts, cuts[1:]))


def reference_frame(tracks: list[BookTrack], stills: list[Still], unit: UnitScale | None) -> int:
    seqs = set(unit.planes) if unit else set()
    members: dict[int, int] = {}
    for track in tracks:
        for ref in track.refs:
            members[ref.frame] = members.get(ref.frame, 0) + 1
    return max(members, key=lambda f: (members[f], stills[f].seq in seqs, stills[f].blur))


def check_one_row(boxes: list[Box]) -> None:
    band = median(middle(b) for b in boxes)
    if any(not b.y1 <= band <= b.y2 for b in boxes):
        raise ValueError("measure_row needs the tracks of exactly one shelf row")


def measure_frame(boxes: list[Box], still: Still, cfg: MeasureConfig, unit: UnitScale | None,
                  run_cm: float | None) -> list[SpineDims]:
    order = sorted(range(len(boxes)), key=lambda i: boxes[i].x1)
    tight = tight_boxes(still.path, [boxes[i] for i in order], cfg.scale)
    scale = frame_scale(still, cfg, unit)
    if scale is not None and run_cm is not None:
        y = mean(middle(b) for b in tight)
        ratio = run_cm / length_cm(scale[0], (tight[0].x1, y), (tight[-1].x2, y))
        scale = np.diag([ratio, ratio, 1.0]) @ scale[0], scale[1]
    if scale is None:
        why = fallback_reason(cfg, unit)
        placed = [prior_placement(box, cfg, why) for box in tight]
    else:
        placed = [Placement(box, scale[0], ScaleMethod.aruco_homography, scale[1], []) for box in tight]
    thickness = [0.0] * len(tight)
    for cluster in clusters(tight, cfg.scale.shared_gap_factor):
        for j, (left, right) in zip(cluster, shared_bounds([tight[j] for j in cluster])):
            y = middle(tight[j])
            thickness[j] = length_cm(placed[j].matrix, (left, y), (right, y))
    dims: dict[int, SpineDims] = {}
    for j, (placement, value) in enumerate(zip(placed, thickness)):
        box = placement.box
        centre = (box.x1 + box.x2) / 2
        dims[order[j]] = SpineDims(
            height_cm=length_cm(placement.matrix, (centre, box.y1), (centre, box.y2)),
            thickness_cm=value,
            scale_method=placement.method,
            confidence=placement.confidence,
            frame_seq=still.seq,
            review=placement.review,
        )
    return [dims[i] for i in range(len(boxes))]


def measure_row(tracks: list[BookTrack], stills: list[Still], cfg: MeasureConfig, unit: UnitScale | None = None,
                run_cm: float | None = None) -> list[SpineDims]:
    if not tracks:
        return []
    frame = reference_frame(tracks, stills, unit)
    inside = {i: ref.box for i, t in enumerate(tracks) for ref in t.refs if ref.frame == frame}
    check_one_row(list(inside.values()))
    dims: dict[int, SpineDims] = dict(zip(inside, measure_frame(list(inside.values()), stills[frame], cfg, unit, run_cm)))
    for index, track in enumerate(tracks):
        if index in inside:
            continue
        alone = measure_row([track], stills, cfg, unit)[0]
        note = f"measured outside the row reference frame {stills[frame].seq}: no shared boundaries"
        dims[index] = alone.model_copy(update={"review": alone.review + [note]})
    return [dims[i] for i in range(len(tracks))]


def measure(track: BookTrack, stills: list[Still], cfg: MeasureConfig, unit: UnitScale | None = None) -> SpineDims:
    return measure_row([track], stills, cfg, unit)[0]


def parse_dimensions(text: str, units: dict[str, float]) -> list[float] | None:
    tokens = text.lower().replace(",", ".").replace("×", " ").replace("x", " ").split()
    if len(tokens) < 2:
        return None
    unit = tokens[-1].rstrip(".")
    if unit not in units:
        return None
    try:
        values = [float(token) for token in tokens[:-1]]
    except ValueError:
        return None
    return [value * units[unit] for value in values]


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
