from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache
from statistics import median

import cv2
import numpy as np

from libclaim.models import Still
from libclaim.packet.models import Opening
from libclaim.room.geometry import (
    NO_FLOOR,
    RoomGeometry,
    ShelfUnit,
    fallback_room,
    load_room_config,
    room_from_points,
    unmeasured,
)
from libclaim.settings import load_settings
from libclaim.vision import masks
from libclaim.vision.boxes import load_vision_config, read_image
from libclaim.vision.scale import aruco_detector, camera_matrix
from libclaim.vision.tracks import pair_homography

CM_PER_M = 100.0
QUAD = 4
PLANE_POINTS = 3
MIN_PAN_FRAMES = 2


class DepthUnavailable(RuntimeError):
    pass


@dataclass
class DepthSet:
    maps: dict[int, np.ndarray]
    scale_method: str
    confidence: float
    review: list[str] = field(default_factory=list)


@cache
def depth_model(model_id: str):
    with masks.MODEL_LOAD_LOCK:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        return AutoImageProcessor.from_pretrained(model_id), AutoModelForDepthEstimation.from_pretrained(model_id).eval().to(masks.device())


def metric_depth(still: Still) -> np.ndarray:
    if not masks.available():
        raise DepthUnavailable("metric room depth needs the [ml] extra (torch, transformers)")
    try:
        processor, model = depth_model(load_room_config().depth.model_id)
    except (ImportError, OSError) as error:
        raise DepthUnavailable(f"metric depth model could not be loaded: {type(error).__name__}") from error
    import torch

    rgb = np.ascontiguousarray(read_image(still.path)[:, :, ::-1])
    inputs = processor(images=rgb, return_tensors="pt").to(masks.device())
    with torch.no_grad():
        outputs = model(**inputs)
    result = processor.post_process_depth_estimation(outputs, target_sizes=[rgb.shape[:2]])
    return result[0]["predicted_depth"].cpu().numpy()


def camera(still: Still, shape: tuple[int, ...]) -> np.ndarray:
    height, width = shape[:2]
    return np.diag([width / still.w, height / still.h, 1.0]) @ camera_matrix(still, load_vision_config().scale)


def backproject(pixels: np.ndarray, values: np.ndarray, k: np.ndarray) -> np.ndarray:
    rays = np.column_stack([pixels, np.ones(len(pixels))]) @ np.linalg.inv(k).T
    return rays * values[:, None]


def marker_edges(quad: np.ndarray, depth: np.ndarray, k: np.ndarray) -> np.ndarray | None:
    centre = quad.mean(axis=0)
    core = centre + (quad - centre) * load_room_config().depth.scale.interior_fraction
    mask = np.zeros(depth.shape, np.uint8)
    cv2.fillConvexPoly(mask, np.rint(core).astype(np.int32), 1)
    rows, cols = np.nonzero(mask)
    values = depth[rows, cols]
    ok = np.isfinite(values) & (values > 0)
    if np.count_nonzero(ok) < PLANE_POINTS:
        return None
    interior = backproject(np.column_stack([cols[ok], rows[ok]]).astype(float), values[ok], k)
    middle = np.median(interior, axis=0)
    normal = np.linalg.svd(interior - middle, full_matrices=False)[2][-1]
    rays = backproject(quad, np.ones(QUAD), k)
    corners = rays * ((normal @ middle) / (rays @ normal))[:, None]
    return np.linalg.norm(corners - np.roll(corners, 1, axis=0), axis=1)


def marker_factors(still: Still, depth: np.ndarray, size_m: float) -> list[float]:
    cfg = load_vision_config().scale
    gray = read_image(still.path, cv2.IMREAD_GRAYSCALE)
    corners, ids, _ = aruco_detector(cfg).detectMarkers(gray)
    if ids is None:
        return []
    k = camera(still, depth.shape)
    to_depth = np.array([depth.shape[1] / gray.shape[1], depth.shape[0] / gray.shape[0]])
    factors = []
    for quad in corners:
        quad = quad.reshape(QUAD, 2)
        if np.mean(np.linalg.norm(quad - np.roll(quad, 1, axis=0), axis=1)) < cfg.min_marker_px:
            continue
        edges = marker_edges(quad * to_depth, depth, k)
        if edges is not None:
            factors.append(size_m / float(np.median(edges)))
    return factors


def scaled_depths(stills: list[Still], depth_fn: Callable[[Still], np.ndarray] = metric_depth,
                  marker_size_cm: float | None = None) -> DepthSet:
    cfg = load_room_config().depth
    try:
        raw = {still.seq: depth_fn(still) for still in stills}
    except DepthUnavailable as error:
        return DepthSet({}, "", 0.0, [str(error)])
    size_cm = marker_size_cm if marker_size_cm is not None else load_settings().aruco_marker_size_cm
    observed: dict[int, list[float]] = {}
    if size_cm is not None:
        for still in stills:
            found = marker_factors(still, raw[still.seq], size_cm / CM_PER_M)
            if found:
                observed[still.seq] = found
    if not observed:
        why = "ARUCO_MARKER_SIZE_CM is not set" if size_cm is None else "no ArUco marker in the room frames"
        return DepthSet(raw, cfg.prior.scale_method, cfg.prior.confidence, [f"{why}: depth scale is the model's own"])
    review = []
    per_frame = {seq: median(found) for seq, found in observed.items()}
    centre = median(per_frame.values())
    kept = {seq: f for seq, f in per_frame.items() if abs(f / centre - 1) <= cfg.scale.factor_band}
    if len(kept) < len(per_frame):
        review.append(
            f"{len(per_frame) - len(kept)} marker scale factors more than {cfg.scale.factor_band:.0%} "
            "from the median were rejected"
        )
    centre = median(kept.values())
    markers = sum(len(observed[seq]) for seq in kept)
    tier = cfg.marker if markers >= cfg.scale.min_markers else cfg.single_marker
    dispersion = median(abs(factor / centre - 1) for factor in kept.values())
    if dispersion > cfg.scale.high_dispersion:
        review.append(f"marker scale varies by {dispersion:.0%} between room frames")
    unscaled = len(raw) - len(kept)
    if unscaled:
        review.append(f"{unscaled} of {len(raw)} room frames had no usable marker; scaled by the median marker factor")
    confidence = tier.confidence * max(0.0, 1 - dispersion / cfg.scale.zero_confidence_dispersion)
    maps = {seq: depth * kept.get(seq, centre) for seq, depth in raw.items()}
    return DepthSet(maps, tier.scale_method, confidence, review)


def rotation_from_homography(matrix: np.ndarray, k_from: np.ndarray, k_to: np.ndarray) -> np.ndarray:
    m = np.linalg.inv(k_to) @ matrix @ k_from
    m = m / np.cbrt(np.linalg.det(m))
    u, _, vt = np.linalg.svd(m)
    if np.linalg.det(u @ vt) < 0:
        u[:, -1] = -u[:, -1]
    return u @ vt


def pan_rotations(stills: list[Still], cameras: list[np.ndarray]) -> list[np.ndarray | None]:
    cfg = load_vision_config().tracks
    grays = [read_image(still.path, cv2.IMREAD_GRAYSCALE) for still in stills]
    placed: list[tuple[int, np.ndarray]] = []
    for index, gray in enumerate(grays):
        group, rotation = index, np.eye(3)
        for earlier in range(index - 1, max(index - 1 - cfg.anchor_lookback, -1), -1):
            step = pair_homography(grays[earlier], gray, cfg)
            if step is not None:
                group = placed[earlier][0]
                rotation = placed[earlier][1] @ rotation_from_homography(step, cameras[index], cameras[earlier])
                break
        placed.append((group, rotation))
    groups = [group for group, _ in placed]
    largest = max(set(groups), key=groups.count, default=None)
    return [rotation if group == largest else None for group, rotation in placed]


def loop_drift(stills: list[Still], cameras: list[np.ndarray], rotations: list[np.ndarray | None]) -> float | None:
    cfg = load_vision_config().tracks
    registered = [index for index, rotation in enumerate(rotations) if rotation is not None]
    if not registered:
        return None
    last = registered[-1]
    for first in registered[: cfg.anchor_lookback]:
        if last - first <= cfg.anchor_lookback:
            continue
        step = pair_homography(read_image(stills[first].path, cv2.IMREAD_GRAYSCALE),
                               read_image(stills[last].path, cv2.IMREAD_GRAYSCALE), cfg)
        if step is not None:
            direct = rotation_from_homography(step, cameras[last], cameras[first])
            residual = direct.T @ rotations[first].T @ rotations[last]
            return float(np.degrees(np.arccos(np.clip((np.trace(residual) - 1) / 2, -1.0, 1.0))))
    return None


def drift_factor(drift: float | None, review: list[str]) -> float:
    cfg = load_room_config().depth.drift
    if drift is None:
        review.append("the room pan did not close a loop; registration drift unknown")
        return cfg.open_pan_factor
    if drift > cfg.review_deg:
        review.append(f"room pan drifted {drift:.1f} degrees around the loop")
    return max(0.0, 1 - drift / cfg.max_deg)


def pan_up(rotations: list[np.ndarray]) -> np.ndarray:
    cfg = load_room_config().depth
    hint = np.mean([r @ cfg.camera_up for r in rotations], axis=0)
    if len(rotations) < MIN_PAN_FRAMES:
        return hint / np.linalg.norm(hint)
    _, spread, vt = np.linalg.svd(np.array([r[:, 0] for r in rotations]))
    if spread[1] < cfg.min_right_spread * spread[0]:
        return hint / np.linalg.norm(hint)
    return vt[-1] if vt[-1] @ hint >= 0 else -vt[-1]


def upright(rotations: list[np.ndarray], up: np.ndarray) -> float:
    return float(np.mean([(r @ load_room_config().depth.camera_up) @ up for r in rotations]))


def frame_points(depth: np.ndarray, k: np.ndarray) -> np.ndarray:
    cfg = load_room_config().depth
    rows, cols = np.mgrid[0 : depth.shape[0] : cfg.stride, 0 : depth.shape[1] : cfg.stride]
    values = depth[rows, cols]
    keep = np.isfinite(values) & (values >= cfg.min_m) & (values <= cfg.max_m)
    pixels = np.column_stack([cols[keep], rows[keep]]).astype(float)
    return backproject(pixels, values[keep], k)


def room_from_depth(stills: list[Still], shelves: list[ShelfUnit], depths: DepthSet,
                    openings: list[Opening] | None = None) -> RoomGeometry:
    if not depths.maps:
        return fallback_room(shelves, depths.review + ["room geometry needs metric depth; room size unknown"])
    frames = [still for still in stills if still.seq in depths.maps]
    cameras = [camera(still, depths.maps[still.seq].shape) for still in frames]
    rotations = pan_rotations(frames, cameras)
    clouds, refs = [], []
    for still, k, rotation in zip(frames, cameras, rotations):
        if rotation is not None:
            cloud = frame_points(depths.maps[still.seq], k) @ rotation.T
            clouds.append(cloud)
            refs += [still.path] * len(cloud)
    review = list(depths.review)
    dropped = sum(rotation is None for rotation in rotations)
    if dropped:
        review.append(f"{dropped} room frames could not be registered to the pan")
    registered = [rotation for rotation in rotations if rotation is not None]
    cfg = load_room_config().depth
    up = pan_up(registered) if registered else None
    if up is not None and upright(registered, up) < cfg.min_image_up_alignment:
        review.append("room frames are not upright (image up is far from vertical): camera orientation mismatch, "
                      "check the phone orientation lock; room size unknown")
        return unmeasured(shelves, depths.scale_method, 0.0, review)
    confidence = depths.confidence * drift_factor(loop_drift(frames, cameras, rotations), review)
    room = room_from_points(
        np.vstack(clouds) if clouds else np.empty((0, 3)),
        refs,
        shelves,
        scale_method=depths.scale_method,
        scale_confidence=confidence,
        origins=np.zeros(3),
        openings=openings,
        up=up,
        max_tilt_deg=cfg.max_tilt_deg,
    )
    if NO_FLOOR in room.review:
        review.append(f"no floor within {cfg.max_tilt_deg:.0f} degrees of the camera up direction: "
                      "possible camera orientation mismatch")
    room.review = review + room.review
    return room
