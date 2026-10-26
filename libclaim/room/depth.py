from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache
from statistics import median

import cv2
import numpy as np

from libclaim.models import Still
from libclaim.packet.models import Opening
from libclaim.room.geometry import RoomGeometry, ShelfUnit, fallback_room, load_room_config, room_from_points
from libclaim.settings import load_settings
from libclaim.vision import masks
from libclaim.vision.boxes import load_vision_config, read_image
from libclaim.vision.scale import aruco_detector, camera_matrix
from libclaim.vision.tracks import pair_homography

CM_PER_M = 100.0
QUAD = 4


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
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    return AutoImageProcessor.from_pretrained(model_id), AutoModelForDepthEstimation.from_pretrained(model_id).eval()


def metric_depth(still: Still) -> np.ndarray:
    if not masks.available():
        raise DepthUnavailable("metric room depth needs the [ml] extra (torch, transformers)")
    try:
        processor, model = depth_model(load_room_config().depth.model_id)
    except (ImportError, OSError) as error:
        raise DepthUnavailable(f"metric depth model could not be loaded: {type(error).__name__}") from error
    import torch

    rgb = np.ascontiguousarray(read_image(still.path)[:, :, ::-1])
    inputs = processor(images=rgb, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    result = processor.post_process_depth_estimation(outputs, target_sizes=[rgb.shape[:2]])
    return result[0]["predicted_depth"].numpy()


def camera(still: Still, shape: tuple[int, ...]) -> np.ndarray:
    height, width = shape[:2]
    return np.diag([width / still.w, height / still.h, 1.0]) @ camera_matrix(still, load_vision_config().scale)


def backproject(pixels: np.ndarray, values: np.ndarray, k: np.ndarray) -> np.ndarray:
    rays = np.column_stack([pixels, np.ones(len(pixels))]) @ np.linalg.inv(k).T
    return rays * values[:, None]


def marker_factor(still: Still, depth: np.ndarray, size_m: float) -> float | None:
    cfg = load_vision_config().scale
    corners, ids, _ = aruco_detector(cfg).detectMarkers(read_image(still.path, cv2.IMREAD_GRAYSCALE))
    if ids is None:
        return None
    k = camera(still, depth.shape)
    height, width = depth.shape
    edges = []
    for quad in corners:
        quad = quad.reshape(QUAD, 2)
        if np.mean(np.linalg.norm(quad - np.roll(quad, 1, axis=0), axis=1)) < cfg.min_marker_px:
            continue
        cells = np.clip(np.rint(quad).astype(int), 0, [width - 1, height - 1])
        points = backproject(quad, depth[cells[:, 1], cells[:, 0]], k)
        edges += np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1).tolist()
    return size_m / median(edges) if edges else None


def scaled_depths(stills: list[Still], depth_fn: Callable[[Still], np.ndarray] = metric_depth,
                  marker_size_cm: float | None = None) -> DepthSet:
    cfg = load_room_config().depth
    try:
        raw = {still.seq: depth_fn(still) for still in stills}
    except DepthUnavailable as error:
        return DepthSet({}, "", 0.0, [str(error)])
    size_cm = marker_size_cm if marker_size_cm is not None else load_settings().aruco_marker_size_cm
    factors: dict[int, float] = {}
    if size_cm is not None:
        for still in stills:
            factor = marker_factor(still, raw[still.seq], size_cm / CM_PER_M)
            if factor is not None:
                factors[still.seq] = factor
    if not factors:
        why = "ARUCO_MARKER_SIZE_CM is not set" if size_cm is None else "no ArUco marker in the room frames"
        return DepthSet(raw, cfg.prior.scale_method, cfg.prior.confidence, [f"{why}: depth scale is the model's own"])
    default = median(factors.values())
    maps = {seq: depth * factors.get(seq, default) for seq, depth in raw.items()}
    return DepthSet(maps, cfg.marker.scale_method, cfg.marker.confidence)


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


def pan_up(rotations: list[np.ndarray]) -> np.ndarray:
    cfg = load_room_config().depth
    hint = np.mean([r @ cfg.camera_up for r in rotations], axis=0)
    _, spread, vt = np.linalg.svd(np.array([r[:, 0] for r in rotations]))
    if len(spread) < 2 or spread[1] < cfg.min_right_spread * spread[0]:
        return hint / np.linalg.norm(hint)
    return vt[-1] if vt[-1] @ hint >= 0 else -vt[-1]


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
    room = room_from_points(
        np.vstack(clouds) if clouds else np.empty((0, 3)),
        refs,
        shelves,
        scale_method=depths.scale_method,
        scale_confidence=depths.confidence,
        origins=np.zeros(3),
        openings=openings,
        up=pan_up(registered) if registered else None,
        max_tilt_deg=load_room_config().depth.max_tilt_deg,
    )
    room.review = review + room.review
    return room
