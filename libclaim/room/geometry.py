import math
from dataclasses import dataclass
from functools import cache

import cv2
import numpy as np
import yaml
from pydantic import BaseModel, Field

from libclaim.packet.models import Opening, Room
from libclaim.settings import CONFIG_DIR

PLANE_SAMPLE = 3
SYMMETRY = 4


class RansacConfig(BaseModel):
    iterations: int
    sample_points: int
    neighbour_pool: int
    local_radius_m: float
    inlier_m: float
    max_tilt_deg: float
    min_inlier_fraction: float
    max_planes: int
    seed: int


class ScaleSource(BaseModel):
    scale_method: str
    confidence: float


class DepthConfig(BaseModel):
    model_id: str
    stride: int
    min_m: float
    max_m: float
    camera_up: list[float]
    min_right_spread: float
    max_tilt_deg: float
    marker: ScaleSource
    prior: ScaleSource


class RoomConfig(BaseModel):
    up_axis: list[float]
    min_points: int
    decimals: int
    ransac: RansacConfig
    ceiling_min_height_m: float
    ceiling_margin_m: float
    band_clearance_m: float
    grid_m: float
    min_cell_points: int
    close_cells: int
    simplify_m: float
    snap_tolerance_deg: float
    wall_tolerance_m: float
    min_edge_m: float
    max_wall_fill_ratio: float
    ceiling_percentile: float
    shapes: dict[int, str]
    other_shape: str
    fallback: ScaleSource
    depth: DepthConfig


class ShelfUnit(BaseModel):
    width_m: float
    height_m: float


class RoomGeometry(Room):
    review: list[str] = Field(default_factory=list)


class RoomUnmeasurable(ValueError):
    pass


@dataclass
class Plane:
    normal: np.ndarray
    offset: float

    def distance(self, points: np.ndarray) -> np.ndarray:
        return points @ self.normal - self.offset


@dataclass
class Line:
    normal: np.ndarray
    offset: float
    length: float
    snapped: bool


@dataclass
class Grid:
    cells: np.ndarray
    origin: np.ndarray
    step: float

    def index(self, q: np.ndarray) -> np.ndarray:
        return np.floor((q - self.origin) / self.step).astype(int)

    def to_metric(self, cells: np.ndarray) -> np.ndarray:
        return self.origin + (cells + 0.5) * self.step


@cache
def load_room_config() -> RoomConfig:
    path = CONFIG_DIR / "room.yaml"
    return RoomConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def unit(vector) -> np.ndarray:
    v = np.asarray(vector, dtype=float)
    return v / np.linalg.norm(v)


def rotation_2d(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])


def shelved_area(shelves: list[ShelfUnit]) -> float | None:
    return sum(s.width_m * s.height_m for s in shelves) if shelves else None


def unmeasured(shelves: list[ShelfUnit], method: str, confidence: float, review: list[str],
               height_m: float | None = None) -> RoomGeometry:
    return RoomGeometry(
        height_m=height_m,
        shelved_wall_area_m2=shelved_area(shelves),
        scale_method=method,
        confidence=confidence,
        review=review,
    )


def fallback_room(shelves: list[ShelfUnit], review: list[str]) -> RoomGeometry:
    source = load_room_config().fallback
    return unmeasured(shelves, source.scale_method, source.confidence, review)


def ransac_plane(points: np.ndarray, up: np.ndarray, max_tilt_deg: float, total: int, cfg: RansacConfig,
                 rng: np.random.Generator) -> Plane | None:
    sample = points[rng.choice(len(points), min(len(points), cfg.sample_points), replace=False)]
    min_cos = math.cos(math.radians(max_tilt_deg))
    best, best_count = None, 0
    pool = sample[: cfg.neighbour_pool]
    for _ in range(cfg.iterations):
        a = pool[rng.integers(len(pool))]
        offsets = pool - a
        near = np.flatnonzero(np.einsum("ij,ij->i", offsets, offsets) < cfg.local_radius_m**2)
        b, c = pool[rng.choice(near, PLANE_SAMPLE - 1)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm == 0:
            continue
        normal = normal / norm
        normal = normal if normal @ up >= 0 else -normal
        if normal @ up < min_cos:
            continue
        count = np.count_nonzero(np.abs(sample @ normal - normal @ a) < cfg.inlier_m)
        if count > best_count:
            best, best_count = Plane(normal, float(normal @ a)), count
    if best is None or best_count * len(points) / len(sample) < cfg.min_inlier_fraction * total:
        return None
    inliers = points[np.abs(best.distance(points)) < cfg.inlier_m]
    centre = inliers.mean(axis=0)
    normal = np.linalg.svd(inliers - centre, full_matrices=False)[2][-1]
    normal = normal if normal @ up >= 0 else -normal
    return Plane(normal, float(normal @ centre))


def horizontal_planes(points: np.ndarray, up: np.ndarray, max_tilt_deg: float, cfg: RansacConfig) -> list[Plane]:
    rng = np.random.default_rng(cfg.seed)
    planes: list[Plane] = []
    remaining = points
    for _ in range(cfg.max_planes):
        if len(remaining) < PLANE_SAMPLE:
            break
        plane = ransac_plane(remaining, up, max_tilt_deg, len(points), cfg, rng)
        if plane is None:
            break
        planes.append(plane)
        remaining = remaining[np.abs(plane.distance(remaining)) >= cfg.inlier_m]
    return planes


def pick_floor(planes: list[Plane], points: np.ndarray) -> Plane | None:
    below_everything = [p for p in planes if np.median(p.distance(points)) > 0]
    return min(below_everything, key=lambda p: p.offset, default=None)


def ceiling_height(planes: list[Plane], floor: Plane, points: np.ndarray, heights: np.ndarray,
                   lowest: float, cfg: RoomConfig) -> tuple[float | None, np.ndarray]:
    best, mask = None, np.zeros(len(points), dtype=bool)
    for plane in planes:
        if plane is floor:
            continue
        inliers = np.abs(plane.distance(points)) < cfg.ransac.inlier_m
        height = float(np.median(heights[inliers]))
        if height >= lowest and (best is None or height > best):
            best, mask = height, inliers
    return best, mask


def occupancy(q: np.ndarray, cfg: RoomConfig) -> Grid:
    cells, counts = np.unique(np.floor(q / cfg.grid_m).astype(np.int64), axis=0, return_counts=True)
    kept = cells[counts >= cfg.min_cell_points]
    if not len(kept):
        raise RoomUnmeasurable("no wall points above the shelving")
    pad = cfg.close_cells + 1
    low = kept.min(axis=0) - pad
    width, height = kept.max(axis=0) - low + pad + 1
    image = np.zeros((height, width), np.uint8)
    image[kept[:, 1] - low[1], kept[:, 0] - low[0]] = 1
    return Grid(image, low * cfg.grid_m, cfg.grid_m)


def outer_polygon(q: np.ndarray, cfg: RoomConfig) -> np.ndarray:
    grid = occupancy(q, cfg)
    kernel = np.ones((cfg.close_cells, cfg.close_cells), np.uint8)
    closed = cv2.morphologyEx(grid.cells, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RoomUnmeasurable("no wall points above the shelving")
    contour = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(closed)
    cv2.drawContours(filled, [contour], -1, 1, cv2.FILLED)
    inside = np.count_nonzero(filled)
    if not inside or np.count_nonzero(closed & filled) / inside > cfg.max_wall_fill_ratio:
        raise RoomUnmeasurable("wall points do not enclose the room; pan along every wall-ceiling line")
    approx = cv2.approxPolyDP(contour, cfg.simplify_m / cfg.grid_m, True).reshape(-1, 2)
    return grid.to_metric(approx)


def manhattan_angle(polygon: np.ndarray) -> float:
    d = np.roll(polygon, -1, axis=0) - polygon
    lengths = np.hypot(d[:, 0], d[:, 1])
    phi = np.arctan2(d[:, 1], d[:, 0])
    return float(np.angle(np.sum(lengths * np.exp(SYMMETRY * 1j * phi))) / SYMMETRY)


def parallel(a: Line, b: Line, tolerance: float) -> bool:
    return abs(a.normal @ b.normal) >= math.cos(tolerance)


def combine(a: Line, b: Line) -> Line:
    sign = 1.0 if a.normal @ b.normal >= 0 else -1.0
    total = a.length + b.length
    return Line(a.normal, (a.offset * a.length + sign * b.offset * b.length) / total, total, a.snapped and b.snapped)


def polygon_lines(polygon: np.ndarray, cfg: RoomConfig) -> list[Line]:
    tolerance = math.radians(cfg.snap_tolerance_deg)
    quarter = math.pi / 2
    lines: list[Line] = []
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        d = end - start
        length = float(np.hypot(*d))
        if length < cfg.min_edge_m:
            continue
        phi = math.atan2(d[1], d[0])
        snapped = abs((phi + quarter / 2) % quarter - quarter / 2) <= tolerance
        if snapped:
            normal = np.array([0.0, 1.0]) if round(phi / quarter) % 2 == 0 else np.array([1.0, 0.0])
        else:
            normal = np.array([-d[1], d[0]]) / length
        lines.append(Line(normal, float(normal @ (start + end) / 2), length, snapped))
    merged: list[Line] = []
    for line in lines:
        if merged and parallel(merged[-1], line, tolerance):
            merged[-1] = combine(merged[-1], line)
        else:
            merged.append(line)
    if len(merged) > 1 and parallel(merged[0], merged[-1], tolerance):
        merged[0] = combine(merged.pop(), merged[0])
    if len(merged) < PLANE_SAMPLE:
        raise RoomUnmeasurable("wall outline has fewer than 3 walls")
    return merged


def vertices(lines: list[Line]) -> np.ndarray:
    try:
        return np.array([
            np.linalg.solve(np.array([prev.normal, line.normal]), np.array([prev.offset, line.offset]))
            for prev, line in zip(lines[-1:] + lines[:-1], lines)
        ])
    except np.linalg.LinAlgError as error:
        raise RoomUnmeasurable("adjacent walls are parallel; outline is ambiguous") from error


def wall_mask(line: Line, start: np.ndarray, end: np.ndarray, q: np.ndarray, tolerance: float) -> np.ndarray:
    length = np.linalg.norm(end - start)
    if length == 0:
        return np.zeros(len(q), dtype=bool)
    along = (q - start) @ ((end - start) / length)
    near = np.abs(q @ line.normal - line.offset) <= tolerance
    return near & (along >= tolerance) & (along <= length - tolerance)


def refine(lines: list[Line], q: np.ndarray, cfg: RoomConfig) -> np.ndarray:
    corners = vertices(lines)
    for i, line in enumerate(lines):
        near = wall_mask(line, corners[i], corners[(i + 1) % len(lines)], q, cfg.wall_tolerance_m)
        if near.any():
            line.offset += float(np.median(q[near] @ line.normal - line.offset))
    return vertices(lines)


def wall_angle(lines: list[Line], corners: np.ndarray, q: np.ndarray, cfg: RoomConfig) -> float:
    angles, weights = [], []
    for i, line in enumerate(lines):
        near = wall_mask(line, corners[i], corners[(i + 1) % len(lines)], q, cfg.wall_tolerance_m)
        if line.snapped and np.count_nonzero(near) >= PLANE_SAMPLE:
            wall = q[near] - q[near].mean(axis=0)
            direction = np.linalg.svd(wall, full_matrices=False)[2][0]
            angles.append(math.atan2(direction[1], direction[0]))
            weights.append(np.count_nonzero(near))
    if not angles:
        return 0.0
    return float(np.angle(np.sum(np.array(weights) * np.exp(SYMMETRY * 1j * np.array(angles)))) / SYMMETRY)


def aligned_outline(points_2d: np.ndarray, band: np.ndarray, angle: float,
                    cfg: RoomConfig) -> tuple[np.ndarray, list[Line], np.ndarray]:
    flat = points_2d @ rotation_2d(-angle).T
    lines = polygon_lines(outer_polygon(flat[band], cfg), cfg)
    return flat, lines, refine(lines, flat[band], cfg)


def shoelace(polygon: np.ndarray) -> float:
    x, y = polygon[:, 0], polygon[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)) / 2)


def perimeter(polygon: np.ndarray) -> float:
    return float(np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1).sum())


def coverage(polygon: np.ndarray, q: np.ndarray, cfg: RoomConfig) -> float:
    grid = occupancy(q, cfg)
    reach = 2 * math.ceil(cfg.wall_tolerance_m / cfg.grid_m) + 1
    near = cv2.dilate(grid.cells, np.ones((reach, reach), np.uint8))
    samples = []
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        count = max(int(np.linalg.norm(end - start) / cfg.grid_m), 1)
        samples.append(start + np.outer((np.arange(count) + 0.5) / count, end - start))
    cells = grid.index(np.vstack(samples))
    rows, cols = near.shape
    inside = (cells[:, 0] >= 0) & (cells[:, 0] < cols) & (cells[:, 1] >= 0) & (cells[:, 1] < rows)
    hits = np.zeros(len(cells), dtype=bool)
    hits[inside] = near[cells[inside, 1], cells[inside, 0]] > 0
    return float(hits.mean())


def ray_plane_height(lines: list[Line], corners: np.ndarray, flat: np.ndarray, heights: np.ndarray,
                     origin_flat: np.ndarray | None, origin_heights: np.ndarray | None, start: float,
                     cfg: RoomConfig) -> float | None:
    per_wall = []
    for i, line in enumerate(lines):
        mask = wall_mask(line, corners[i], corners[(i + 1) % len(lines)], flat, cfg.wall_tolerance_m)
        mask &= heights >= start
        if origin_flat is None:
            hits = heights[mask]
        else:
            reach = (flat[mask] - origin_flat[mask]) @ line.normal
            ok = np.abs(reach) >= cfg.wall_tolerance_m
            t = (line.offset - origin_flat[mask][ok] @ line.normal) / reach[ok]
            base = origin_heights[mask][ok]
            hits = base + t * (heights[mask][ok] - base)
        if len(hits) >= cfg.min_cell_points:
            per_wall.append(float(np.percentile(hits, cfg.ceiling_percentile)))
    return float(np.median(per_wall)) if per_wall else None


def room_from_points(points, frames, shelves: list[ShelfUnit], *, scale_method: str, scale_confidence: float,
                     origins=None, openings: list[Opening] | None = None, up=None,
                     max_tilt_deg: float | None = None) -> RoomGeometry:
    cfg = load_room_config()
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    refs = np.asarray(frames, dtype=str)
    finite = np.isfinite(pts).all(axis=1)
    origin_pts = None if origins is None else np.broadcast_to(np.asarray(origins, dtype=float), pts.shape)[finite]
    pts, refs = pts[finite], refs[finite]
    review: list[str] = []
    if len(pts) < cfg.min_points:
        return unmeasured(shelves, scale_method, 0.0, [f"only {len(pts)} room points, need {cfg.min_points}"])
    tilt = cfg.ransac.max_tilt_deg if max_tilt_deg is None else max_tilt_deg
    planes = horizontal_planes(pts, unit(cfg.up_axis if up is None else up), tilt, cfg.ransac)
    floor = pick_floor(planes, pts)
    if floor is None:
        return unmeasured(shelves, scale_method, 0.0, ["no floor plane found in the room points"])
    up_axis = floor.normal
    heights = pts @ up_axis - floor.offset
    shelf_top = max((s.height_m for s in shelves), default=0.0)
    start = shelf_top + cfg.band_clearance_m
    ceiling, ceiling_mask = ceiling_height(
        planes, floor, pts, heights, max(start, cfg.ceiling_min_height_m), cfg
    )
    band = heights >= start
    if ceiling is not None:
        band &= heights <= ceiling - cfg.ceiling_margin_m
    side = unit(np.cross(up_axis, np.eye(3)[np.argmin(np.abs(up_axis))]))
    basis = np.array([side, np.cross(up_axis, side)])
    plan = pts @ basis.T
    try:
        angle = manhattan_angle(outer_polygon(plan[band], cfg))
        flat, lines, corners = aligned_outline(plan, band, angle, cfg)
        angle += wall_angle(lines, corners, flat[band], cfg)
        flat, lines, corners = aligned_outline(plan, band, angle, cfg)
    except RoomUnmeasurable as error:
        return unmeasured(shelves, scale_method, 0.0, [str(error)], ceiling)
    if ceiling is None:
        origin_flat = None if origin_pts is None else origin_pts @ basis.T @ rotation_2d(-angle).T
        origin_heights = None if origin_pts is None else origin_pts @ up_axis - floor.offset
        ceiling = ray_plane_height(lines, corners, flat, heights, origin_flat, origin_heights, start, cfg)
        review.append("no ceiling plane; height from wall points intersected with the wall planes")
    if ceiling is None:
        review.append("ceiling height could not be measured")
    snapped = all(line.snapped for line in lines)
    shape = cfg.shapes.get(len(lines), cfg.other_shape) if snapped else cfg.other_shape
    gross = None if ceiling is None else perimeter(corners) * ceiling
    net = None
    if openings is None:
        review.append("doors and windows not measured; net wall area unknown")
    elif gross is not None:
        net = gross - sum(o.width_m * o.height_m for o in openings if o.width_m and o.height_m)
    used = band | (np.abs(floor.distance(pts)) < cfg.ransac.inlier_m) | ceiling_mask
    extent = corners.max(axis=0) - corners.min(axis=0)

    def rounded(value: float | None) -> float | None:
        return None if value is None else round(value, cfg.decimals)

    return RoomGeometry(
        length_m=rounded(float(extent.max())),
        width_m=rounded(float(extent.min())),
        height_m=rounded(ceiling),
        floor_area_m2=rounded(shoelace(corners)),
        wall_area_m2=rounded(gross),
        wall_area_net_m2=rounded(net),
        shelved_wall_area_m2=rounded(shelved_area(shelves)),
        openings=openings or [],
        shape=shape,
        polygon_m=[[rounded(x), rounded(y)] for x, y in corners - corners.min(axis=0)],
        frame_refs=sorted(set(refs[used].tolist())),
        scale_method=scale_method,
        confidence=rounded(scale_confidence * coverage(corners, flat[band], cfg)),
        review=review,
    )
