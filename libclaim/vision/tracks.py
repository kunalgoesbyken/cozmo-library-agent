import cv2
import numpy as np
from pydantic import BaseModel

from libclaim.models import Still
from libclaim.vision.boxes import Box, TrackConfig, containment, iou, load_vision_config, read_image

HOMOGRAPHY_MIN_MATCHES = 4


class Frame(BaseModel):
    still: Still
    boxes: list[Box]


class BoxRef(BaseModel):
    frame: int
    box: Box


class BookTrack(BaseModel):
    track_id: int
    refs: list[BoxRef]
    best_frame: Still
    best_box: Box


def plausible(matrix: np.ndarray, width: int, height: int, cfg: TrackConfig) -> bool:
    if not np.isfinite(matrix).all():
        return False
    if not cfg.det_min <= abs(np.linalg.det(matrix[:2, :2])) <= cfg.det_max:
        return False
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]]).reshape(-1, 1, 2)
    moved = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
    if not np.isfinite(moved).all():
        return False
    ratio = cv2.contourArea(moved.astype(np.float32)) / (width * height)
    return cfg.area_ratio_min <= ratio <= cfg.area_ratio_max


def pair_homography(prev: np.ndarray, cur: np.ndarray, cfg: TrackConfig) -> np.ndarray | None:
    orb = cv2.ORB_create(cfg.orb_features)
    kp_prev, des_prev = orb.detectAndCompute(prev, None)
    kp_cur, des_cur = orb.detectAndCompute(cur, None)
    if des_prev is None or des_cur is None:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des_cur, des_prev, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < cfg.match_ratio * p[1].distance]
    if len(good) < HOMOGRAPHY_MIN_MATCHES:
        return None
    src = np.float32([kp_cur[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_prev[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, cfg.ransac_reproj_px)
    if matrix is None or int(mask.sum()) < cfg.min_inliers:
        return None
    height, width = cur.shape[:2]
    return matrix if plausible(matrix, width, height, cfg) else None


def project(box: Box, matrix: np.ndarray) -> Box:
    corners = np.float32([[box.x1, box.y1], [box.x2, box.y1], [box.x2, box.y2], [box.x1, box.y2]]).reshape(-1, 1, 2)
    moved = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
    return Box(x1=moved[:, 0].min(), y1=moved[:, 1].min(), x2=moved[:, 0].max(), y2=moved[:, 1].max())


def register(grays: list[np.ndarray], cfg: TrackConfig) -> list[tuple[int, np.ndarray]]:
    placed: list[tuple[int, np.ndarray]] = []
    for index, gray in enumerate(grays):
        group, matrix = index, np.eye(3)
        for earlier in range(index - 1, max(index - 1 - cfg.anchor_lookback, -1), -1):
            step = pair_homography(grays[earlier], gray, cfg)
            if step is not None:
                group, matrix = placed[earlier][0], placed[earlier][1] @ step
                break
        placed.append((group, matrix))
    return placed


def touched_edges(box: Box, width: int, height: int, cfg: TrackConfig) -> frozenset[str]:
    hits = {
        "left": box.x1 <= cfg.edge_px,
        "top": box.y1 <= cfg.edge_px,
        "right": box.x2 >= width - cfg.edge_px,
        "bottom": box.y2 >= height - cfg.edge_px,
    }
    return frozenset(edge for edge, hit in hits.items() if hit)


def track_overlap(box: Box, cut: frozenset[str], track: list[tuple[int, Box, Box, int, frozenset[str]]], group: int, cfg: TrackConfig) -> float:
    best = 0.0
    for _, _, ref, g, ref_cut in track:
        if g != group:
            continue
        overlap = iou(box, ref)
        if overlap < cfg.track_iou and (cut or ref_cut) and containment(box, ref) >= cfg.cut_containment:
            overlap = cfg.track_iou
        best = max(best, overlap)
    return best


def views_of(track: list[tuple[int, Box, Box, int, frozenset[str]]], frames: list[Frame], placed: list[tuple[int, np.ndarray]], cfg: TrackConfig) -> int:
    group = track[0][3]
    cx = float(np.mean([(m[2].x1 + m[2].x2) / 2 for m in track]))
    cy = float(np.mean([(m[2].y1 + m[2].y2) / 2 for m in track]))
    views = 0
    for frame, (g, matrix) in zip(frames, placed):
        if g != group:
            continue
        field = project(Box(x1=0, y1=0, x2=frame.still.w, y2=frame.still.h), matrix)
        views += field.x1 + cfg.edge_px < cx < field.x2 - cfg.edge_px and field.y1 < cy < field.y2
    return views


def is_noise(track: list[tuple[int, Box, Box, int, frozenset[str]]], frames: list[Frame], placed: list[tuple[int, np.ndarray]], cfg: TrackConfig) -> bool:
    views = views_of(track, frames, placed, cfg)
    return views >= cfg.noise_min_views and len(track) / views <= cfg.noise_max_rate


def merge_tracks(frames: list[Frame]) -> list[BookTrack]:
    cfg = load_vision_config().tracks
    grays = [read_image(f.still.path, cv2.IMREAD_GRAYSCALE) for f in frames]
    placed = register(grays, cfg)
    members: list[list[tuple[int, Box, Box, int, frozenset[str]]]] = []
    for index, frame in enumerate(frames):
        group, matrix = placed[index]
        moved = [project(box, matrix) for box in frame.boxes]
        cuts = [touched_edges(box, frame.still.w, frame.still.h, cfg) for box in frame.boxes]
        candidates = sorted(
            (overlap, bi, ti)
            for bi, box in enumerate(moved)
            for ti, track in enumerate(members)
            for overlap in [track_overlap(box, cuts[bi], track, group, cfg)]
            if overlap >= cfg.track_iou
        )[::-1]
        assigned: dict[int, int] = {}
        taken: set[int] = set()
        for _, bi, ti in candidates:
            if bi not in assigned and ti not in taken:
                assigned[bi] = ti
                taken.add(ti)
        for bi, box in enumerate(frame.boxes):
            if bi not in assigned:
                members.append([])
                assigned[bi] = len(members) - 1
            members[assigned[bi]].append((index, box, moved[bi], group, cuts[bi]))
    members = [track for track in members if not is_noise(track, frames, placed, cfg)]
    tracks = []
    for track_id, track in enumerate(members):
        sharpest = max(track, key=lambda m: (-len(m[4]), frames[m[0]].still.blur))
        tracks.append(
            BookTrack(
                track_id=track_id,
                refs=[BoxRef(frame=i, box=b) for i, b, _, _, _ in track],
                best_frame=frames[sharpest[0]].still,
                best_box=sharpest[1],
            )
        )
    return tracks
