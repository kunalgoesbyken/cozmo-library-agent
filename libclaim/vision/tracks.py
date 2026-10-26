import cv2
import numpy as np
from pydantic import BaseModel

from libclaim.models import Still
from libclaim.vision.boxes import Box, TrackConfig, iou, load_vision_config

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
    return matrix


def project(box: Box, matrix: np.ndarray) -> Box:
    corners = np.float32([[box.x1, box.y1], [box.x2, box.y1], [box.x2, box.y2], [box.x1, box.y2]]).reshape(-1, 1, 2)
    moved = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
    return Box(x1=moved[:, 0].min(), y1=moved[:, 1].min(), x2=moved[:, 0].max(), y2=moved[:, 1].max())


def merge_tracks(frames: list[Frame]) -> list[BookTrack]:
    cfg = load_vision_config().tracks
    grays = [cv2.imread(f.still.path, cv2.IMREAD_GRAYSCALE) for f in frames]
    to_reference: list[np.ndarray | None] = []
    for index, gray in enumerate(grays):
        if index == 0:
            to_reference.append(np.eye(3))
            continue
        step = pair_homography(grays[index - 1], gray, cfg)
        before = to_reference[index - 1]
        to_reference.append(None if step is None or before is None else before @ step)
    members: list[list[tuple[int, Box, Box | None]]] = []
    for index, frame in enumerate(frames):
        matrix = to_reference[index]
        taken: set[int] = set()
        for box in frame.boxes:
            moved = project(box, matrix) if matrix is not None else None
            best, best_iou = None, cfg.track_iou
            if moved is not None:
                for track_id, track in enumerate(members):
                    if track_id in taken:
                        continue
                    overlap = max((iou(moved, ref) for _, _, ref in track if ref is not None), default=0.0)
                    if overlap >= best_iou:
                        best, best_iou = track_id, overlap
            if best is None:
                members.append([])
                best = len(members) - 1
            taken.add(best)
            members[best].append((index, box, moved))
    tracks = []
    for track_id, track in enumerate(members):
        sharpest = max(track, key=lambda m: frames[m[0]].still.blur)
        tracks.append(
            BookTrack(
                track_id=track_id,
                refs=[BoxRef(frame=i, box=b) for i, b, _ in track],
                best_frame=frames[sharpest[0]].still,
                best_box=sharpest[1],
            )
        )
    return tracks
