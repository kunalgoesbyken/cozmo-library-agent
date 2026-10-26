import cv2
import numpy as np

from libclaim.models import Still
from libclaim.vision.boxes import Box
from libclaim.vision.tracks import Frame, merge_tracks

SPINES = 8
SPINE_W = 60
SPINE_H = 300
SCENE_W = 800
SCENE_H = 400
FRAME_W = 700
SHIFTS = [0, 18, 36]
BLURS = [0.3, 0.9, 0.5]


def scene():
    rng = np.random.default_rng(7)
    canvas = np.full((SCENE_H, SCENE_W, 3), 40, np.uint8)
    spans = []
    for index in range(SPINES):
        x1 = 40 + index * (SPINE_W + 20)
        canvas[40 : 40 + SPINE_H, x1 : x1 + SPINE_W] = rng.integers(60, 255, 3, dtype=np.uint8)
        noise = rng.integers(0, 90, (SPINE_H, SPINE_W, 3), dtype=np.uint8)
        canvas[40 : 40 + SPINE_H, x1 : x1 + SPINE_W] += noise
        cv2.putText(canvas, f"T{index}", (x1 + 5, 120 + index * 7), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        spans.append((x1, 40, x1 + SPINE_W, 40 + SPINE_H))
    return canvas, spans


def make_frames(tmp_path):
    canvas, spans = scene()
    frames = []
    for seq, (shift, blur) in enumerate(zip(SHIFTS, BLURS)):
        img = canvas[:, shift : shift + FRAME_W].copy()
        path = tmp_path / f"{seq}.png"
        cv2.imwrite(str(path), img)
        still = Still(seq=seq, t_ms=seq * 500, path=str(path), w=FRAME_W, h=SCENE_H, blur=blur,
                      glare=0.0, motion_dps=0.0)
        boxes = [
            Box(x1=x1 - shift, y1=y1, x2=x2 - shift, y2=y2)
            for x1, y1, x2, y2 in spans
            if x2 - shift <= FRAME_W
        ]
        frames.append(Frame(still=still, boxes=boxes))
    return frames


def test_three_overlapping_frames_give_n_tracks_with_sharpest_best_frame(tmp_path):
    frames = make_frames(tmp_path)
    tracks = merge_tracks(frames)
    assert len(tracks) == SPINES
    assert all(len(t.refs) == 3 for t in tracks)
    assert all(t.best_frame.seq == 1 for t in tracks)
    assert all(t.best_box in frames[1].boxes for t in tracks)
    assert sorted(t.track_id for t in tracks) == list(range(SPINES))


def test_unrelated_frames_do_not_merge(tmp_path):
    frames = make_frames(tmp_path)
    rng = np.random.default_rng(99)
    other = rng.integers(0, 256, (SCENE_H, FRAME_W, 3), dtype=np.uint8)
    cv2.imwrite(frames[2].still.path, other)
    tracks = merge_tracks(frames)
    assert len(tracks) == SPINES + SPINES
