import cv2
import numpy as np
import pytest

from libclaim.models import Still
from libclaim.vision import tracks as tracks_module
from libclaim.vision.boxes import Box, ImageUnreadable, load_vision_config
from libclaim.vision.tracks import Frame, merge_tracks, plausible

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


def test_mid_chain_registration_failure_still_gives_n_tracks(tmp_path, monkeypatch):
    frames = make_frames(tmp_path)
    real = tracks_module.pair_homography
    calls = []

    def flaky(prev, cur, cfg):
        calls.append(1)
        return None if len(calls) == 2 else real(prev, cur, cfg)

    monkeypatch.setattr(tracks_module, "pair_homography", flaky)
    tracks = merge_tracks(frames)
    assert len(tracks) == SPINES
    assert all(len(t.refs) == 3 for t in tracks)


def test_identical_adjacent_books_do_not_merge(tmp_path):
    rng = np.random.default_rng(3)
    tile = rng.integers(0, 256, (SPINE_H, SPINE_W, 3), dtype=np.uint8)
    canvas = np.full((SCENE_H, SCENE_W, 3), 40, np.uint8)
    spans = []
    for index in range(SPINES):
        x1 = 40 + index * SPINE_W
        canvas[40 : 40 + SPINE_H, x1 : x1 + SPINE_W] = tile
        spans.append((x1, 40, x1 + SPINE_W, 40 + SPINE_H))
    cv2.putText(canvas, "ANCHOR", (50, 380), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    frames = []
    for seq, shift in enumerate(SHIFTS):
        path = tmp_path / f"{seq}.png"
        cv2.imwrite(str(path), canvas[:, shift : shift + FRAME_W])
        still = Still(seq=seq, t_ms=seq, path=str(path), w=FRAME_W, h=SCENE_H, blur=1.0 + seq,
                      glare=0.0, motion_dps=0.0)
        frames.append(Frame(still=still, boxes=[Box(x1=a - shift, y1=b, x2=c - shift, y2=d) for a, b, c, d in spans]))
    tracks = merge_tracks(frames)
    assert len(tracks) == SPINES
    assert all(len(t.refs) == 3 for t in tracks)


def test_degenerate_homography_is_rejected():
    cfg = load_vision_config().tracks
    collapsed = np.array([[0.01, 0, 0], [0, 0.01, 0], [0, 0, 1.0]])
    flipped = np.array([[1.0, 0, 0], [0, 1.0, 0], [0.01, 0, 1.0]])
    assert not plausible(collapsed, FRAME_W, SCENE_H, cfg)
    assert not plausible(flipped, FRAME_W, SCENE_H, cfg)
    assert plausible(np.eye(3), FRAME_W, SCENE_H, cfg)


def test_unreadable_frame_raises_clear_error(tmp_path):
    frames = make_frames(tmp_path)
    frames[1].still.path = str(tmp_path / "gone.png")
    with pytest.raises(ImageUnreadable):
        merge_tracks(frames)


def test_books_cut_by_the_frame_edge_do_not_become_extra_tracks(tmp_path):
    canvas, spans = scene()
    width = 620
    frames = []
    for seq, shift in enumerate([0, 60]):
        path = tmp_path / f"cut{seq}.png"
        cv2.imwrite(str(path), canvas[:, shift : shift + width])
        still = Still(seq=seq, t_ms=seq * 500, path=str(path), w=width, h=SCENE_H, blur=0.5, glare=0.0, motion_dps=0.0)
        boxes = []
        for x1, y1, x2, y2 in spans:
            left, right = max(x1 - shift, 0), min(x2 - shift, width)
            if right - left > 5:
                boxes.append(Box(x1=left, y1=y1, x2=right, y2=y2))
        frames.append(Frame(still=still, boxes=boxes))
    found = merge_tracks(frames)
    assert len(found) == SPINES
    for track in found:
        assert track.best_box.width == SPINE_W
