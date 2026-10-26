import math

import cv2
import numpy as np
import pytest

from libclaim.models import Still
from libclaim.settings import Settings
from libclaim.vision import masks
from libclaim.vision.boxes import Box, load_vision_config
from libclaim.vision.scale import (
    ScaleMethod,
    SpineDims,
    cross_check,
    load_measure_config,
    measure,
    measure_row,
    unit_scale,
)
from libclaim.vision.tracks import BookTrack, BoxRef

MARKER_CM = 16.0
PX_PER_CM = 20


@pytest.fixture(autouse=True)
def no_ml_extra(monkeypatch):
    monkeypatch.setattr(masks, "find_spec", lambda name: None)


@pytest.fixture
def cfg():
    return load_measure_config(Settings(aruco_marker_size_cm=MARKER_CM))


def dictionary():
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, load_vision_config().scale.aruco_dictionary))


def scene(width_cm, height_cm, px_per_cm, books, marker=None, textured=False):
    rng = np.random.default_rng(3)
    shape = (round(height_cm * px_per_cm), round(width_cm * px_per_cm))
    if textured:
        canvas = cv2.GaussianBlur(rng.integers(0, 256, shape, dtype=np.uint8), (3, 3), 0)
    else:
        canvas = np.full(shape, 230, np.uint8)
    if marker is not None:
        marker_id, x_cm, y_cm = marker
        side = round(MARKER_CM * px_per_cm)
        x, y = round(x_cm * px_per_cm), round(y_cm * px_per_cm)
        pad = px_per_cm
        canvas[y - pad : y + side + pad, x - pad : x + side + pad] = 255
        canvas[y : y + side, x : x + side] = cv2.aruco.generateImageMarker(dictionary(), marker_id, side)
    boxes = []
    for index, (x_cm, y_cm, w_cm, h_cm) in enumerate(books):
        x1, y1 = round(x_cm * px_per_cm), round(y_cm * px_per_cm)
        x2, y2 = round((x_cm + w_cm) * px_per_cm), round((y_cm + h_cm) * px_per_cm)
        canvas[y1:y2, x1:x2] = 40 + 50 * (index % 3)
        boxes.append(Box(x1=x1, y1=y1, x2=x2, y2=y2))
    return canvas, boxes


def still(tmp_path, img, seq=0, blur=1.0):
    path = tmp_path / f"{seq}.png"
    cv2.imwrite(str(path), img)
    return Still(seq=seq, t_ms=seq * 500, path=str(path), w=img.shape[1], h=img.shape[0], blur=blur,
                 glare=0.0, motion_dps=0.0)


def track(track_id, refs, stills):
    best = max(refs, key=lambda r: stills[r[0]].blur)
    return BookTrack(
        track_id=track_id,
        refs=[BoxRef(frame=f, box=b) for f, b in refs],
        best_frame=stills[best[0]],
        best_box=best[1],
    )


def rotation_warp(img, boxes, degrees):
    h, w = img.shape[:2]
    f = float(w)
    k = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    t = math.radians(degrees)
    r = np.array([[math.cos(t), 0, math.sin(t)], [0, 1, 0], [-math.sin(t), 0, math.cos(t)]])
    matrix = k @ r @ np.linalg.inv(k)
    corners = cv2.perspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2), matrix)
    x0, y0 = corners.reshape(-1, 2).min(axis=0)
    x1, y1 = corners.reshape(-1, 2).max(axis=0)
    matrix = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]]) @ matrix
    warped = cv2.warpPerspective(img, matrix, (int(x1 - x0), int(y1 - y0)), borderValue=230)
    moved = []
    for box in boxes:
        pts = np.float32([[box.x1, box.y1], [box.x2, box.y1], [box.x2, box.y2], [box.x1, box.y2]])
        out = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        moved.append(Box(x1=out[:, 0].min(), y1=out[:, 1].min(), x2=out[:, 0].max(), y2=out[:, 1].max()))
    return warped, moved


def close(value, expected, fraction):
    return abs(value - expected) <= fraction * expected


def test_marker_and_book_measure_within_three_percent(tmp_path, cfg):
    img, boxes = scene(60, 32, PX_PER_CM, [(30, 6, 3, 20)], marker=(0, 4, 8))
    frames = [still(tmp_path, img)]
    unit = unit_scale(0, frames, cfg)
    dims = measure(track(0, [(0, boxes[0])], frames), frames, cfg, unit)
    assert dims.scale_method == ScaleMethod.aruco_homography
    assert close(dims.height_cm, 20.0, 0.03)
    assert close(dims.thickness_cm, 3.0, 0.03)
    assert dims.review == []
    assert dims.frame_seq == 0


def test_no_marker_falls_back_to_format_prior_with_review(tmp_path, cfg):
    img, boxes = scene(60, 32, PX_PER_CM, [(30, 4, 3.5, 23.5)])
    frames = [still(tmp_path, img)]
    unit = unit_scale(0, frames, cfg)
    assert unit.planes == {}
    for given in (unit, None):
        dims = measure(track(0, [(0, boxes[0])], frames), frames, cfg, given)
        assert dims.scale_method == ScaleMethod.book_format_prior
        assert dims.confidence <= 0.4
        assert dims.review
    box = boxes[0]
    aspect = box.height / box.width
    nearest = min(cfg.formats.values(), key=lambda f: abs(math.log(aspect * f.thickness_cm / f.height_cm)))
    assert dims.height_cm == pytest.approx(nearest.height_cm)
    assert close(dims.thickness_cm, nearest.height_cm / aspect, 1e-6)


def test_unset_marker_size_falls_back_to_prior(tmp_path):
    cfg = load_measure_config(Settings())
    img, boxes = scene(60, 32, PX_PER_CM, [(30, 6, 3, 20)], marker=(0, 4, 8))
    frames = [still(tmp_path, img)]
    dims = measure(track(0, [(0, boxes[0])], frames), frames, cfg, unit_scale(0, frames, cfg))
    assert dims.scale_method == ScaleMethod.book_format_prior
    assert dims.review


def test_perspective_warp_measures_within_five_percent(tmp_path, cfg):
    img, boxes = scene(60, 32, PX_PER_CM, [(30, 6, 3, 20)], marker=(0, 4, 8))
    warped, moved = rotation_warp(img, boxes, 25)
    frames = [still(tmp_path, warped)]
    dims = measure(track(0, [(0, moved[0])], frames), frames, cfg, unit_scale(0, frames, cfg))
    assert dims.scale_method == ScaleMethod.aruco_homography
    assert close(dims.height_cm, 20.0, 0.05)
    assert close(dims.thickness_cm, 3.0, 0.05)


def test_two_units_with_different_scales_are_not_mixed(tmp_path, cfg):
    near, near_boxes = scene(40, 32, 20, [(25, 6, 3, 20)], marker=(0, 4, 8))
    far, far_boxes = scene(40, 32, 12, [(25, 6, 3, 20)], marker=(1, 4, 8))
    canvas = np.full((near.shape[0], near.shape[1] + far.shape[1]), 230, np.uint8)
    canvas[:, : near.shape[1]] = near
    canvas[: far.shape[0], near.shape[1] :] = far
    offset = near.shape[1]
    far_box = far_boxes[0].model_copy(update={"x1": far_boxes[0].x1 + offset, "x2": far_boxes[0].x2 + offset})
    frames = [still(tmp_path, canvas)]
    unit_a, unit_b = unit_scale(0, frames, cfg), unit_scale(1, frames, cfg)
    book_a, book_b = track(0, [(0, near_boxes[0])], frames), track(1, [(0, far_box)], frames)
    assert close(measure(book_a, frames, cfg, unit_a).height_cm, 20.0, 0.03)
    assert close(measure(book_b, frames, cfg, unit_b).height_cm, 20.0, 0.03)
    assert not close(measure(book_b, frames, cfg, unit_a).height_cm, 20.0, 0.2)


def test_marker_in_overlapping_frame_is_chained(tmp_path, cfg):
    img, boxes = scene(60, 30, 15, [(40, 5, 3, 20)], marker=(0, 2, 6), textured=True)
    shift = 300
    first = still(tmp_path, img[:, :700], seq=0, blur=0.5)
    second = still(tmp_path, img[:, shift:], seq=1, blur=0.9)
    frames = [first, second]
    unit = unit_scale(0, frames, cfg)
    assert list(unit.planes) == [0]
    box = boxes[0].model_copy(update={"x1": boxes[0].x1 - shift, "x2": boxes[0].x2 - shift})
    dims = measure(track(0, [(1, box)], frames), frames, cfg, unit)
    direct = measure(track(0, [(0, boxes[0])], frames), frames, cfg, unit)
    assert dims.scale_method == ScaleMethod.aruco_homography
    assert dims.frame_seq == 1
    assert dims.confidence < direct.confidence
    assert close(dims.height_cm, 20.0, 0.03)


ROW = [2.0, 3.5, 1.5, 4.0, 2.5]
LOOSE = [(3, 3), (2, 4), (3, 2), (4, 3), (2, 2)]


def row_scene(tmp_path):
    books, x = [], 22.0
    for thickness in ROW:
        books.append((x, 6, thickness, 20))
        x += thickness
    img, boxes = scene(60, 32, PX_PER_CM, books, marker=(0, 2, 8))
    loose = [b.model_copy(update={"x1": b.x1 - out, "x2": b.x2 + back}) for b, (out, back) in zip(boxes, LOOSE)]
    frames = [still(tmp_path, img)]
    tracks = [track(i, [(0, b)], frames) for i, b in enumerate(loose)]
    return frames, tracks, loose


def test_shared_boundaries_beat_box_widths(tmp_path, cfg):
    frames, tracks, loose = row_scene(tmp_path)
    unit = unit_scale(0, frames, cfg)
    row = measure_row(tracks, frames, cfg, unit)
    single = [measure(t, frames, cfg, unit) for t in tracks]
    shared_error = sum(abs(d.thickness_cm - t) for d, t in zip(row, ROW))
    box_error = sum(abs(d.thickness_cm - t) for d, t in zip(single, ROW))
    assert shared_error < box_error
    assert all(close(d.thickness_cm, t, 0.1) for d, t in zip(row, ROW))
    assert close(sum(d.thickness_cm for d in row), sum(ROW), 0.03)


def test_row_sum_is_constrained_to_the_measured_run(tmp_path, cfg):
    frames, tracks, _ = row_scene(tmp_path)
    unit = unit_scale(0, frames, cfg)
    run = 13.0
    row = measure_row(tracks, frames, cfg, unit, run_cm=run)
    assert sum(d.thickness_cm for d in row) == pytest.approx(run)
    free = measure_row(tracks, frames, cfg, unit)
    ratios = [r.thickness_cm / f.thickness_cm for r, f in zip(row, free)]
    assert max(ratios) == pytest.approx(min(ratios))


def test_large_gap_is_not_absorbed_by_neighbours(tmp_path, cfg):
    img, boxes = scene(60, 32, PX_PER_CM, [(22, 6, 2, 20), (24, 6, 3, 20), (33, 6, 3, 20)], marker=(0, 2, 8))
    frames = [still(tmp_path, img)]
    tracks = [track(i, [(0, b)], frames) for i, b in enumerate(boxes)]
    row = measure_row(tracks, frames, cfg, unit_scale(0, frames, cfg))
    assert [round(d.thickness_cm, 1) for d in row] == [2.0, 3.0, 3.0]


def test_mask_adapter_is_none_without_ml_extra(tmp_path):
    img, boxes = scene(20, 20, PX_PER_CM, [(5, 2, 3, 15)])
    assert masks.spine_edges(still(tmp_path, img).path, boxes) is None


def test_mask_edges_use_the_middle_band():
    mask = np.zeros((100, 60), bool)
    mask[10:90, 20:35] = True
    mask[10:20, 5:50] = True
    left, right = masks.mask_edges(mask, Box(x1=15, y1=10, x2=40, y2=90), 0.5)
    assert (left, right) == (20.0, 35.0)


def dims_of(height, thickness):
    return SpineDims(height_cm=height, thickness_cm=thickness, scale_method=ScaleMethod.aruco_homography,
                     confidence=0.9, frame_seq=0)


def test_open_library_cross_check_flags_large_differences():
    assert cross_check(dims_of(20.0, 3.0), "20.5 x 13 x 3.1 centimeters") == []
    reasons = cross_check(dims_of(24.0, 3.0), "20 x 13 x 3 centimeters")
    assert len(reasons) == 1 and "height" in reasons[0]
    assert len(cross_check(dims_of(20.0, 1.0), "7.9 x 5.1 x 1.2 inches")) == 1
    assert cross_check(dims_of(20.0, 3.0), "") == []
    assert cross_check(dims_of(20.0, 3.0), "about a hand tall") == []
