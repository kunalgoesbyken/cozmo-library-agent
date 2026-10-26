import math

import cv2
import numpy as np
import pytest

from libclaim.models import Still
from libclaim.settings import Settings
from libclaim.vision import masks
from libclaim.vision.boxes import Box, load_vision_config
from libclaim.vision.scale import (
    NO_MARKER,
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


def configured(recess=0.0, trim=0.0, marker=MARKER_CM):
    cfg = load_measure_config(Settings(aruco_marker_size_cm=marker))
    scale = cfg.scale.model_copy(update={"spine_recess_cm": recess, "box_height_trim": trim})
    return cfg.model_copy(update={"scale": scale})


@pytest.fixture
def cfg():
    return configured()


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


def still(tmp_path, img, seq=0, blur=1.0, intrinsics=None):
    path = tmp_path / f"{seq}.png"
    cv2.imwrite(str(path), img)
    return Still(seq=seq, t_ms=seq * 500, path=str(path), w=img.shape[1], h=img.shape[0], blur=blur,
                 glare=0.0, motion_dps=0.0, intrinsics=intrinsics)


def track(track_id, refs, stills):
    best = max(refs, key=lambda r: stills[r[0]].blur)
    return BookTrack(
        track_id=track_id,
        refs=[BoxRef(frame=f, box=b) for f, b in refs],
        best_frame=stills[best[0]],
        best_box=best[1],
    )


def shifted(box, dx):
    return box.model_copy(update={"x1": box.x1 + dx, "x2": box.x2 + dx})


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


FOCAL = 1500.0
SIZE = (1600, 1200)


def pinhole(yaw_deg, distance_cm, recess_cm):
    w, h = SIZE
    k = np.array([[FOCAL, 0, w / 2], [0, FOCAL, h / 2], [0, 0, 1]])
    a = math.radians(yaw_deg)
    r = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    t = np.array([0, 0, distance_cm]) - r @ np.array([22.0, 14.0, 0])

    def plane(depth):
        return k @ np.column_stack([r[:, 0], r[:, 1], r[:, 2] * depth + t])

    texture_px = 20
    sheet = np.full((18 * texture_px, 18 * texture_px), 255, np.uint8)
    sheet[texture_px : 17 * texture_px, texture_px : 17 * texture_px] = cv2.aruco.generateImageMarker(
        dictionary(), 0, 16 * texture_px)
    to_plane = np.array([[1 / texture_px, 0, -1.0], [0, 1 / texture_px, 4.0 - 1.0], [0, 0, 1]])
    img = np.full((h, w), 200, np.uint8)
    warped = cv2.warpPerspective(sheet, plane(0) @ to_plane, (w, h), borderValue=0)
    cover = cv2.warpPerspective(np.full(sheet.shape, 255, np.uint8), plane(0) @ to_plane, (w, h)) > 0
    img[cover] = warped[cover]
    corners = np.float64([[30, 4, 1], [33, 4, 1], [33, 24, 1], [30, 24, 1]])
    projected = (plane(recess_cm) @ corners.T).T
    projected = projected[:, :2] / projected[:, 2:]
    cv2.fillConvexPoly(img, np.round(projected).astype(np.int32), 50)
    box = Box(x1=projected[:, 0].min(), y1=projected[:, 1].min(), x2=projected[:, 0].max(), y2=projected[:, 1].max())
    return img, box


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


@pytest.mark.parametrize("yaw", [0, 30])
def test_pinhole_recess_behind_marker_plane_within_three_percent(tmp_path, yaw):
    recess, distance = 4.0, 70.0
    img, box = pinhole(yaw, distance, recess)
    frames = [still(tmp_path, img, intrinsics={"fx": FOCAL})]
    corrected, plain = configured(recess=recess), configured(recess=0.0)
    unit = unit_scale(0, frames, corrected)
    assert close(unit.planes[0].plane_distance_cm, distance * math.cos(math.radians(yaw)), 0.02)
    book = track(0, [(0, box)], frames)
    assert close(measure(book, frames, corrected, unit).height_cm, 20.0, 0.03)
    assert not close(measure(book, frames, plain, unit_scale(0, frames, plain)).height_cm, 20.0, 0.04)


def test_default_recess_comes_from_config():
    assert load_measure_config(Settings()).scale.spine_recess_cm == 2.5


def test_no_marker_falls_back_to_format_prior_with_review(tmp_path, cfg):
    img, boxes = scene(60, 32, PX_PER_CM, [(30, 4, 3.5, 23.5)])
    frames = [still(tmp_path, img)]
    unit = unit_scale(0, frames, cfg)
    assert unit.planes == {}
    for given in (unit, None):
        dims = measure(track(0, [(0, boxes[0])], frames), frames, cfg, given)
        assert dims.scale_method == ScaleMethod.book_format_prior
        assert dims.confidence <= 0.4
        assert dims.fallback_reason == NO_MARKER
        assert dims.review == []
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
    assert dims.fallback_reason == NO_MARKER
    assert "ARUCO" not in dims.fallback_reason


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
    frames = [still(tmp_path, canvas)]
    unit_a, unit_b = unit_scale(0, frames, cfg), unit_scale(1, frames, cfg)
    book_a = track(0, [(0, near_boxes[0])], frames)
    book_b = track(1, [(0, shifted(far_boxes[0], near.shape[1]))], frames)
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
    dims = measure(track(0, [(1, shifted(boxes[0], -shift))], frames), frames, cfg, unit)
    direct = measure(track(0, [(0, boxes[0])], frames), frames, cfg, unit)
    assert dims.scale_method == ScaleMethod.aruco_homography
    assert dims.frame_seq == 1
    assert dims.confidence < direct.confidence
    assert close(dims.height_cm, 20.0, 0.03)


ROW = [2.0, 3.5, 1.5, 4.0, 2.5]
LOOSE = [(3, 3), (2, 4), (3, 2), (4, 3), (2, 2)]


def row_scene(tmp_path, books=None):
    books = books or []
    if not books:
        x = 22.0
        for thickness in ROW:
            books.append((x, 6, thickness, 20))
            x += thickness
    img, boxes = scene(60, 32, PX_PER_CM, books, marker=(0, 2, 8))
    return [still(tmp_path, img)], boxes


def loosened(boxes):
    return [b.model_copy(update={"x1": b.x1 - out, "x2": b.x2 + back}) for b, (out, back) in zip(boxes, LOOSE)]


def test_shared_boundaries_beat_box_widths(tmp_path, cfg):
    frames, boxes = row_scene(tmp_path)
    tracks = [track(i, [(0, b)], frames) for i, b in enumerate(loosened(boxes))]
    unit = unit_scale(0, frames, cfg)
    row = measure_row(tracks, frames, cfg, unit)
    single = [measure(t, frames, cfg, unit) for t in tracks]
    shared_error = sum(abs(d.thickness_cm - t) for d, t in zip(row, ROW))
    box_error = sum(abs(d.thickness_cm - t) for d, t in zip(single, ROW))
    assert shared_error < box_error
    assert all(close(d.thickness_cm, t, 0.1) for d, t in zip(row, ROW))
    assert close(sum(d.thickness_cm for d in row), sum(ROW), 0.03)


def test_adjacent_books_with_different_sharpest_frames_share_one_reference_frame(tmp_path, cfg):
    frames, boxes = row_scene(tmp_path)
    img = cv2.imread(frames[0].path, cv2.IMREAD_GRAYSCALE)
    frames += [still(tmp_path, img, seq=1, blur=2.0), still(tmp_path, img, seq=2, blur=3.0)]
    first, second = loosened(boxes)[:2]
    tracks = [track(0, [(0, first), (1, first)], frames), track(1, [(0, second), (2, second)], frames)]
    unit = unit_scale(0, frames, cfg)
    row = measure_row(tracks, frames, cfg, unit)
    assert [d.frame_seq for d in row] == [0, 0]
    assert all(d.review == [] for d in row)
    single = [measure(t, frames, cfg, unit) for t in tracks]
    assert [d.frame_seq for d in single] == [1, 2]
    shared_error = sum(abs(d.thickness_cm - t) for d, t in zip(row, ROW))
    box_error = sum(abs(d.thickness_cm - t) for d, t in zip(single, ROW))
    assert shared_error < box_error


def test_track_missing_from_the_reference_frame_is_flagged(tmp_path, cfg):
    frames, boxes = row_scene(tmp_path)
    img = cv2.imread(frames[0].path, cv2.IMREAD_GRAYSCALE)
    frames.append(still(tmp_path, img, seq=1, blur=0.5))
    tracks = [track(0, [(0, boxes[0])], frames), track(1, [(0, boxes[1])], frames), track(2, [(1, boxes[2])], frames)]
    row = measure_row(tracks, frames, cfg, unit_scale(0, frames, cfg))
    assert [d.frame_seq for d in row] == [0, 0, 1]
    assert row[0].review == [] and row[1].review == []
    assert any("reference frame" in reason for reason in row[2].review)
    assert close(row[2].thickness_cm, ROW[2], 0.03)


def test_tracks_from_two_rows_are_rejected(tmp_path, cfg):
    img, boxes = scene(60, 60, 10, [(20, 2, 3, 20), (24, 30, 3, 20)], marker=(0, 2, 2))
    frames = [still(tmp_path, img)]
    tracks = [track(i, [(0, b)], frames) for i, b in enumerate(boxes)]
    with pytest.raises(ValueError, match="one shelf row"):
        measure_row(tracks, frames, cfg, unit_scale(0, frames, cfg))


def test_measured_run_scales_the_whole_span_including_gaps(tmp_path, cfg):
    books = [(22, 6, 2, 20), (24, 6, 3, 20), (33, 6, 3, 20)]
    frames, boxes = row_scene(tmp_path, books)
    tracks = [track(i, [(0, b)], frames) for i, b in enumerate(boxes)]
    unit = unit_scale(0, frames, cfg)
    free = measure_row(tracks, frames, cfg, unit)
    assert [round(d.thickness_cm, 1) for d in free] == [2.0, 3.0, 3.0]
    exact = measure_row(tracks, frames, cfg, unit, run_cm=14.0)
    assert [round(d.thickness_cm, 2) for d in exact] == [round(d.thickness_cm, 2) for d in free]
    assert sum(d.thickness_cm for d in exact) == pytest.approx(8.0, abs=0.05)
    longer = measure_row(tracks, frames, cfg, unit, run_cm=15.4)
    for scaled, base in zip(longer, free):
        assert scaled.thickness_cm == pytest.approx(base.thickness_cm * 1.1, rel=0.01)
        assert scaled.height_cm == pytest.approx(base.height_cm * 1.1, rel=0.01)


def test_loose_box_heights_are_trimmed_by_the_configured_fraction(tmp_path):
    loose = 0.04
    img, boxes = scene(60, 32, PX_PER_CM, [(30, 6, 3, 20)], marker=(0, 4, 8))
    box = boxes[0]
    grown = box.model_copy(update={"y1": box.y1 - loose * box.height, "y2": box.y2 + loose * box.height})
    frames = [still(tmp_path, img)]
    book = track(0, [(0, grown)], frames)
    raw = configured()
    trimmed = configured(trim=2 * loose / (1 + 2 * loose))
    assert close(measure(book, frames, raw, unit_scale(0, frames, raw)).height_cm, 20.0 * (1 + 2 * loose), 0.01)
    assert close(measure(book, frames, trimmed, unit_scale(0, frames, trimmed)).height_cm, 20.0, 0.01)


def test_mask_adapter_is_none_without_ml_extra(tmp_path):
    img, boxes = scene(20, 20, PX_PER_CM, [(5, 2, 3, 15)])
    assert masks.spine_boxes(still(tmp_path, img).path, boxes) is None


def test_mask_adapter_is_none_when_model_is_missing(tmp_path, monkeypatch):
    img, boxes = scene(20, 20, PX_PER_CM, [(5, 2, 3, 15)])
    monkeypatch.setattr(masks, "find_spec", lambda name: object())

    def missing(model_id):
        raise OSError(model_id)

    monkeypatch.setattr(masks, "sam_model", missing)
    assert masks.spine_boxes(still(tmp_path, img).path, boxes) is None


def test_mask_box_uses_the_middle_bands():
    mask = np.zeros((100, 60), bool)
    mask[12:88, 20:35] = True
    mask[12:20, 5:50] = True
    tight = masks.mask_box(mask, Box(x1=15, y1=8, x2=40, y2=92), 0.5)
    assert (tight.x1, tight.x2, tight.y1, tight.y2) == (20.0, 35.0, 12.0, 88.0)


def dims_of(height, thickness):
    return SpineDims(height_cm=height, thickness_cm=thickness, scale_method=ScaleMethod.aruco_homography,
                     confidence=0.9, frame_seq=0)


def test_open_library_cross_check_flags_large_differences():
    assert cross_check(dims_of(20.0, 3.0), "20.5 x 13 x 3.1 centimeters") == []
    reasons = cross_check(dims_of(24.0, 3.0), "20 x 13 x 3 centimeters")
    assert len(reasons) == 1 and "height" in reasons[0]
    assert len(cross_check(dims_of(20.0, 1.0), "7.9 x 5.1 x 1.2 inches")) == 1
    assert cross_check(dims_of(20.0, 3.0), "7,9 X 5,1 X 1,2 In.") == []
    assert cross_check(dims_of(20.0, 3.0), "20 × 13 × 3 CM") == []
    assert cross_check(dims_of(20.0, 3.0), "") == []
    assert cross_check(dims_of(20.0, 3.0), "about a hand tall") == []


def bleeding(factor_height, factor_width):
    def fake(path, boxes):
        return [
            Box(x1=b.x1 + b.width * (1 - factor_width) / 2, x2=b.x2 - b.width * (1 - factor_width) / 2,
                y1=b.y1, y2=b.y1 + b.height * factor_height)
            for b in boxes
        ]

    return fake


def marker_scene(tmp_path):
    img, boxes = scene(60, 32, PX_PER_CM, [(30, 4, 3, 20)], marker=(0, 4, 8))
    return [still(tmp_path, img)], boxes


def test_mask_bleeding_over_other_objects_does_not_change_the_height(tmp_path, cfg, monkeypatch):
    frames, boxes = marker_scene(tmp_path)
    book = track(0, [(0, boxes[0])], frames)
    honest = measure(book, frames, cfg, unit_scale(0, frames, cfg)).height_cm
    monkeypatch.setattr(masks, "spine_boxes", bleeding(2.0, 1.0))
    assert measure(book, frames, cfg, unit_scale(0, frames, cfg)).height_cm == pytest.approx(honest)


def test_mask_covering_a_sliver_does_not_change_the_height(tmp_path, cfg, monkeypatch):
    frames, boxes = marker_scene(tmp_path)
    book = track(0, [(0, boxes[0])], frames)
    honest = measure(book, frames, cfg, unit_scale(0, frames, cfg))
    monkeypatch.setattr(masks, "spine_boxes", bleeding(0.4, 0.3))
    dims = measure(book, frames, cfg, unit_scale(0, frames, cfg))
    assert (dims.height_cm, dims.thickness_cm) == pytest.approx((honest.height_cm, honest.thickness_cm))


def test_agreeing_mask_is_used(tmp_path, cfg, monkeypatch):
    frames, boxes = marker_scene(tmp_path)
    book = track(0, [(0, boxes[0])], frames)
    honest = measure(book, frames, cfg, unit_scale(0, frames, cfg)).height_cm
    monkeypatch.setattr(masks, "spine_boxes", bleeding(0.95, 1.0))
    assert measure(book, frames, cfg, unit_scale(0, frames, cfg)).height_cm == pytest.approx(honest * 0.95, rel=0.01)


def test_masks_never_change_the_format_class_of_a_prior(tmp_path, cfg, monkeypatch):
    img, boxes = scene(60, 32, PX_PER_CM, [(30, 4, 3.5, 17.5)])
    frames = [still(tmp_path, img)]
    book = track(0, [(0, boxes[0])], frames)
    honest = measure(book, frames, cfg, None)
    monkeypatch.setattr(masks, "spine_boxes", bleeding(1.0, 0.55))
    narrow = measure(book, frames, cfg, None)
    assert (narrow.book_format, narrow.height_cm) == (honest.book_format, honest.height_cm)
    assert honest.book_format == "hardcover"
