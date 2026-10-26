import cv2
import numpy as np
import pytest

from libclaim.models import Still
from libclaim.settings import Settings
from libclaim.vision import masks
from libclaim.vision.boxes import Box, load_vision_config
from libclaim.vision.rows import continues, flat_dims, is_row, marker_unit, same_row, shelf_units, split_rows
from libclaim.vision.scale import ScaleMethod, SpineDims, load_measure_config
from libclaim.vision.tracks import BookTrack, BoxRef

SIZE = (400, 700)


@pytest.fixture(autouse=True)
def no_ml(monkeypatch):
    monkeypatch.setattr(masks, "find_spec", lambda name: None)


def still_at(tmp_path, seq, img):
    path = tmp_path / f"{seq:05d}_{seq}.png"
    cv2.imwrite(str(path), img)
    return Still(seq=seq, t_ms=seq * 1000, path=str(path), w=img.shape[1], h=img.shape[0], blur=1.0, glare=0.0,
                 motion_dps=0.0)


def texture(seed, width=900):
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.integers(0, 256, (SIZE[0], width), dtype=np.uint8), (3, 3), 0)


def track(index, box, frame=0, seq=0):
    still = Still(seq=seq, t_ms=0, path="", w=SIZE[1], h=SIZE[0], blur=1.0, glare=0.0, motion_dps=0.0)
    return BookTrack(track_id=index, refs=[BoxRef(frame=frame, box=box)], best_frame=still, best_box=box)


def box(x1, y1, x2, y2):
    return Box(x1=x1, y1=y1, x2=x2, y2=y2)


def test_overlapping_pan_continues_and_unrelated_view_does_not(tmp_path):
    scene = texture(1)
    first = still_at(tmp_path, 1, scene[:, :700])
    second = still_at(tmp_path, 2, scene[:, 150:850])
    other = still_at(tmp_path, 3, texture(2)[:, :700])
    cfg = load_vision_config().tracks
    assert continues([first], second, cfg)
    assert not continues([first, second], other, cfg)


def test_same_row_needs_mutual_mid_height_overlap():
    tall, short, above = box(0, 100, 30, 300), box(40, 180, 70, 300), box(0, 0, 30, 90)
    assert same_row(tall, short)
    assert not same_row(tall, above)


def test_split_rows_orders_rows_top_down_books_left_right_and_places_flat_books():
    anchors = [
        (0, box(200, 50, 230, 200)),
        (0, box(100, 60, 130, 200)),
        (0, box(100, 260, 130, 400)),
        (0, box(140, 370, 260, 400)),
        (1, box(100, 60, 130, 200)),
    ]
    vertical = [True, True, True, False, True]
    rows, flat = split_rows(anchors, vertical)
    assert rows == [[1, 0], [2, 3], [4]]
    assert flat == [3]


def test_row_frames_need_tall_spines_so_wide_shots_are_room_frames():
    still = Still(seq=0, t_ms=0, path="", w=960, h=540, blur=1.0, glare=0.0, motion_dps=0.0)
    close = [box(0, 100, 30, 300)]
    wide = [box(x, 200, x + 8, 240) for x in range(0, 400, 10)]
    assert is_row(close, still, 1, 0.2)
    assert not is_row(wide, still, 1, 0.2)
    assert not is_row([], still, 1, 0.2)


def test_marker_unit_is_the_most_frequent_marker(tmp_path):
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, load_vision_config().scale.aruco_dictionary))
    stills = []
    for seq, marker_id in enumerate([3, 3, 5]):
        canvas = np.full(SIZE, 255, np.uint8)
        canvas[100:260, 100:260] = cv2.aruco.generateImageMarker(dictionary, marker_id, 160)
        stills.append(still_at(tmp_path, seq, canvas))
    stills.append(still_at(tmp_path, 9, texture(4)[:, :700]))
    assert marker_unit(stills, load_vision_config().scale) == 3
    assert marker_unit(stills[3:], load_vision_config().scale) is None


def test_flat_book_without_marker_uses_its_long_side_for_height():
    cfg = load_measure_config(Settings(aruco_marker_size_cm=None))
    lying = track(0, box(100, 300, 310, 330))
    dims = flat_dims(lying, [lying.best_frame], cfg, None)
    assert dims.scale_method == ScaleMethod.book_format_prior
    assert dims.height_cm > dims.thickness_cm
    assert dims.height_cm in {fmt.height_cm for fmt in cfg.formats.values()}
    assert dims.fallback_reason


def test_shelf_units_are_lower_bounds_from_metric_rows_only():
    def dims(height, thickness, method=ScaleMethod.aruco_homography):
        return SpineDims(height_cm=height, thickness_cm=thickness, scale_method=method, confidence=0.8, frame_seq=0)

    rows = {
        1: [[dims(20, 3), dims(24, 4)], [dims(30, 5)]],
        2: [[dims(21, 2, ScaleMethod.book_format_prior)]],
    }
    units = shelf_units(rows, {0: "frames/00000_0.jpg"})
    assert len(units) == 1 and units[0].frame_ref == "frames/00000_0.jpg"
    assert units[0].width_m == pytest.approx(0.07)
    assert units[0].height_m == pytest.approx(0.54)
