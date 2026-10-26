import cv2
import numpy as np
import pytest

from libclaim.capture.frame_store import FrameHeader, FrameStore, Motion
from libclaim.capture.quality import AlertCode, Quality, gate, load_capture_config, score
from libclaim.sweep import SweepDir

CFG = load_capture_config()


def checkerboard(size=256, square=16):
    cells = (np.indices((size, size)) // square).sum(axis=0) % 2
    return (cells * 190 + 30).astype(np.uint8)


def jpeg(img):
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_sharp_checkerboard_passes():
    q = score(checkerboard(), CFG)
    assert gate(q, 0.0, CFG) == []


def test_blurred_checkerboard_flags_blur():
    blurred = cv2.GaussianBlur(checkerboard(), (0, 0), 5)
    q = score(blurred, CFG)
    assert q.blur < score(checkerboard(), CFG).blur
    assert gate(q, 0.0, CFG) == [AlertCode.blur]


def test_clipped_area_flags_glare():
    img = checkerboard()
    img[:, : int(256 * 0.2)] = 255
    q = score(img, CFG)
    assert q.glare > CFG.glare_clipped_fraction
    assert AlertCode.glare in gate(q, 0.0, CFG)


def test_small_clipped_area_is_not_glare():
    img = checkerboard()
    img[:, : int(256 * 0.02)] = 255
    assert AlertCode.glare not in gate(score(img, CFG), 0.0, CFG)


def test_fast_rotation_flags_too_fast():
    q = score(checkerboard(), CFG)
    assert gate(q, 61.0, CFG) == [AlertCode.too_fast]
    assert gate(q, 59.0, CFG) == []


def test_gate_reports_every_failure():
    q = Quality(blur=0.0, glare=0.5)
    assert gate(q, 90.0, CFG) == [AlertCode.blur, AlertCode.glare, AlertCode.too_fast]


def test_color_input_is_scored_like_gray():
    gray = checkerboard()
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    assert score(color, CFG) == score(gray, CFG)


def test_store_writes_frame_and_returns_still(tmp_path):
    sweep = SweepDir(tmp_path)
    store = FrameStore(sweep, CFG)
    header = FrameHeader(
        seq=3,
        t_ms=1500,
        w=256,
        h=256,
        motion=Motion(rot_dps=12.5, acc=0.1),
        pose=[0.0, 0.0, 0.0],
    )
    still = store.add(header, jpeg(checkerboard()))
    assert still.path == str(sweep.frame_path(3, 1500))
    assert still.seq == 3 and still.t_ms == 1500
    assert (still.w, still.h) == (256, 256)
    assert still.motion_dps == 12.5
    assert still.pose == [0.0, 0.0, 0.0]
    assert still.glare == 0.0
    assert still.blur > CFG.blur_floor
    assert sweep.frame_path(3, 1500).read_bytes() == jpeg(checkerboard())


def test_store_rejects_undecodable_jpeg(tmp_path):
    store = FrameStore(SweepDir(tmp_path), CFG)
    header = FrameHeader(seq=1, t_ms=0, w=1, h=1)
    with pytest.raises(ValueError):
        store.add(header, b"not a jpeg")
