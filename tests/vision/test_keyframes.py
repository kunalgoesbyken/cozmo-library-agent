import cv2
import numpy as np
import pytest

from libclaim.models import Still
from libclaim.vision import keyframes as keyframes_module
from libclaim.vision.keyframes import Keyframes, KeyframeStream, Phase, PhaseMarker, select_keyframes

WINDOW_MS = 500


@pytest.fixture(autouse=True)
def half_second_windows(monkeypatch):
    config = keyframes_module.load_keyframe_config().model_copy(update={"window_ms": WINDOW_MS})
    monkeypatch.setattr(keyframes_module, "load_keyframe_config", lambda: config)


def noise(seed, size=64):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (size, size, 3), dtype=np.uint8)


def make_still(tmp_path, seq, t_ms, img, blur):
    path = tmp_path / f"{seq}.png"
    cv2.imwrite(str(path), img)
    return Still(seq=seq, t_ms=t_ms, path=str(path), w=64, h=64, blur=blur, glare=0.0, motion_dps=0.0)


def test_thirty_stills_give_ten_keyframes(tmp_path):
    stills = [
        make_still(tmp_path, seq, seq * 160, noise(seq // 3), blur=float(seq % 3))
        for seq in range(30)
    ]
    events = [PhaseMarker(t_ms=0, phase=Phase.rows)]
    result = select_keyframes(stills, events)
    assert isinstance(result, Keyframes)
    assert len(result.rows) + len(result.room) == 10
    assert all(s.blur == max(x.blur for x in stills if x.t_ms // WINDOW_MS == s.t_ms // WINDOW_MS) for s in result.rows)


def test_sharpest_per_window_and_phase_split(tmp_path):
    stills = [
        make_still(tmp_path, 0, 0, noise(1), 1.0),
        make_still(tmp_path, 1, 200, noise(2), 5.0),
        make_still(tmp_path, 2, 1000, noise(3), 2.0),
    ]
    events = [PhaseMarker(t_ms=0, phase=Phase.rows), PhaseMarker(t_ms=900, phase=Phase.room)]
    result = select_keyframes(stills, events)
    assert [s.seq for s in result.rows] == [1]
    assert [s.seq for s in result.room] == [2]


def test_duplicates_are_dropped(tmp_path):
    img = noise(7)
    stills = [make_still(tmp_path, i, i * 600, img, 1.0) for i in range(4)]
    result = select_keyframes(stills, [PhaseMarker(t_ms=0, phase=Phase.rows)])
    assert [s.seq for s in result.rows] == [0]


def test_near_duplicate_within_threshold_dropped_distinct_kept(tmp_path):
    img = noise(7)
    tweaked = np.clip(img.astype(np.int16) + 2, 0, 255).astype(np.uint8)
    stills = [
        make_still(tmp_path, 0, 0, img, 1.0),
        make_still(tmp_path, 1, 600, tweaked, 1.0),
        make_still(tmp_path, 2, 1200, noise(99), 1.0),
    ]
    result = select_keyframes(stills, [PhaseMarker(t_ms=0, phase=Phase.rows)])
    assert [s.seq for s in result.rows] == [0, 2]


def test_stills_before_first_marker_are_ignored(tmp_path):
    stills = [make_still(tmp_path, 0, 0, noise(1), 1.0), make_still(tmp_path, 1, 1000, noise(2), 1.0)]
    result = select_keyframes(stills, [PhaseMarker(t_ms=800, phase=Phase.room)])
    assert result.rows == []
    assert [s.seq for s in result.room] == [1]


def bars(vertical, size=64, period=8):
    cells = (np.indices((size, size))[1 if vertical else 0] // period) % 2
    gray = (cells * 200 + 20).astype(np.uint8)
    return np.dstack([gray] * 3)


def test_structured_distinct_images_both_survive(tmp_path):
    stills = [
        make_still(tmp_path, 0, 0, bars(True), 1.0),
        make_still(tmp_path, 1, 600, bars(False), 1.0),
    ]
    result = select_keyframes(stills, [PhaseMarker(t_ms=0, phase=Phase.rows)])
    assert [s.seq for s in result.rows] == [0, 1]


def test_stream_emits_each_window_when_it_closes_and_matches_batch(tmp_path):
    stills = [
        make_still(tmp_path, seq, seq * 160, noise(seq // 3), blur=float(seq % 3))
        for seq in range(30)
    ]
    stream = KeyframeStream(keyframes_module.load_keyframe_config())
    emitted = []
    for still in stills:
        out = stream.add(still)
        assert all(kept.t_ms // WINDOW_MS < still.t_ms // WINDOW_MS for kept in out)
        emitted += out
    emitted += stream.flush()
    assert stream.flush() == []
    batch = select_keyframes(stills, [PhaseMarker(t_ms=0, phase=Phase.rows)])
    assert [s.seq for s in emitted] == [s.seq for s in batch.rows]


def test_unreadable_still_raises(tmp_path):
    still = Still(seq=0, t_ms=0, path=str(tmp_path / "missing.png"), w=1, h=1, blur=1.0, glare=0.0, motion_dps=0.0)
    with pytest.raises(ValueError):
        select_keyframes([still], [PhaseMarker(t_ms=0, phase=Phase.rows)])



def frame_still(tmp_path, seq, blur, value):
    path = tmp_path / f"{seq}.jpg"
    image = np.zeros((64, 64), np.uint8)
    image[:, : 8 * seq % 64] = value
    cv2.imwrite(str(path), image)
    return Still(seq=seq, t_ms=seq * 10_000, path=str(path), w=64, h=64, blur=blur, glare=0, motion_dps=0)


def test_a_window_far_blurrier_than_the_sweep_gives_no_keyframe(tmp_path):
    blurred = KeyframeStream(keyframes_module.load_keyframe_config(), blur_floor=0.005)
    blurred.sharpness = [0.006]
    blurred.add(frame_still(tmp_path, 3, 0.0002, 90))
    assert blurred.flush() == []


def test_a_uniformly_soft_sweep_keeps_its_keyframes(tmp_path):
    stream = KeyframeStream(keyframes_module.load_keyframe_config(), blur_floor=0.005)
    stream.sharpness = [0.004]
    stream.add(frame_still(tmp_path, 4, 0.0039, 150))
    assert len(stream.flush()) == 1
