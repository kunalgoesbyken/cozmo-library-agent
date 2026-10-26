import math

import cv2
import numpy as np
import pytest

from libclaim.models import Still
from libclaim.room import depth
from libclaim.room.depth import (
    DepthSet,
    DepthUnavailable,
    metric_depth,
    rotation_from_homography,
    room_from_depth,
    scaled_depths,
)
from libclaim.room.geometry import ShelfUnit
from libclaim.vision import masks
from libclaim.vision.boxes import load_vision_config

WIDTH, HEIGHT = 1280, 720
MARKER_CM = 16.0
ROOM = {"x": (-1.5, 2.5), "y": (-1.4, 1.3), "z": (-1.2, 1.8)}


def focal(width=WIDTH):
    return width / 2 / math.tan(math.radians(load_vision_config().scale.fallback_long_side_fov_deg) / 2)


def still(tmp_path, seq, image=None, width=WIDTH, height=HEIGHT):
    path = tmp_path / f"{seq:05d}_0.png"
    cv2.imwrite(str(path), np.full((height, width), 255, np.uint8) if image is None else image)
    return Still(seq=seq, t_ms=seq, path=str(path), w=width, h=height, blur=1.0, glare=0.0, motion_dps=0.0)


def marker_image(distance_m):
    side = round(focal() * MARKER_CM / 100 / distance_m)
    cfg = load_vision_config().scale
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.aruco_dictionary))
    marker = cv2.aruco.generateImageMarker(dictionary, 3, side)
    image = np.full((HEIGHT, WIDTH), 255, np.uint8)
    top, left = (HEIGHT - side) // 2, (WIDTH - side) // 2
    image[top : top + side, left : left + side] = marker
    return image


def rotation(yaw_deg, pitch_deg):
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    yaw = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    pitch = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    return yaw @ pitch


def room_depth(r, width, height):
    f = focal(width)
    v, u = np.mgrid[0:height, 0:width].astype(float)
    rays = np.stack([(u - width / 2) / f, (v - height / 2) / f, np.ones_like(u)], axis=-1)
    world = rays @ r.T
    t = np.full(u.shape, np.inf)
    for axis, (lo, hi) in enumerate(ROOM.values()):
        component = world[..., axis]
        with np.errstate(divide="ignore", invalid="ignore"):
            for wall in (lo, hi):
                hit = wall / component
                t = np.where((hit > 0) & (hit < t), hit, t)
    return t


def test_metric_depth_without_ml_extra_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(masks, "available", lambda: False)
    with pytest.raises(DepthUnavailable, match=r"\[ml\]"):
        metric_depth(still(tmp_path, 0))


def test_unavailable_depth_falls_back_to_low_confidence_room_with_review(tmp_path):
    def unavailable(_):
        raise DepthUnavailable("metric depth needs the [ml] extra")

    stills = [still(tmp_path, 0)]
    depths = scaled_depths(stills, unavailable, MARKER_CM)
    assert depths.maps == {}
    room = room_from_depth(stills, [ShelfUnit(width_m=1.0, height_m=2.0)], depths)
    assert room.floor_area_m2 is None
    assert room.shelved_wall_area_m2 == pytest.approx(2.0)
    assert room.scale_method == "aruco_only"
    assert room.confidence <= 0.1
    assert any("[ml]" in reason for reason in room.review)


def test_marker_rescales_model_depth_to_metric(tmp_path):
    true_m = 1.2
    stills = [still(tmp_path, 0, marker_image(true_m)), still(tmp_path, 1)]
    depths = scaled_depths(stills, lambda s: np.full((HEIGHT, WIDTH), true_m * 1.25), MARKER_CM)
    assert depths.scale_method == "depth_aruco"
    assert np.median(depths.maps[0]) == pytest.approx(true_m, rel=0.02)
    assert np.median(depths.maps[1]) == pytest.approx(true_m, rel=0.02)
    assert depths.review == []


def test_no_marker_keeps_model_scale_flagged(tmp_path):
    stills = [still(tmp_path, 0)]
    depths = scaled_depths(stills, lambda s: np.full((HEIGHT, WIDTH), 2.0), MARKER_CM)
    assert depths.scale_method == "depth_metric_prior"
    assert depths.confidence <= 0.3
    assert depths.review
    assert np.median(depths.maps[0]) == pytest.approx(2.0)


def test_rotation_from_pure_rotation_homography():
    k = np.array([[focal(), 0, WIDTH / 2], [0, focal(), HEIGHT / 2], [0, 0, 1]])
    expected = rotation(12, -7)
    homography = 3.0 * k @ expected @ np.linalg.inv(k)
    assert rotation_from_homography(homography, k, k) == pytest.approx(expected, abs=1e-9)


def test_pan_rotations_recover_camera_yaw_from_images(tmp_path):
    width, height = 640, 360
    rng = np.random.default_rng(0)
    texture = cv2.GaussianBlur(rng.integers(0, 256, (height, width), dtype=np.uint8), (3, 3), 0)
    k = np.array([[focal(width), 0, width / 2], [0, focal(width), height / 2], [0, 0, 1]])
    expected = rotation(8, 3)
    turned = cv2.warpPerspective(texture, k @ expected @ np.linalg.inv(k), (width, height), flags=cv2.WARP_INVERSE_MAP)
    stills = [still(tmp_path, 0, texture, width, height), still(tmp_path, 1, turned, width, height)]
    first, second = depth.pan_rotations(stills, [k, k])
    assert first == pytest.approx(np.eye(3))
    assert second == pytest.approx(expected, abs=0.01)


def test_unregistered_frames_are_dropped_with_review(tmp_path, monkeypatch):
    width, height = 320, 180
    stills = [still(tmp_path, i, width=width, height=height) for i in range(2)]
    maps = {s.seq: room_depth(rotation(0, -25), width, height) for s in stills}
    monkeypatch.setattr(depth, "pan_rotations", lambda frames, cameras: [np.eye(3), None])
    room = room_from_depth(stills, [], DepthSet(maps, "depth_aruco", 0.7))
    assert any("could not be registered" in reason for reason in room.review)
    assert stills[1].path not in room.frame_refs


def test_panned_depth_frames_measure_the_room(tmp_path, monkeypatch):
    width, height = 320, 180
    poses = [rotation(yaw, pitch) for pitch in (-25, 35) for yaw in range(0, 360, 45)]
    stills = [still(tmp_path, i, width=width, height=height) for i in range(len(poses))]
    maps = {s.seq: room_depth(r, width, height) for s, r in zip(stills, poses)}
    monkeypatch.setattr(depth, "pan_rotations", lambda frames, cameras: [poses[f.seq] for f in frames])
    monkeypatch.setattr(depth.load_room_config().depth, "stride", 2)
    room = room_from_depth(stills, [], DepthSet(maps, "depth_aruco", 0.7))
    assert room.floor_area_m2 == pytest.approx(12.0, abs=0.3)
    assert room.height_m == pytest.approx(2.7, abs=0.05)
    assert room.wall_area_m2 == pytest.approx(37.8, abs=1.0)
    assert room.shape == "rectangle"
    assert room.scale_method == "depth_aruco"
    assert set(room.frame_refs) <= {s.path for s in stills}
