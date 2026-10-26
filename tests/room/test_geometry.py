import math

import numpy as np
import pytest

from libclaim.packet.models import Opening
from libclaim.room.geometry import ShelfUnit, fallback_room, room_from_points

DENSITY = 1500
HEIGHT = 2.7
SHELF = {"z": (0.5, 2.5), "depth": 0.35, "top": 2.0}
FRAME = "frames/{:05d}_0.jpg"


def face(rng, fixed, value, ranges, keep=None):
    (a, (a_lo, a_hi)), (b, (b_lo, b_hi)) = ranges.items()
    count = int((a_hi - a_lo) * (b_hi - b_lo) * DENSITY)
    pts = np.zeros((count, 3))
    pts[:, fixed] = value
    pts[:, a] = rng.uniform(a_lo, a_hi, count)
    pts[:, b] = rng.uniform(b_lo, b_hi, count)
    return pts if keep is None else pts[keep(pts)]


def behind_shelf(p):
    z_lo, z_hi = SHELF["z"]
    return (p[:, 2] >= z_lo) & (p[:, 2] <= z_hi) & (p[:, 1] < SHELF["top"])


def under_shelf(p):
    z_lo, z_hi = SHELF["z"]
    return (p[:, 0] < SHELF["depth"]) & (p[:, 2] >= z_lo) & (p[:, 2] <= z_hi)


def shelf_room(rng, ceiling=True):
    x, y, z = 0, 1, 2
    z_lo, z_hi = SHELF["z"]
    depth, top = SHELF["depth"], SHELF["top"]
    faces = [
        face(rng, y, 0.0, {x: (0, 4), z: (0, 3)}, lambda p: ~under_shelf(p)),
        face(rng, x, 0.0, {y: (0, HEIGHT), z: (0, 3)}, lambda p: ~behind_shelf(p)),
        face(rng, x, 4.0, {y: (0, HEIGHT), z: (0, 3)}),
        face(rng, z, 0.0, {x: (0, 4), y: (0, HEIGHT)}),
        face(rng, z, 3.0, {x: (0, 4), y: (0, HEIGHT)}),
        face(rng, x, depth, {y: (0, top), z: (z_lo, z_hi)}),
        face(rng, y, top, {x: (0, depth), z: (z_lo, z_hi)}),
        face(rng, z, z_lo, {x: (0, depth), y: (0, top)}),
        face(rng, z, z_hi, {x: (0, depth), y: (0, top)}),
    ]
    if ceiling:
        faces.append(face(rng, y, HEIGHT, {x: (0, 4), z: (0, 3)}))
    return labelled(faces)


def labelled(faces):
    points = np.vstack(faces)
    frames = [FRAME.format(i) for i, f in enumerate(faces) for _ in range(len(f))]
    return points, frames


def l_room(rng, height=2.5):
    x, y, z = 0, 1, 2
    corners = [(0, 0), (4, 0), (4, 2), (2, 2), (2, 3.5), (0, 3.5)]

    def inside(p):
        return ((p[:, 0] < 4) & (p[:, 2] < 2)) | ((p[:, 0] < 2) & (p[:, 2] < 3.5))

    faces = [
        face(rng, y, 0.0, {x: (0, 4), z: (0, 3.5)}, inside),
        face(rng, y, height, {x: (0, 4), z: (0, 3.5)}, inside),
    ]
    for (x1, z1), (x2, z2) in zip(corners, corners[1:] + corners[:1]):
        if x1 == x2:
            faces.append(face(rng, x, x1, {y: (0, height), z: (min(z1, z2), max(z1, z2))}))
        else:
            faces.append(face(rng, z, z1, {x: (min(x1, x2), max(x1, x2)), y: (0, height)}))
    return labelled(faces)


def transform(points, yaw_deg, shift):
    a = math.radians(yaw_deg)
    rotation = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    return points @ rotation.T + np.asarray(shift)


def measure(points, frames, shelves, **kwargs):
    return room_from_points(points, frames, shelves, scale_method="test", scale_confidence=1.0, **kwargs)


SHELVES = [ShelfUnit(width_m=2.0, height_m=SHELF["top"])]


def test_box_room_with_proud_shelf_gives_floor_and_gross_wall_area():
    rng = np.random.default_rng(1)
    points, frames = shelf_room(rng)
    room = measure(transform(points, 20, (1.0, 0.4, -2.0)), frames, SHELVES)
    assert room.floor_area_m2 == pytest.approx(12.0, abs=0.3)
    assert room.wall_area_m2 == pytest.approx(37.8, abs=1.0)
    assert room.height_m == pytest.approx(HEIGHT, abs=0.05)
    assert room.length_m == pytest.approx(4.0, abs=0.1)
    assert room.width_m == pytest.approx(3.0, abs=0.1)
    assert room.shape == "rectangle"
    assert len(room.polygon_m) == 4
    assert room.shelved_wall_area_m2 == pytest.approx(4.0)
    assert room.scale_method == "test"
    assert 0.9 <= room.confidence <= 1.0
    assert set(room.frame_refs) <= set(frames) and room.frame_refs


def test_openings_unknown_leave_net_area_empty_with_review():
    rng = np.random.default_rng(2)
    points, frames = shelf_room(rng)
    room = measure(points, frames, SHELVES)
    assert room.wall_area_net_m2 is None
    assert room.openings == []
    assert room.review


def test_openings_are_subtracted_for_net_area():
    rng = np.random.default_rng(3)
    points, frames = shelf_room(rng)
    door = Opening(kind="door", width_m=0.9, height_m=2.0, frame_ref=FRAME.format(0))
    room = measure(points, frames, SHELVES, openings=[door])
    assert room.wall_area_net_m2 == pytest.approx(room.wall_area_m2 - 1.8, abs=1e-3)
    assert room.openings == [door]


def test_ceiling_from_ray_plane_when_ceiling_has_no_points():
    rng = np.random.default_rng(4)
    points, frames = shelf_room(rng, ceiling=False)
    camera = np.array([2.0, 1.4, 1.5])
    room = measure(points, frames, SHELVES, origins=camera)
    assert room.height_m == pytest.approx(HEIGHT, abs=0.05)
    assert room.wall_area_m2 == pytest.approx(37.8, abs=1.0)
    assert room.floor_area_m2 == pytest.approx(12.0, abs=0.3)


def test_noise_and_outliers_are_tolerated():
    rng = np.random.default_rng(5)
    points, frames = shelf_room(rng)
    noisy = points + rng.normal(0, 0.02, points.shape)
    count = int(0.03 * len(points))
    outliers = rng.uniform([-1, -1, -1], [5, HEIGHT + 1, 4], (count, 3))
    cloud = transform(np.vstack([noisy, outliers]), -35, (0.0, 1.2, 0.0))
    room = measure(cloud, frames + [FRAME.format(99)] * count, SHELVES)
    assert room.floor_area_m2 == pytest.approx(12.0, abs=0.3)
    assert room.wall_area_m2 == pytest.approx(37.8, abs=1.0)
    assert room.shape == "rectangle"


def test_l_shaped_room_gives_l_and_shoelace_area():
    rng = np.random.default_rng(6)
    points, frames = l_room(rng)
    room = measure(transform(points, 30, (0.5, 0.0, 0.5)), frames, [])
    assert room.shape == "L"
    assert len(room.polygon_m) == 6
    assert room.floor_area_m2 == pytest.approx(11.0, abs=0.3)
    assert room.wall_area_m2 == pytest.approx(15.0 * 2.5, abs=1.0)
    assert room.shelved_wall_area_m2 is None


@pytest.mark.parametrize("count", [0, 10])
def test_too_few_points_is_unmeasured_not_a_crash(count):
    rng = np.random.default_rng(7)
    points = rng.uniform(0, 3, (count, 3))
    room = measure(points, [FRAME.format(0)] * count, SHELVES)
    assert room.floor_area_m2 is None and room.wall_area_m2 is None
    assert room.confidence == 0.0
    assert room.shelved_wall_area_m2 == pytest.approx(4.0)
    assert room.review


def test_nan_points_are_dropped():
    rng = np.random.default_rng(8)
    points, frames = shelf_room(rng)
    points[::50] = np.nan
    room = measure(points, frames, SHELVES)
    assert room.floor_area_m2 == pytest.approx(12.0, abs=0.3)


def test_single_wall_does_not_enclose_a_room():
    rng = np.random.default_rng(9)
    faces = [
        face(rng, 1, 0.0, {0: (0, 4), 2: (0, 3)}),
        face(rng, 2, 0.0, {0: (0, 4), 1: (0, HEIGHT)}),
    ]
    points, frames = labelled(faces)
    room = measure(points, frames, [])
    assert room.floor_area_m2 is None
    assert room.confidence == 0.0
    assert room.review


def test_no_floor_plane_is_unmeasured():
    rng = np.random.default_rng(10)
    points, frames = labelled([face(rng, 0, 0.0, {1: (0, HEIGHT), 2: (0, 3)})])
    room = measure(points, frames, [])
    assert room.floor_area_m2 is None and room.height_m is None
    assert room.review


def test_shelving_taller_than_the_walls_leaves_no_wall_band():
    rng = np.random.default_rng(11)
    points, frames = shelf_room(rng)
    room = measure(points, frames, [ShelfUnit(width_m=2.0, height_m=HEIGHT)])
    assert room.floor_area_m2 is None
    assert room.review


def test_far_outliers_do_not_blow_up_the_grid():
    rng = np.random.default_rng(12)
    points, frames = shelf_room(rng)
    cloud = np.vstack([points, [[1e6, 2.5, 1e6], [-1e6, 2.4, 3e5]]])
    room = measure(cloud, frames + [FRAME.format(99)] * 2, SHELVES)
    assert room.floor_area_m2 == pytest.approx(12.0, abs=0.3)


def test_fallback_room_keeps_shelved_area_and_flags_low_confidence():
    room = fallback_room(SHELVES, ["metric depth unavailable"])
    assert room.floor_area_m2 is None
    assert room.shelved_wall_area_m2 == pytest.approx(4.0)
    assert room.scale_method == "aruco_only"
    assert room.confidence <= 0.1
    assert room.review == ["metric depth unavailable"]
