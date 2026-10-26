import cv2
import numpy as np
import pytest

from libclaim.pipeline import run_pipeline
from libclaim.stages import KeyframesStage
from libclaim.vision import rows as rows_module
from libclaim.vision import tracks as tracks_module
from tests import sweep12 as fx
from tests.test_pipeline import fast_config, offline_models, online, services

FIXTURES = (offline_models, online)


async def count(sweep, tmp_path, **config):
    packet = await run_pipeline(sweep, services=services(sweep, tmp_path, fx.Web(), local=fx.fake_local),
                                config=fast_config(**config))
    keyframes = len(list(sweep.frames.glob("*.jpg")))
    assert len(sweep.read_stage("keyframes", KeyframesStage).keyframes) == keyframes
    return packet


async def test_pan_longer_than_a_segment_counts_each_book_once(tmp_path, online):
    pan = fx.Pan(seed=21, slots=dict(enumerate(fx.upright(18))))
    sweep = fx.write_pans(tmp_path / "run", [(pan, offset) for offset in (0, 120, 240, 360, 480)])
    packet = await count(sweep, tmp_path, segment_max_frames=2)
    assert packet.totals.book_count == 18
    assert len({book.shelf for book in packet.books}) == 1


async def test_second_pass_over_a_row_is_not_recounted(tmp_path, online):
    first = fx.Pan(seed=22, slots=dict(enumerate(fx.upright(10))))
    other = fx.Pan(seed=23, slots=dict(enumerate(fx.upright(8, start=10))))
    views = [(first, 0), (first, 240), (other, 0), (other, 240), (first, 120), (first, 360)]
    packet = await count(fx.write_pans(tmp_path / "run", views), tmp_path)
    assert packet.totals.book_count == 18


async def test_sparse_frame_mid_pan_neither_splits_nor_loses_books(tmp_path, online):
    slots = dict(enumerate(fx.upright(15))) | {34: fx.upright(1, start=15)[0]}
    pan = fx.Pan(seed=24, slots=slots)
    packet = await count(fx.write_pans(tmp_path / "run", [(pan, 0), (pan, 1500), (pan, 240)]), tmp_path)
    assert packet.totals.book_count == 16


async def test_registration_failure_inside_a_marked_unit_is_deduplicated_on_the_marker_plane(
    tmp_path, online, monkeypatch
):
    books = fx.upright(17)
    slots = dict(zip([*range(8), *range(11, 20)], books))
    pan = fx.Pan(seed=25, slots=slots, marker_slot=8)
    sweep = fx.write_pans(tmp_path / "run", [(pan, offset) for offset in (0, 120, 240, 360)])
    broken = cv2.imread(str(sorted(sweep.frames.glob("*.jpg"))[2]), cv2.IMREAD_GRAYSCALE)
    real = tracks_module.pair_homography

    def fragile(prev, cur, cfg):
        if np.array_equal(prev, broken) or np.array_equal(cur, broken):
            return None
        return real(prev, cur, cfg)

    monkeypatch.setattr(tracks_module, "pair_homography", fragile)
    monkeypatch.setattr(rows_module, "pair_homography", fragile)
    packet = await count(sweep, tmp_path)
    assert packet.totals.book_count == 17
    assert any("already counted" in entry.reason for entry in packet.review_queue)


@pytest.mark.parametrize("limit", [1])
async def test_one_frame_segments_still_count_each_book_once(tmp_path, online, limit):
    pan = fx.Pan(seed=26, slots=dict(enumerate(fx.upright(14))))
    sweep = fx.write_pans(tmp_path / "run", [(pan, offset) for offset in (0, 120, 240)])
    packet = await count(sweep, tmp_path, segment_max_frames=limit)
    assert packet.totals.book_count == 14
