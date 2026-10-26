from libclaim.capture.coverage import CoverageMap
from libclaim.capture.quality import load_capture_config
from libclaim.models import Still

CFG = load_capture_config()


def still(seq, blur):
    return Still(seq=seq, t_ms=seq, path="p", w=1, h=1, blur=blur, glare=0.0, motion_dps=0.0)


def test_region_with_only_blurry_stills_is_missed():
    cov = CoverageMap(CFG)
    cov.record("A1", still(1, CFG.blur_floor * 10))
    cov.record("A2", still(2, CFG.blur_floor / 10))
    assert cov.missed_regions() == ["A2"]
    cov.record("A2", still(3, CFG.blur_floor * 10))
    assert cov.missed_regions() == []


def test_expected_regions_never_seen_are_missed():
    cov = CoverageMap(CFG)
    cov.record("A1", still(1, CFG.blur_floor * 10))
    assert cov.missed_regions(["A1", "A3", "A2"]) == ["A2", "A3"]


def test_a_still_moved_to_a_shelf_leaves_the_unassigned_region():
    cov = CoverageMap(CFG)
    cov.record("sweep", still(1, CFG.blur_floor * 10))
    cov.record("sweep", still(2, CFG.blur_floor / 10))
    cov.assign(1, "S1.1")
    assert cov.seen == {"sweep", "S1.1"}
    assert cov.missed_regions() == ["sweep"]
    cov.assign(2, "S1.1")
    assert cov.seen == {"S1.1"} and cov.missed_regions() == []


def test_assigning_an_unknown_still_changes_nothing():
    cov = CoverageMap(CFG)
    cov.assign(9, "S1.1")
    assert cov.seen == set()
