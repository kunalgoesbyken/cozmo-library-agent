import json

import pytest

from libclaim.models import Still
from libclaim.sweep import SweepDir


def make_still(path):
    return Still(seq=1, t_ms=250, path=path, w=640, h=480, blur=0.1, glare=0.0, motion_dps=2.5)


def test_frame_path_is_under_root_and_distinct(tmp_path):
    sweep = SweepDir(tmp_path)
    first = sweep.frame_path(1, 250)
    assert tmp_path in first.parents
    assert first != sweep.frame_path(2, 250)
    assert first != sweep.frame_path(1, 500)


def test_stage_round_trip(tmp_path):
    sweep = SweepDir(tmp_path)
    still = make_still(str(sweep.frame_path(1, 250)))
    sweep.write_stage("capture", still)
    assert sweep.read_stage("capture", Still) == still
    assert (tmp_path / "stages" / "capture.json").is_file()


def test_optional_fields_default_to_none():
    still = make_still("a.jpg")
    assert still.pose is None
    assert still.intrinsics is None
    assert still.depth_path is None


def test_ledger_written_to_stages(tmp_path):
    sweep = SweepDir(tmp_path)
    sweep.ledger.add("spines", 1.5, 1200, 0.002)
    sweep.ledger.add("ocr", 0.5, 0, 0.0)
    data = json.loads((tmp_path / "stages" / "_ledger.json").read_text())
    assert [row["stage"] for row in data] == ["spines", "ocr"]
    assert data[0]["tokens"] == 1200
    assert data[0]["usd_at_paid_rates"] == 0.002


def test_ledger_reloads_existing_entries(tmp_path):
    SweepDir(tmp_path).ledger.add("spines", 1.0, 10, 0.1)
    reopened = SweepDir(tmp_path)
    reopened.ledger.add("ocr", 1.0, 5, 0.05)
    assert len(reopened.ledger.entries) == 2
    assert reopened.ledger.total_usd() == pytest.approx(0.15)
