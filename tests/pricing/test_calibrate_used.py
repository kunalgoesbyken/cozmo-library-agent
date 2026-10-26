import csv
import importlib.util
import json
from decimal import Decimal

import pytest

from libclaim.pricing.config import load_pricing_config
from libclaim.pricing.sources.bookchor import Bookchor
from libclaim.settings import ROOT
from tests.pricing.factories import ISBN


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("calibrate_used", ROOT / "tools" / "calibrate_used.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pair(tool, used):
    return tool.Pair(
        isbn=str(used), source="Bookchor", new=Decimal(100), used=Decimal(used), new_url="n", used_url="u", new_cache_ref="n", used_cache_ref="c"
    )


async def test_pairs_come_from_observed_used_and_new_on_the_same_source(tool, http):
    pairs = await tool.collect_pairs([Bookchor(http)], [ISBN, "9780099590088"])
    assert [(p.isbn, p.source, p.new, p.used) for p in pairs] == [(ISBN, "Bookchor", Decimal(1571), Decimal(661))]
    assert pairs[0].new_url.endswith("everything-i-never-told-you") and pairs[0].used_cache_ref


def test_factor_is_the_median_ratio_and_other_grades_keep_their_proportion(tool):
    cfg = load_pricing_config()
    factors = tool.build_factors([pair(tool, 30), pair(tool, 40), pair(tool, 50)], cfg)
    old = cfg.condition_factors["v1"]
    assert factors["good"] == Decimal("0.4")
    assert factors["new"] == Decimal(1)
    assert factors["very_good"] == (Decimal("0.4") * old["very_good"] / old["good"]).quantize(Decimal("0.001"))


def test_isbns_in_ground_truth_are_excluded_and_requests_are_capped(tool, tmp_path):
    cfg = load_pricing_config()
    truth = tmp_path / "prices15.csv"
    with truth.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([cfg.calibration.isbn_column, "price"])
        writer.writerow(["111", "1"])
    chosen = tool.choose_isbns([str(n) for n in range(100, 140)], cfg, truth)
    assert "111" not in chosen
    sources = len(cfg.calibration.sources)
    assert (len(chosen) + 1) * sources <= cfg.calibration.max_requests


def test_missing_ground_truth_file_excludes_nothing(tool, tmp_path):
    assert tool.choose_isbns(["111"], load_pricing_config(), tmp_path / "absent.csv") == ["111"]


async def test_snapshot_records_pairs_and_factors_apart_from_ground_truth(tool, http, tmp_path):
    cfg = load_pricing_config()
    cfg = cfg.model_copy(update={"calibration": cfg.calibration.model_copy(update={"min_pairs": 1})})
    path = await tool.calibrate([ISBN], [Bookchor(http)], cfg, tmp_path / "out")
    assert path.parent == tmp_path / "out" and path.name == "condition_factors.v1.json"
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    assert snapshot["version"] == "v1" and len(snapshot["pairs"]) == 1
    assert Decimal(snapshot["condition_factors"]["good"]) == (Decimal(661) / Decimal(1571)).quantize(Decimal("0.001"))
    assert snapshot["pairs"][0]["used_cache_ref"] and snapshot["pairs"][0]["new_url"]


async def test_too_few_pairs_writes_nothing(tool, http, tmp_path):
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        await tool.calibrate([ISBN], [Bookchor(http)], load_pricing_config(), out)
    assert not out.exists()
