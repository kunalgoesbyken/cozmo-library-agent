import asyncio

import pytest

from libclaim.live.bridge import load_live_config
from libclaim.live.genai_session import build_config
from libclaim.live.tools import (
    EXCLUSIONS_STAGE,
    LOCALE_STAGE,
    MONITOR_NAME,
    SPECS,
    ExclusionLog,
    LiveTools,
    LocaleChoice,
    StatementKind,
    StatementLog,
)
from libclaim.models import Still
from libclaim.sweep import SweepDir

CONFIG = load_live_config()


def still(seq, t_ms):
    return Still(seq=seq, t_ms=t_ms, path=f"{seq}.jpg", w=1, h=1, blur=1.0, glare=0.0, motion_dps=0.0)


class Harness:
    def __init__(self, tmp_path, now_ms=10_000):
        self.sweep = SweepDir(tmp_path / "sweep")
        self.stills = [still(1, 5_000), still(2, 6_000), still(3, 7_500), still(4, 10_000), still(5, 11_000)]
        self.now_ms = now_ms
        self.ended = 0
        self.shelves = {"B2", "S1.1"}
        self.tools = LiveTools(
            self.sweep, CONFIG, lambda: self.stills, lambda: self.now_ms, lambda: setattr(self, "ended", self.ended + 1),
            lambda: self.shelves,
        )

    def call(self, name, args):
        return asyncio.run(self.tools.call(name, args))


def test_first_edition_attaches_to_window_stills(tmp_path):
    h = Harness(tmp_path)
    result = h.call("record_user_statement", {"subject": "book", "kind": "first_edition", "value": "first edition"})
    assert result == {"recorded": True, "needs_appraisal": True}
    [statement] = h.sweep.read_stage("statements", StatementLog).items
    assert statement.kind is StatementKind.first_edition
    assert (statement.t_ms, statement.still_seq_from, statement.still_seq_to) == (10_000, 2, 4)
    assert statement.needs_appraisal


def test_window_length_comes_from_config(tmp_path):
    h = Harness(tmp_path)
    h.tools.config = CONFIG.model_copy(update={"statement_window_ms": 6_000})
    h.call("record_user_statement", {"subject": "book", "kind": "signed", "value": "signed"})
    [statement] = h.sweep.read_stage("statements", StatementLog).items
    assert (statement.still_seq_from, statement.still_seq_to) == (1, 4)


def test_other_statement_is_not_appraisal_and_empty_window_is_none(tmp_path):
    h = Harness(tmp_path, now_ms=60_000)
    result = h.call("record_user_statement", {"subject": "book", "kind": "other", "value": "mine"})
    assert result["needs_appraisal"] is False
    [statement] = h.sweep.read_stage("statements", StatementLog).items
    assert statement.still_seq_from is None and statement.still_seq_to is None


def test_art_print_statement_is_not_appraisal(tmp_path):
    h = Harness(tmp_path, now_ms=60_000)
    result = h.call("record_user_statement", {"subject": "framed poster", "kind": "art_print", "value": "print"})
    assert result["needs_appraisal"] is False


def test_statements_accumulate(tmp_path):
    h = Harness(tmp_path)
    for _ in range(2):
        h.call("record_user_statement", {"subject": "book", "kind": "signed", "value": "x"})
    assert len(h.sweep.read_stage("statements", StatementLog).items) == 2


def test_exclude_region_is_persisted(tmp_path):
    h = Harness(tmp_path)
    assert h.call("exclude_region", {"description": "bottom shelf", "shelf_id": "B2"}) == {"excluded": True}
    h.call("exclude_region", {"description": "top shelf"})
    items = h.sweep.read_stage(EXCLUSIONS_STAGE, ExclusionLog).items
    assert [(i.description, i.shelf_id) for i in items] == [("bottom shelf", "B2"), ("top shelf", None)]


def test_exclusion_with_an_invented_shelf_is_an_error_the_model_can_recover_from(tmp_path):
    h = Harness(tmp_path)
    result = h.call("exclude_region", {"description": "left shelf", "shelf_id": "current_shelf"})
    assert "error" in result and "current_shelf" in result["error"]
    assert result["known_shelves"] == ["B2", "S1.1"]
    assert not (h.sweep.stages / f"{EXCLUSIONS_STAGE}.json").is_file()
    assert h.call("exclude_region", {"description": "left shelf"}) == {"excluded": True}


def test_exclusion_shelf_is_matched_ignoring_case_and_stored_as_the_known_label(tmp_path):
    h = Harness(tmp_path)
    assert h.call("exclude_region", {"description": "bottom", "shelf_id": "s1.1"}) == {"excluded": True}
    [exclusion] = h.sweep.read_stage(EXCLUSIONS_STAGE, ExclusionLog).items
    assert exclusion.shelf_id == "S1.1"


def test_set_locale_persists_and_rejects_unknown_country(tmp_path):
    h = Harness(tmp_path)
    assert h.call("set_locale", {"country": "GB", "currency": "GBP"}) == {"country": "GB", "currency": "GBP"}
    assert h.sweep.read_stage(LOCALE_STAGE, LocaleChoice) == LocaleChoice(country="GB", currency="GBP")
    assert "error" in h.call("set_locale", {"country": "ZZ", "currency": "ZZZ"})


def test_status_counts_come_from_sweep_dir_only(tmp_path):
    h = Harness(tmp_path)
    assert h.call("get_sweep_status", {}) == {"frames": 0, "statements": 0, "exclusions": 0}
    h.sweep.frame_path(1, 100).parent.mkdir(parents=True)
    h.sweep.frame_path(1, 100).write_bytes(b"x")
    h.sweep.frame_path(2, 200).write_bytes(b"x")
    h.call("exclude_region", {"description": "shelf"})
    h.call("record_user_statement", {"subject": "b", "kind": "signed", "value": "x"})
    assert h.call("get_sweep_status", {}) == {"frames": 2, "statements": 1, "exclusions": 1}


def test_end_sweep_signals_callback(tmp_path):
    h = Harness(tmp_path)
    assert h.call("end_sweep", {}) == {"ended": True}
    assert h.ended == 1


def test_monitor_tool_is_idempotent_and_unknown_or_bad_args_return_errors(tmp_path):
    h = Harness(tmp_path)
    assert h.call(MONITOR_NAME, {}) == h.call(MONITOR_NAME, {}) == {"monitoring": True}
    assert "error" in h.call("nope", {})
    assert "error" in h.call("record_user_statement", {"subject": "b"})
    assert "error" in h.call("record_user_statement", {"subject": "b", "kind": "bogus", "value": "x"})


def test_specs_are_shared_with_bridge_config_and_monitor_is_non_blocking():
    names = {spec.name for spec in SPECS}
    assert names == {"record_user_statement", "exclude_region", "set_locale", "get_sweep_status", "end_sweep", MONITOR_NAME}
    assert [spec.name for spec in SPECS if spec.non_blocking] == [MONITOR_NAME]
    config = build_config(CONFIG, SPECS, None, prompt="x")
    declared = {d.name for d in config.tools[0].function_declarations}
    assert declared == names
    assert LiveTools.monitor_name == MONITOR_NAME


@pytest.mark.parametrize("spec", SPECS)
def test_spec_parameters_are_objects_or_absent(spec):
    assert spec.parameters is None or spec.parameters["type"] == "object"


@pytest.mark.parametrize("country", ["../IN", "IN/../GB", "", "/etc/passwd", "C:/x"])
def test_set_locale_rejects_traversal_with_generic_error(tmp_path, country):
    h = Harness(tmp_path)
    result = h.call("set_locale", {"country": country, "currency": "INR"})
    assert result == {"error": "set_locale rejected: invalid values"}
    assert not (h.sweep.stages / "locale.json").exists()


def test_set_locale_lowercase_and_currency_mismatch(tmp_path):
    h = Harness(tmp_path)
    assert h.call("set_locale", {"country": "in", "currency": "inr"}) == {"country": "IN", "currency": "INR"}
    assert "error" in h.call("set_locale", {"country": "IN", "currency": "USD"})


def test_storage_error_is_returned_not_raised(tmp_path):
    h = Harness(tmp_path)

    def boom(*_):
        raise OSError("disk")

    h.sweep.write_stage = boom
    assert h.call("exclude_region", {"description": "x"}) == {"error": "exclude_region failed: storage unavailable"}


def test_bug_inside_handler_is_not_masked_as_bad_arguments(tmp_path):
    h = Harness(tmp_path)

    def bug(*_):
        raise TypeError("bug")

    h.sweep.write_stage = bug
    with pytest.raises(TypeError):
        h.call("exclude_region", {"description": "x"})


def test_unexpected_argument_is_rejected(tmp_path):
    h = Harness(tmp_path)
    assert h.call("end_sweep", {"extra": 1}) == {"error": "end_sweep rejected: invalid arguments"}


def test_set_locale_schema_asks_for_iso_codes_not_names():
    spec = next(spec for spec in SPECS if spec.name == "set_locale")
    for field in ("country", "currency"):
        assert "ISO" in spec.parameters["properties"][field]["description"]


def test_tool_calls_are_logged_with_arguments_and_result(tmp_path, caplog):
    h = Harness(tmp_path)
    with caplog.at_level("INFO", logger="libclaim.live.tools"):
        h.call("set_locale", {"country": "India", "currency": "rupees"})
    assert "set_locale" in caplog.text and "India" in caplog.text and "invalid values" in caplog.text
