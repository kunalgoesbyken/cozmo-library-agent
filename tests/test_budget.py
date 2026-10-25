import pytest

from libclaim.budget import Budget, BudgetExceeded


def test_record_returns_usd_and_totals_add_up(tmp_path):
    budget = Budget(tmp_path / "spend.jsonl")
    first = budget.record("gemini-3.8-flash", 1_000_000, 0)
    second = budget.record("gemini-3.8-flash", 0, 1_000_000)
    assert first == pytest.approx(0.50)
    assert second == pytest.approx(3.00)
    assert budget.total_usd() == pytest.approx(3.50)


def test_modality_pricing(tmp_path):
    budget = Budget(tmp_path / "spend.jsonl")
    assert budget.record("gemini-3.8-live", 1_000_000, 0, "audio") == pytest.approx(3.00)
    assert budget.record("gemini-3.8-live", 1_000_000, 0, "video") == pytest.approx(1.00)


def test_gemma_is_free(tmp_path):
    budget = Budget(tmp_path / "spend.jsonl")
    assert budget.record("gemma-4-31b-it", 5_000_000, 5_000_000) == 0


def test_cap_trips(tmp_path):
    budget = Budget(tmp_path / "spend.jsonl", cap_usd=0.80)
    budget.record("gemini-3.8-flash", 2_000_000, 0)
    with pytest.raises(BudgetExceeded):
        budget.check()


def test_under_cap_passes(tmp_path):
    budget = Budget(tmp_path / "spend.jsonl", cap_usd=0.80)
    budget.record("gemini-3.5-flash-lite", 1000, 1000)
    budget.check()


def test_unknown_model_priced_at_highest(tmp_path):
    budget = Budget(tmp_path / "spend.jsonl")
    assert budget.record("mystery-model", 1_000_000, 0) == pytest.approx(3.00)
    assert budget.record("mystery-model", 0, 1_000_000) == pytest.approx(12.00)


def test_unknown_modality_priced_conservatively(tmp_path):
    budget = Budget(tmp_path / "spend.jsonl")
    assert budget.record("gemini-3.5-flash-lite", 1_000_000, 0, "audio") == pytest.approx(3.00)


def test_persisted_across_instances(tmp_path):
    path = tmp_path / "nested" / "spend.jsonl"
    Budget(path).record("gemini-3.8-flash", 1_000_000, 0)
    assert Budget(path).total_usd() == pytest.approx(0.50)
