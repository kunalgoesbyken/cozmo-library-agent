import json
from datetime import UTC, datetime, timedelta

import pytest
from google.genai import errors
from pydantic import BaseModel

from libclaim.budget import Budget, BudgetExceeded
from libclaim.llm import Llm, LlmResponse, LiveCallBlocked, load_llm_config
from libclaim.sweep import Ledger

CFG = load_llm_config()
FIRST, SECOND = list(CFG.models)[:2]


class Answer(BaseModel):
    value: int


def rate_limited(delay="2s"):
    body = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": [
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay}]}}
    return errors.ClientError(429, body)


class FakeClient:
    def __init__(self, script):
        self.script = script
        self.calls = []

    async def generate(self, model, parts, schema):
        self.calls.append(model)
        outcome = self.script[model]
        if isinstance(outcome, list):
            outcome = outcome.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return LlmResponse(text=json.dumps({"value": outcome}), input_tokens=100, output_tokens=20)


class Clock:
    def __init__(self):
        self.now = datetime(2026, 10, 26, 12, tzinfo=UTC)
        self.slept = []

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.slept.append(seconds)


def make(tmp_path, script, clock=None):
    clock = clock or Clock()
    client = FakeClient(script)
    ledger = Ledger(tmp_path / "ledger.json")
    llm = Llm(
        client=client,
        budget=Budget(tmp_path / "spend.jsonl"),
        ledger=ledger,
        quota_path=tmp_path / "_quota.json",
        now=clock,
        sleep=clock.sleep,
    )
    return llm, client, ledger, clock


def counts(tmp_path):
    return json.loads((tmp_path / "_quota.json").read_text())["counts"]


async def test_429_fails_over_and_quota_counts_both(tmp_path):
    llm, client, ledger, _ = make(tmp_path, {FIRST: rate_limited(), SECOND: 7})
    result = await llm.structured([FIRST, SECOND], ["hi"], Answer)
    assert result == Answer(value=7)
    assert client.calls == [FIRST, SECOND]
    assert counts(tmp_path) == {FIRST: 1, SECOND: 1}
    assert len(ledger.entries) == 1
    assert ledger.entries[0].tokens == 120
    assert llm.budget.total_usd() > 0


async def test_rpd_exhausted_model_is_skipped(tmp_path):
    llm, client, _, clock = make(tmp_path, {FIRST: 1, SECOND: 2})
    (tmp_path / "_quota.json").write_text(
        json.dumps({"day": clock.now.date().isoformat(), "counts": {FIRST: CFG.models[FIRST].rpd}})
    )
    result = await llm.structured([FIRST, SECOND], ["hi"], Answer)
    assert result.value == 2
    assert client.calls == [SECOND]


async def test_quota_resets_on_utc_day_rollover(tmp_path):
    llm, client, _, clock = make(tmp_path, {FIRST: 1, SECOND: 2})
    (tmp_path / "_quota.json").write_text(
        json.dumps({"day": clock.now.date().isoformat(), "counts": {FIRST: CFG.models[FIRST].rpd}})
    )
    clock.now += timedelta(days=1)
    result = await llm.structured([FIRST, SECOND], ["hi"], Answer)
    assert result.value == 1
    assert counts(tmp_path) == {FIRST: 1}


async def test_all_models_exhausted_raises(tmp_path):
    llm, client, _, clock = make(tmp_path, {FIRST: 1})
    (tmp_path / "_quota.json").write_text(
        json.dumps({"day": clock.now.date().isoformat(), "counts": {FIRST: CFG.models[FIRST].rpd}})
    )
    with pytest.raises(RuntimeError):
        await llm.structured([FIRST], ["hi"], Answer)
    assert client.calls == []


async def test_all_429_sleeps_capped_retry_delay_then_retries(tmp_path):
    script = {FIRST: [rate_limited("999s"), 5], SECOND: [rate_limited("3s"), rate_limited("3s")]}
    llm, client, _, clock = make(tmp_path, script)
    result = await llm.structured([FIRST, SECOND], ["hi"], Answer)
    assert result.value == 5
    assert clock.slept == [CFG.retry_delay_cap_seconds]
    assert client.calls == [FIRST, SECOND, FIRST]


async def test_non_rate_limit_error_propagates(tmp_path):
    boom = errors.ClientError(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT"}})
    llm, client, _, _ = make(tmp_path, {FIRST: boom, SECOND: 1})
    with pytest.raises(errors.ClientError):
        await llm.structured([FIRST, SECOND], ["hi"], Answer)
    assert client.calls == [FIRST]


async def test_budget_cap_blocks_before_call(tmp_path):
    llm, client, _, _ = make(tmp_path, {FIRST: 1})
    llm.budget.cap_usd = 0.0
    with pytest.raises(BudgetExceeded):
        await llm.structured([FIRST], ["hi"], Answer)
    assert client.calls == []


async def test_offline_guard_raises_for_live_client(tmp_path):
    llm = Llm(budget=Budget(tmp_path / "spend.jsonl"), quota_path=tmp_path / "_quota.json")
    with pytest.raises(LiveCallBlocked):
        await llm.structured([FIRST], ["hi"], Answer)
    assert not (tmp_path / "_quota.json").exists()
