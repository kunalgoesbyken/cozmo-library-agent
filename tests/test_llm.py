import asyncio
import json
from datetime import UTC, datetime, timedelta

import pytest
from google.genai import errors
from pydantic import BaseModel

from libclaim.budget import Budget, BudgetExceeded
from libclaim.llm import (
    Llm,
    LiveCallBlocked,
    LlmReplyInvalid,
    LlmResponse,
    QuotaExhausted,
    UnknownModel,
    load_llm_config,
)
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
        await asyncio.sleep(0)
        outcome = self.script[model]
        if isinstance(outcome, list):
            outcome = outcome.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, str):
            return LlmResponse(text=outcome, input_tokens=10, output_tokens=0)
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


def seed_quota(tmp_path, clock, model, used):
    (tmp_path / "_quota.json").write_text(
        json.dumps({"day": clock.now.date().isoformat(), "counts": {model: used}})
    )


async def test_concurrent_calls_cannot_overshoot_budget(tmp_path):
    llm, client, _, _ = make(tmp_path, {FIRST: 1})
    llm.budget.cap_usd = CFG.reserve_usd * 1.5
    results = await asyncio.gather(*[llm.structured([FIRST], ["hi"], Answer) for _ in range(3)], return_exceptions=True)
    assert sum(isinstance(r, Answer) for r in results) == 1
    assert sum(isinstance(r, BudgetExceeded) for r in results) == 2
    assert llm.budget.pending == 0


async def test_concurrent_calls_respect_rpd(tmp_path):
    llm, client, _, clock = make(tmp_path, {FIRST: 1, SECOND: 2})
    seed_quota(tmp_path, clock, FIRST, CFG.models[FIRST].rpd - 1)
    await asyncio.gather(*[llm.structured([FIRST, SECOND], ["hi"], Answer) for _ in range(3)])
    assert client.calls.count(FIRST) == 1
    assert counts(tmp_path)[FIRST] == CFG.models[FIRST].rpd


@pytest.mark.parametrize("content", ["{not json", '{"day": "2026-10-26"}', '{"counts": {}}', "[]"])
async def test_corrupt_quota_resets_to_empty(tmp_path, content):
    llm, client, _, _ = make(tmp_path, {FIRST: 1})
    (tmp_path / "_quota.json").write_text(content)
    assert (await llm.structured([FIRST], ["hi"], Answer)).value == 1
    assert counts(tmp_path) == {FIRST: 1}
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("reply", ["", '{"value": ', '{"value": "x"}'])
async def test_invalid_reply_fails_over_then_raises(tmp_path, reply):
    llm, client, _, _ = make(tmp_path, {FIRST: reply, SECOND: reply})
    with pytest.raises(LlmReplyInvalid):
        await llm.structured([FIRST, SECOND], ["hi"], Answer)
    assert client.calls == [FIRST, SECOND]


async def test_invalid_reply_on_first_model_succeeds_on_second(tmp_path):
    llm, client, _, _ = make(tmp_path, {FIRST: "{", SECOND: 3})
    assert (await llm.structured([FIRST, SECOND], ["hi"], Answer)).value == 3


async def test_server_error_fails_over(tmp_path):
    boom = errors.ServerError(503, {"error": {"code": 503, "status": "UNAVAILABLE"}})
    llm, client, _, _ = make(tmp_path, {FIRST: boom, SECOND: 4})
    assert (await llm.structured([FIRST, SECOND], ["hi"], Answer)).value == 4


async def test_unknown_model_rejected_up_front(tmp_path):
    llm, client, _, _ = make(tmp_path, {})
    with pytest.raises(UnknownModel):
        await llm.structured(["nope"], ["hi"], Answer)
    assert client.calls == []


async def test_no_sleep_after_final_round(tmp_path):
    limited = rate_limited("1s")
    llm, client, _, clock = make(tmp_path, {FIRST: [limited] * CFG.max_rounds})
    with pytest.raises(QuotaExhausted):
        await llm.structured([FIRST], ["hi"], Answer)
    assert len(clock.slept) == CFG.max_rounds - 1


async def test_retry_info_without_delay_uses_default(tmp_path):
    body = {"error": {"code": 429, "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo"}]}}
    llm, client, _, clock = make(tmp_path, {FIRST: [errors.ClientError(429, body), 2]})
    assert (await llm.structured([FIRST], ["hi"], Answer)).value == 2
    assert clock.slept == [CFG.default_retry_delay_seconds]
