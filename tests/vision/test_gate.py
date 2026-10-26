import pytest

from libclaim.llm import DailyQuotaExhausted, QuotaExhausted
from libclaim.vision.gate import VisionGate

MODELS = ["a", "b"]
COOLDOWN = 30.0


class Recorder:
    def __init__(self, error=None):
        self.error = error
        self.orders = []

    async def structured(self, models, parts, schema):
        self.orders.append(models)
        if self.error:
            raise self.error
        return "ok"


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


async def test_calls_alternate_between_models():
    llm = Recorder()
    gate = VisionGate(llm, COOLDOWN)
    for _ in range(3):
        await gate.structured(MODELS, [], None)
    assert llm.orders == [["a", "b"], ["b", "a"], ["a", "b"]]


async def test_rate_limit_cools_down_then_retries():
    llm, clock = Recorder(QuotaExhausted("rate limited")), Clock()
    gate = VisionGate(llm, COOLDOWN, clock)
    for _ in range(2):
        with pytest.raises(QuotaExhausted):
            await gate.structured(MODELS, [], None)
    assert len(llm.orders) == 1
    clock.now += COOLDOWN + 1
    with pytest.raises(QuotaExhausted):
        await gate.structured(["b", "a"], [], None)
    assert len(llm.orders) == 2


async def test_daily_exhaustion_trips_for_the_rest_of_the_sweep():
    llm, clock = Recorder(DailyQuotaExhausted("daily")), Clock()
    gate = VisionGate(llm, COOLDOWN, clock)
    with pytest.raises(QuotaExhausted):
        await gate.structured(MODELS, [], None)
    clock.now += 10 * COOLDOWN
    with pytest.raises(QuotaExhausted):
        await gate.structured(MODELS, [], None)
    assert len(llm.orders) == 1
