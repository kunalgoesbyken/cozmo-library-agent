import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from libclaim.budget import BudgetExceeded
from libclaim.llm import DailyQuotaExhausted, QuotaExhausted

PERMANENT = float("inf")


class VisionGate:
    def __init__(self, llm: Any, cooldown_s: float, clock: Callable[[], float] = time.monotonic):
        self.llm = llm
        self.cooldown_s = cooldown_s
        self.clock = clock
        self.turn = 0
        self.down: dict[tuple[str, ...], tuple[float, str]] = {}

    def rotated(self, models: list[str]) -> list[str]:
        shift = self.turn % len(models)
        self.turn += 1
        return models[shift:] + models[:shift]

    async def structured(self, models: list[str], parts: list[Any], schema: type[BaseModel]) -> BaseModel:
        key = tuple(sorted(models))
        until, reason = self.down.get(key, (0.0, ""))
        if self.clock() < until:
            raise QuotaExhausted(f"{reason} earlier in this sweep for {list(key)}")
        try:
            return await self.llm.structured(self.rotated(models), parts, schema)
        except (DailyQuotaExhausted, BudgetExceeded) as error:
            self.down[key] = (PERMANENT, type(error).__name__)
            raise
        except QuotaExhausted as error:
            self.down[key] = (self.clock() + self.cooldown_s, type(error).__name__)
            raise
