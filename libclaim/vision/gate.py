import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from libclaim.budget import BudgetExceeded
from libclaim.llm import DailyQuotaExhausted, QuotaExhausted

PERMANENT = float("inf")


class StageView:
    def __init__(self, gate: "VisionGate", stage: str):
        self.gate = gate
        self.stage = stage

    async def structured(self, models: list[str], parts: list[Any], schema: type[BaseModel]) -> BaseModel:
        return await self.gate.structured(models, parts, schema, self.stage)


def part_digest(part: Any) -> str:
    if isinstance(part, str):
        return hashlib.sha256(part.encode("utf-8")).hexdigest()
    blob = part.inline_data
    return hashlib.sha256(blob.mime_type.encode("utf-8") + blob.data).hexdigest()


def reply_key(models: list[str], parts: list[Any], schema: type[BaseModel]) -> str:
    material = json.dumps([sorted(models), schema.__name__, [part_digest(part) for part in parts]])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class VisionGate:
    def __init__(self, llm: Any, cooldown_s: float, clock: Callable[[], float] = time.monotonic,
                 cache_dir: Path | None = None):
        self.llm = llm
        self.cooldown_s = cooldown_s
        self.clock = clock
        self.cache_dir = cache_dir
        self.turn = 0
        self.down: dict[tuple[str, ...], tuple[float, str]] = {}

    def rotated(self, models: list[str]) -> list[str]:
        shift = self.turn % len(models)
        self.turn += 1
        return models[shift:] + models[:shift]

    def cached(self, models: list[str], parts: list[Any], schema: type[BaseModel]) -> Path | None:
        return None if self.cache_dir is None else self.cache_dir / f"{reply_key(models, parts, schema)}.json"

    def at(self, stage: str) -> "StageView":
        return StageView(self, stage)

    async def structured(self, models: list[str], parts: list[Any], schema: type[BaseModel],
                         stage: str | None = None) -> BaseModel:
        path = self.cached(models, parts, schema)
        if path is not None and path.is_file():
            return schema.model_validate_json(path.read_text(encoding="utf-8"))
        reply = await self.ask(models, parts, schema, stage)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(reply.model_dump_json(), encoding="utf-8")
        return reply

    async def ask(self, models: list[str], parts: list[Any], schema: type[BaseModel], stage: str | None) -> BaseModel:
        key = tuple(sorted(models))
        until, reason = self.down.get(key, (0.0, ""))
        if self.clock() < until:
            raise QuotaExhausted(f"{reason} earlier in this sweep for {list(key)}")
        try:
            return await self.llm.structured(self.rotated(models), parts, schema, stage=stage)
        except (DailyQuotaExhausted, BudgetExceeded) as error:
            self.down[key] = (PERMANENT, type(error).__name__)
            raise
        except QuotaExhausted as error:
            self.down[key] = (self.clock() + self.cooldown_s, type(error).__name__)
            raise
