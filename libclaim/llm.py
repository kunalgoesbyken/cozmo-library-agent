import asyncio
import json
import os
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any, Protocol, TypeVar

import yaml
from aiolimiter import AsyncLimiter
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from libclaim.budget import Budget
from libclaim.settings import CONFIG_DIR, load_settings
from libclaim.sweep import Ledger

LLM_CONFIG_PATH = CONFIG_DIR / "llm.yaml"
SECONDS_PER_MINUTE = 60
RETRY_INFO_SUFFIX = "RetryInfo"

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LiveCallBlocked(RuntimeError):
    pass


class QuotaExhausted(RuntimeError):
    pass


class DailyQuotaExhausted(QuotaExhausted):
    pass


class UnknownModel(ValueError):
    pass


class LlmReplyInvalid(RuntimeError):
    pass


class ModelLimits(BaseModel):
    rpm: int
    rpd: int


class LlmConfig(BaseModel):
    models: dict[str, ModelLimits]
    quota_path: Path
    retry_delay_cap_seconds: float
    default_retry_delay_seconds: float
    max_rounds: int
    ledger_stage: str
    offline_env: str
    rate_limit_code: int
    offline_value: str
    reserve_usd: float
    failover_min_code: int
    max_invalid_replies: int


class LlmResponse(BaseModel):
    text: str
    input_tokens: int
    output_tokens: int


class LlmClient(Protocol):
    async def generate(self, model: str, parts: list[Any], schema: type[BaseModel]) -> LlmResponse: ...


class GenaiClient:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    async def generate(self, model: str, parts: list[Any], schema: type[BaseModel]) -> LlmResponse:
        response = await self.client.aio.models.generate_content(
            model=model,
            contents=parts,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        usage = response.usage_metadata
        return LlmResponse(
            text=response.text or "",
            input_tokens=(usage.prompt_token_count or 0) if usage else 0,
            output_tokens=((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)) if usage else 0,
        )


def load_llm_config(path: Path = LLM_CONFIG_PATH) -> LlmConfig:
    return LlmConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def retry_delay_seconds(error: errors.APIError, default: float) -> float:
    body = error.details if isinstance(error.details, dict) else {}
    for detail in body.get("error", {}).get("details", []):
        if detail.get("@type", "").endswith(RETRY_INFO_SUFFIX) and detail.get("retryDelay"):
            return float(str(detail["retryDelay"]).removesuffix("s"))
    return default


class Quota:
    def __init__(self, path: Path, now: Callable[[], datetime]):
        self.path = path
        self.now = now

    def _load(self) -> dict[str, Any]:
        today = self.now().astimezone(UTC).date().isoformat()
        if self.path.is_file():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if data["day"] == today:
                    return {"day": today, "counts": dict(data["counts"])}
            except (OSError, ValueError, KeyError, TypeError):
                logger.warning("quota file %s unreadable, resetting counts", self.path)
        return {"day": today, "counts": {}}

    def used(self, model: str) -> int:
        return self._load()["counts"].get(model, 0)

    def add(self, model: str) -> None:
        data = self._load()
        data["counts"][model] = data["counts"].get(model, 0) + 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(temp, self.path)


def live_client() -> LlmClient:
    config = load_llm_config()
    if os.environ.get(config.offline_env) == config.offline_value:
        raise LiveCallBlocked("live Gemini call attempted while offline")
    return GenaiClient(load_settings().gemini_api_key)


class Llm:
    def __init__(
        self,
        client: LlmClient | None = None,
        budget: Budget | None = None,
        ledger: Ledger | None = None,
        quota_path: Path | None = None,
        config: LlmConfig | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self.config = config or load_llm_config()
        self.client = client
        self.budget = budget or Budget()
        self.ledger = ledger
        self.quota = Quota(quota_path or self.config.quota_path, now)
        self.sleep = sleep
        self.limiters = {
            name: AsyncLimiter(limits.rpm, SECONDS_PER_MINUTE) for name, limits in self.config.models.items()
        }

    async def structured(self, models: list[str], parts: list[Any], schema: type[T], stage: str | None = None) -> T:
        unknown = [name for name in models if name not in self.config.models]
        if unknown:
            raise UnknownModel(f"models not in llm config: {unknown}")
        client = self.client or live_client()
        invalid = 0
        for round_number in range(self.config.max_rounds):
            delays = []
            for model in models:
                try:
                    result = await self._call(client, model, parts, schema, stage or self.config.ledger_stage)
                except errors.APIError as error:
                    if error.code != self.config.rate_limit_code and error.code < self.config.failover_min_code:
                        raise
                    delays.append(retry_delay_seconds(error, self.config.default_retry_delay_seconds))
                except ValidationError:
                    invalid += 1
                    if invalid >= self.config.max_invalid_replies:
                        raise LlmReplyInvalid(f"{invalid} invalid replies from {models}") from None
                else:
                    if result is not None:
                        return result
            if not delays:
                if invalid:
                    raise LlmReplyInvalid(f"{invalid} invalid replies from {models}")
                raise DailyQuotaExhausted(f"daily quota exhausted for {models}")
            if round_number + 1 < self.config.max_rounds:
                await self.sleep(min(max(delays), self.config.retry_delay_cap_seconds))
        raise QuotaExhausted(f"rate limited on every model in {models}")

    async def _call(self, client: LlmClient, model: str, parts: list[Any], schema: type[T], stage: str) -> T | None:
        async with self.limiters[model]:
            if self.quota.used(model) >= self.config.models[model].rpd:
                return None
            self.budget.reserve(self.config.reserve_usd)
            try:
                self.quota.add(model)
                started = time.monotonic()
                response = await client.generate(model, parts, schema)
                usd = self.budget.record(model, response.input_tokens, response.output_tokens)
            finally:
                self.budget.release(self.config.reserve_usd)
        if self.ledger is not None:
            self.ledger.add(
                stage,
                time.monotonic() - started,
                response.input_tokens + response.output_tokens,
                usd,
            )
        return schema.model_validate_json(response.text)


@cache
def default_llm() -> Llm:
    return Llm()


async def structured(models: list[str], parts: list[Any], schema: type[T]) -> T:
    return await default_llm().structured(models, parts, schema)
