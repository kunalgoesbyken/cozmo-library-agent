import hashlib
import json
import logging
import os
from pathlib import Path
from collections.abc import Awaitable
from typing import Any, TypeVar

import httpx
from aiolimiter import AsyncLimiter
from pydantic import BaseModel

from libclaim.llm import load_llm_config
from libclaim.resolve_config import CatalogConfig, load_resolve_config
from libclaim.settings import load_settings

PER_SECOND = 1.0

logger = logging.getLogger(__name__)
T = TypeVar("T")


def describe(error: httpx.HTTPError) -> str:
    if isinstance(error, httpx.HTTPStatusError):
        return f"status {error.response.status_code}"
    return type(error).__name__


class CatalogBadResponse(httpx.HTTPError):
    pass


async def guarded(call: Awaitable[list[T]], name: str) -> tuple[list[T], bool]:
    try:
        return await call, False
    except httpx.HTTPError as error:
        logger.warning("%s failed: %s", name, describe(error))
        return [], True


class CatalogOffline(RuntimeError):
    pass


class Candidate(BaseModel):
    source: str
    title: str
    author: str = ""
    publisher: str = ""
    isbn: str = ""
    year: int | None = None
    url: str = ""
    work: str = ""


class Edition(BaseModel):
    publisher: str
    isbn: str
    url: str
    year: int | None = None


YEAR_DIGITS = 4


def year_of(text: str) -> int | None:
    words = "".join(char if char.isalnum() else " " for char in text).split()
    return next((int(word) for word in words if len(word) == YEAR_DIGITS and word.isdigit()), None)


class CatalogHttp:
    def __init__(
        self,
        cache_dir: Path,
        client: httpx.AsyncClient | None = None,
        config: CatalogConfig | None = None,
        user_agent: str | None = None,
    ):
        self.config = config or load_resolve_config().catalog
        self.cache_dir = Path(cache_dir)
        self.client = client or httpx.AsyncClient(
            headers={"User-Agent": user_agent or load_settings().user_agent},
            timeout=self.config.timeout_seconds,
            follow_redirects=True,
        )
        self.limiters: dict[str, AsyncLimiter] = {}

    def cache_path(self, url: str) -> Path:
        return self.cache_dir / f"{hashlib.sha1(url.encode('utf-8')).hexdigest()}.json"

    async def get_json(self, url: str, params: dict[str, Any], headers: dict[str, str] | None = None) -> Any:
        request = httpx.URL(url, params=params)
        path = self.cache_path(str(request))
        if path.is_file():
            try:
                return json.loads(path.read_bytes())
            except ValueError:
                path.unlink()
        llm = load_llm_config()
        if os.environ.get(llm.offline_env) == llm.offline_value:
            raise CatalogOffline(f"catalogue request not cached while offline: {request}")
        limiter = self.limiters.setdefault(request.host, AsyncLimiter(self.config.requests_per_second, PER_SECOND))
        async with limiter:
            response = await self.client.get(request, headers=headers)
        response.raise_for_status()
        try:
            body = json.loads(response.content)
        except ValueError:
            raise CatalogBadResponse("catalogue reply is not JSON") from None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
        return body

    async def aclose(self) -> None:
        await self.client.aclose()
