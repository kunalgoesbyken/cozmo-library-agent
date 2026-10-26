import asyncio
import hashlib
import json
import os
from datetime import UTC, datetime
from functools import cached_property
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from aiolimiter import AsyncLimiter
from protego import Protego

from libclaim.catalog.http import CatalogHttp, CatalogOffline
from libclaim.llm import load_llm_config
from libclaim.pricing.config import PricingConfig, load_pricing_config
from libclaim.pricing.markup import Page, jsonld_blocks
from libclaim.settings import load_settings

HTML_TYPES = ("text/html", "application/xhtml")
ERROR_CLASS = 400


class PriceFetchError(RuntimeError):
    pass


class RobotsDisallowed(PriceFetchError):
    pass


def digest(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


def origin_of(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


class CachedResponse:
    def __init__(self, meta: dict[str, Any], body: bytes, config: PricingConfig):
        self.meta = meta
        self.body = body
        self.config = config

    @property
    def url(self) -> str:
        return self.meta["url"]

    @property
    def final_url(self) -> str:
        return self.meta["final_url"]

    @property
    def status(self) -> int:
        return self.meta["status"]

    @property
    def cache_ref(self) -> str:
        return self.meta["key"]

    @property
    def retrieved_at(self) -> datetime:
        return datetime.fromisoformat(self.meta["retrieved_at"])

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")

    @property
    def jsonld(self) -> list[Any]:
        return self.meta["jsonld"]

    @cached_property
    def page(self) -> Page:
        return Page(self.text)

    @property
    def canonical(self) -> str:
        return self.page.canonical or self.final_url

    def json(self, **kwargs: Any) -> Any:
        return json.loads(self.body, **kwargs)

    def next_data(self) -> dict[str, Any]:
        raw = self.page.script_with(id="__NEXT_DATA__")
        return json.loads(raw) if raw else {}


def is_offline() -> bool:
    llm = load_llm_config()
    return os.environ.get(llm.offline_env) == llm.offline_value


class PriceHttp(CatalogHttp):
    def __init__(
        self,
        cache_dir: Path,
        client: httpx.AsyncClient | None = None,
        config: PricingConfig | None = None,
        user_agent: str | None = None,
    ):
        self.pricing = config or load_pricing_config()
        client = client or httpx.AsyncClient(
            headers={
                "User-Agent": user_agent or load_settings().user_agent,
                "Accept-Language": self.pricing.http.accept_language,
            },
            timeout=self.pricing.http.timeout_seconds,
            follow_redirects=True,
        )
        super().__init__(cache_dir, client=client)
        self.robots: dict[str, Protego] = {}

    def body_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.body"

    def meta_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def read(self, url: str) -> CachedResponse | None:
        key = digest(url)
        if not self.meta_path(key).is_file() or not self.body_path(key).is_file():
            return None
        meta = json.loads(self.meta_path(key).read_bytes())
        return CachedResponse(meta, self.body_path(key).read_bytes(), self.pricing)

    def interval(self, host: str) -> float:
        cfg = self.pricing.http
        return cfg.host_min_interval_seconds.get(host, cfg.min_interval_seconds)

    def limiter(self, host: str) -> AsyncLimiter:
        if host not in self.limiters:
            self.limiters[host] = AsyncLimiter(1, self.interval(host))
        return self.limiters[host]

    async def send(self, url: str, headers: dict[str, str] | None) -> httpx.Response:
        cfg = self.pricing.http
        host = urlsplit(url).netloc
        failure = "no response"
        for attempt in range(cfg.max_retries + 1):
            if attempt:
                await asyncio.sleep(cfg.backoff_seconds * 2 ** (attempt - 1))
            try:
                async with self.limiter(host):
                    response = await self.client.get(url, headers=headers)
            except httpx.TransportError as error:
                failure = type(error).__name__
                continue
            if response.status_code not in cfg.retry_statuses:
                return response
            failure = f"status {response.status_code}"
        raise PriceFetchError(f"{host}: {failure} after {cfg.max_retries} retries")

    async def store(self, url: str, headers: dict[str, str] | None) -> CachedResponse:
        if is_offline():
            raise CatalogOffline(f"price request not cached while offline: {urlsplit(url).netloc}")
        response = await self.send(url, headers)
        key = digest(url)
        content_type = response.headers.get("content-type", "")
        blocks, repairs = [], []
        if content_type.startswith(HTML_TYPES):
            blocks, repairs = jsonld_blocks(Page(response.text), self.pricing.jsonld_repairs)
        meta = {
            "key": key,
            "url": url,
            "final_url": str(response.url),
            "status": response.status_code,
            "content_type": content_type,
            "retrieved_at": datetime.now(UTC).isoformat(),
            "sha256": hashlib.sha256(response.content).hexdigest(),
            "bytes": len(response.content),
            "jsonld": blocks,
            "jsonld_repairs": repairs,
        }
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.body_path(key).write_bytes(response.content)
        self.meta_path(key).write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        return CachedResponse(meta, response.content, self.pricing)

    async def robots_for(self, url: str) -> Protego:
        origin = origin_of(url)
        if origin not in self.robots:
            robots_url = origin + self.pricing.http.robots_path
            response = await self.fetch(robots_url, None)
            if response.status >= 500:
                raise PriceFetchError(f"{urlsplit(url).netloc}: robots.txt status {response.status}")
            parsed = Protego.parse(response.text if response.status < ERROR_CLASS else "")
            self.robots[origin] = parsed
            delay = parsed.crawl_delay(self.pricing.http.robots_agent)
            if delay:
                host = urlsplit(url).netloc
                self.limiters[host] = AsyncLimiter(1, max(delay, self.interval(host)))
        return self.robots[origin]

    async def fetch(self, url: str, headers: dict[str, str] | None) -> CachedResponse:
        return self.read(url) or await self.store(url, headers)

    async def cached_get(self, url: str, headers: dict[str, str] | None = None) -> CachedResponse:
        hit = self.read(url)
        if hit:
            return hit
        robots = await self.robots_for(url)
        if not robots.can_fetch(url, self.pricing.http.robots_agent):
            raise RobotsDisallowed(f"robots.txt disallows {urlsplit(url).netloc}{urlsplit(url).path}")
        return await self.store(url, headers)
