import asyncio
import hashlib
import ipaddress
import socket
import json
import os
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from functools import cached_property
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx
from aiolimiter import AsyncLimiter
from protego import Protego

from libclaim.catalog.http import CatalogHttp, CatalogOffline
from libclaim.llm import load_llm_config
from libclaim.pricing.config import PricingConfig, load_pricing_config
from libclaim.pricing.markup import Page, jsonld_blocks
from libclaim.settings import load_settings

HTML_TYPES = ("text/html", "application/xhtml")
REDIRECT_HEADER = "location"
CLOSED_ROBOTS = "User-agent: *\nDisallow: /\n"
NOT_FOUND = 404
REDIRECT_FLOOR = 300
SERVER_ERROR = 500


class PriceFetchError(RuntimeError):
    pass


class RobotsDisallowed(PriceFetchError):
    pass


class UnsafeTarget(PriceFetchError):
    pass


Resolver = Callable[[str], Awaitable[list[str]]]


async def system_resolver(host: str) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return [info[4][0] for info in infos]


def normalized_host(url: str) -> str:
    return (urlsplit(url).hostname or "").lower().rstrip(".")


def is_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


BLOCKED_IPV6 = [ipaddress.ip_network(net) for net in ("64:ff9b::/96", "64:ff9b:1::/48", "::ffff:0:0/96", "::/96")]


def embedded_ipv4(ip: ipaddress.IPv6Address) -> list[ipaddress.IPv4Address]:
    return ([ip.sixtofour] if ip.sixtofour else []) + (list(ip.teredo) if ip.teredo else [])


def is_public(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%")[0])
    if isinstance(ip, ipaddress.IPv6Address) and any(ip in net for net in BLOCKED_IPV6):
        return False
    embedded = embedded_ipv4(ip) if isinstance(ip, ipaddress.IPv6Address) else []
    return ip.is_global and not ip.is_multicast and all(is_public(str(inner)) for inner in embedded)


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
    def blocked(self) -> str:
        return self.meta.get("blocked", "")

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
        resolver: Resolver = system_resolver,
    ):
        self.pricing = config or load_pricing_config()
        client = client or httpx.AsyncClient(
            headers={
                "User-Agent": user_agent or load_settings().user_agent,
                "Accept-Language": self.pricing.http.accept_language,
            },
            timeout=self.pricing.http.timeout_seconds,
            trust_env=False,
        )
        super().__init__(cache_dir, client=client)
        self.resolver = resolver
        self.allowed_hosts = self.configured_hosts()
        self.robots: dict[str, Protego] = {}
        self.inflight: dict[str, asyncio.Future] = {}

    def configured_hosts(self) -> set[str]:
        sources = (self.pricing.sources.bookchor, self.pricing.sources.bookswagon, self.pricing.sources.pustakkosh, self.pricing.sources.penguin_uk, self.pricing.sources.fx)
        return {normalized_host(source.url) for source in sources}

    def allow_hosts(self, hosts: set[str]) -> None:
        self.allowed_hosts |= {host.lower().rstrip(".") for host in hosts}

    async def guard(self, url: str, origin: str) -> None:
        host = normalized_host(url)
        if urlsplit(url).scheme not in self.pricing.http.allowed_schemes or not host:
            raise UnsafeTarget(f"scheme not allowed for {host or 'empty host'}")
        if host != normalized_host(origin) and host not in self.allowed_hosts:
            raise UnsafeTarget(f"{host} is not a configured source host")
        try:
            addresses = [host] if is_literal(host) else await self.resolver(host)
        except OSError as error:
            raise PriceFetchError(f"{host}: {type(error).__name__}") from error
        if not addresses or not all(is_public(address) for address in addresses):
            raise UnsafeTarget(f"{host} resolves to a non-public address")

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

    def fresh(self, response: CachedResponse) -> bool:
        if is_offline():
            return True
        cfg = self.pricing.cache
        ttl = (
            timedelta(hours=cfg.not_found_ttl_hours)
            if response.blocked or response.status == NOT_FOUND
            else timedelta(days=cfg.price_ttl_days)
        )
        return datetime.now(UTC) - response.retrieved_at <= ttl

    def interval(self, host: str) -> float:
        cfg = self.pricing.http
        return cfg.host_min_interval_seconds.get(host, cfg.min_interval_seconds)

    def limiter(self, host: str) -> AsyncLimiter:
        if host not in self.limiters:
            self.limiters[host] = AsyncLimiter(1, self.interval(host))
        return self.limiters[host]

    def pause(self, response: httpx.Response, attempt: int) -> float:
        cfg = self.pricing.http
        backoff = cfg.backoff_seconds * 2**attempt
        advised = response.headers.get("retry-after", "")
        if advised.isdigit():
            return min(max(float(advised), backoff), cfg.retry_after_max_seconds)
        return backoff

    async def send(self, url: str, headers: dict[str, str] | None) -> httpx.Response:
        cfg = self.pricing.http
        host = urlsplit(url).netloc
        failure, wait = "no response", 0.0
        for attempt in range(cfg.max_retries + 1):
            if attempt:
                await asyncio.sleep(wait)
            try:
                async with self.limiter(host):
                    response = await self.client.get(url, headers=headers, follow_redirects=False)
            except httpx.TransportError as error:
                failure, wait = type(error).__name__, cfg.backoff_seconds * 2**attempt
                continue
            if response.status_code not in cfg.retry_statuses:
                return response
            failure, wait = f"status {response.status_code}", self.pause(response, attempt)
        raise PriceFetchError(f"{host}: {failure} after {cfg.max_retries} retries")

    async def follow(self, url: str, headers: dict[str, str] | None, enforce: bool) -> tuple[httpx.Response, str]:
        current = url
        for _ in range(self.pricing.http.max_redirects + 1):
            await self.guard(current, url)
            if enforce:
                await self.require_allowed(current)
            response = await self.send(current, headers)
            target = response.headers.get(REDIRECT_HEADER)
            if not response.is_redirect or not target:
                return response, current
            current = urljoin(current, target)
        raise PriceFetchError(f"{urlsplit(url).netloc}: too many redirects")

    async def store(self, url: str, headers: dict[str, str] | None, enforce: bool) -> CachedResponse:
        if is_offline():
            raise CatalogOffline(f"price request not cached while offline: {urlsplit(url).netloc}")
        key = digest(url)
        try:
            response, final = await self.follow(url, headers, enforce)
        except RobotsDisallowed as error:
            self.write_meta(key, {"key": key, "url": url, "final_url": url, "blocked": str(error),
                                  "retrieved_at": datetime.now(UTC).isoformat()}, b"")
            raise
        content_type = response.headers.get("content-type", "")
        blocks, repairs = [], []
        if content_type.startswith(HTML_TYPES):
            blocks, repairs = jsonld_blocks(Page(response.text), self.pricing.jsonld_repairs.get(urlsplit(final).netloc, []))
        cacheable = response.status_code not in self.pricing.cache.uncached_statuses
        meta = {
            "key": key if cacheable else "",
            "url": url,
            "final_url": final,
            "status": response.status_code,
            "content_type": content_type,
            "retrieved_at": datetime.now(UTC).isoformat(),
            "sha256": hashlib.sha256(response.content).hexdigest(),
            "bytes": len(response.content),
            "jsonld": blocks,
            "jsonld_repairs": repairs,
        }
        if cacheable:
            self.write_meta(key, meta, response.content)
        return CachedResponse(meta, response.content, self.pricing)

    def write_meta(self, key: str, meta: dict[str, Any], body: bytes) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.body_path(key).write_bytes(body)
        self.meta_path(key).write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    def once(self, key: str, factory: Callable[[], Awaitable[Any]]) -> asyncio.Future:
        if key not in self.inflight:
            task = asyncio.ensure_future(factory())
            self.inflight[key] = task
            task.add_done_callback(lambda _: self.inflight.pop(key, None))
        return self.inflight[key]

    async def load_robots(self, origin: str) -> Protego:
        cfg = self.pricing.http
        response = await self.fetch(origin + cfg.robots_path)
        if response.status < REDIRECT_FLOOR:
            return Protego.parse(response.text)
        if response.status in cfg.robots_open_statuses:
            return Protego.parse("")
        if response.status >= SERVER_ERROR:
            raise PriceFetchError(f"{urlsplit(origin).netloc}: robots.txt status {response.status}")
        return Protego.parse(CLOSED_ROBOTS)

    async def robots_for(self, url: str) -> Protego:
        origin = origin_of(url)
        if origin not in self.robots:
            self.robots[origin] = await self.once("robots " + origin, lambda: self.load_robots(origin))
            delay = self.robots[origin].crawl_delay(self.pricing.http.robots_agent)
            if delay:
                host = urlsplit(url).netloc
                self.limiters[host] = AsyncLimiter(1, max(delay, self.interval(host)))
        return self.robots[origin]

    async def require_allowed(self, url: str) -> None:
        robots = await self.robots_for(url)
        if not robots.can_fetch(url, self.pricing.http.robots_agent):
            raise RobotsDisallowed(f"robots.txt disallows {urlsplit(url).netloc}{urlsplit(url).path}")

    async def get(self, url: str, headers: dict[str, str] | None, enforce: bool) -> CachedResponse:
        hit = self.read(url)
        if hit and self.fresh(hit):
            if hit.blocked:
                raise RobotsDisallowed(hit.blocked)
            return hit
        return await self.once(f"{enforce} {url}", lambda: self.store(url, headers, enforce))

    async def fetch(self, url: str, headers: dict[str, str] | None = None) -> CachedResponse:
        return await self.get(url, headers, False)

    async def cached_get(self, url: str, headers: dict[str, str] | None = None) -> CachedResponse:
        return await self.get(url, headers, True)
