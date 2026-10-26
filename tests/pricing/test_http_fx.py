import hashlib
from datetime import date
from decimal import Decimal

import httpx
import pytest

from libclaim.catalog.http import CatalogOffline
from libclaim.models import NotFound
from libclaim.pricing.http import PriceFetchError, PriceHttp, RobotsDisallowed
from libclaim.pricing.sources.bookchor import Bookchor
from libclaim.pricing.sources.fx import Fx
from libclaim.pricing.sources.penguin_uk import PenguinUk
from libclaim.pricing.sources.pustakkosh import Pustakkosh
from tests.pricing.conftest import public_resolver, Site, fast_config, make_http

PAGE = "https://www.bookchor.com/book/9780143127550/x"
BLOCK_BOOKS = {"https://www.bookchor.com/robots.txt": {"status": 200, "type": "text/plain", "body": "User-agent: *\nDisallow: /book/\n"}}


async def test_fx_rate_from_ecb_via_frankfurter(http):
    record = await Fx(http).rate("GBP", "INR", date(2026, 10, 23))
    assert (record.base, record.quote, record.rate, record.rate_date) == ("GBP", "INR", Decimal("126.93"), date(2026, 10, 23))
    assert "providers=ECB" in record.url and "date=2026-10-23" in record.url
    assert record.cache_ref and record.provider == "ECB via Frankfurter v2"


async def test_fx_without_a_rate_raises(http):
    with pytest.raises(PriceFetchError):
        await Fx(http).rate("GBP", "XXX", date(2026, 10, 23))


async def test_cache_key_is_sha1_of_url_and_keeps_body_and_jsonld(http, tmp_path):
    response = await http.cached_get(PAGE)
    key = hashlib.sha1(PAGE.encode()).hexdigest()
    assert response.cache_ref == key
    assert (tmp_path / "cache" / f"{key}.body").read_bytes() == response.body
    assert any(block.get("@type") == "Product" for block in response.jsonld if isinstance(block, dict))


async def test_robots_is_fetched_once_per_host(http, site):
    await http.cached_get(PAGE)
    await http.cached_get("https://www.bookchor.com/book/9780099590088/x")
    assert site.requests.count("https://www.bookchor.com/robots.txt") == 1


async def test_robots_disallow_blocks_the_page_fetch(tmp_path, online):
    site = Site(BLOCK_BOOKS)
    with pytest.raises(RobotsDisallowed):
        await make_http(tmp_path, site).cached_get(PAGE)
    assert PAGE not in site.requests


async def test_blocked_source_returns_not_found_blocked(tmp_path, online):
    site = Site(BLOCK_BOOKS)
    result = await Bookchor(make_http(tmp_path, site)).quote("9780143127550", "new")
    assert isinstance(result, NotFound) and result.reason == "blocked"


async def test_rate_limited_response_is_retried_then_served(tmp_path, online):
    calls = []

    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        calls.append(1)
        return httpx.Response(429) if len(calls) < 3 else httpx.Response(200, content=b"[]")

    http = PriceHttp(tmp_path / "cache", resolver=public_resolver, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), config=fast_config())
    response = await http.cached_get("https://example.test/a")
    assert response.status == 200 and len(calls) == 3


async def test_persistent_server_error_raises_and_is_not_cached(tmp_path, online):
    def handler(request):
        return httpx.Response(404) if request.url.path == "/robots.txt" else httpx.Response(503)

    http = PriceHttp(
        tmp_path / "cache",
        resolver=public_resolver,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        config=fast_config(max_retries=2),
    )
    with pytest.raises(PriceFetchError):
        await http.cached_get("https://example.test/a")
    assert http.read("https://example.test/a") is None


async def test_source_reports_unavailable_when_the_host_keeps_failing(tmp_path, online):
    def handler(request):
        return httpx.Response(404) if request.url.path == "/robots.txt" else httpx.Response(429)

    http = PriceHttp(
        tmp_path / "cache",
        resolver=public_resolver,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        config=fast_config(max_retries=1),
    )
    result = await Bookchor(http).quote("9780143127550", "new")
    assert isinstance(result, NotFound) and result.reason == "unavailable"


def snapshot(cache):
    return {path.name: path.read_bytes() for path in sorted(cache.iterdir())}


async def run_all(http):
    return [
        await Bookchor(http).quote("9780143127550", "new"),
        await Bookchor(http).quote("9780143127550", "used"),
        await Pustakkosh(http).quote("9788173711466", "used"),
        await PenguinUk(http).quote("9780099590088", "new"),
        await Fx(http).rate("GBP", "INR", date(2026, 10, 23)),
    ]


async def test_replay_with_no_network_is_byte_identical(tmp_path, site, online, monkeypatch):
    first = await run_all(make_http(tmp_path, site))
    before = snapshot(tmp_path / "cache")
    monkeypatch.setenv("LIBCLAIM_OFFLINE", "1")

    def refuse(request):
        raise AssertionError("network used during replay")

    offline = PriceHttp(tmp_path / "cache", resolver=public_resolver, client=httpx.AsyncClient(transport=httpx.MockTransport(refuse)), config=fast_config())
    second = await run_all(offline)
    assert [m.model_dump_json() for m in first] == [m.model_dump_json() for m in second]
    assert snapshot(tmp_path / "cache") == before


async def test_uncached_request_while_offline_raises(tmp_path):
    with pytest.raises(CatalogOffline):
        await make_http(tmp_path, Site()).cached_get(PAGE)
