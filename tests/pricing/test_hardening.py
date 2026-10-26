import asyncio
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import httpx
import pytest

from libclaim.models import NotFound, PriceQuote
from libclaim.pricing.http import PriceFetchError, PriceHttp, RobotsDisallowed
from libclaim.pricing.sources.bookchor import Bookchor
from libclaim.pricing.sources.pustakkosh import Pustakkosh
from tests.pricing.conftest import Site, fast_config, make_http

ISBN = "9780143127550"
OTHER = "9780141439518"
PAGE = f"https://www.bookchor.com/book/{ISBN}/x"


def product(isbn, offers):
    return {"@type": "Product", "isbn": isbn, "offers": offers}


def offer(price, condition="NewCondition", stock="InStock", currency="INR"):
    return {
        "@type": "Offer",
        "price": str(price),
        "priceCurrency": currency,
        "itemCondition": f"https://schema.org/{condition}",
        "availability": f"https://schema.org/{stock}",
    }


def page(*products):
    blocks = "".join(f'<script type="application/ld+json">{json.dumps(p)}</script>' for p in products)
    return f"<html><head>{blocks}</head></html>"


def serve(tmp_path, routes, config=None):
    def handler(request):
        url = str(request.url)
        if url not in routes:
            return httpx.Response(404)
        status, body, headers = routes[url]
        return httpx.Response(status, content=body, headers=headers)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return PriceHttp(tmp_path / "cache", client=client, config=config or fast_config())


def html(body):
    return (200, body.encode(), {"content-type": "text/html"})


@pytest.fixture(autouse=True)
def online(monkeypatch):
    monkeypatch.delenv("LIBCLAIM_OFFLINE")


async def test_offers_come_only_from_the_product_with_the_queried_isbn(tmp_path):
    body = page(product(OTHER, offer(99)), product(ISBN, offer(500)))
    result = await Bookchor(serve(tmp_path, {PAGE: html(body)})).quote(ISBN, "new")
    assert isinstance(result, PriceQuote) and result.amount == Decimal("500")


async def test_related_product_offers_never_price_a_book_without_its_own_offer(tmp_path):
    body = page(product(ISBN, []), product(OTHER, offer(99)))
    result = await Bookchor(serve(tmp_path, {PAGE: html(body)})).quote(ISBN, "new")
    assert isinstance(result, NotFound) and result.reason == "not_found"


async def test_lowest_in_stock_offer_wins_over_cheaper_out_of_stock(tmp_path):
    offers = [offer(100, stock="OutOfStock"), offer(700), offer(650), offer(300, condition="UsedCondition")]
    result = await Bookchor(serve(tmp_path, {PAGE: html(page(product(ISBN, offers)))})).quote(ISBN, "new")
    assert (result.amount, result.availability) == (Decimal("650"), "InStock")


async def test_out_of_stock_is_used_only_when_nothing_is_in_stock(tmp_path):
    offers = [offer(400, stock="OutOfStock"), offer(250, stock="OutOfStock")]
    result = await Bookchor(serve(tmp_path, {PAGE: html(page(product(ISBN, offers)))})).quote(ISBN, "new")
    assert (result.amount, result.availability) == (Decimal("250"), "OutOfStock")


async def test_offer_in_another_currency_is_rejected(tmp_path):
    body = page(product(ISBN, [offer(12, currency="USD")]))
    result = await Bookchor(serve(tmp_path, {PAGE: html(body)})).quote(ISBN, "new")
    assert isinstance(result, NotFound) and result.reason == "currency_mismatch"


@pytest.mark.parametrize("status", [401, 403])
async def test_robots_bot_wall_fails_closed(tmp_path, status):
    routes = {"https://www.bookchor.com/robots.txt": (status, b"", {}), PAGE: html(page(product(ISBN, [offer(5)])))}
    result = await Bookchor(serve(tmp_path, routes)).quote(ISBN, "new")
    assert isinstance(result, NotFound) and result.reason == "blocked"


async def test_robots_gone_means_allowed(tmp_path):
    routes = {"https://www.bookchor.com/robots.txt": (410, b"", {}), PAGE: html(page(product(ISBN, [offer(5)])))}
    assert isinstance(await Bookchor(serve(tmp_path, routes)).quote(ISBN, "new"), PriceQuote)


async def test_cross_host_redirect_is_checked_against_the_target_host(tmp_path):
    start = "https://one.test/a"
    routes = {
        start: (302, b"", {"location": "https://two.test/private/x"}),
        "https://two.test/robots.txt": (200, b"User-agent: *\nDisallow: /private/\n", {}),
        "https://two.test/private/x": html("secret"),
    }
    http = serve(tmp_path, routes)
    with pytest.raises(RobotsDisallowed):
        await http.cached_get(start)


async def test_cross_host_redirect_uses_the_target_hosts_robots_and_records_final_url(tmp_path):
    routes = {
        "https://one.test/a": (302, b"", {"location": "https://two.test/ok"}),
        "https://two.test/ok": html("fine"),
    }
    response = await serve(tmp_path, routes).cached_get("https://one.test/a")
    assert response.final_url == "https://two.test/ok" and response.text == "fine"


async def test_redirect_loop_is_bounded(tmp_path):
    routes = {"https://one.test/a": (302, b"", {"location": "https://one.test/a"})}
    with pytest.raises(PriceFetchError):
        await serve(tmp_path, routes).cached_get("https://one.test/a")


@pytest.mark.parametrize("status", [401, 403])
async def test_bot_wall_responses_are_returned_but_not_cached(tmp_path, status):
    http = serve(tmp_path, {"https://one.test/a": (status, b"blocked", {})})
    response = await http.cached_get("https://one.test/a")
    assert response.status == status and response.cache_ref == ""
    assert http.read("https://one.test/a") is None


def age(http, url, delta):
    path = http.meta_path(__import__("hashlib").sha1(url.encode()).hexdigest())
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["retrieved_at"] = (datetime.now(UTC) - delta).isoformat()
    path.write_text(json.dumps(meta), encoding="utf-8")


async def test_stale_entry_is_refreshed_online_but_replayed_unchanged_offline(tmp_path, monkeypatch):
    url = "https://one.test/a"
    http = serve(tmp_path, {url: html("v1")})
    first = await http.cached_get(url)
    age(http, url, timedelta(days=30))
    stale = json.loads(http.meta_path(first.cache_ref).read_text(encoding="utf-8"))["retrieved_at"]
    monkeypatch.setenv("LIBCLAIM_OFFLINE", "1")
    replay = await serve(tmp_path, {}).cached_get(url)
    assert replay.meta["retrieved_at"] == stale and replay.text == "v1"
    monkeypatch.delenv("LIBCLAIM_OFFLINE")
    refreshed = await serve(tmp_path, {url: html("v2")}).cached_get(url)
    assert refreshed.text == "v2" and refreshed.meta["retrieved_at"] != stale


async def test_not_found_pages_expire_sooner_than_prices(tmp_path):
    url = "https://one.test/missing"
    http = serve(tmp_path, {})
    await http.cached_get(url)
    age(http, url, timedelta(days=2))
    routes = {url: html("now here")}
    assert (await serve(tmp_path, routes).cached_get(url)).text == "now here"


async def test_retry_after_is_honoured_within_the_bound(tmp_path, monkeypatch):
    waits = []
    real = asyncio.sleep

    async def record(seconds):
        waits.append(seconds)
        await real(0)

    monkeypatch.setattr("libclaim.pricing.http.asyncio.sleep", record)
    calls = []

    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        calls.append(1)
        return httpx.Response(429, headers={"retry-after": "7"}) if len(calls) == 1 else httpx.Response(200, content=b"ok")

    http = PriceHttp(tmp_path / "cache", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), config=fast_config())
    assert (await http.cached_get("https://one.test/a")).text == "ok"
    assert waits == [7.0]


async def test_concurrent_identical_requests_and_robots_are_single_flight(tmp_path):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(404) if request.url.path == "/robots.txt" else httpx.Response(200, content=b"ok")

    http = PriceHttp(tmp_path / "cache", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), config=fast_config())
    await asyncio.gather(*[http.cached_get("https://one.test/a") for _ in range(5)], http.cached_get("https://one.test/b"))
    assert seen.count("https://one.test/a") == 1 and seen.count("https://one.test/robots.txt") == 1


SKU = ISBN


def wc(**fields):
    base = {
        "sku": SKU,
        "permalink": "https://pustakkosh.com/products/x/",
        "is_in_stock": True,
        "prices": {"price": "500", "currency_minor_unit": 0, "currency_code": "INR"},
    }
    return json.dumps([{**base, **fields}]).encode()


API = f"https://pustakkosh.com/wp-json/wc/store/v1/products?search={ISBN}"


@pytest.mark.parametrize("body", [b"<html>nope</html>", b'{"code":"rest_error"}'])
async def test_pustakkosh_garbage_body_is_unavailable_with_evidence(tmp_path, body):
    result = await Pustakkosh(serve(tmp_path, {API: (200, body, {})})).quote(ISBN, "new")
    assert isinstance(result, NotFound) and result.reason == "unavailable" and result.evidence.cache_ref


async def test_pustakkosh_missing_fields_do_not_raise(tmp_path):
    body = json.dumps([{"sku": SKU}, 5, {"sku": 7}]).encode()
    result = await Pustakkosh(serve(tmp_path, {API: (200, body, {})})).quote(ISBN, "new")
    assert isinstance(result, NotFound) and result.reason == "unavailable"


async def test_pustakkosh_prefers_in_stock_and_compares_isbn_case_insensitively(tmp_path):
    isbn10 = "014312755X"
    api = f"https://pustakkosh.com/wp-json/wc/store/v1/products?search={isbn10}"
    rows = [
        {"sku": isbn10.lower(), "permalink": "https://p/1", "is_in_stock": False, "prices": {"price": "100", "currency_minor_unit": 0, "currency_code": "INR"}},
        {"sku": isbn10, "permalink": "https://p/2", "is_in_stock": True, "prices": {"price": "300", "currency_minor_unit": 0, "currency_code": "INR"}},
    ]
    result = await Pustakkosh(serve(tmp_path, {api: (200, json.dumps(rows).encode(), {})})).quote(isbn10, "new")
    assert isinstance(result, PriceQuote) and result.url == "https://p/2"


async def test_pustakkosh_other_currency_is_rejected(tmp_path):
    body = wc(prices={"price": "5", "currency_minor_unit": 0, "currency_code": "USD"})
    result = await Pustakkosh(serve(tmp_path, {API: (200, body, {})})).quote(ISBN, "new")
    assert isinstance(result, NotFound) and result.reason == "currency_mismatch"


SHOP = "https://shop.test/collections/c/products.json"


def shopify(*products):
    rows = [{"title": f"Book case {i}", "handle": f"h{i}", "variants": variants} for i, variants in enumerate(products)]
    return (200, json.dumps({"products": rows}).encode(), {"content-type": "application/json"})


def book_sources():
    from libclaim.pricing.config import CategorySpec, ItemSources, ListingSpec

    spec = CategorySpec(must=["book case"], exclude=[], listings=[ListingSpec(kind="shopify", url=SHOP)])
    return {"IN": ItemSources(categories={"shelf": spec})}


async def test_listing_prices_far_below_the_median_are_dropped_and_unavailable_variants_ignored(tmp_path):
    from libclaim.pricing.sources.listings import Listings

    variants = [[{"price": str(p)}] for p in (5, 1000, 1100, 1200, 1300, 1400, 1500)]
    variants.append([{"price": "50", "available": False}, {"price": "1600", "available": True}])
    result = await Listings(serve(tmp_path, {SHOP: shopify(*variants)}), item_sources=book_sources()).range("shelf", "IN")
    prices = [row.amount for row in result.rows]
    assert Decimal("5") not in prices and Decimal("50") not in prices and Decimal("1600") in prices
