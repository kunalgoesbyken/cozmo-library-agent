import asyncio
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import httpx
import pytest

from libclaim.models import NotFound, PriceQuote
from libclaim.pricing.http import PriceFetchError, PriceHttp, RobotsDisallowed, UnsafeTarget, is_public, normalized_host
from libclaim.pricing.sources.bookchor import Bookchor
from libclaim.pricing.sources.pustakkosh import Pustakkosh
from tests.pricing.conftest import public_resolver, Site, fast_config, make_http

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
    return PriceHttp(tmp_path / "cache", resolver=public_resolver, client=client, config=config or fast_config())


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
    http.allow_hosts({"two.test"})
    with pytest.raises(RobotsDisallowed):
        await http.cached_get(start)


async def test_robots_block_is_cached_evidence_and_replays_without_network(tmp_path):
    start = "https://one.test/a"
    routes = {
        start: (302, b"", {"location": "https://one.test/private/x"}),
        "https://one.test/robots.txt": (200, b"User-agent: *\nDisallow: /private/\n", {}),
    }
    with pytest.raises(RobotsDisallowed):
        await serve(tmp_path, routes).cached_get(start)
    replay = serve(tmp_path, {})
    replay.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: pytest.fail(str(request.url))))
    with pytest.raises(RobotsDisallowed):
        await replay.cached_get(start)


async def test_cross_host_redirect_uses_the_target_hosts_robots_and_records_final_url(tmp_path):
    routes = {
        "https://one.test/a": (302, b"", {"location": "https://two.test/ok"}),
        "https://two.test/ok": html("fine"),
    }
    http = serve(tmp_path, routes)
    http.allow_hosts({"two.test"})
    response = await http.cached_get("https://one.test/a")
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

    http = PriceHttp(tmp_path / "cache", resolver=public_resolver, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), config=fast_config())
    assert (await http.cached_get("https://one.test/a")).text == "ok"
    assert waits == [7.0]


async def test_concurrent_identical_requests_and_robots_are_single_flight(tmp_path):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(404) if request.url.path == "/robots.txt" else httpx.Response(200, content=b"ok")

    http = PriceHttp(tmp_path / "cache", resolver=public_resolver, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), config=fast_config())
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


def resolving(tmp_path, routes, mapping):
    async def resolver(host):
        return mapping.get(host, ["93.184.216.34"])

    def handler(request):
        seen.append(str(request.url))
        status, body, headers = routes.get(str(request.url), (404, b"", {}))
        return httpx.Response(status, content=body, headers=headers)

    seen = []
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return PriceHttp(tmp_path / "cache", client=client, config=fast_config(), resolver=resolver), seen


@pytest.mark.parametrize(
    "target",
    [
        "http://169.254.169.254/latest/meta-data/",
        "https://169.254.169.254/",
        "https://localhost/",
        "https://[::1]/",
        "https://[::ffff:127.0.0.1]/",
        "https://[fe80::1]/",
        "https://[fd00::1]/",
        "https://10.0.0.5/",
        "https://192.168.1.1/",
        "https://0.0.0.0/",
        "https://[2002:7f00:1::1]/",
        "https://[2002:a00:1::]/",
        "https://[2001:0:4136:e378:8000:63bf:f5ff:fffe]/",
        "https://[64:ff9b::7f00:1]/",
        "https://[64:ff9b::a9fe:a9fe]/",
        "https://[::7f00:1]/",
        "https://[::a00:1]/",
        "https://[::ffff:a9fe:a9fe]/",
    ],
)
async def test_redirect_hop_to_a_private_or_unsafe_address_is_refused(tmp_path, target):
    http, seen = resolving(tmp_path, {"https://one.test/a": (302, b"", {"location": target})}, {"localhost": ["127.0.0.1"]})
    http.allow_hosts({normalized_host(target)})
    with pytest.raises(UnsafeTarget):
        await http.cached_get("https://one.test/a")
    assert seen == ["https://one.test/robots.txt", "https://one.test/a"]


async def test_redirect_to_a_host_outside_the_allowlist_is_refused(tmp_path):
    http, seen = resolving(tmp_path, {"https://one.test/a": (302, b"", {"location": "https://evil.example/x"})}, {})
    with pytest.raises(UnsafeTarget):
        await http.cached_get("https://one.test/a")
    assert not any("evil.example" in url for url in seen)


async def test_hostname_resolving_to_loopback_is_refused_even_if_allowlisted(tmp_path):
    http, seen = resolving(tmp_path, {"https://one.test/a": (302, b"", {"location": "https://rebind.test/x"})}, {"rebind.test": ["127.0.0.1"]})
    http.allow_hosts({"rebind.test"})
    with pytest.raises(UnsafeTarget):
        await http.cached_get("https://one.test/a")
    assert not any("rebind.test" in url for url in seen)


async def test_plain_http_is_refused(tmp_path):
    http, seen = resolving(tmp_path, {}, {})
    with pytest.raises(UnsafeTarget):
        await http.cached_get("http://one.test/a")
    assert seen == []


async def test_cache_hits_skip_dns(tmp_path, monkeypatch):
    http, _ = resolving(tmp_path, {"https://one.test/a": html("x")}, {})
    await http.cached_get("https://one.test/a")

    async def broken(host):
        raise AssertionError("dns used")

    http.resolver = broken
    assert (await http.cached_get("https://one.test/a")).text == "x"


URBAN = "https://www.urbanladder.com/collection/bookshelves"
GOOD = "https://www.urbanladder.com/product/good-bookshelf-1"


def itemlist(*urls):
    entries = [{"@type": "ListItem", "name": f"Solid bookshelf {i}", "url": u} for i, u in enumerate(urls)]
    return html(page({"@type": "ItemList", "itemListElement": entries}))


def good_product():
    return html(page({"@type": "Product", "name": "Solid bookshelf 9", "offers": {"@type": "Offer", "price": "1000", "priceCurrency": "INR"}}))


async def test_itemlist_entries_off_origin_are_skipped_and_same_origin_relative_works(tmp_path):
    from libclaim.pricing.config import CategorySpec, ItemSources, ListingSpec
    from libclaim.pricing.sources.listings import Listings

    hostile = [
        "http://169.254.169.254/",
        "http://localhost:8000/product/x",
        "https://evil.example/product/x",
        "https://rebind.example/product/x",
        "http://www.urbanladder.com/product/downgraded",
    ]
    routes = {
        URBAN: itemlist(*hostile, "/product/relative-bookshelf-2", GOOD),
        GOOD: good_product(),
        "https://www.urbanladder.com/product/relative-bookshelf-2": good_product(),
    }
    http, seen = resolving(tmp_path, routes, {"rebind.example": ["127.0.0.1"]})
    spec = CategorySpec(must=["bookshelf"], exclude=[], listings=[ListingSpec(kind="itemlist", url=URBAN)])
    result = await Listings(http, item_sources={"IN": ItemSources(categories={"shelf": spec})}).range("shelf", "IN")
    assert sorted(row.url for row in result.rows) == sorted([GOOD, "https://www.urbanladder.com/product/relative-bookshelf-2"])
    assert not any(bad in url for url in seen for bad in ("169.254", "localhost", "evil.example", "rebind.example", "downgraded"))


@pytest.mark.parametrize("address", ["93.184.216.34", "2606:4700:4700::1111"])
def test_public_addresses_stay_allowed_including_embedded_public_ipv4(address):
    assert is_public(address)


def test_price_client_ignores_proxy_environment_variables(tmp_path):
    assert PriceHttp(tmp_path / "cache").client.trust_env is False


@pytest.mark.parametrize(
    "address",
    [
        "64:ff9b::a00:1", "64:ff9b::808:808", "64:ff9b::7f00:1",
        "64:ff9b:1::a00:1", "64:ff9b:1:0:0:0:808:808", "64:ff9b:1:ffff::7f00:1",
        "::ffff:10.0.0.1", "::ffff:8.8.8.8", "::ffff:127.0.0.1",
        "::10.0.0.1", "::8.8.8.8", "::127.0.0.1", "::a9fe:a9fe",
    ],
)
def test_nat64_mapped_and_compatible_ipv6_ranges_are_never_public(address):
    assert not is_public(address)
