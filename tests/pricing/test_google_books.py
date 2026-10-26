import json
from datetime import UTC, datetime
from decimal import Decimal

import httpx
import pytest

from libclaim.catalog.googlebooks import GoogleBooks
from libclaim.catalog.http import CatalogHttp
from libclaim.models import NotFound, PriceQuote
from libclaim.pricing.books import Market, price_book
from libclaim.pricing.config import load_pricing_config
from libclaim.pricing.sources.fx import Fx
from libclaim.pricing.sources.google_books import GoogleBooksPrice
from libclaim.settings import load_locale
from tests.pricing.conftest import FIXTURES
from tests.pricing.factories import FakeSource, quote, reading, resolution

ISBN = "9780099590088"
KEY = "AIzaTestKeyNotReal123456"
IN = load_locale("IN")
VOLUMES = "https://www.googleapis.com/books/v1/volumes"


def enabled_config():
    config = load_pricing_config()
    on = config.sources.google_books.model_copy(update={"enabled": True})
    return config.model_copy(update={"sources": config.sources.model_copy(update={"google_books": on})})


def source_for(tmp_path, fixture, locale=IN, config=None, status=200):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(status, content=(FIXTURES / fixture).read_bytes())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), headers={"User-Agent": "test"})
    books = GoogleBooks(CatalogHttp(tmp_path / "cache", client=client), api_key=KEY)
    return GoogleBooksPrice(books, locale, config or enabled_config()), seen


@pytest.fixture(autouse=True)
def online(monkeypatch):
    monkeypatch.delenv("LIBCLAIM_OFFLINE")


async def test_retail_price_is_a_new_quote_with_buy_link_date_and_country(tmp_path):
    source, seen = source_for(tmp_path, "google_books_for_sale.json")
    result = await source.quote(ISBN, "new")
    assert isinstance(result, PriceQuote)
    assert (result.amount, result.currency, result.condition, result.kind) == (Decimal("349.5"), "INR", "new", "listing")
    assert result.source == "Google Books" and result.isbn_on_page == ISBN
    assert result.url.startswith("https://play.google.com/store/books/details?id=ABC123")
    assert result.retrieved_at.tzinfo is not None and abs(datetime.now(UTC) - result.retrieved_at).total_seconds() < 60
    assert (tmp_path / "cache" / f"{result.cache_ref}.json").is_file()
    request = seen[0]
    assert request.url.params["q"] == f"isbn:{ISBN}" and request.url.params["country"] == "IN"
    assert request.headers["x-goog-api-key"] == KEY and KEY not in str(request.url)


async def test_list_price_is_used_when_there_is_no_retail_price_and_currency_comes_from_the_reply(tmp_path):
    source, _ = source_for(tmp_path, "google_books_list_only_gbp.json")
    result = await source.quote(ISBN, "new")
    assert isinstance(result, PriceQuote)
    assert (result.amount, result.currency) == (Decimal("8.99"), "GBP")
    assert result.url == "https://www.googleapis.com/books/v1/volumes/ABC123"


@pytest.mark.parametrize(
    ("fixture", "reason"),
    [
        ("google_books_not_for_sale.json", "not_found"),
        ("google_books_no_sale_info.json", "not_found"),
        ("google_books_empty.json", "not_found"),
    ],
)
async def test_nothing_is_invented_without_a_for_sale_volume(tmp_path, fixture, reason):
    source, _ = source_for(tmp_path, fixture)
    result = await source.quote(ISBN, "new")
    assert isinstance(result, NotFound) and result.reason == reason and result.source == "Google Books"


async def test_an_ebook_price_is_never_a_replacement_cost(tmp_path):
    payload = json.loads((FIXTURES / "google_books_for_sale.json").read_text(encoding="utf-8"))
    for item in payload["items"]:
        item["saleInfo"]["isEbook"] = True
    (FIXTURES / "google_books_ebook_tmp.json").write_text(json.dumps(payload), encoding="utf-8")
    try:
        source, _ = source_for(tmp_path, "google_books_ebook_tmp.json")
        result = await source.quote(ISBN, "new")
    finally:
        (FIXTURES / "google_books_ebook_tmp.json").unlink()
    assert isinstance(result, NotFound) and result.reason == "unsupported_condition"


async def test_not_for_sale_keeps_the_evidence(tmp_path):
    source, _ = source_for(tmp_path, "google_books_not_for_sale.json")
    result = await source.quote(ISBN, "new")
    assert result.evidence.isbn_on_page == ISBN and result.evidence.cache_ref


async def test_a_volume_for_another_isbn_is_never_priced(tmp_path):
    source, _ = source_for(tmp_path, "google_books_for_sale.json")
    result = await source.quote("9781111111111", "new")
    assert isinstance(result, NotFound) and result.reason == "not_found"


async def test_used_condition_is_unsupported_and_makes_no_request(tmp_path):
    source, seen = source_for(tmp_path, "google_books_for_sale.json")
    result = await source.quote(ISBN, "used")
    assert isinstance(result, NotFound) and result.reason == "unsupported_condition"
    assert seen == []


async def test_disabled_flag_makes_no_request(tmp_path):
    config = load_pricing_config()
    off = config.sources.google_books.model_copy(update={"enabled": False})
    config = config.model_copy(update={"sources": config.sources.model_copy(update={"google_books": off})})
    source, seen = source_for(tmp_path, "google_books_for_sale.json", config=config)
    result = await source.quote(ISBN, "new")
    assert isinstance(result, NotFound) and result.reason == "disabled" and seen == []


async def test_http_failure_is_unavailable(tmp_path):
    source, _ = source_for(tmp_path, "google_books_for_sale.json", status=503)
    result = await source.quote(ISBN, "new")
    assert isinstance(result, NotFound) and result.reason == "unavailable"


async def test_offline_without_a_cached_reply_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("LIBCLAIM_OFFLINE", "1")
    source, seen = source_for(tmp_path, "google_books_for_sale.json")
    result = await source.quote(ISBN, "new")
    assert isinstance(result, NotFound) and result.reason == "unavailable" and seen == []


async def test_malformed_price_is_unavailable(tmp_path):
    body = json.loads((FIXTURES / "google_books_for_sale.json").read_text(encoding="utf-8"))
    del body["items"][1]["saleInfo"]["retailPrice"]["currencyCode"]
    (tmp_path / "bad.json").write_text(json.dumps(body), encoding="utf-8")
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, content=(tmp_path / "bad.json").read_bytes())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), headers={"User-Agent": "test"})
    source = GoogleBooksPrice(GoogleBooks(CatalogHttp(tmp_path / "cache", client=client), api_key=KEY), IN, enabled_config())
    result = await source.quote(ISBN, "new")
    assert isinstance(result, NotFound) and result.reason == "unavailable"


async def test_foreign_currency_google_price_goes_through_the_converted_path(http):
    gbp = quote("10.00", currency="GBP", source="Google Books")
    market = Market(IN, {"bookchor": FakeSource(), "google_books": FakeSource(new=gbp)})
    result = await price_book(resolution(isbn=ISBN), market, reading=reading(), statements=[], fx=Fx(http))
    assert result.replacement_cost.converted is True
    assert result.replacement_cost.source == "Google Books" and result.replacement_cost.amount == Decimal("1269.30")
    assert result.original.currency == "GBP" and result.fx.rate == Decimal("126.93")
    assert any(reason.startswith("converted: GBP to INR") for reason in result.review)


async def test_local_currency_google_price_is_not_converted():
    inr = quote("349.50", source="Google Books")
    market = Market(IN, {"google_books": FakeSource(new=inr)})
    result = await price_book(resolution(isbn=ISBN), market, reading=reading(), statements=[])
    assert result.replacement_cost.converted is False and result.replacement_cost.amount == Decimal("349.50")
    assert result.fx is None and result.original is None


def test_pipeline_registers_google_books_for_locales_that_list_it():
    from types import SimpleNamespace

    from libclaim.pipeline import SweepRun

    services = SimpleNamespace(http=object(), catalogs=SimpleNamespace(google_books=GoogleBooks(None, api_key=KEY)))
    market = SweepRun.market(SimpleNamespace(services=services), IN)
    assert isinstance(market.sources["google_books"], GoogleBooksPrice)
    assert market.sources["google_books"].locale is IN
