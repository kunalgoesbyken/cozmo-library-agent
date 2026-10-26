import json
from decimal import Decimal

from libclaim.packet.models import Book, BookReplacementCost, ClaimPacket, Item
from libclaim.pricing.books import Market
from libclaim.pricing.items import price_item, second_country
from libclaim.pricing.sources.fx import Fx
from libclaim.pricing.sources.listings import Listings
from libclaim.settings import load_locale
from tests.pricing.conftest import Site, make_http
from tests.pricing.factories import FakeSource, not_found, quote
from tests.pricing.test_listings import NILKAMAL, sources

IN = load_locale("IN")
GB = load_locale("GB")
EMPTY = FakeSource()
INR_GBP = "https://api.frankfurter.dev/v2/rates?base=INR&quotes=GBP&providers=ECB&date=2026-10-23"
FX_ROW = [{"base": "INR", "quote": "GBP", "rate": 0.01, "date": "2026-10-23"}]


def products(prices, handles=None):
    handles = handles or [f"shelf-{i}" for i in range(len(prices))]
    return {
        "products": [
            {"title": f"Book Case {h}", "handle": h, "variants": [{"price": str(p), "available": True}]}
            for h, p in zip(handles, prices)
        ]
    }


def site_with(listing=None):
    extra = {INR_GBP: {"status": 200, "type": "application/json", "body": json.dumps(FX_ROW)}}
    if listing is not None:
        extra[NILKAMAL] = {"status": 200, "type": "application/json", "body": json.dumps(listing)}
    return Site(extra)


def listings_for(http):
    return Listings(http, item_sources=sources(("shopify", NILKAMAL), must=("book case",)))


def shelf(**changes):
    return Item(**{"id": "i1", "category": "bookshelf", "status": "range", **changes})


async def test_seven_rows_give_nearest_rank_p25_p75_with_urls(tmp_path, online):
    site = site_with(products([700, 100, 300, 500, 200, 600, 400]))
    result = await price_item(shelf(), IN, listings=listings_for(make_http(tmp_path, site)))
    assert result.status == "range" and result.range.n == 7
    assert (result.replacement_cost.low, result.replacement_cost.high) == (Decimal(200), Decimal(600))
    assert result.range.low.url == "https://www.nilkamalfurniture.com/products/shelf-4"
    assert result.range.high.url == "https://www.nilkamalfurniture.com/products/shelf-5"
    assert result.replacement_cost.url == result.range.low.url
    assert result.replacement_cost.source == "www.nilkamalfurniture.com"
    assert len(result.evidence) == 7 and all(row.url for row in result.evidence)


async def test_two_rows_need_appraisal_and_keep_the_rows(tmp_path, online):
    site = site_with(products([300, 500]))
    result = await price_item(shelf(), IN, listings=listings_for(make_http(tmp_path, site)))
    assert result.status == "needs_appraisal" and result.range is None
    assert result.replacement_cost.low is None and result.replacement_cost.high is None
    assert sorted(row.amount for row in result.evidence) == [300, 500] and all(row.url for row in result.evidence)
    assert result.review == ["too_few_listings"]


async def test_art_flagged_for_appraisal_is_not_priced_or_fetched(tmp_path, online):
    site = site_with(products([100, 200, 300, 400, 500, 600, 700]))
    art = shelf(category="framed wall art print", status="needs_appraisal")
    result = await price_item(art, IN, listings=listings_for(make_http(tmp_path, site)))
    assert result.status == "needs_appraisal" and result.evidence == [] and result.review == ["flagged_for_appraisal"]
    assert site.requests == []


async def test_same_listing_url_counts_once(tmp_path, online):
    handles = ["a", "b", "c", "d", "a", "b"]
    site = site_with(products([100, 200, 300, 400, 100, 200], handles))
    result = await price_item(shelf(), IN, listings=listings_for(make_http(tmp_path, site)))
    assert result.status == "needs_appraisal"
    assert len(result.evidence) == 4 and len({row.url for row in result.evidence}) == 4


def book(index, amount, isbn=None, **changes):
    fields = {
        "id": f"b{index}",
        "status": "identified",
        "isbn": isbn if isbn is not None else f"97800000000{index:02d}",
        "replacement_cost": BookReplacementCost(amount=Decimal(amount) if amount is not None else None),
    }
    return Book(**{**fields, **changes})


def packet_of(*books):
    return ClaimPacket(books=list(books))


class ByIsbn:
    def __init__(self, prices):
        self.prices = prices

    async def quote(self, isbn, condition):
        if isbn in self.prices and condition == "new":
            return quote(self.prices[isbn], currency="GBP", source="Penguin UK")
        return not_found()


class Chor:
    async def quote(self, isbn, condition):
        return quote(1000, source="Bookchor") if condition == "new" else not_found()


def markets(penguin):
    return (
        Market(GB, {"penguin_uk": penguin}),
        Market(IN, {"bookchor": Chor(), "pustakkosh": EMPTY, "bookswagon": EMPTY}),
    )


async def test_local_gb_listing_and_converted_are_labelled(tmp_path, online):
    local_isbn, other_isbn = "9780000000001", "9780000000002"
    gb, india = markets(ByIsbn({local_isbn: "12.99"}))
    packet = packet_of(book(1, 900, local_isbn), book(2, 800, other_isbn))
    fx = Fx(make_http(tmp_path, site_with()))
    result = await second_country(packet, "GB", 10, market=gb, other=india, fx=fx)
    assert packet.second_country is result and result.country == "GB" and result.currency == "GBP"
    local, converted = result.books
    assert local.source_type == "local_listing" and local.fx is None
    assert local.replacement_cost.amount == Decimal("12.99") and not local.replacement_cost.converted
    assert local.original_amount == Decimal(900) and local.original_currency == "INR"
    assert converted.source_type == "converted" and converted.replacement_cost.converted
    assert converted.replacement_cost.amount == Decimal("10.00")
    assert converted.original_amount == Decimal(1000) and converted.original_currency == "INR"
    assert converted.fx.rate == Decimal("0.01") and converted.fx.base == "INR" and converted.fx.quote == "GBP"
    assert converted.fx.cache_ref in converted.evidence


async def test_sampling_is_highest_value_first_then_isbn_and_stable(tmp_path, online):
    gb, india = markets(ByIsbn({}))
    books = [
        book(1, 100),
        book(2, 500, "9780000000009"),
        book(3, 500, "9780000000003"),
        book(4, 300),
        book(5, None),
        book(6, 900, status="needs_appraisal"),
        book(7, 700, "", status="identified"),
    ]
    fx = Fx(make_http(tmp_path, site_with()))
    first = await second_country(packet_of(*books), "GB", 3, market=gb, other=india, fx=fx)
    again = await second_country(packet_of(*reversed(books)), "GB", 3, market=gb, other=india, fx=fx)
    assert [b.isbn for b in first.books] == ["9780000000003", "9780000000009", "9780000000004"]
    assert [b.book_id for b in first.books] == ["b3", "b2", "b4"]
    assert first == again


async def test_limit_n_is_respected(tmp_path, online):
    gb, india = markets(ByIsbn({}))
    packet = packet_of(*[book(i, 100 + i) for i in range(1, 6)])
    fx = Fx(make_http(tmp_path, site_with()))
    result = await second_country(packet, "GB", 2, market=gb, other=india, fx=fx)
    assert [b.book_id for b in result.books] == ["b5", "b4"]
