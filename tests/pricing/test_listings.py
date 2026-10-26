from decimal import Decimal

from libclaim.models import ItemRange, NotEnough
from libclaim.pricing.config import CategorySpec, ItemSources, ListingSpec
from libclaim.pricing.sources.listings import Listings, nearest_rank

IKEA = "https://www.ikea.com/in/en/cat/bookcases-10382/"
NILKAMAL = "https://www.nilkamalfurniture.com/collections/book-cases/products.json?limit=50"
URBAN = "https://www.urbanladder.com/collection/bookshelves"


def sources(*specs, must=("bookcase",), exclude=()):
    category = CategorySpec(
        must=list(must), exclude=list(exclude), listings=[ListingSpec(kind=k, url=u) for k, u in specs]
    )
    return {"IN": ItemSources(categories={"bookshelf": category})}


async def test_ikea_listing_gives_p25_p75_rows_that_each_carry_a_url(http):
    result = await Listings(http, item_sources=sources(("ikea_cards", IKEA))).range("bookshelf", "IN")
    assert isinstance(result, ItemRange)
    assert result.n >= 5 and result.currency == "INR"
    prices = [row.amount for row in result.rows]
    assert prices == sorted(prices)
    assert result.low == nearest_rank(result.rows, 0.25) and result.high == nearest_rank(result.rows, 0.75)
    for bound in (result.low, result.high):
        assert bound.url.startswith("https://www.ikea.com/in/en/p/") and bound.cache_ref
        assert bound in result.rows
    assert result.low.amount < result.high.amount


async def test_fewer_than_five_matches_is_not_enough_and_keeps_the_rows(http):
    result = await Listings(http, item_sources=sources(("ikea_cards", IKEA), must=("kallax", "brimnes"))).range("bookshelf", "IN")
    assert isinstance(result, NotEnough) and result.needed == 5
    assert 0 < len(result.rows) < 5 and all(row.url for row in result.rows)


async def test_unknown_category_is_not_enough_without_requests(http, site):
    result = await Listings(http, item_sources=sources(("ikea_cards", IKEA))).range("unicycle", "IN")
    assert isinstance(result, NotEnough) and result.rows == []
    assert site.requests == []


async def test_exclude_terms_remove_rows(http):
    everything = await Listings(http, item_sources=sources(("ikea_cards", IKEA))).range("bookshelf", "IN")
    trimmed = await Listings(http, item_sources=sources(("ikea_cards", IKEA), exclude=("billy",))).range("bookshelf", "IN")
    assert {row.url for row in trimmed.rows} < {row.url for row in everything.rows}
    assert not any("billy" in row.name.lower() for row in trimmed.rows)


async def test_shopify_products_json_rows_cite_product_pages(http):
    result = await Listings(http, item_sources=sources(("shopify", NILKAMAL), must=("book",))).range("bookshelf", "IN")
    assert isinstance(result, ItemRange)
    assert all(row.url.startswith("https://www.nilkamalfurniture.com/products/") for row in result.rows)
    assert all(isinstance(row.amount, Decimal) for row in result.rows)


async def test_itemlist_listing_fetches_product_pages_for_prices(http, site):
    result = await Listings(http, item_sources=sources(("itemlist", URBAN), must=("bookshelf",))).range("bookshelf", "IN")
    assert isinstance(result, ItemRange)
    assert all(row.url.startswith("https://www.urbanladder.com/product/") for row in result.rows)
    assert URBAN in site.requests and len(site.requests) > 3


async def test_sources_are_merged_and_bounds_are_real_rows(http):
    result = await Listings(
        http, item_sources=sources(("ikea_cards", IKEA), ("shopify", NILKAMAL), must=("bookcase", "book case"))
    ).range("bookshelf", "IN")
    assert isinstance(result, ItemRange)
    assert {row.source for row in result.rows} >= {"www.ikea.com", "www.nilkamalfurniture.com"}
    assert result.low in result.rows and result.high in result.rows
