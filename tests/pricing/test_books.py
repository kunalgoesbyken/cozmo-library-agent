from decimal import Decimal

from libclaim.models import NotFound, PriceQuote
from libclaim.pricing.sources.bookchor import Bookchor
from libclaim.pricing.sources.bookswagon import Bookswagon
from libclaim.pricing.sources.penguin_uk import PenguinUk
from libclaim.pricing.sources.pustakkosh import Pustakkosh
from tests.pricing.conftest import Site, fast_config, make_http

EINTY = "9780143127550"
SAPIENS = "9780099590088"
BAD = "9789354892227"
BOOKCHOR = "https://www.bookchor.com/book/{}/x"


async def test_bookchor_new_and_used_offers_carry_isbn_and_evidence(http, tmp_path):
    source = Bookchor(http)
    new = await source.quote(EINTY, "new")
    used = await source.quote(EINTY, "used")
    assert isinstance(new, PriceQuote) and isinstance(used, PriceQuote)
    assert (new.amount, new.currency, new.condition) == (Decimal("1571"), "INR", "new")
    assert (used.amount, used.condition) == (Decimal("661"), "used")
    assert new.isbn_on_page == used.isbn_on_page == EINTY
    assert new.url == "https://www.bookchor.com/book/9780143127550/everything-i-never-told-you"
    assert new.kind == "listing" and new.availability == "InStock"
    assert (tmp_path / "cache" / f"{new.cache_ref}.body").is_file()
    assert (tmp_path / "cache" / f"{new.cache_ref}.json").read_text(encoding="utf-8").count("UsedCondition") >= 1


async def test_bookchor_without_used_stock_is_not_found_with_evidence(http):
    result = await Bookchor(http).quote(SAPIENS, "used")
    assert isinstance(result, NotFound) and result.reason == "not_found"
    assert result.evidence.isbn_on_page == SAPIENS and result.evidence.cache_ref


async def test_search_redirect_page_is_not_found(http):
    result = await Bookchor(http).quote(BAD, "new")
    assert isinstance(result, NotFound) and result.reason == "not_found"
    assert result.evidence.url.endswith("/search/?query=9789354892227")


async def test_page_for_a_different_isbn_is_equivalent_edition_not_a_price(tmp_path, online):
    other = "9780141439518"
    site = Site({BOOKCHOR.format(other): {"file": "bookchor_sapiens.html", "status": 200, "type": "text/html"}})
    result = await Bookchor(make_http(tmp_path, site)).quote(other, "new")
    assert isinstance(result, NotFound) and result.reason == "equivalent_edition"
    assert result.evidence.isbn_on_page == SAPIENS


async def test_bookswagon_is_disabled_by_default_and_makes_no_request(http, site):
    result = await Bookswagon(http).quote("9788129104595", "new")
    assert isinstance(result, NotFound) and result.reason == "disabled"
    assert site.requests == []


async def test_bookswagon_out_of_stock_offer_is_returned_with_availability(tmp_path, site, online):
    config = fast_config()
    config = config.model_copy(update={"bookswagon": config.bookswagon.model_copy(update={"enabled": True})})
    source = Bookswagon(make_http(tmp_path, site, config), config)
    quote = await source.quote("9788129104595", "new")
    assert isinstance(quote, PriceQuote)
    assert (quote.amount, quote.availability, quote.isbn_on_page) == (Decimal("87"), "OutOfStock", "9788129104595")
    meta = (tmp_path / "cache" / f"{quote.cache_ref}.json").read_text(encoding="utf-8")
    assert "unquoted_url_value" in meta


async def test_bookswagon_blank_page_is_not_found(tmp_path, site, online):
    config = fast_config()
    config = config.model_copy(update={"bookswagon": config.bookswagon.model_copy(update={"enabled": True})})
    result = await Bookswagon(make_http(tmp_path, site, config), config).quote(BAD, "new")
    assert isinstance(result, NotFound) and result.reason == "not_found"


async def test_pustakkosh_used_sku_and_new_sku_are_told_apart(http):
    source = Pustakkosh(http)
    used = await source.quote("9788173711466", "used")
    assert isinstance(used, PriceQuote) and (used.amount, used.condition) == (Decimal("251"), "used")
    assert used.url == "https://pustakkosh.com/products/wings-of-fire-an-autobiography-of-abdul-kalam/"
    assert isinstance(await source.quote("9788173711466", "new"), NotFound)
    new = await source.quote(SAPIENS, "new")
    assert isinstance(new, PriceQuote) and new.amount == Decimal("630")
    empty = await source.quote("9780141439518", "new")
    assert isinstance(empty, NotFound) and empty.reason == "not_found"


async def test_penguin_uk_reads_rrp_from_next_data(http):
    quote = await PenguinUk(http).quote(SAPIENS, "new")
    assert isinstance(quote, PriceQuote)
    assert (quote.amount, quote.currency, quote.isbn_on_page) == (Decimal("12.99"), "GBP", SAPIENS)
    assert quote.url.endswith(f"/{SAPIENS}")


async def test_penguin_uk_search_results_redirect_is_not_found(http):
    result = await PenguinUk(http).quote(EINTY, "new")
    assert isinstance(result, NotFound) and result.reason == "not_found"
    assert result.evidence.url.endswith("/search-results")


async def test_penguin_uk_has_no_used_price(http, site):
    result = await PenguinUk(http).quote(SAPIENS, "used")
    assert isinstance(result, NotFound) and result.reason == "unsupported_condition"
    assert site.requests == []
