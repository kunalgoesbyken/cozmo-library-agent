from decimal import Decimal, InvalidOperation

from libclaim.models import NotFound, PriceQuote
from libclaim.pricing.config import PricingConfig, load_pricing_config
from libclaim.pricing.http import PriceFetchError, PriceHttp, RobotsDisallowed
from libclaim.pricing.sources.common import OK, not_found

NAME = "Penguin UK"
NEW = "new"


class PenguinUk:
    name = NAME

    def __init__(self, http: PriceHttp, config: PricingConfig | None = None):
        self.http = http
        self.config = config or load_pricing_config()

    def parse_rrp(self, rrp: str) -> tuple[Decimal, str] | None:
        symbols = self.config.sources.penguin_uk.currency_symbols
        currency = symbols.get(rrp[:1])
        if currency is None:
            return None
        try:
            return Decimal(rrp[1:].replace(",", "")), currency
        except InvalidOperation:
            return None

    async def quote(self, isbn: str, condition: str) -> PriceQuote | NotFound:
        if condition != NEW:
            return not_found(NAME, isbn, "unsupported_condition")
        try:
            response = await self.http.cached_get(self.config.sources.penguin_uk.url.format(isbn=isbn))
        except RobotsDisallowed:
            return not_found(NAME, isbn, "blocked")
        except PriceFetchError:
            return not_found(NAME, isbn, "unavailable")
        props = response.next_data().get("props", {}).get("pageProps", {}) if response.status == OK else {}
        found = str(props.get("isbn", ""))
        parsed = self.parse_rrp(str(props.get("rrp", "")))
        if not found or parsed is None:
            return not_found(NAME, isbn, "not_found", response)
        if found != isbn:
            return not_found(NAME, isbn, "equivalent_edition", response, found)
        return PriceQuote(
            amount=parsed[0],
            currency=parsed[1],
            source=NAME,
            url=response.final_url,
            retrieved_at=response.retrieved_at,
            kind="listing",
            condition=NEW,
            isbn_on_page=found,
            availability="",
            cache_ref=response.cache_ref,
        )
