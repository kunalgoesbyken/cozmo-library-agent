from decimal import Decimal, InvalidOperation

from libclaim.models import NotFound, PriceQuote
from libclaim.pricing.config import PricingConfig, load_pricing_config
from libclaim.pricing.http import CachedResponse, PriceFetchError, PriceHttp, RobotsDisallowed
from libclaim.pricing.sources.common import OK, not_found

NAME = "Pustakkosh"
NEW = "new"
USED = "used"
MALFORMED = (KeyError, TypeError, ValueError, InvalidOperation, AttributeError)


class Pustakkosh:
    name = NAME

    def __init__(self, http: PriceHttp, config: PricingConfig | None = None):
        self.http = http
        self.config = config or load_pricing_config()

    def condition_of(self, sku: str, isbn: str) -> str | None:
        suffix = sku.strip().lower().removeprefix(isbn.lower())
        if not suffix:
            return NEW
        return USED if suffix == self.config.sources.pustakkosh.used_sku_suffix.lower() else None

    def quote_of(self, product: dict, isbn: str, condition: str, response: CachedResponse) -> PriceQuote:
        cfg = self.config.sources.pustakkosh
        prices = product["prices"]
        return PriceQuote(
            amount=Decimal(prices["price"]) / 10 ** int(prices["currency_minor_unit"]),
            currency=prices["currency_code"],
            source=NAME,
            url=product["permalink"],
            retrieved_at=response.retrieved_at,
            kind="listing",
            condition=condition,
            isbn_on_page=isbn,
            availability=cfg.in_stock if product.get("is_in_stock") else cfg.out_of_stock,
            cache_ref=response.cache_ref,
        )

    async def quote(self, isbn: str, condition: str) -> PriceQuote | NotFound:
        cfg = self.config.sources.pustakkosh
        try:
            response = await self.http.cached_get(cfg.url.format(isbn=isbn))
        except RobotsDisallowed:
            return not_found(NAME, isbn, "blocked")
        except PriceFetchError:
            return not_found(NAME, isbn, "unavailable")
        if response.status != OK:
            return not_found(NAME, isbn, "not_found", response)
        try:
            products = response.json()
        except ValueError:
            return not_found(NAME, isbn, "unavailable", response)
        if not isinstance(products, list):
            return not_found(NAME, isbn, "unavailable", response)
        matches = [
            product
            for product in products
            if isinstance(product, dict)
            and isinstance(product.get("sku"), str)
            and self.condition_of(product["sku"], isbn) == condition
            and product["sku"].strip().lower().startswith(isbn.lower())
        ]
        if not matches:
            return not_found(NAME, isbn, "not_found", response)
        quotes = []
        for product in matches:
            try:
                quotes.append(self.quote_of(product, isbn, condition, response))
            except MALFORMED:
                continue
        if not quotes:
            return not_found(NAME, isbn, "unavailable", response, isbn)
        quotes.sort(key=lambda quote: (quote.availability != cfg.in_stock, quote.amount))
        if quotes[0].currency != cfg.currency:
            return not_found(NAME, isbn, "currency_mismatch", response, isbn)
        return quotes[0]
