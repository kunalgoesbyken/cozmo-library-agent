from decimal import Decimal

from libclaim.models import NotFound, PriceQuote
from libclaim.pricing.config import PricingConfig, load_pricing_config
from libclaim.pricing.http import PriceFetchError, PriceHttp, RobotsDisallowed
from libclaim.pricing.sources.common import OK, not_found

NAME = "Pustakkosh"
NEW = "new"
USED = "used"


class Pustakkosh:
    name = NAME

    def __init__(self, http: PriceHttp, config: PricingConfig | None = None):
        self.http = http
        self.config = config or load_pricing_config()

    def condition_of(self, sku: str, isbn: str) -> str | None:
        suffix = sku.strip().lower().removeprefix(isbn)
        if not suffix:
            return NEW
        return USED if suffix == self.config.sources.pustakkosh.used_sku_suffix.lower() else None

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
        matches = [
            product
            for product in response.json()
            if str(product.get("sku", "")).strip().lower().startswith(isbn)
            and self.condition_of(str(product["sku"]), isbn) == condition
        ]
        if not matches:
            return not_found(NAME, isbn, "not_found", response)
        product = matches[0]
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
