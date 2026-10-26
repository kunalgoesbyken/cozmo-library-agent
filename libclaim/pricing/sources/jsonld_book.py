from decimal import Decimal, InvalidOperation

from libclaim.models import NotFound, PriceQuote
from libclaim.pricing.config import PricingConfig, load_pricing_config
from libclaim.pricing.http import CachedResponse, PriceFetchError, PriceHttp, RobotsDisallowed
from libclaim.pricing.markup import has_type, walk
from libclaim.pricing.sources.common import OK, not_found, schema_name

IN_STOCK = "InStock"


class JsonLdBookSource:
    name: str

    def __init__(self, http: PriceHttp, config: PricingConfig | None = None):
        self.http = http
        self.config = config or load_pricing_config()

    @property
    def template(self) -> str:
        raise NotImplementedError

    @property
    def enabled(self) -> bool:
        return True

    def identifiers(self, response: CachedResponse, products: list[dict]) -> list[str]:
        raise NotImplementedError

    async def quote(self, isbn: str, condition: str) -> PriceQuote | NotFound:
        if not self.enabled:
            return not_found(self.name, isbn, "disabled")
        suffix = self.config.conditions.get(condition)
        if suffix is None:
            return not_found(self.name, isbn, "unsupported_condition")
        try:
            response = await self.http.cached_get(self.template.format(isbn=isbn))
        except RobotsDisallowed:
            return not_found(self.name, isbn, "blocked")
        except PriceFetchError:
            return not_found(self.name, isbn, "unavailable")
        products = [node for node in walk(response.jsonld) if has_type(node, "Product")]
        if response.status != OK or not products:
            return not_found(self.name, isbn, "not_found", response)
        found = self.identifiers(response, products)
        if isbn not in found:
            return not_found(self.name, isbn, "equivalent_edition" if found else "not_found", response, found[0] if found else "")
        offer = self.pick_offer(products, suffix)
        if offer is None:
            return not_found(self.name, isbn, "not_found", response, isbn)
        return PriceQuote(
            amount=offer[0],
            currency=offer[1],
            source=self.name,
            url=response.canonical,
            retrieved_at=response.retrieved_at,
            kind="listing",
            condition=condition,
            isbn_on_page=isbn,
            availability=offer[2],
            cache_ref=response.cache_ref,
        )

    def pick_offer(self, products: list[dict], suffix: str) -> tuple[Decimal, str, str] | None:
        offers = []
        for product in products:
            for node in walk(product.get("offers")):
                if has_type(node, "Offer") and schema_name(node.get("itemCondition")) == suffix:
                    try:
                        offers.append((Decimal(str(node["price"])), node["priceCurrency"], schema_name(node.get("availability"))))
                    except (KeyError, InvalidOperation):
                        continue
        offers.sort(key=lambda offer: offer[2] != IN_STOCK)
        return offers[0] if offers else None
