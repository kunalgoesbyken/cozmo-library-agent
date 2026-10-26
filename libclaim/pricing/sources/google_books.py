from decimal import Decimal, InvalidOperation

import httpx

from libclaim.catalog.googlebooks import ISBN_13, GoogleBooks
from libclaim.models import Evidence, NotFound, PriceQuote
from libclaim.pricing.config import PricingConfig, load_pricing_config
from libclaim.settings import Locale

NEW = "new"
MALFORMED = (KeyError, TypeError, ValueError, InvalidOperation, AttributeError)


class GoogleBooksPrice:
    def __init__(self, books: GoogleBooks, locale: Locale, config: PricingConfig | None = None):
        self.books = books
        self.locale = locale
        self.config = (config or load_pricing_config()).sources.google_books
        self.name = self.config.name

    def missing(self, isbn: str, reason: str, evidence: Evidence | None = None) -> NotFound:
        return NotFound(source=self.name, isbn=isbn, reason=reason, evidence=evidence)

    def evidence(self, params: dict, url: str, isbn: str) -> Evidence:
        cache_ref, retrieved_at = self.books.retrieval(params)
        return Evidence(url=url, cache_ref=cache_ref, retrieved_at=retrieved_at, isbn_on_page=isbn)

    def volume_of(self, items: list[dict], isbn: str) -> dict | None:
        for item in items:
            identifiers = item.get("volumeInfo", {}).get("industryIdentifiers", [])
            if any(i.get("type") == ISBN_13 and i.get("identifier") == isbn for i in identifiers):
                return item
        return None

    async def quote(self, isbn: str, condition: str) -> PriceQuote | NotFound:
        cfg = self.config
        if not cfg.enabled:
            return self.missing(isbn, "disabled")
        if condition != NEW:
            return self.missing(isbn, "unsupported_condition")
        params = {
            "q": cfg.isbn_query.format(isbn=isbn),
            cfg.country_param: self.locale.country,
            "maxResults": cfg.max_results,
        }
        try:
            items = await self.books.volumes(params)
        except httpx.HTTPError:
            return self.missing(isbn, "unavailable")
        volume = self.volume_of(items, isbn)
        if volume is None:
            return self.missing(isbn, "not_found")
        url = volume.get("selfLink", "")
        evidence = self.evidence(params, url, isbn)
        sale = volume.get("saleInfo") or {}
        if sale.get("saleability") != cfg.sale_status:
            return self.missing(isbn, "not_found", evidence)
        if sale.get("isEbook"):
            return self.missing(isbn, "unsupported_condition", evidence)
        try:
            price = sale.get("retailPrice") or sale["listPrice"]
            amount, currency = Decimal(str(price["amount"])), price["currencyCode"]
        except MALFORMED:
            return self.missing(isbn, "unavailable", evidence)
        return PriceQuote(
            amount=amount,
            currency=currency,
            source=self.name,
            url=sale.get("buyLink") or url,
            retrieved_at=evidence.retrieved_at,
            kind="listing",
            condition=NEW,
            isbn_on_page=isbn,
            availability=sale["saleability"],
            cache_ref=evidence.cache_ref,
        )
