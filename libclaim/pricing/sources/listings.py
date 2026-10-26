import logging
import math
from decimal import Decimal, InvalidOperation
from statistics import median
from urllib.parse import urljoin, urlsplit

from libclaim.models import ItemRange, ListingRow, NotEnough
from libclaim.pricing.config import ItemSources, ListingSpec, PricingConfig, CategorySpec, load_item_sources, load_pricing_config
from libclaim.pricing.http import CachedResponse, PriceFetchError, PriceHttp, normalized_host
from libclaim.pricing.markup import has_type, walk
from libclaim.pricing.sources.common import OK
from libclaim.settings import load_locale

logger = logging.getLogger(__name__)

HTTPS = "https"



def host_of(url: str) -> str:
    return urlsplit(url).netloc


def decimal_or_none(value: object) -> Decimal | None:
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


class Listings:
    def __init__(
        self,
        http: PriceHttp,
        config: PricingConfig | None = None,
        item_sources: dict[str, ItemSources] | None = None,
    ):
        self.http = http
        self.config = config or load_pricing_config()
        self.item_sources = item_sources or {}

    def sources_for(self, locale: str) -> ItemSources:
        return self.item_sources.get(locale) or load_item_sources(locale)

    async def get(self, url: str) -> CachedResponse | None:
        try:
            response = await self.http.cached_get(url)
        except PriceFetchError as error:
            logger.warning("listing fetch failed: %s", error)
            return None
        return response if response.status == OK else None

    def ikea_rows(self, response: CachedResponse) -> list[ListingRow]:
        cfg = self.config.listings.ikea_cards
        rows, tags = [], response.page.tags
        starts = [i for i, (tag, attrs) in enumerate(tags) if tag == "div" and cfg.card_class in attrs.get("class", "").split()]
        for start, end in zip(starts, starts[1:] + [len(tags)]):
            card = tags[start][1]
            inner = tags[start + 1 : end]
            label = next((a[cfg.label_attr] for _, a in inner if cfg.label_attr in a), "")
            link = next((urljoin(response.final_url, a["href"]) for t, a in inner if t == "a" and a.get("href")), "")
            amount = decimal_or_none(card.get(cfg.price_attr))
            if amount is None or not link:
                continue
            description = label.split(cfg.label_separator, 1)[-1]
            rows.append(
                ListingRow(
                    source=host_of(link),
                    name=f"{card.get(cfg.name_attr, '')} {description}".strip(),
                    amount=amount,
                    currency=card.get(cfg.currency_attr, ""),
                    url=link,
                    cache_ref=response.cache_ref,
                )
            )
        return rows

    def shopify_rows(self, response: CachedResponse, currency: str) -> list[ListingRow]:
        path = self.config.listings.shopify_product_path
        rows = []
        for product in response.json().get("products", []):
            prices = [decimal_or_none(v.get("price")) for v in product.get("variants", []) if v.get("available", True)]
            prices = [price for price in prices if price is not None]
            amount = min(prices) if prices else None
            if amount is not None:
                url = urljoin(response.final_url, path + product["handle"])
                rows.append(
                    ListingRow(
                        source=host_of(url), name=product["title"], amount=amount, currency=currency, url=url, cache_ref=response.cache_ref
                    )
                )
        return rows

    def itemlist_entries(self, response: CachedResponse, allowed: set[str]) -> list[tuple[str, str]]:
        marker = self.config.listings.product_url_marker
        entries = {}
        for node in walk(response.jsonld):
            if has_type(node, "ListItem"):
                url = node.get("url") or node.get("item")
                url = url.get("@id") or url.get("url") if isinstance(url, dict) else url
                if not isinstance(url, str):
                    continue
                url = urljoin(response.final_url, url)
                if marker not in url:
                    continue
                if urlsplit(url).scheme != HTTPS or normalized_host(url) not in allowed:
                    logger.warning("skipped off-origin listing entry: %s", normalized_host(url))
                    continue
                entries.setdefault(url, str(node.get("name", "")))
        return list(entries.items())

    def product_offer(self, response: CachedResponse) -> tuple[str, Decimal, str] | None:
        for product in (node for node in walk(response.jsonld) if has_type(node, "Product")):
            for node in walk(product.get("offers")):
                if has_type(node, "Offer"):
                    keys = ("price",)
                elif has_type(node, "AggregateOffer"):
                    keys = ("price", "lowPrice")
                else:
                    continue
                for key in keys:
                    amount = decimal_or_none(node[key]) if node.get(key) not in (None, "") else None
                    if amount is not None:
                        return str(product.get("name", "")), amount, str(node.get("priceCurrency", ""))
        return None

    async def itemlist_rows(self, response: CachedResponse, category: CategorySpec, allowed: set[str]) -> list[ListingRow]:
        rows = []
        for url, listed_name in self.itemlist_entries(response, allowed)[: self.config.listings.max_product_pages]:
            if listed_name and not matches(listed_name, category):
                continue
            product = await self.get(url)
            offer = self.product_offer(product) if product else None
            if offer and matches(listed_name or offer[0], category):
                rows.append(
                    ListingRow(
                        source=host_of(url),
                        name=listed_name or offer[0],
                        amount=offer[1],
                        currency=offer[2],
                        url=url,
                        cache_ref=product.cache_ref,
                    )
                )
        return rows

    async def rows_for(self, spec: ListingSpec, category: CategorySpec, currency: str) -> list[ListingRow]:
        self.http.allow_hosts({normalized_host(spec.url), *spec.allowed_hosts})
        response = await self.get(spec.url)
        if response is None:
            return []
        if spec.kind == "ikea_cards":
            return self.ikea_rows(response)
        if spec.kind == "shopify":
            return self.shopify_rows(response, currency)
        return await self.itemlist_rows(response, category, {normalized_host(spec.url), *spec.allowed_hosts})

    async def range(self, category: str, locale: str) -> ItemRange | NotEnough:
        cfg = self.config.listings
        spec = self.sources_for(locale).categories.get(category)
        currency = load_locale(locale).currency
        found: dict[str, ListingRow] = {}
        for listing in spec.listings if spec else []:
            for row in await self.rows_for(listing, spec, currency):
                if row.currency == currency and matches(row.name, spec):
                    found.setdefault(row.url, row)
        rows = sorted(found.values(), key=lambda row: (row.amount, row.url))
        if rows:
            floor = median(row.amount for row in rows) * Decimal(str(cfg.min_price_ratio_of_median))
            rows = [row for row in rows if row.amount >= floor]
        if len(rows) < cfg.min_matches:
            return NotEnough(category=category, locale=locale, needed=cfg.min_matches, rows=rows)
        return ItemRange(
            category=category,
            locale=locale,
            currency=currency,
            low=nearest_rank(rows, cfg.low_percentile),
            high=nearest_rank(rows, cfg.high_percentile),
            n=len(rows),
            rows=rows,
        )


def matches(name: str, category: CategorySpec) -> bool:
    lowered = name.lower()
    return any(term in lowered for term in category.must) and not any(term in lowered for term in category.exclude)


def nearest_rank(rows: list[ListingRow], percentile: float) -> ListingRow:
    return rows[max(math.ceil(percentile * len(rows)), 1) - 1]
