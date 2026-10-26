from functools import cache

import yaml
from pydantic import BaseModel

from libclaim.settings import CONFIG_DIR


class HttpConfig(BaseModel):
    timeout_seconds: float
    min_interval_seconds: float
    host_min_interval_seconds: dict[str, float]
    max_retries: int
    backoff_seconds: float
    retry_statuses: list[int]
    accept_language: str
    robots_agent: str
    robots_path: str


class Repair(BaseModel):
    name: str
    pattern: str
    replacement: str


class UrlSource(BaseModel):
    url: str


class BookswagonSource(UrlSource):
    isbn_label: str


class PustakkoshSource(UrlSource):
    used_sku_suffix: str
    in_stock: str
    out_of_stock: str


class PenguinSource(UrlSource):
    currency_symbols: dict[str, str]


class FxSource(UrlSource):
    provider: str
    provider_label: str


class Sources(BaseModel):
    bookchor: UrlSource
    bookswagon: BookswagonSource
    pustakkosh: PustakkoshSource
    penguin_uk: PenguinSource
    fx: FxSource


class IkeaCards(BaseModel):
    card_class: str
    price_attr: str
    currency_attr: str
    name_attr: str
    label_attr: str
    label_separator: str


class ListingsConfig(BaseModel):
    min_matches: int
    low_percentile: float
    high_percentile: float
    max_product_pages: int
    shopify_product_path: str
    ikea_cards: IkeaCards
    product_url_marker: str


class Flag(BaseModel):
    enabled: bool


class PricingConfig(BaseModel):
    bookswagon: Flag
    http: HttpConfig
    conditions: dict[str, str]
    jsonld_repairs: list[Repair]
    sources: Sources
    listings: ListingsConfig


class ListingSpec(BaseModel):
    kind: str
    url: str


class CategorySpec(BaseModel):
    must: list[str]
    exclude: list[str]
    listings: list[ListingSpec]


class ItemSources(BaseModel):
    categories: dict[str, CategorySpec]


@cache
def load_pricing_config() -> PricingConfig:
    return PricingConfig.model_validate(yaml.safe_load((CONFIG_DIR / "pricing.yaml").read_text(encoding="utf-8")))


@cache
def load_item_sources(country: str) -> ItemSources:
    path = CONFIG_DIR / "item_sources" / f"{country.upper()}.yaml"
    return ItemSources.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
