import json
from decimal import Decimal
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
    allowed_schemes: list[str]
    robots_open_statuses: list[int]
    max_redirects: int
    retry_after_max_seconds: float


class Repair(BaseModel):
    name: str
    pattern: str
    replacement: str


class UrlSource(BaseModel):
    url: str
    currency: str = ""


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
    min_price_ratio_of_median: float
    low_percentile: float
    high_percentile: float
    max_product_pages: int
    shopify_product_path: str
    ikea_cards: IkeaCards
    product_url_marker: str


class CacheConfig(BaseModel):
    price_ttl_days: float
    not_found_ttl_hours: float
    uncached_statuses: list[int]


class OffersConfig(BaseModel):
    in_stock_availability: str


class Flag(BaseModel):
    enabled: bool


class BooksConfig(BaseModel):
    factors_snapshot: str
    factor_version: str
    default_condition: str
    disagreement_ratio: Decimal
    used_new_ratio: Decimal
    rounding: Decimal
    equivalent_source: str
    basis_notes: dict[str, str]
    appraisal_triggers: list[str]


class CalibrationConfig(BaseModel):
    grade: str
    max_requests: int
    min_pairs: int
    sources: list[str]
    snapshot_dir: str
    ground_truth_csv: str
    isbn_column: str


class PricingConfig(BaseModel):
    condition_factors: dict[str, Decimal]
    books: BooksConfig
    calibration: CalibrationConfig
    bookswagon: Flag
    http: HttpConfig
    conditions: dict[str, str]
    cache: CacheConfig
    offers: OffersConfig
    jsonld_repairs: dict[str, list[Repair]]
    sources: Sources
    listings: ListingsConfig


class ListingSpec(BaseModel):
    kind: str
    url: str
    allowed_hosts: list[str] = []


class CategorySpec(BaseModel):
    must: list[str]
    exclude: list[str]
    listings: list[ListingSpec]


class ItemSources(BaseModel):
    categories: dict[str, CategorySpec]


@cache
def load_pricing_config() -> PricingConfig:
    raw = yaml.safe_load((CONFIG_DIR / "pricing.yaml").read_text(encoding="utf-8"))
    path = CONFIG_DIR / raw["calibration"]["snapshot_dir"] / f"{raw['books']['factors_snapshot']}.json"
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    raw["condition_factors"] = snapshot["condition_factors"]
    raw["books"]["factor_version"] = snapshot["version"]
    return PricingConfig.model_validate(raw)


@cache
def load_item_sources(country: str) -> ItemSources:
    path = CONFIG_DIR / "item_sources" / f"{country.upper()}.yaml"
    return ItemSources.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
