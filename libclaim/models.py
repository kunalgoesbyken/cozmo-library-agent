from datetime import date, datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel


class Still(BaseModel):
    seq: int
    t_ms: int
    path: str
    w: int
    h: int
    blur: float
    glare: float
    motion_dps: float
    pose: list[float] | None = None
    intrinsics: dict[str, float] | None = None
    depth_path: str | None = None


class Evidence(BaseModel):
    url: str
    cache_ref: str
    retrieved_at: datetime
    isbn_on_page: str = ""


class PriceQuote(BaseModel):
    amount: Decimal
    currency: str
    source: str
    url: str
    retrieved_at: datetime
    kind: Literal["listing", "derived_from_new", "converted"]
    condition: str
    isbn_on_page: str = ""
    availability: str = ""
    cache_ref: str


class NotFound(BaseModel):
    source: str
    isbn: str
    reason: Literal[
        "not_found",
        "equivalent_edition",
        "unsupported_condition",
        "disabled",
        "blocked",
        "unavailable",
        "currency_mismatch",
    ]
    evidence: Evidence | None = None


class ListingRow(BaseModel):
    source: str
    name: str
    amount: Decimal
    currency: str
    url: str
    cache_ref: str
    retrieved_at: datetime


class ItemRange(BaseModel):
    category: str
    locale: str
    currency: str
    low: ListingRow
    high: ListingRow
    n: int
    rows: list[ListingRow]


class NotEnough(BaseModel):
    category: str
    locale: str
    needed: int
    rows: list[ListingRow]


class FxRecord(BaseModel):
    base: str
    quote: str
    rate: Decimal
    rate_date: date
    provider: str
    url: str
    retrieved_at: datetime
    cache_ref: str
