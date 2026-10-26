from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, Field, PlainSerializer

Money = Annotated[Decimal, PlainSerializer(float, return_type=float, when_used="json")]


class SweepInfo(BaseModel):
    id: str = ""
    captured_at: str = ""
    device: str = ""
    duration_s: float | None = None
    country: str = ""
    currency: str = ""


class Opening(BaseModel):
    kind: str = ""
    width_m: float | None = None
    height_m: float | None = None
    frame_ref: str = ""


class Room(BaseModel):
    length_m: float | None = None
    width_m: float | None = None
    height_m: float | None = None
    floor_area_m2: float | None = None
    wall_area_m2: float | None = None
    wall_area_net_m2: float | None = None
    shelved_wall_area_m2: float | None = None
    openings: list[Opening] = Field(default_factory=list)
    shape: str = ""
    polygon_m: list[list[float]] = Field(default_factory=list)
    frame_refs: list[str] = Field(default_factory=list)
    scale_method: str = ""
    confidence: float | None = None


class BookReplacementCost(BaseModel):
    amount: Money | None = None
    source: str = ""
    url: str = ""
    retrieved_at: str = ""
    converted: bool = False


class UsedValue(BaseModel):
    amount: Money | None = None
    source: str = ""
    url: str = ""
    retrieved_at: str = ""
    condition_assumed: str = ""
    method: str = ""
    source_type: Literal["", "observed", "derived"] = ""


class Book(BaseModel):
    id: str
    shelf: str = ""
    position: int | None = None
    frame_ref: str = ""
    status: Literal["identified", "unidentified", "needs_appraisal"]
    title: str = ""
    author: str = ""
    edition: str = ""
    isbn: str = ""
    spine_height_cm: float | None = None
    spine_thickness_cm: float | None = None
    scale_method: str = ""
    id_confidence: float | None = None
    replacement_cost: BookReplacementCost = Field(default_factory=BookReplacementCost)
    used_value: UsedValue = Field(default_factory=UsedValue)
    evidence: list[str] = Field(default_factory=list)


class Dimensions(BaseModel):
    w: float | None = None
    h: float | None = None
    d: float | None = None


class ItemReplacementCost(BaseModel):
    low: Money | None = None
    high: Money | None = None
    source: str = ""
    url: str = ""
    retrieved_at: str = ""


class Item(BaseModel):
    id: str
    category: str = ""
    description: str = ""
    brand_model: str = ""
    frame_ref: str = ""
    dimensions_cm: Dimensions = Field(default_factory=Dimensions)
    dimensions_method: str = ""
    status: Literal["priced", "range", "needs_appraisal"]
    replacement_cost: ItemReplacementCost = Field(default_factory=ItemReplacementCost)
    confidence: float | None = None
    evidence: list[str] = Field(default_factory=list)


class Totals(BaseModel):
    book_count: int = 0
    books_identified: int = 0
    books_unidentified: int = 0
    books_needs_appraisal: int = 0
    shelf_run_m: float = 0
    books_replacement_cost: Money = Decimal(0)
    books_used_value: Money = Decimal(0)
    books_used_value_observed: Money = Decimal(0)
    books_used_value_derived: Money = Decimal(0)
    items_replacement_cost_low: Money = Decimal(0)
    items_replacement_cost_high: Money = Decimal(0)
    excluded_from_totals: int = 0


class ReviewEntry(BaseModel):
    id: str
    ref_id: str
    severity: Literal["high", "medium", "low"]
    reason: str
    line_ref: str
    frame_ref: str = ""


class Costs(BaseModel):
    usd_at_paid_rates: float = 0
    tokens: int = 0
    by_stage_usd: dict[str, float] = Field(default_factory=dict)


class ClaimPacket(BaseModel):
    sweep: SweepInfo = Field(default_factory=SweepInfo)
    second_country: str = ""
    room: Room = Field(default_factory=Room)
    books: list[Book] = Field(default_factory=list)
    items: list[Item] = Field(default_factory=list)
    totals: Totals = Field(default_factory=Totals)
    review_queue: list[ReviewEntry] = Field(default_factory=list)
    costs: Costs = Field(default_factory=Costs)
    latency_s: dict[str, float] = Field(default_factory=dict)
