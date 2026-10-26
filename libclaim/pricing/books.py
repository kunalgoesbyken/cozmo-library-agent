from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from libclaim.models import FxRecord, NotFound, PriceQuote
from libclaim.packet.models import BookReplacementCost, UsedValue
from libclaim.pricing.appraisal import NEW, appraisal_reasons
from libclaim.pricing.config import PricingConfig, load_pricing_config
from libclaim.pricing.http import PriceFetchError
from libclaim.pricing.sources.fx import Fx
from libclaim.settings import Locale

USED = "used"
CONVERTED = "converted"
DERIVED = "derived"
PSEUDO_SOURCES = (CONVERTED, DERIVED)
LISTING = "listing"
DERIVED_KIND = "derived_from_new"
EQUIVALENT_EDITION = "equivalent_edition"


class Source(Protocol):
    async def quote(self, isbn: str, condition: str) -> PriceQuote | NotFound: ...


@dataclass
class Market:
    locale: Locale
    sources: dict[str, Source]

    async def quotes(self, names: list[str], isbn: str, condition: str) -> list[PriceQuote | NotFound]:
        return [await self.sources[name].quote(isbn, condition) for name in names if name not in PSEUDO_SOURCES]


class BookPrices(BaseModel):
    status: Literal["identified", "unidentified", "needs_appraisal"]
    replacement_cost: BookReplacementCost = Field(default_factory=BookReplacementCost)
    used_value: UsedValue = Field(default_factory=UsedValue)
    original: PriceQuote | None = None
    fx: FxRecord | None = None
    review: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    excluded: bool = False


def derive_used(new_quote: PriceQuote, condition: str, cfg: PricingConfig) -> PriceQuote:
    factor = cfg.condition_factors[cfg.books.factor_version][condition]
    amount = (new_quote.amount * factor).quantize(cfg.books.rounding)
    return new_quote.model_copy(update={"amount": amount, "kind": DERIVED_KIND, "condition": condition})


def used_value_of(quote: PriceQuote, cfg: PricingConfig) -> UsedValue:
    derived = quote.kind == DERIVED_KIND
    return UsedValue(
        amount=quote.amount,
        source=quote.source,
        url=quote.url,
        retrieved_at=quote.retrieved_at.isoformat(),
        condition_assumed=quote.condition,
        method=f"condition_factor:{cfg.books.factor_version}:{quote.condition}" if derived else LISTING,
        source_type="derived" if derived else "observed",
    )


def replacement_cost_of(quote: PriceQuote) -> BookReplacementCost:
    return BookReplacementCost(
        amount=quote.amount,
        source=quote.source,
        url=quote.url,
        retrieved_at=quote.retrieved_at.isoformat(),
        converted=quote.kind == CONVERTED,
    )


def listings(found: list[PriceQuote | NotFound]) -> list[PriceQuote]:
    return [item for item in found if isinstance(item, PriceQuote)]


async def convert_new(
    market: Market, other: Market, isbn: str, fx: Fx, cfg: PricingConfig
) -> tuple[PriceQuote, PriceQuote, FxRecord] | None:
    found = listings(await other.quotes(other.locale.new_sources, isbn, NEW))
    if not found:
        return None
    original = found[0]
    try:
        record = await fx.rate(original.currency, market.locale.currency, original.retrieved_at.date())
    except PriceFetchError:
        return None
    amount = (original.amount * record.rate).quantize(cfg.books.rounding)
    converted = original.model_copy(update={"amount": amount, "currency": market.locale.currency, "kind": CONVERTED})
    return converted, original, record


def disagreement(found: list[PriceQuote], cfg: PricingConfig) -> list[str]:
    if len(found) < 2:
        return []
    low, high = min(found, key=lambda q: q.amount), max(found, key=lambda q: q.amount)
    if (high.amount - low.amount) / low.amount <= cfg.books.disagreement_ratio:
        return []
    return [f"retailer_disagreement: {high.source} {high.amount} vs {low.source} {low.amount}"]


async def price_book(
    resolution,
    market: Market,
    *,
    reading,
    statements,
    other: Market | None = None,
    fx: Fx | None = None,
    condition: str | None = None,
    config: PricingConfig | None = None,
) -> BookPrices:
    cfg = config or load_pricing_config()
    locale = market.locale
    if resolution.status != "identified":
        return BookPrices(status="unidentified")
    reasons = appraisal_reasons(resolution, reading, statements, [], locale)
    if reasons:
        return BookPrices(status="needs_appraisal", review=reasons)
    new_found = await market.quotes(locale.new_sources, resolution.isbn, NEW)
    used_found = await market.quotes(locale.used_sources, resolution.isbn, USED)
    review = [f"{EQUIVALENT_EDITION}: {item.source}" for item in new_found + used_found if isinstance(item, NotFound) and item.reason == EQUIVALENT_EDITION]
    local = listings(new_found)
    new, original, record = (local[0] if local else None), None, None
    if new is None and CONVERTED in locale.new_sources and other is not None and fx is not None:
        new, original, record = await convert_new(market, other, resolution.isbn, fx, cfg) or (None, None, None)
    used = next(iter(listings(used_found)), None)
    if used is None and new is not None and DERIVED in locale.used_sources:
        used = derive_used(new, condition or cfg.books.default_condition, cfg)
    reasons = appraisal_reasons(resolution, reading, statements, [q for q in (new, used) if q], locale)
    if reasons:
        return BookPrices(status="needs_appraisal", review=review + reasons)
    if new is None and used is None:
        return BookPrices(status="identified", review=review + ["no_price_source"], excluded=True)
    if record is not None:
        review.append(f"converted: {original.currency} to {locale.currency} at {record.rate} on {record.rate_date}")
    sources = [original or new, used] + ([record] if record else [])
    evidence = list(dict.fromkeys(item.cache_ref for item in sources if item))
    return BookPrices(
        status="identified",
        replacement_cost=replacement_cost_of(new) if new else BookReplacementCost(),
        used_value=used_value_of(used, cfg) if used else UsedValue(),
        original=original,
        fx=record,
        review=review + disagreement(local, cfg),
        evidence=evidence,
    )
