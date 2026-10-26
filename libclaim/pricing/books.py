from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from libclaim.live.tools import Statement
from libclaim.models import FxRecord, NotFound, PriceQuote
from libclaim.packet.models import BookReplacementCost, UsedValue
from libclaim.pricing.appraisal import NEW, appraisal_reasons
from libclaim.pricing.config import PricingConfig, load_pricing_config
from libclaim.pricing.sources.fx import Fx
from libclaim.settings import Locale
from libclaim.vision.reading import Reading
from libclaim.vision.resolve import Resolution

USED = "used"
CONVERTED = "converted"
DERIVED = "derived"
LISTING = "listing"
DERIVED_KIND = "derived_from_new"
EQUIVALENT_EDITION = "equivalent_edition"
UNAVAILABLE = "unavailable"
FROM_CONVERTED = "from_converted"


class Source(Protocol):
    async def quote(self, isbn: str, condition: str) -> PriceQuote | NotFound: ...


@dataclass
class Market:
    locale: Locale
    sources: dict[str, Source]

    async def quotes(self, names: list[str], isbn: str, condition: str) -> list[PriceQuote | NotFound]:
        found: list[PriceQuote | NotFound] = []
        for name in names:
            source = self.sources.get(name)
            if source is None:
                continue
            try:
                found.append(await source.quote(isbn, condition))
            except Exception:
                found.append(NotFound(source=name, isbn=isbn, reason=UNAVAILABLE))
        return found


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
    factor = cfg.condition_factors[condition]
    amount = (new_quote.amount * factor).quantize(cfg.books.rounding)
    return new_quote.model_copy(update={"amount": amount, "kind": DERIVED_KIND, "condition": condition})


def used_value_of(quote: PriceQuote, cfg: PricingConfig, from_converted: bool = False) -> UsedValue:
    derived = quote.kind == DERIVED_KIND
    method = f"condition_factor:{cfg.books.factor_version}:{quote.condition}" if derived else LISTING
    return UsedValue(
        amount=quote.amount,
        source=quote.source,
        url=quote.url,
        retrieved_at=quote.retrieved_at.isoformat(),
        condition_assumed=quote.condition,
        method=f"{method}:{FROM_CONVERTED}" if derived and from_converted else method,
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


def notes_of(found: list[PriceQuote | NotFound]) -> list[str]:
    notes = {EQUIVALENT_EDITION: EQUIVALENT_EDITION, UNAVAILABLE: "source_unavailable"}
    return [f"{notes[item.reason]}: {item.source}" for item in found if isinstance(item, NotFound) and item.reason in notes]


def converted_quote(original: PriceQuote, record: FxRecord, locale: Locale, cfg: PricingConfig) -> PriceQuote:
    amount = (original.amount * record.rate).quantize(cfg.books.rounding)
    return original.model_copy(update={"amount": amount, "currency": locale.currency, "kind": CONVERTED})


def disagreement(found: list[PriceQuote], cfg: PricingConfig) -> list[str]:
    if len(found) < 2:
        return []
    low, high = min(found, key=lambda q: q.amount), max(found, key=lambda q: q.amount)
    if low.amount <= 0 or (high.amount - low.amount) / low.amount <= cfg.books.disagreement_ratio:
        return []
    return [f"retailer_disagreement: {high.source} {high.amount} vs {low.source} {low.amount}"]


async def price_book(
    resolution: Resolution,
    market: Market,
    *,
    reading: Reading,
    statements: list[Statement],
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
    review = list(dict.fromkeys(notes_of(new_found + used_found))) + (["year_unknown"] if resolution.year is None else [])
    local = listings(new_found)
    new, original, record = (local[0] if local else None), None, None
    if new is None and CONVERTED in locale.new_sources and other is not None and fx is not None:
        other_found = await other.quotes(other.locale.new_sources, resolution.isbn, NEW)
        review += notes_of(other_found)
        original = next(iter(listings(other_found)), None)
        if original is not None:
            try:
                record = await fx.rate(original.currency, locale.currency, original.retrieved_at.date())
            except Exception:
                review.append("fx_unavailable")
            else:
                new = converted_quote(original, record, locale, cfg)
                review.append(f"converted: {original.currency} to {locale.currency} at {record.rate} on {record.rate_date}")
    used = next(iter(listings(used_found)), None)
    if used is None and new is not None and DERIVED in locale.used_sources:
        used = derive_used(new, condition or cfg.books.default_condition, cfg)
    sourced = [*local, original, used, record]
    evidence = list(dict.fromkeys(item.cache_ref for item in sourced if item))
    reasons = appraisal_reasons(resolution, reading, statements, [q for q in (new, used) if q], locale)
    if reasons:
        return BookPrices(status="needs_appraisal", review=review + reasons, evidence=evidence)
    if new is None and used is None:
        return BookPrices(status="identified", original=original, review=review + ["no_price_source"], evidence=evidence, excluded=True)
    if new is None:
        review.append("no_new_price")
    return BookPrices(
        status="identified",
        replacement_cost=replacement_cost_of(new) if new else BookReplacementCost(),
        used_value=used_value_of(used, cfg, new is not None and new.kind == CONVERTED) if used else UsedValue(),
        original=original,
        fx=record,
        review=review + disagreement(local, cfg),
        evidence=evidence,
    )
