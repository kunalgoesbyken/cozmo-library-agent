from typing import Literal

from pydantic import BaseModel, Field

from libclaim.models import ItemRange, ListingRow
from libclaim.packet.models import ClaimPacket, Item, ItemReplacementCost, SecondCountry, SecondCountryBook
from libclaim.pricing.books import Market, price_book
from libclaim.pricing.sources.fx import Fx
from libclaim.pricing.sources.listings import Listings
from libclaim.settings import Locale
from libclaim.vision.reading import Reading
from libclaim.vision.resolve import Resolution

FLAGGED = "flagged_for_appraisal"
TOO_FEW = "too_few_listings"
LOCAL = "local_listing"
CONVERTED = "converted"


class ItemPrice(BaseModel):
    status: Literal["range", "needs_appraisal"]
    replacement_cost: ItemReplacementCost = Field(default_factory=ItemReplacementCost)
    range: ItemRange | None = None
    evidence: list[ListingRow] = Field(default_factory=list)
    review: list[str] = Field(default_factory=list)


async def price_item(item: Item, locale: Locale, *, listings: Listings) -> ItemPrice:
    if item.status == "needs_appraisal":
        return ItemPrice(status="needs_appraisal", review=[FLAGGED])
    found = await listings.range(item.category, locale.country)
    if not isinstance(found, ItemRange):
        return ItemPrice(status="needs_appraisal", evidence=found.rows, review=[TOO_FEW])
    return ItemPrice(
        status="range",
        replacement_cost=ItemReplacementCost(
            low=found.low.amount,
            high=found.high.amount,
            source=", ".join(dict.fromkeys(row.source for row in (found.low, found.high))),
            url=found.low.url,
            url_high=found.high.url,
            retrieved_at=max(found.low.retrieved_at, found.high.retrieved_at).isoformat(),
        ),
        range=found,
        evidence=found.rows,
    )


def sample(packet: ClaimPacket, n: int):
    priced = [book for book in packet.books if book.status == "identified" and book.isbn and book.replacement_cost.amount is not None]
    return sorted(priced, key=lambda book: (-book.replacement_cost.amount, book.isbn, book.id))[:n]


async def second_country(
    packet: ClaimPacket, n: int = 10, *, market: Market, other: Market, fx: Fx,
    resolutions: dict[str, Resolution] | None = None,
) -> SecondCountry:
    rows = []
    for book in sample(packet, n):
        resolution = (resolutions or {}).get(book.id) or Resolution(
            status="identified", title=book.title, author=book.author, edition=book.edition, isbn=book.isbn
        )
        priced = await price_book(resolution, market, reading=Reading(track_id=0), statements=[], other=other, fx=fx)
        converted = priced.replacement_cost.converted
        has_price = priced.replacement_cost.amount is not None
        rows.append(
            SecondCountryBook(
                book_id=book.id,
                title=book.title,
                author=book.author,
                edition=book.edition,
                isbn=book.isbn,
                status="needs_appraisal" if priced.status == "needs_appraisal" else "identified",
                source_type=(CONVERTED if converted else LOCAL) if has_price else None,
                replacement_cost=priced.replacement_cost,
                used_value=priced.used_value,
                original_amount=priced.original.amount if priced.original else book.replacement_cost.amount,
                original_currency=packet.sweep.currency or other.locale.currency,
                fx=priced.fx,
                review=priced.review,
                evidence=priced.evidence,
            )
        )
    result = SecondCountry(country=market.locale.country, currency=market.locale.currency, books=rows)
    packet.second_country = result
    return result
