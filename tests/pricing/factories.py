from datetime import UTC, datetime
from decimal import Decimal

from libclaim.live.tools import Statement, StatementKind
from libclaim.models import NotFound, PriceQuote
from libclaim.vision.reading import Reading
from libclaim.vision.resolve import Resolution

ISBN = "9780143127550"
RETRIEVED = datetime(2026, 10, 23, 9, 30, tzinfo=UTC)


def resolution(**changes) -> Resolution:
    base = {"status": "identified", "title": "Everything I Never Told You", "isbn": ISBN, "year": 2015}
    return Resolution(**{**base, **changes})


def reading(**changes) -> Reading:
    base = {"track_id": 1, "title_verbatim": "Everything I Never Told You", "author_verbatim": "Celeste Ng", "legible": True}
    return Reading(**{**base, **changes})


def statement(kind=StatementKind.signed) -> Statement:
    return Statement(subject="this one", kind=kind, value="yes", t_ms=1)


def quote(amount, currency="INR", source="Bookchor", condition="new", kind="listing") -> PriceQuote:
    return PriceQuote(
        amount=Decimal(str(amount)),
        currency=currency,
        source=source,
        url=f"https://example.test/{source}/{condition}",
        retrieved_at=RETRIEVED,
        kind=kind,
        condition=condition,
        isbn_on_page=ISBN,
        cache_ref=f"{source}-{condition}",
    )


def not_found(reason="not_found", source="fake") -> NotFound:
    return NotFound(source=source, isbn=ISBN, reason=reason)


class FakeSource:
    def __init__(self, **by_condition):
        self.by_condition = by_condition

    async def quote(self, isbn, condition):
        return self.by_condition.get(condition) or not_found()
