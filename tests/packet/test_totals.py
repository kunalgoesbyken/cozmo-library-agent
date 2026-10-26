from decimal import Decimal

import pytest
from pydantic import ValidationError

from libclaim.packet.models import Book, Item, ItemReplacementCost
from libclaim.packet.totals import compute_totals


def book(id, status="identified", amount=None, thickness=None, used=None, used_type="derived"):
    fields = {"id": id, "status": status, "spine_thickness_cm": thickness}
    if amount is not None:
        fields["replacement_cost"] = {"amount": amount}
    if used is not None:
        fields["used_value"] = {"amount": used, "source_type": used_type}
    return Book.model_validate(fields)


def item(id, status, low=None, high=None):
    return Item(
        id=id,
        status=status,
        replacement_cost=ItemReplacementCost(
            low=None if low is None else Decimal(low),
            high=None if high is None else Decimal(high),
        ),
    )


def test_needs_appraisal_and_priceless_books_are_excluded_and_counted():
    books = [
        book("a", amount=Decimal("10.50")),
        book("b", "needs_appraisal", amount=Decimal("500")),
        book("c"),
        book("d", "unidentified"),
    ]
    totals = compute_totals(books, [])
    assert totals.books_replacement_cost == Decimal("10.50")
    assert totals.excluded_from_totals == 3


def test_status_counts_add_up_to_book_count():
    books = [book("a"), book("b", "unidentified"), book("c", "unidentified"), book("d", "needs_appraisal")]
    totals = compute_totals(books, [])
    assert (totals.books_identified, totals.books_unidentified, totals.books_needs_appraisal) == (1, 2, 1)
    assert (
        totals.books_identified + totals.books_unidentified + totals.books_needs_appraisal
        == totals.book_count
        == 4
    )


def test_shelf_run_is_sum_of_thicknesses_in_metres():
    books = [book("a", thickness=2.5), book("b", thickness=3.5), book("c")]
    assert compute_totals(books, []).shelf_run_m == 0.06


def test_observed_plus_derived_equals_used_value():
    books = [
        book("a", used=Decimal("4.10"), used_type="observed"),
        book("b", used=Decimal("2.20"), used_type="derived"),
        book("c", used=Decimal("1.05"), used_type="derived"),
        book("d"),
    ]
    totals = compute_totals(books, [])
    assert totals.books_used_value_observed == Decimal("4.10")
    assert totals.books_used_value_derived == Decimal("3.25")
    assert totals.books_used_value == totals.books_used_value_observed + totals.books_used_value_derived


def test_range_items_add_low_and_high_and_appraisal_items_are_excluded():
    items = [
        item("i1", "range", "100", "200"),
        item("i2", "priced", "50", "50"),
        item("i3", "needs_appraisal", "900", "1000"),
        item("i4", "priced"),
    ]
    totals = compute_totals([], items)
    assert totals.items_replacement_cost_low == Decimal("150")
    assert totals.items_replacement_cost_high == Decimal("250")
    assert totals.excluded_from_totals == 2


def test_used_value_amount_requires_source_type():
    with pytest.raises(ValidationError):
        Book.model_validate({"id": "a", "status": "identified", "used_value": {"amount": "3"}})
