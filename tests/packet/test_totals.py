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


def test_only_identified_books_without_a_price_count_as_excluded():
    books = [
        book("a", amount=Decimal("10.50")),
        book("b", "needs_appraisal", amount=Decimal("500")),
        book("c"),
        book("d", "unidentified"),
    ]
    totals = compute_totals(books, [])
    assert totals.books_replacement_cost == Decimal("10.50")
    assert totals.excluded_from_totals == 1
    assert totals.books_needs_appraisal == 1 and totals.books_unidentified == 1


def test_unidentified_and_appraisal_lines_are_never_excluded_from_totals_by_the_counter():
    books = [book("a", "unidentified"), book("b", "needs_appraisal")]
    assert compute_totals(books, []).excluded_from_totals == 0


def test_status_counts_add_up_to_book_count():
    books = [book("a"), book("b", "unidentified"), book("c", "unidentified"), book("d", "needs_appraisal")]
    totals = compute_totals(books, [])
    assert (totals.books_identified, totals.books_unidentified, totals.books_needs_appraisal) == (1, 2, 1)
    assert (
        totals.books_identified + totals.books_unidentified + totals.books_needs_appraisal
        == totals.book_count
        == 4
    )


def test_shelf_run_is_sum_of_measured_thicknesses_in_metres():
    books = [book("a", thickness=2.5), book("b", thickness=3.5), book("c")]
    totals = compute_totals(books, [])
    assert totals.shelf_run_m == 0.06
    assert totals.shelf_run_estimated_m is None


def test_shelf_run_is_empty_when_nothing_was_measured():
    assert compute_totals([book("a"), book("b")], []).shelf_run_m is None
    assert compute_totals([], []).shelf_run_m is None


def test_prior_thicknesses_only_count_in_the_estimated_run():
    estimate = {"height_cm": 21.0, "thickness_cm": 2.5, "book_format": "trade_paperback"}
    prior = Book.model_validate({"id": "p", "status": "unidentified", "dims_estimated_from_prior": estimate})
    measured = book("m", thickness=3.0)
    totals = compute_totals([prior, prior.model_copy(update={"id": "q"}), measured], [])
    assert totals.shelf_run_m == 0.03
    assert totals.shelf_run_estimated_m == 0.05


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
    assert totals.excluded_from_totals == 1


def test_used_value_amount_requires_source_type():
    with pytest.raises(ValidationError):
        Book.model_validate({"id": "a", "status": "identified", "used_value": {"amount": "3"}})
