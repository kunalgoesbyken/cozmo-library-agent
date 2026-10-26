from decimal import Decimal

from libclaim.packet.models import Book, Item, Totals


def _book_excluded(book: Book) -> bool:
    return book.status == "needs_appraisal" or book.replacement_cost.amount is None


def _item_excluded(item: Item) -> bool:
    cost = item.replacement_cost
    return item.status == "needs_appraisal" or cost.low is None or cost.high is None


def _sum(values) -> Decimal:
    return sum(values, Decimal(0))


def compute_totals(books: list[Book], items: list[Item]) -> Totals:
    counted_books = [book for book in books if not _book_excluded(book)]
    used = [
        book.used_value
        for book in books
        if book.status != "needs_appraisal" and book.used_value.amount is not None
    ]
    counted_items = [item for item in items if not _item_excluded(item)]
    return Totals(
        book_count=len(books),
        books_identified=sum(book.status == "identified" for book in books),
        books_unidentified=sum(book.status == "unidentified" for book in books),
        books_needs_appraisal=sum(book.status == "needs_appraisal" for book in books),
        shelf_run_m=round(
            sum(book.spine_thickness_cm or 0 for book in books) / 100, 4
        ),
        books_replacement_cost=_sum(book.replacement_cost.amount for book in counted_books),
        books_used_value=_sum(value.amount for value in used),
        books_used_value_observed=_sum(v.amount for v in used if v.source_type == "observed"),
        books_used_value_derived=_sum(v.amount for v in used if v.source_type == "derived"),
        items_replacement_cost_low=_sum(item.replacement_cost.low for item in counted_items),
        items_replacement_cost_high=_sum(item.replacement_cost.high for item in counted_items),
        excluded_from_totals=sum(map(_book_excluded, books)) + sum(map(_item_excluded, items)),
    )
