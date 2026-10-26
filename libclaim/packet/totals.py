from decimal import Decimal

from libclaim.packet.models import Book, Item, Totals

CM_PER_M = 100
RUN_DECIMALS = 4


def _book_counted(book: Book) -> bool:
    return book.status != "needs_appraisal" and book.replacement_cost.amount is not None


def _book_unpriced(book: Book) -> bool:
    return book.status == "identified" and book.replacement_cost.amount is None


def _item_counted(item: Item) -> bool:
    cost = item.replacement_cost
    return item.status != "needs_appraisal" and cost.low is not None and cost.high is not None


def _item_unpriced(item: Item) -> bool:
    return item.status != "needs_appraisal" and not _item_counted(item)


def _sum(values) -> Decimal:
    return sum(values, Decimal(0))


def _run_m(centimetres: list[float]) -> float | None:
    return round(sum(centimetres) / CM_PER_M, RUN_DECIMALS) if centimetres else None


def compute_totals(books: list[Book], items: list[Item]) -> Totals:
    counted_books = [book for book in books if _book_counted(book)]
    used = [
        book.used_value
        for book in books
        if book.status != "needs_appraisal" and book.used_value.amount is not None
    ]
    counted_items = [item for item in items if _item_counted(item)]
    measured = [book.spine_thickness_cm for book in books if book.spine_thickness_cm is not None]
    estimated = [
        book.dims_estimated_from_prior.thickness_cm
        for book in books
        if book.spine_thickness_cm is None and book.dims_estimated_from_prior is not None
    ]
    return Totals(
        book_count=len(books),
        books_identified=sum(book.status == "identified" for book in books),
        books_unidentified=sum(book.status == "unidentified" for book in books),
        books_needs_appraisal=sum(book.status == "needs_appraisal" for book in books),
        shelf_run_m=_run_m(measured),
        shelf_run_estimated_m=_run_m(estimated),
        books_replacement_cost=_sum(book.replacement_cost.amount for book in counted_books),
        books_used_value=_sum(value.amount for value in used),
        books_used_value_observed=_sum(v.amount for v in used if v.source_type == "observed"),
        books_used_value_derived=_sum(v.amount for v in used if v.source_type == "derived"),
        items_replacement_cost_low=_sum(item.replacement_cost.low for item in counted_items),
        items_replacement_cost_high=_sum(item.replacement_cost.high for item in counted_items),
        excluded_from_totals=sum(map(_book_unpriced, books)) + sum(map(_item_unpriced, items)),
    )
