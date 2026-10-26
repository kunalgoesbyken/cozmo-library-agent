from decimal import Decimal

import pytest

from libclaim.live.tools import StatementKind
from libclaim.pricing.appraisal import appraisal_reasons
from libclaim.settings import load_locale
from tests.pricing.factories import quote, reading, resolution, statement

IN = load_locale("IN")


def reasons(res=None, read=None, statements=(), quotes=()):
    return appraisal_reasons(res or resolution(), read or reading(), list(statements), list(quotes), IN)


def test_ordinary_book_has_no_reasons():
    assert reasons(quotes=[quote(500), quote(250, condition="used")]) == []


@pytest.mark.parametrize("kind", [StatementKind.signed, StatementKind.first_edition, StatementKind.original_art])
def test_user_statement_requires_appraisal(kind):
    assert reasons(statements=[statement(kind)]) == ["user_statement"]


def test_statement_of_kind_other_is_not_a_reason():
    assert reasons(statements=[statement(StatementKind.other)]) == []


@pytest.mark.parametrize("title", ["Signed Copy", "THE FIRST EDITION", "first, edition"])
def test_spine_wording_requires_appraisal(title):
    assert reasons(read=reading(title_verbatim=title)) == ["spine_wording"]


def test_spine_wording_found_in_ocr_text():
    assert reasons(res=resolution(ocr_text="Limited Signed by the author")) == ["spine_wording"]


def test_words_that_only_contain_a_trigger_do_not_fire():
    assert reasons(read=reading(title_verbatim="Designed Sign Language")) == []


def test_year_before_antiquarian_year():
    assert reasons(res=resolution(year=1949)) == ["antiquarian_year"]
    assert reasons(res=resolution(year=1950)) == []
    assert reasons(res=resolution(year=None)) == []


def test_missing_isbn():
    assert reasons(res=resolution(isbn="")) == ["no_isbn"]


def test_used_at_threshold():
    threshold = IN.appraisal_threshold
    assert reasons(quotes=[quote(threshold, condition="used")]) == ["used_over_threshold"]
    assert reasons(quotes=[quote(threshold - 1, condition="used")]) == []


def test_used_at_one_and_a_half_times_new():
    assert reasons(quotes=[quote(100), quote(150, condition="used")]) == ["used_over_new"]
    assert reasons(quotes=[quote(100), quote(Decimal("149.99"), condition="used")]) == []


def test_reasons_accumulate():
    assert reasons(res=resolution(isbn="", year=1900), statements=[statement()]) == ["user_statement", "antiquarian_year", "no_isbn"]
