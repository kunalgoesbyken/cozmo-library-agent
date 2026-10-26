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


@pytest.mark.parametrize(
    "title",
    [
        "Signed Copy",
        "THE FIRST EDITION",
        "first, edition",
        "Author's Signature",
        "AUTOGRAPHED!",
        "Inscribed to Tom",
        "An Inscription",
        "1st Edition",
        "First Printing",
        "First Print",
        "Numbered 12/500",
        "Limited Edition",
        "Uncorrected Proof",
        "Advance Copy",
        "Rare Find",
        "Antiquarian Books",
    ],
)
def test_wording_families_require_appraisal(title):
    assert reasons(read=reading(title_verbatim=title)) == ["spine_wording"]


@pytest.mark.parametrize("field", ["ocr_text", "vlm_text", "edition"])
def test_wording_is_found_in_every_text_source(field):
    assert reasons(res=resolution(**{field: "Limited Signed by the author"})) == ["spine_wording"]


@pytest.mark.parametrize("field", ["author_verbatim", "publisher_verbatim"])
def test_wording_is_found_in_author_and_publisher_reads(field):
    assert reasons(read=reading(**{field: "First Edition Society"})) == ["spine_wording"]


@pytest.mark.parametrize("title", ["Designed Sign Language", "Unsigned", "Rarely Numbering"])
def test_words_that_only_contain_a_trigger_do_not_fire(title):
    assert reasons(read=reading(title_verbatim=title)) == []


def test_year_before_antiquarian_year():
    assert reasons(res=resolution(year=1949)) == ["antiquarian_year"]
    assert reasons(res=resolution(year=1950)) == []
    assert reasons(res=resolution(year=None)) == []


def test_missing_isbn():
    assert reasons(res=resolution(isbn="")) == ["no_isbn"]


def test_equivalent_edition_of_a_modern_work_needs_no_appraisal():
    assert reasons(res=resolution(equivalent_edition=True, edition_unverified=True, work_year=1993)) == []


def test_equivalent_edition_of_an_old_work_without_publisher_evidence_needs_appraisal():
    old = resolution(equivalent_edition=True, edition_unverified=True, work_year=1942, year=2016)
    assert reasons(res=old) == ["old_work_edition_unverified"]


def test_old_work_whose_printed_publisher_matched_the_equivalent_is_priced():
    old = resolution(equivalent_edition=True, edition_unverified=True, work_year=1942, year=2016, edition="Award")
    assert reasons(res=old) == []


def test_used_at_threshold():
    threshold = IN.appraisal_threshold
    assert reasons(quotes=[quote(threshold, condition="used")]) == ["used_over_threshold"]
    assert reasons(quotes=[quote(threshold - 1, condition="used")]) == []


def test_used_at_one_and_a_half_times_new():
    assert reasons(quotes=[quote(100), quote(150, condition="used")]) == ["used_over_new"]
    assert reasons(quotes=[quote(100), quote(Decimal("149.99"), condition="used")]) == []


def test_reasons_accumulate():
    assert reasons(res=resolution(isbn="", year=1900), statements=[statement()]) == ["user_statement", "antiquarian_year", "no_isbn"]
