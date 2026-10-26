from datetime import date
from decimal import Decimal

import pytest

from libclaim.live.tools import StatementKind
from libclaim.pricing.books import Market, derive_used, price_book
from libclaim.pricing.config import load_pricing_config
from libclaim.pricing.http import PriceFetchError
from libclaim.pricing.sources.fx import Fx
from libclaim.settings import load_locale
from tests.pricing.factories import ISBN, FakeSource, not_found, quote, reading, resolution, statement

IN = load_locale("IN")
GB = load_locale("GB")
EMPTY = FakeSource()


def india(**sources):
    return Market(IN, {"bookchor": EMPTY, "pustakkosh": EMPTY, "bookswagon": EMPTY, **sources})


def britain(**sources):
    return Market(GB, {"penguin_uk": EMPTY, **sources})


async def price(res=None, market=None, other=None, statements=(), fx=None, read=None):
    return await price_book(
        res or resolution(),
        market or india(),
        reading=read or reading(),
        statements=list(statements),
        other=other,
        fx=fx,
    )


async def test_local_new_price_and_observed_used(http):
    chor = FakeSource(new=quote(1571), used=quote(661, condition="used"))
    result = await price(market=india(bookchor=chor), fx=Fx(http))
    assert result.status == "identified" and not result.excluded
    assert result.replacement_cost.amount == Decimal("1571") and not result.replacement_cost.converted
    assert result.replacement_cost.source == "Bookchor" and result.replacement_cost.url.endswith("/new")
    assert (result.used_value.amount, result.used_value.source_type) == (Decimal("661"), "observed")
    assert result.used_value.method == "listing"


async def test_used_is_derived_from_new_and_labelled(http):
    chor = FakeSource(new=quote(1000))
    result = await price(market=india(bookchor=chor), fx=Fx(http))
    factor = load_pricing_config().condition_factors["good"]
    assert result.used_value.source_type == "derived"
    assert result.used_value.amount == Decimal(1000) * factor
    assert result.used_value.method == "condition_factor:v1:good"
    assert result.used_value.condition_assumed == "good"
    assert result.used_value.url.endswith("/new")


async def test_retailer_sources_follow_locale_order():
    first = FakeSource(new=quote(500, source="Bookchor"))
    second = FakeSource(new=quote(510, source="Pustakkosh"))
    result = await price(market=india(bookchor=first, pustakkosh=second))
    assert result.replacement_cost.source == "Bookchor"
    result = await price(market=india(bookchor=EMPTY, pustakkosh=second))
    assert result.replacement_cost.source == "Pustakkosh"


async def test_disagreement_over_a_quarter_is_flagged_for_review():
    chor = FakeSource(new=quote(242, source="Bookchor"))
    kosh = FakeSource(new=quote(193, source="Pustakkosh"))
    result = await price(market=india(bookchor=chor, pustakkosh=kosh))
    assert result.replacement_cost.amount == Decimal(242)
    assert any(reason.startswith("retailer_disagreement") for reason in result.review)


async def test_agreement_within_a_quarter_is_not_flagged():
    chor = FakeSource(new=quote(240, source="Bookchor"))
    kosh = FakeSource(new=quote(192, source="Pustakkosh"))
    result = await price(market=india(bookchor=chor, pustakkosh=kosh))
    assert result.review == []


async def test_no_local_price_converts_from_the_other_market_at_the_ecb_rate(http):
    penguin = FakeSource(new=quote("10.00", currency="GBP", source="Penguin UK"))
    result = await price(market=india(), other=britain(penguin_uk=penguin), fx=Fx(http))
    assert result.replacement_cost.converted is True
    assert result.replacement_cost.amount == Decimal("1269.30")
    assert result.original.currency == "GBP" and result.original.amount == Decimal("10.00")
    assert result.fx.rate == Decimal("126.93") and result.fx.rate_date == date(2026, 10, 23)
    assert "api.frankfurter.dev" in result.fx.url and "providers=ECB" in result.fx.url
    assert result.fx.cache_ref in result.evidence and result.original.cache_ref in result.evidence
    assert result.used_value.source_type == "derived" and result.used_value.amount is not None
    assert result.used_value.method == "condition_factor:v1:good:from_converted"
    assert any(reason.startswith("converted") for reason in result.review)


async def test_no_price_anywhere_is_excluded_with_empty_amounts(http):
    result = await price(fx=Fx(http), other=britain())
    assert result.status == "identified" and result.excluded is True
    assert result.replacement_cost.amount is None and result.used_value.amount is None
    assert result.review == ["no_price_source"]


async def test_signed_spine_needs_appraisal_and_has_no_amounts(http):
    chor = FakeSource(new=quote(500), used=quote(200, condition="used"))
    result = await price(market=india(bookchor=chor), read=reading(title_verbatim="Signed"), fx=Fx(http))
    assert result.status == "needs_appraisal"
    assert result.replacement_cost.amount is None and result.used_value.amount is None
    assert result.review == ["spine_wording"]
    assert not result.excluded


@pytest.mark.parametrize("kind", [StatementKind.original_art, StatementKind.signed, StatementKind.first_edition])
async def test_art_and_rare_are_never_priced(kind):
    chor = FakeSource(new=quote(500))
    result = await price(market=india(bookchor=chor), statements=[statement(kind)])
    assert result.status == "needs_appraisal"
    assert result.replacement_cost.amount is None and result.used_value.amount is None and not result.excluded


async def test_used_over_threshold_after_pricing_needs_appraisal():
    chor = FakeSource(new=quote(9000), used=quote(5000, condition="used"))
    result = await price(market=india(bookchor=chor))
    assert result.status == "needs_appraisal" and result.review == ["used_over_threshold"]
    assert result.replacement_cost.amount is None and result.used_value.amount is None


async def test_equivalent_edition_goes_to_review_not_price():
    chor = FakeSource(new=not_found("equivalent_edition", "Bookchor"))
    result = await price(market=india(bookchor=chor))
    assert result.excluded is True
    assert "equivalent_edition: Bookchor" in result.review


async def test_equivalent_edition_is_priced_from_its_isbn_and_labelled_in_the_line(http):
    seen = []

    class Recording:
        async def quote(self, isbn, condition):
            seen.append(isbn)
            return quote(400, source="Bookchor") if condition == "new" else not_found()

    equivalent = resolution(isbn="9780000000002", edition="", edition_unverified=True, equivalent_edition=True,
                            work_year=1993)
    result = await price(res=equivalent, market=india(bookchor=Recording()), fx=Fx(http))
    assert result.status == "identified" and result.replacement_cost.amount == Decimal(400)
    assert set(seen) == {"9780000000002"}
    assert "equivalent edition" in result.replacement_cost.source and "9780000000002" in result.replacement_cost.source
    assert "equivalent edition" in result.used_value.source
    assert result.replacement_cost.url.endswith("/new")
    assert any(note.startswith("equivalent_edition_priced") and "9780000000002" in note for note in result.review)


async def test_catalogue_only_identification_is_noted_for_review():
    chor = FakeSource(new=quote(400))
    result = await price(res=resolution(corroboration="catalogue_only"), market=india(bookchor=chor))
    assert result.replacement_cost.amount == Decimal(400)
    assert any(note.startswith("identified_from_catalogue_only") for note in result.review)


async def test_equivalent_edition_of_an_old_work_is_not_priced():
    chor = FakeSource(new=quote(400))
    old = resolution(isbn="9780000000002", edition_unverified=True, equivalent_edition=True, work_year=1942)
    result = await price(res=old, market=india(bookchor=chor))
    assert result.status == "needs_appraisal"
    assert result.replacement_cost.amount is None
    assert "old_work_edition_unverified" in result.review


async def test_unidentified_book_is_not_priced():
    result = await price(res=resolution(status="unidentified", isbn=""))
    assert result.status == "unidentified" and not result.excluded and result.review == []


async def test_no_isbn_needs_appraisal():
    result = await price(res=resolution(isbn=""))
    assert result.status == "needs_appraisal" and result.review == ["no_isbn"]


def test_derive_used_uses_the_versioned_factor_and_labels_it():
    cfg = load_pricing_config()
    derived = derive_used(quote(1571), "very_good", cfg)
    assert derived.kind == "derived_from_new" and derived.condition == "very_good"
    assert derived.amount == (Decimal(1571) * cfg.condition_factors["very_good"]).quantize(cfg.books.rounding)
    assert derived.url.endswith("/new") and derived.cache_ref == "Bookchor-new" and derived.isbn_on_page == ISBN


def test_derive_used_rejects_an_unknown_grade():
    with pytest.raises(KeyError):
        derive_used(quote(100), "mint", load_pricing_config())


class BrokenFx:
    async def rate(self, base, quote, on_or_before):
        raise PriceFetchError("ecb down")


class Exploding:
    async def quote(self, isbn, condition):
        raise RuntimeError("boom")


async def test_fx_failure_keeps_the_foreign_price_as_evidence_and_says_why():
    penguin = FakeSource(new=quote("10.00", currency="GBP", source="Penguin UK"))
    result = await price(market=india(), other=britain(penguin_uk=penguin), fx=BrokenFx())
    assert result.excluded is True and result.replacement_cost.amount is None
    assert result.review == ["fx_unavailable", "no_price_source"]
    assert result.original.amount == Decimal("10.00") and result.evidence == ["Penguin UK-new"]


async def test_old_edition_year_needs_appraisal_and_unknown_year_is_only_a_review_reason():
    chor = FakeSource(new=quote(500))
    old = await price(res=resolution(year=1920), market=india(bookchor=chor))
    assert old.status == "needs_appraisal" and old.review == ["antiquarian_year"]
    assert old.replacement_cost.amount is None
    unknown = await price(res=resolution(year=None), market=india(bookchor=chor))
    assert unknown.status == "identified" and unknown.replacement_cost.amount == Decimal(500)
    assert unknown.review == ["year_unknown"]


async def test_source_errors_and_missing_sources_become_review_reasons_not_crashes():
    market = Market(IN, {"bookchor": Exploding(), "pustakkosh": FakeSource(new=quote(300, source="Pustakkosh"))})
    result = await price(market=market)
    assert result.replacement_cost.source == "Pustakkosh"
    assert result.review == ["source_unavailable: bookchor"]


async def test_observed_used_without_new_is_priced_for_used_only_and_not_excluded():
    chor = FakeSource(used=quote(251, condition="used"))
    result = await price(market=india(bookchor=chor))
    assert result.status == "identified" and not result.excluded
    assert result.replacement_cost.amount is None and result.used_value.amount == Decimal(251)
    assert result.review == ["no_new_price"]


async def test_disagreement_is_relative_to_the_lower_price_and_skips_a_zero_price():
    def pair(high, low):
        return india(bookchor=FakeSource(new=quote(high)), pustakkosh=FakeSource(new=quote(low, source="Pustakkosh")))

    just_over, exactly, zero = [await price(market=pair(h, low)) for h, low in (("125.1", 100), (125, 100), (125, 0))]
    assert [bool(r.review) for r in (just_over, exactly, zero)] == [True, False, False]


async def test_evidence_lists_every_retailer_consulted():
    chor = FakeSource(new=quote(500, source="Bookchor"))
    kosh = FakeSource(new=quote(510, source="Pustakkosh"))
    result = await price(market=india(bookchor=chor, pustakkosh=kosh))
    assert {"Bookchor-new", "Pustakkosh-new"} <= set(result.evidence)
