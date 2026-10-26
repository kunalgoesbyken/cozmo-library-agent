from pathlib import Path

import pytest

from libclaim.catalog.googlebooks import GoogleBooks
from libclaim.catalog.http import CatalogHttp
from libclaim.catalog.openlibrary import OpenLibrary
from libclaim.vision.reading import Reading
from libclaim.vision.resolve import Catalogs, SpineDims, accept_identification, rank, resolve

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "catalog"


@pytest.fixture
def catalogs():
    http = CatalogHttp(FIXTURES)
    return Catalogs(OpenLibrary(http), GoogleBooks(http, api_key=""))


def reading(title, author="", publisher="", legible=True):
    return Reading(
        track_id=1, title_verbatim=title, author_verbatim=author, publisher_verbatim=publisher, legible=legible
    )


async def test_sapiens_resolves_to_vintage_edition(catalogs):
    result = await resolve(reading("SAPIENS", "HARARI", "VINTAGE"), ("SAPIENS HARARI VINTAGE", 0.98), None, catalogs)
    assert result.status == "identified"
    assert result.title == "Sapiens"
    assert "Harari" in result.author
    assert "Vintage" in result.edition
    assert len(result.isbn) == 13
    assert result.match_score >= 95
    assert result.margin >= 8
    assert 0.9 <= result.id_confidence <= 1.0
    assert any("openlibrary.org" in url for url in result.source_urls)


async def test_common_title_resolves(catalogs):
    result = await resolve(
        reading("PRIDE AND PREJUDICE", "AUSTEN"), ("PRIDE AND PREJUDICE AUSTEN", 0.9), None, catalogs
    )
    assert result.status == "identified"
    assert result.title == "Pride and Prejudice"
    assert result.edition == ""


async def test_ambiguous_short_title_alone_is_unidentified(catalogs):
    result = await resolve(reading("EMMA"), ("EMMA", 0.95), None, catalogs)
    assert result.status == "unidentified"
    assert result.reason == "not_confident"
    assert result.vlm_text == "EMMA"


async def test_vlm_read_ocr_does_not_support_keeps_both_texts(catalogs):
    result = await resolve(reading("SAPIENS", "HARARI"), ("ZEBRA CROSSING", 0.9), None, catalogs)
    assert result.status == "unidentified"
    assert result.reason == "ocr_disagrees"
    assert result.vlm_text == "SAPIENS HARARI"
    assert result.ocr_text == "ZEBRA CROSSING"


async def test_unreadable_spine_is_unidentified(catalogs):
    result = await resolve(reading("", legible=False), ("", 0.0), None, catalogs)
    assert result.status == "unidentified"
    assert result.reason == "illegible"


async def test_garbage_title_is_a_miss_everywhere(catalogs):
    result = await resolve(reading("XQZVK PLORBWEN"), ("XQZVK PLORBWEN", 0.9), None, catalogs)
    assert result.status == "unidentified"
    assert result.reason == "no_catalogue_match"


async def test_google_books_fallback_on_open_library_miss(catalogs):
    result = await resolve(
        reading("THE PSYCHOLOGY OF MONY", "HOUSEL"), ("THE PSYCHOLOGY OF MONEY HOUSEL", 0.9), None, catalogs
    )
    assert result.status == "identified"
    assert result.title == "The Psychology of Money"
    assert all("openlibrary" not in url for url in result.source_urls)


async def test_replay_is_identical_and_needs_no_network(catalogs):
    args = (reading("SAPIENS", "HARARI", "VINTAGE"), ("SAPIENS HARARI VINTAGE", 0.98), SpineDims(height_cm=19.7))
    first = await resolve(*args, catalogs)
    second = await resolve(*args, catalogs)
    assert first == second
    assert first.dims == SpineDims(height_cm=19.7)


async def test_open_library_hit_does_not_query_google_books(catalogs):
    class Failing:
        async def search(self, title, author):
            raise AssertionError("google books must not be queried")

    catalogs.google_books = Failing()
    result = await resolve(reading("SAPIENS", "HARARI"), ("SAPIENS HARARI", 0.9), None, catalogs)
    assert result.status == "identified"


async def test_accept_gates_are_individually_enforced(catalogs):
    sapiens = reading("SAPIENS", "HARARI")
    ranked = rank(sapiens, await catalogs.open_library.search("sapiens", "harari"))
    assert accept_identification(sapiens, ("SAPIENS HARARI", 0.9), ranked)
    assert not accept_identification(sapiens, ("OTHER WORDS", 0.9), ranked)
    assert not accept_identification(sapiens, ("SAPIENS HARARI", 0.9), [])
    assert not accept_identification(sapiens, ("SAPIENS HARARI", 0.9), ranked[:1] + [ranked[0]])
