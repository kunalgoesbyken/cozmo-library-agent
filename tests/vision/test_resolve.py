from pathlib import Path

import httpx
import pytest

from libclaim.catalog.googlebooks import GoogleBooks
from libclaim.catalog.http import Candidate, CatalogHttp, Edition
from libclaim.catalog.openlibrary import OpenLibrary
from libclaim.vision.reading import Reading
import libclaim.vision.resolve as resolve_module
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
    assert result.year is not None and result.year >= 1900
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


def cand(title, author="Yuval Noah Harari", source="openlibrary", **fields):
    return Candidate(source=source, title=title, author=author, url=f"https://x/{title}", work=title, **fields)


class FakeLibrary:
    def __init__(self, found=(), editions=()):
        self.found = list(found)
        self.edition_list = list(editions)

    async def search(self, title, author):
        return self.found

    async def editions(self, work):
        return self.edition_list


class FakeGoogle:
    def __init__(self, found=()):
        self.found = list(found)
        self.calls = 0

    async def search(self, title, author):
        self.calls += 1
        return self.found


def fakes(library=(), google=(), editions=()):
    return Catalogs(FakeLibrary(library, editions), FakeGoogle(google))


def gates_with(monkeypatch, **changes):
    config = resolve_module.load_resolve_config()
    patched = config.model_copy(update={"gates": config.gates.model_copy(update=changes)})
    monkeypatch.setattr(resolve_module, "load_resolve_config", lambda: patched)


async def test_sibling_title_does_not_inflate_the_score():
    library = [cand("Sapiens: A Graphic History"), cand("Sapiens")]
    result = await resolve(reading("SAPIENS", "HARARI"), ("SAPIENS HARARI", 0.9), None, fakes(library))
    assert result.status == "identified"
    assert result.title == "Sapiens"
    assert result.margin >= 8


async def test_only_a_longer_sibling_in_the_catalogue_is_unidentified():
    library = [cand("Sapiens: A Graphic History")]
    result = await resolve(reading("SAPIENS", "HARARI"), ("SAPIENS HARARI", 0.9), None, fakes(library, library))
    assert result.status == "unidentified"


async def test_partial_read_does_not_match_a_longer_title():
    library = [cand("Homo Deus")]
    result = await resolve(reading("HOMO", "HARARI"), ("HOMO HARARI", 0.9), None, fakes(library, library))
    assert result.status == "unidentified"


async def test_short_generic_title_is_unidentified():
    library = [cand("It", "Stephen King"), cand("It", "Someone Else")]
    result = await resolve(reading("IT"), ("IT", 0.9), None, fakes(library, library))
    assert result.status == "unidentified"


async def test_author_only_read_is_unidentified():
    result = await resolve(reading("", "HARARI"), ("HARARI", 0.9), None, fakes([cand("Sapiens")]))
    assert result.status == "unidentified"
    assert result.reason == "illegible"


async def test_lone_catalogue_hit_needs_a_stricter_score_or_corroboration(monkeypatch):
    sapiens = reading("SAPIENS", "HARARI")
    lone = rank(sapiens, [cand("Sapiens")])
    assert accept_identification(sapiens, ("SAPIENS HARARI", 0.9), lone)
    gates_with(monkeypatch, lone_score_min=101)
    assert not accept_identification(sapiens, ("SAPIENS HARARI", 0.9), lone)
    corroborated = rank(sapiens, [cand("Sapiens"), cand("Sapiens", source="googlebooks")])
    assert accept_identification(sapiens, ("SAPIENS HARARI", 0.9), corroborated)


@pytest.mark.parametrize(
    "gate", ["ocr_overlap_min", "score_min", "margin_min", "confident_score"]
)
async def test_each_gate_can_reject_on_its_own(monkeypatch, catalogs, gate):
    spine = reading("SAPIENS", "HARARI")
    ranked = rank(spine, await catalogs.open_library.search("sapiens", "harari"))[:2]
    assert accept_identification(spine, ("SAPIENS HARARI", 0.9), ranked)
    gates_with(monkeypatch, **{gate: 101})
    assert not accept_identification(spine, ("SAPIENS HARARI", 0.9), ranked)


async def test_one_distinctive_token_without_author_needs_the_gate_relaxed(monkeypatch, catalogs):
    spine = reading("SAPIENS")
    ranked = rank(spine, await catalogs.open_library.search("sapiens", "harari"))[:2]
    assert not accept_identification(spine, ("SAPIENS", 0.9), ranked)
    gates_with(monkeypatch, distinctive_min_tokens=1)
    assert accept_identification(spine, ("SAPIENS", 0.9), ranked)


async def test_rate_limited_catalogues_are_reported_as_unavailable(tmp_path, monkeypatch):
    monkeypatch.delenv("LIBCLAIM_OFFLINE")
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(429)))
    http = CatalogHttp(tmp_path / "cache", client=client)
    catalogs = Catalogs(OpenLibrary(http), GoogleBooks(http, api_key="k"))
    result = await resolve(reading("SAPIENS", "HARARI"), ("SAPIENS HARARI", 0.9), None, catalogs)
    assert result.status == "unidentified"
    assert result.reason == "catalogue_unavailable"


@pytest.mark.parametrize(
    ("spine", "text", "reason"),
    [(reading("", legible=False), "", "illegible"), (reading("SAPIENS", "HARARI"), "ZEBRA CROSSING", "ocr_disagrees")],
)
async def test_early_rejections_keep_the_dims(catalogs, spine, text, reason):
    dims = SpineDims(height_cm=20.0, thickness_cm=3.0)
    result = await resolve(spine, (text, 0.9), dims, catalogs)
    assert result.reason == reason
    assert result.dims == dims


async def test_failed_reading_is_reported():
    spine = Reading(track_id=1, failure="QuotaExhausted")
    result = await resolve(spine, ("", 0.0), None, fakes())
    assert result.reason == "reading_failed"


async def test_ambiguous_publisher_gives_no_isbn_claim(catalogs):
    found = await catalogs.open_library.search("sapiens", "harari")
    editions = [Edition(publisher="Penguin", isbn="111", url="u1"), Edition(publisher="Penguin", isbn="222", url="u2")]
    result = await resolve(
        reading("SAPIENS", "HARARI", "PENGUIN"), ("SAPIENS HARARI", 0.9), None, fakes(found, editions=editions)
    )
    assert result.status == "identified"
    assert result.edition_unverified
    assert result.isbn == "" and result.edition == ""


async def test_query_keeps_punctuation(tmp_path, monkeypatch):
    seen = []

    class Spy(FakeLibrary):
        async def search(self, title, author):
            seen.append(title)
            return []

    catalogs = Catalogs(Spy(), FakeGoogle())
    await resolve(reading("C++ PRIMER", "LIPPMAN"), ("C++ PRIMER LIPPMAN", 0.9), None, catalogs)
    assert seen == ["c++ primer"]


async def test_picked_edition_carries_open_library_dimensions(catalogs):
    found = await catalogs.open_library.search("sapiens", "harari")
    editions = [Edition(publisher="Vintage", isbn="9780099590088", url="u1", physical_dimensions="19.8 x 12.9 x 3 cm")]
    result = await resolve(
        reading("SAPIENS", "HARARI", "VINTAGE"), ("SAPIENS HARARI VINTAGE", 0.9), None, fakes(found, editions=editions)
    )
    assert result.isbn == "9780099590088"
    assert result.physical_dimensions == "19.8 x 12.9 x 3 cm"
