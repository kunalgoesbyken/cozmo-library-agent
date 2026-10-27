from pathlib import Path

import httpx
import pytest

from libclaim.catalog.googlebooks import GoogleBooks
from libclaim.catalog.http import Candidate, CatalogHttp, Edition
from libclaim.catalog.openlibrary import OpenLibrary
from libclaim.resolve_config import load_resolve_config
from libclaim.vision.reading import Reading
import libclaim.vision.resolve as resolve_module
from libclaim.vision.resolve import Catalogs, SpineDims, accept_identification, ocr_overlap, rank, resolve

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
    assert any("books.google.com" in url for url in result.source_urls)
    assert not any("search" in url for url in result.source_urls)


async def test_common_title_resolves(catalogs):
    result = await resolve(
        reading("PRIDE AND PREJUDICE", "AUSTEN"), ("PRIDE AND PREJUDICE AUSTEN", 0.9), None, catalogs
    )
    assert result.status == "identified"
    assert result.title == "Pride and Prejudice"
    assert result.edition == ""


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


async def test_misspelled_title_resolves_through_google_books(catalogs):
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


async def test_accept_gates_are_individually_enforced(catalogs):
    sapiens = reading("SAPIENS", "HARARI")
    ranked = rank(sapiens, await catalogs.google_books.search("sapiens", "harari"))
    assert accept_identification(sapiens, ("SAPIENS HARARI", 0.9), ranked)
    assert not accept_identification(sapiens, ("OTHER WORDS", 0.9), ranked)
    assert not accept_identification(sapiens, ("SAPIENS HARARI", 0.9), [])
    rival = ranked[0].model_copy(update={"best": ranked[0].best.model_copy(update={"title": "Homo Deus"})})
    assert not accept_identification(sapiens, ("SAPIENS HARARI", 0.9), ranked[:1] + [rival])


async def test_same_title_from_another_catalogue_author_is_not_a_rival():
    merin = cand("Modern German Art", author="Peter Merin")
    thoene = cand("Modern German Art", author="Peter Thoene")
    read = reading("MODERN GERMAN ART", "PETER")
    ranked = rank(read, [merin, thoene])
    assert accept_identification(read, ("PETER PELICAN MODERN GERMAN ART", 0.9), ranked)


async def test_author_inside_a_catalogue_title_does_not_lower_the_score():
    read = reading("CHARTRES", "GEORGE HENDERSON")
    ranked = rank(read, [cand("Chartres, By George Henderson", author="George Henderson")])
    assert ranked[0].score >= 95


def cand(title, author="Yuval Noah Harari", source="googlebooks", isbn="9780000000000", **fields):
    return Candidate(source=source, title=title, author=author, isbn=isbn, url=f"https://x/{title}", work=title, **fields)


class FakeLibrary:
    def __init__(self, editions=()):
        self.edition_list = list(editions)
        self.isbns = []

    async def work_of(self, isbn):
        self.isbns.append(isbn)
        return "OL1W"

    async def editions(self, work):
        return self.edition_list


class FakeGoogle:
    def __init__(self, found=()):
        self.found = list(found)
        self.calls = 0

    async def search(self, title, author):
        self.calls += 1
        return self.found


def fakes(found=(), editions=()):
    return Catalogs(FakeLibrary(editions), FakeGoogle(found))


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
    result = await resolve(reading("SAPIENS", "HARARI"), ("SAPIENS HARARI", 0.9), None, fakes(library))
    assert result.status == "unidentified"


async def test_partial_read_does_not_match_a_longer_title():
    library = [cand("Homo Deus")]
    result = await resolve(reading("HOMO", "HARARI"), ("HOMO HARARI", 0.9), None, fakes(library))
    assert result.status == "unidentified"


async def test_short_generic_title_is_unidentified():
    library = [cand("It", "Stephen King"), cand("It", "Someone Else")]
    result = await resolve(reading("IT"), ("IT", 0.9), None, fakes(library))
    assert result.status == "unidentified"


async def test_author_only_read_is_unidentified():
    result = await resolve(reading("", "HARARI"), ("HARARI", 0.9), None, fakes([cand("Sapiens")]))
    assert result.status == "unidentified"
    assert result.reason == "illegible"


async def test_lone_catalogue_hit_needs_a_stricter_score(monkeypatch):
    sapiens = reading("SAPIENS", "HARARI")
    lone = rank(sapiens, [cand("Sapiens")])
    assert accept_identification(sapiens, ("SAPIENS HARARI", 0.9), lone)
    gates_with(monkeypatch, lone_score_min=101)
    assert not accept_identification(sapiens, ("SAPIENS HARARI", 0.9), lone)


@pytest.mark.parametrize(
    "changes",
    [
        {"ocr_overlap_min": 101, "catalogue_only_score": 101},
        {"ocr_overlap_min": 101, "catalogue_only_margin": 101},
        {"score_min": 101},
        {"margin_min": 101},
        {"confident_score": 101},
    ],
)
async def test_each_gate_can_reject_on_its_own(monkeypatch, catalogs, changes):
    spine = reading("SAPIENS", "HARARI")
    ranked = rank(spine, await catalogs.google_books.search("sapiens", "harari"))[:2]
    assert accept_identification(spine, ("SAPIENS HARARI", 0.9), ranked)
    gates_with(monkeypatch, **changes)
    assert not accept_identification(spine, ("SAPIENS HARARI", 0.9), ranked)


async def test_one_distinctive_token_without_author_needs_the_gate_relaxed(monkeypatch, catalogs):
    spine = reading("SAPIENS")
    ranked = rank(spine, await catalogs.google_books.search("sapiens", "harari"))[:2]
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


async def test_ambiguous_publisher_prices_an_equivalent_edition_of_that_publisher(catalogs):
    found = await catalogs.google_books.search("sapiens", "harari")
    editions = [
        Edition(publisher="Penguin", isbn="9780000000111", url="u1", year=2014),
        Edition(publisher="Penguin", isbn="9780000000222", url="u2", year=2019),
        Edition(publisher="Vintage", isbn="9780000000333", url="u3", year=2023),
    ]
    result = await resolve(
        reading("SAPIENS", "HARARI", "PENGUIN"), ("SAPIENS HARARI", 0.9), None, fakes(found, editions=editions)
    )
    assert result.status == "identified"
    assert result.edition_unverified and result.equivalent_edition
    assert result.edition == "Penguin"
    assert result.isbn == "9780000000222"


async def test_query_keeps_punctuation(tmp_path, monkeypatch):
    seen = []

    class Spy(FakeGoogle):
        async def search(self, title, author):
            seen.append(title)
            return []

    catalogs = Catalogs(FakeLibrary(), Spy())
    await resolve(reading("C++ PRIMER", "LIPPMAN"), ("C++ PRIMER LIPPMAN", 0.9), None, catalogs)
    assert seen == ["c++ primer"]


async def test_picked_edition_carries_open_library_dimensions(catalogs):
    found = await catalogs.google_books.search("sapiens", "harari")
    editions = [Edition(publisher="Vintage", isbn="9780099590088", url="u1", physical_dimensions="19.8 x 12.9 x 3 cm")]
    result = await resolve(
        reading("SAPIENS", "HARARI", "VINTAGE"), ("SAPIENS HARARI VINTAGE", 0.9), None, fakes(found, editions=editions)
    )
    assert result.isbn == "9780099590088"
    assert result.physical_dimensions == "19.8 x 12.9 x 3 cm"


class StubBooks:
    def __init__(self, known):
        self.known = known

    async def search(self, title, author):
        return [
            Candidate(source="googlebooks", title=known, author="Sven Hedin", work="g1", url="https://books.google.com/g1")
            for known in self.known
            if title == known.casefold()
        ]


async def test_volume_number_on_spine_falls_back_to_the_bare_title():
    stub = Catalogs(FakeLibrary(), StubBooks(["Transhimalaja"]))
    result = await resolve(reading("Transhimalaja 2", "Sven Hedin"), ("Transhimalaja 2 Sven Hedin", 0.9), None, stub)
    assert result.status == "identified"
    assert result.title == "Transhimalaja"


async def test_all_digit_title_is_not_stripped():
    stub = Catalogs(FakeLibrary(), StubBooks(["Transhimalaja"]))
    result = await resolve(reading("1984", "Sven Hedin"), ("1984 Sven Hedin", 0.9), None, stub)
    assert result.reason == "no_catalogue_match"


def blyton(title, **fields):
    return cand(title, "Enid Blyton", **fields)


SKYTOP = [blyton("The Secret of Skytop Hill and Other Stories"), blyton("The Secret of Skytop Hill and Other Stories"),
          blyton("The Secret of Spiggy Holes")]


async def test_empty_ocr_with_a_strong_catalogue_match_is_identified_from_the_catalogue_only():
    spine = reading("The Secret of Skytop Hill and other stories", "Enid Blyton", "Award")
    result = await resolve(spine, ("", 0.0), None, fakes(SKYTOP))
    assert result.status == "identified"
    assert result.title == "The Secret of Skytop Hill and Other Stories"
    assert result.corroboration == "catalogue_only"
    assert 0 < result.id_confidence <= 0.5


async def test_ocr_corroborated_read_records_ocr_and_keeps_the_full_confidence():
    spine = reading("The Secret of Skytop Hill and other stories", "Enid Blyton")
    result = await resolve(spine, ("The Secret of Skytop Hill and other stories", 0.9), None, fakes(SKYTOP))
    assert result.corroboration == "ocr"
    assert result.id_confidence > 0.9


async def test_hallucinated_title_that_matches_nothing_is_unidentified():
    spine = reading("The Moonlit Lighthouse of Kerrigan Bay", "Enid Blyton")
    result = await resolve(spine, ("", 0.0), None, fakes(SKYTOP))
    assert result.status == "unidentified"
    assert result.corroboration == ""


async def test_real_title_by_a_different_author_is_unidentified():
    spine = reading("The Secret of Skytop Hill and other stories", "Enid Blyton")
    other = [cand("The Secret of Skytop Hill and Other Stories", "Margaret Jennings"),
             cand("The Secret of Skytop Hill and Other Stories", "Margaret Jennings")]
    result = await resolve(spine, ("", 0.0), None, fakes(other))
    assert result.status == "unidentified"


@pytest.mark.parametrize("title", ["IT", "EMMA"])
async def test_short_generic_title_without_author_and_without_ocr_is_unidentified(title):
    library = [cand(title.title(), "Stephen King"), cand(title.title(), "Jane Austen")]
    result = await resolve(reading(title), ("", 0.0), None, fakes(library))
    assert result.status == "unidentified"
    assert result.reason == "not_corroborated"


async def test_ocr_that_reads_other_words_blocks_the_catalogue_only_path():
    spine = reading("The Secret of Skytop Hill and other stories", "Enid Blyton")
    result = await resolve(spine, ("ZEBRA CROSSING TIMETABLE", 0.9), None, fakes(SKYTOP))
    assert result.status == "unidentified"
    assert result.reason == "ocr_disagrees"


async def test_catalogue_only_needs_a_clear_margin():
    spine = reading("The Secret Seven", "Enid Blyton")
    library = [blyton("The Secret Seven"), blyton("The Secret Seven Win Through")]
    result = await resolve(spine, ("", 0.0), None, fakes(library))
    assert result.status == "unidentified"


async def test_partial_ocr_support_still_allows_the_catalogue_path():
    spine = reading("Ein ewiger Traum", "Borges", "Hanser")
    library = [cand("Ein ewiger Traum", "Jorge Luis Borges"), cand("Ein ewiger Traum", "Jorge Luis Borges")]
    result = await resolve(spine, ("Hanser Traum", 0.9), None, fakes(library))
    assert result.status == "identified"
    assert result.corroboration == "catalogue_only"


NECKLACE = [blyton("The Secret of the Lost Necklace", year=2014)]


def edition(isbn, year, publisher="Hodder", title="The Secret of the Lost Necklace", physical_format="Paperback"):
    return Edition(publisher=publisher, isbn=isbn, url=f"https://openlibrary.org/books/{isbn}", year=year, title=title,
                   physical_format=physical_format)


async def test_unknown_printed_edition_takes_the_most_recent_isbn_edition_as_equivalent():
    editions = [edition("9780000000001", 2001), edition("9780000000003", 2019, physical_format="Hardcover"),
                edition("9780000000002", 2016), edition("9780000000004", 2022, title="Das Geheimnis der Halskette")]
    spine = reading("The Secret of the LOST NECKLACE", "Enid Blyton")
    result = await resolve(spine, ("The Secret of the LOST NECKLACE", 0.9), None, fakes(NECKLACE, editions=editions))
    assert result.status == "identified"
    assert result.isbn == "9780000000002"
    assert result.edition == "" and result.edition_unverified and result.equivalent_edition
    assert result.year == 2016
    assert result.work_year == 2014
    assert "https://openlibrary.org/books/9780000000002" in result.source_urls


async def test_printed_publisher_that_matches_one_edition_is_that_edition_not_an_equivalent():
    editions = [edition("9780000000001", 2001, publisher="Award Publications"), edition("9780000000002", 2016)]
    spine = reading("The Secret of the LOST NECKLACE", "Enid Blyton", "AWARD")
    result = await resolve(spine, ("The Secret of the LOST NECKLACE", 0.9), None, fakes(NECKLACE, editions=editions))
    assert result.isbn == "9780000000001"
    assert result.edition == "Award Publications"
    assert not result.equivalent_edition and not result.edition_unverified


async def test_no_edition_with_an_isbn_leaves_the_isbn_empty():
    spine = reading("The Secret of the LOST NECKLACE", "Enid Blyton")
    result = await resolve(spine, ("The Secret of the LOST NECKLACE", 0.9), None, fakes(NECKLACE))
    assert result.status == "identified"
    assert result.isbn == "" and not result.equivalent_edition


def test_ocr_words_run_together_still_support_the_read():
    spine = reading("Gemeinsame Werke 1", "J.L.Borges")
    assert ocr_overlap(spine, "Hanser GemeinsameWerke1 J.L.Borges") == 1.0


def test_short_title_is_not_supported_by_a_longer_ocr_word():
    assert ocr_overlap(reading("IT"), "WITCHES") == 0.0


async def test_series_name_run_together_by_ocr_does_not_block_the_catalogue_path():
    spine = reading("Five on a Treasure Island", "Enid Blyton", "The Famous Five")
    library = [blyton("Five on a Treasure Island"), blyton("Five on a Treasure Island"), blyton("Five Go to Camp")]
    result = await resolve(spine, ("FAMOUSFIVE", 1.0), None, fakes(library))
    assert result.status == "identified"
    assert result.corroboration == "catalogue_only"


async def test_recorded_blyton_spine_with_empty_ocr_is_identified_from_the_catalogue(catalogs):
    spine = reading("The Secret of the LOST NECKLACE", "Enid Blyton", "AWARD")
    result = await resolve(spine, ("", 0.0), None, catalogs)
    assert result.status == "identified"
    assert result.title == "The Secret of the Lost Necklace"
    assert result.corroboration == "catalogue_only" and result.id_confidence == 0.5
    assert result.equivalent_edition and result.edition == "" and result.isbn == "9780753727041"


async def test_recorded_google_books_reply_identifies_well_done_secret_seven_without_ocr(catalogs):
    result = await resolve(reading("Well Done, Secret Seven", "Enid Blyton"), ("", 0.0), None, catalogs)
    assert result.status == "identified"
    assert result.corroboration == "catalogue_only"
    assert result.equivalent_edition and len(result.isbn) == 13


async def test_recorded_runner_up_one_word_longer_keeps_amelia_jane_unidentified_without_ocr(catalogs):
    result = await resolve(reading("Naughty Amelia Jane", "Enid Blyton"), ("", 0.0), None, catalogs)
    assert result.status == "unidentified"
    assert result.reason == "not_corroborated"
    assert result.margin < load_resolve_config().gates.catalogue_only_margin


async def test_no_request_goes_to_a_disallowed_open_library_search_path(tmp_path, monkeypatch):
    monkeypatch.delenv("LIBCLAIM_OFFLINE")
    seen = []

    def handler(request):
        seen.append(request.url)
        if request.url.host == "www.googleapis.com":
            body = {"items": [{"id": "a", "volumeInfo": {"title": "Sapiens", "authors": ["Yuval Noah Harari"],
                                                         "industryIdentifiers": [{"type": "ISBN_13", "identifier": "9780000000002"}]}}]}
        elif request.url.path.startswith("/isbn/"):
            body = {"works": [{"key": "/works/OL1W"}]}
        else:
            body = {"entries": []}
        return httpx.Response(200, json=body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    http = CatalogHttp(tmp_path / "cache", client=client)
    catalogs = Catalogs(OpenLibrary(http), GoogleBooks(http, api_key=""))
    result = await resolve(reading("SAPIENS", "HARARI"), ("SAPIENS HARARI", 0.9), None, catalogs)
    assert result.status == "identified"
    assert any(url.host == "openlibrary.org" for url in seen)
    assert not any(url.host == "openlibrary.org" and url.path.startswith("/search") for url in seen)


def test_shown_title_drops_an_author_name_only_when_the_spine_agrees():
    from libclaim.vision.resolve import shown_title
    by = cand("Chartres, By George Henderson", author="George Henderson")
    assert shown_title(reading("CHARTRES", "GEORGE HENDERSON"), by) == "Chartres"
    keats = cand("Selected Poems of John Keats", author="John Keats")
    assert shown_title(reading("SELECTED POEMS OF JOHN KEATS"), keats) == "Selected Poems of John Keats"


def test_shown_title_keeps_the_printed_words_in_catalogue_casing():
    from libclaim.vision.resolve import shown_title
    mosella = cand("Die Mosella des Decimus Magnus Ausonius", author="Decimus Magnus Ausonius")
    assert shown_title(reading("MOSELLA", "AUSONIUS"), mosella) == "Mosella"
    misread = cand("Die Mosella des Decimus Magnus Ausonius", author="Decimus Magnus Ausonius")
    assert shown_title(reading("MOSELA", "AUSONIUS"), misread) == "Die Mosella"


async def test_catalogue_title_with_the_author_appended_is_identified_without_ocr():
    library = [cand("Chartres, By George Henderson", author="George Henderson")]
    result = await resolve(reading("Chartres", "George Henderson"), ("George Henderson Charea", 0.9), None, fakes(library))
    assert result.status == "identified"
    assert result.corroboration == "catalogue_only"
    assert result.title == "Chartres"

