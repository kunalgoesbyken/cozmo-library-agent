import logging
from pathlib import Path

import httpx
import pytest

from libclaim.catalog.googlebooks import GoogleBooks
from libclaim.catalog.http import CatalogBadResponse, CatalogHttp, CatalogOffline, guarded, year_of
from libclaim.catalog.openlibrary import OpenLibrary

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "catalog"
KEY = "AIzaTestKeyNotReal123456"
BODY = b'{"items": [{"id": "x", "volumeInfo": {"title": "Sapiens", "authors": ["Harari"]}}], "docs": []}'


def http_with(tmp_path, handler):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), headers={"User-Agent": "test"})
    return CatalogHttp(tmp_path / "cache", client=client)


@pytest.fixture
def online(monkeypatch):
    monkeypatch.delenv("LIBCLAIM_OFFLINE")


def cache_text(tmp_path):
    return "".join(p.read_text(encoding="utf-8") + p.name for p in (tmp_path / "cache").glob("*"))


async def test_key_goes_in_header_and_never_in_url_or_cache(tmp_path, online):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, content=BODY)

    http = http_with(tmp_path, handler)
    found = await GoogleBooks(http, api_key=KEY).search("sapiens", "harari")
    assert found[0].title == "Sapiens"
    assert seen[0].headers["x-goog-api-key"] == KEY
    assert KEY not in str(seen[0].url)
    assert KEY not in cache_text(tmp_path)


async def test_second_call_is_served_from_cache(tmp_path, online):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, content=BODY)

    google = GoogleBooks(http_with(tmp_path, handler), api_key=KEY)
    first = await google.search("sapiens", "harari")
    second = await google.search("sapiens", "harari")
    assert first == second
    assert len(calls) == 1


@pytest.mark.parametrize("failure", ["status", "connect"])
async def test_failures_never_leak_the_key(tmp_path, online, caplog, failure):
    def handler(request):
        if failure == "connect":
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(429)

    http = http_with(tmp_path, handler)
    with caplog.at_level(logging.DEBUG):
        google, google_down = await guarded(GoogleBooks(http, api_key=KEY).search("sapiens", "harari"), "google books search")
        library, library_down = await guarded(OpenLibrary(http).editions("OL1W"), "open library editions")
    assert google == [] and library == []
    assert google_down and library_down
    assert KEY not in caplog.text
    assert KEY not in cache_text(tmp_path)
    assert "google books search failed" in caplog.text


async def test_status_error_message_has_no_key(tmp_path, online):
    http = http_with(tmp_path, lambda request: httpx.Response(429))
    with pytest.raises(httpx.HTTPStatusError) as raised:
        await http.get_json("https://www.googleapis.com/books/v1/volumes", {"q": "x"}, {"X-goog-api-key": KEY})
    assert KEY not in str(raised.value)


async def test_offline_miss_raises_and_fixtures_replay(tmp_path):
    http = CatalogHttp(FIXTURES)
    found = await GoogleBooks(http, api_key="").search("sapiens", "harari")
    assert found[0].title == "Sapiens"
    assert await OpenLibrary(http).work_of(found[0].isbn) == "OL17075811W"
    with pytest.raises(CatalogOffline):
        await GoogleBooks(http, api_key="").search("never recorded", "")


def test_fixtures_hold_no_credentials():
    for path in FIXTURES.glob("*.json"):
        assert "AIza" not in path.read_text(encoding="utf-8")


async def test_html_reply_is_not_cached_and_is_reported(tmp_path, online):
    http = http_with(tmp_path, lambda request: httpx.Response(200, content=b"<html>verify you are human</html>"))
    found, down = await guarded(GoogleBooks(http, api_key="").search("sapiens", "harari"), "google books search")
    assert found == [] and down
    assert not list((tmp_path / "cache").glob("*"))
    with pytest.raises(CatalogBadResponse):
        await http.get_json("https://openlibrary.org/works/OL1W/editions.json", {"limit": 1})


async def test_edition_with_empty_publishers_keeps_its_isbn(tmp_path, online):
    body = b'{"entries": [{"key": "/books/OL1M", "publishers": [], "isbn_13": ["9780000000002"]}]}'
    http = http_with(tmp_path, lambda request: httpx.Response(200, content=body))
    editions = await OpenLibrary(http).editions("OL1W")
    assert [(e.publisher, e.isbn) for e in editions] == [("", "9780000000002")]


@pytest.mark.parametrize(
    ("text", "year"), [("2022", 2022), ("March 5, 2010", 2010), ("1998-03", 1998), ("c. 1850", 1850), ("", None), ("n.d.", None)]
)
def test_year_of_publish_date_text(text, year):
    assert year_of(text) == year


async def test_edition_year_comes_from_its_own_publish_date(tmp_path, online):
    body = b'{"entries": [{"key": "/books/OL1M", "publish_date": "May 2, 1931", "isbn_13": ["9780000000002"]}]}'
    http = http_with(tmp_path, lambda request: httpx.Response(200, content=body))
    editions = await OpenLibrary(http).editions("OL1W")
    assert [e.year for e in editions] == [1931]


async def test_edition_keeps_its_physical_dimensions_text(tmp_path, online):
    body = (
        b'{"entries": [{"key": "/books/OL1M", "isbn_13": ["9780000000002"], "physical_dimensions": "20 x 13 x 2 cm"},'
        b' {"key": "/books/OL2M", "isbn_13": ["9780000000019"]}]}'
    )
    http = http_with(tmp_path, lambda request: httpx.Response(200, content=body))
    editions = await OpenLibrary(http).editions("OL1W")
    assert [e.physical_dimensions for e in editions] == ["20 x 13 x 2 cm", ""]


async def test_work_of_isbn_reads_the_work_key_from_the_isbn_endpoint(tmp_path, online):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, content=b'{"works": [{"key": "/works/OL1W"}], "isbn_13": ["9780000000002"]}')

    assert await OpenLibrary(http_with(tmp_path, handler)).work_of("9780000000002") == "OL1W"
    assert seen == ["https://openlibrary.org/isbn/9780000000002.json"]


async def test_work_of_isbn_without_works_is_empty(tmp_path, online):
    http = http_with(tmp_path, lambda request: httpx.Response(200, content=b'{"isbn_13": ["9780000000002"]}'))
    assert await OpenLibrary(http).work_of("9780000000002") == ""


def test_open_library_has_no_search_because_robots_disallows_it():
    assert not hasattr(OpenLibrary, "search")
