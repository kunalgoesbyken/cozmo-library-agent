import logging
from pathlib import Path

import httpx
import pytest

from libclaim.catalog.googlebooks import GoogleBooks
from libclaim.catalog.http import CatalogHttp, CatalogOffline
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
        google = await GoogleBooks(http, api_key=KEY).search("sapiens", "harari")
        library = await OpenLibrary(http).search("sapiens", "harari")
    assert google == [] and library == []
    assert KEY not in caplog.text
    assert KEY not in cache_text(tmp_path)
    assert all(KEY not in str(record.args) + record.getMessage() for record in caplog.records)
    assert "google books search failed" in caplog.text


async def test_status_error_message_has_no_key(tmp_path, online):
    http = http_with(tmp_path, lambda request: httpx.Response(429))
    with pytest.raises(httpx.HTTPStatusError) as raised:
        await http.get_json("https://www.googleapis.com/books/v1/volumes", {"q": "x"}, {"X-goog-api-key": KEY})
    assert KEY not in str(raised.value)


async def test_offline_miss_raises_and_fixtures_replay(tmp_path):
    http = CatalogHttp(FIXTURES)
    found = await OpenLibrary(http).search("sapiens", "harari")
    assert found[0].title == "Sapiens"
    with pytest.raises(CatalogOffline):
        await OpenLibrary(http).search("never recorded", "")


def test_fixtures_hold_no_credentials():
    for path in FIXTURES.glob("*.json"):
        assert "AIza" not in path.read_text(encoding="utf-8")
