import asyncio
from pathlib import Path

from pydantic import BaseModel

from libclaim.catalog.googlebooks import GoogleBooks
from libclaim.catalog.http import CatalogHttp
from libclaim.catalog.openlibrary import OpenLibrary
from libclaim.settings import ROOT

FIXTURES = ROOT / "tests" / "fixtures" / "catalog"


class Case(BaseModel):
    title: str
    author: str = ""
    editions: bool = False
    google: bool = False


CASES = [
    Case(title="sapiens", author="harari", editions=True),
    Case(title="pride and prejudice", author="austen", editions=True),
    Case(title="emma", google=True),
    Case(title="xqzvk plorbwen", google=True),
    Case(title="the psychology of mony", author="housel", google=True),
]


async def record(cache_dir: Path) -> None:
    http = CatalogHttp(cache_dir)
    try:
        open_library = OpenLibrary(http)
        google_books = GoogleBooks(http)
        for case in CASES:
            found = await open_library.search(case.title, case.author)
            if case.editions and found:
                await open_library.editions(found[0].work)
            if case.google:
                await google_books.search(case.title, case.author)
    finally:
        await http.aclose()


if __name__ == "__main__":
    asyncio.run(record(FIXTURES))
