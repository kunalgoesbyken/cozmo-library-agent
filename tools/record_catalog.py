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


CASES = [
    Case(title="sapiens", author="harari", editions=True),
    Case(title="pride and prejudice", author="austen", editions=True),
    Case(title="the secret of the lost necklace", author="enid blyton", editions=True),
    Case(title="xqzvk plorbwen"),
    Case(title="the psychology of mony", author="housel"),
]


async def record(cache_dir: Path) -> None:
    http = CatalogHttp(cache_dir)
    try:
        open_library = OpenLibrary(http)
        google_books = GoogleBooks(http)
        for case in CASES:
            found = await google_books.search(case.title, case.author)
            isbn = next((candidate.isbn for candidate in found if candidate.isbn), "")
            if case.editions and isbn:
                work = await open_library.work_of(isbn)
                if work:
                    await open_library.editions(work)
    finally:
        await http.aclose()


if __name__ == "__main__":
    asyncio.run(record(FIXTURES))
