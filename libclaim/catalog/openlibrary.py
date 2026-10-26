from libclaim.catalog.http import CatalogHttp, Edition, year_of
from libclaim.resolve_config import load_resolve_config

WORKS_PREFIX = "/works/"
ISBN_13 = "isbn_13"


class OpenLibrary:
    def __init__(self, http: CatalogHttp):
        self.http = http
        self.config = load_resolve_config().catalog

    async def work_of(self, isbn: str) -> str:
        cfg = self.config.openlibrary
        body = await self.http.get_json(cfg.isbn_url.format(isbn=isbn), {})
        works = body.get("works") or [{}]
        return works[0].get("key", "").removeprefix(WORKS_PREFIX)

    async def editions(self, work: str) -> list[Edition]:
        cfg = self.config.openlibrary
        url = cfg.editions_url.format(work=work)
        body = await self.http.get_json(url, {"limit": self.config.editions_limit})
        return [
            Edition(
                publisher=(entry.get("publishers") or [""])[0],
                isbn=entry[ISBN_13][0],
                url=cfg.site_url + entry["key"],
                year=year_of(str(entry.get("publish_date", ""))),
                physical_dimensions=str(entry.get("physical_dimensions", "")),
                title=str(entry.get("title", "")),
                physical_format=str(entry.get("physical_format", "")),
            )
            for entry in body.get("entries", [])
            if entry.get(ISBN_13) and entry.get("key")
        ]
