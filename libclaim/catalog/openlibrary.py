from libclaim.catalog.http import Candidate, CatalogHttp, Edition, year_of
from libclaim.resolve_config import load_resolve_config

SOURCE = "openlibrary"
WORKS_PREFIX = "/works/"
ISBN_13 = "isbn_13"


class OpenLibrary:
    def __init__(self, http: CatalogHttp):
        self.http = http
        self.config = load_resolve_config().catalog

    async def search(self, title: str, author: str) -> list[Candidate]:
        cfg = self.config.openlibrary
        params = {"title": title, "limit": self.config.search_limit, "fields": cfg.search_fields}
        if author:
            params["author"] = author
        body = await self.http.get_json(cfg.search_url, params)
        return [
            Candidate(
                source=SOURCE,
                title=doc.get("title", ""),
                author=", ".join(doc.get("author_name", [])),
                year=doc.get("first_publish_year"),
                url=cfg.site_url + doc["key"],
                work=doc["key"].removeprefix(WORKS_PREFIX),
            )
            for doc in body.get("docs", [])
            if doc.get("key")
        ]

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
            )
            for entry in body.get("entries", [])
            if entry.get(ISBN_13) and entry.get("key")
        ]
