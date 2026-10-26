import logging

import httpx

from libclaim.catalog.http import CatalogHttp, Candidate, Edition, describe
from libclaim.resolve_config import load_resolve_config

logger = logging.getLogger(__name__)

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
        try:
            body = await self.http.get_json(cfg.search_url, params)
        except httpx.HTTPError as error:
            logger.warning("open library search failed: %s", describe(error))
            return []
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
        try:
            body = await self.http.get_json(url, {"limit": self.config.editions_limit})
        except httpx.HTTPError as error:
            logger.warning("open library editions failed: %s", describe(error))
            return []
        return [
            Edition(publisher=publisher, isbn=entry[ISBN_13][0], url=cfg.site_url + entry["key"])
            for entry in body.get("entries", [])
            if entry.get(ISBN_13) and entry.get("key")
            for publisher in entry.get("publishers", [""])[:1]
        ]
