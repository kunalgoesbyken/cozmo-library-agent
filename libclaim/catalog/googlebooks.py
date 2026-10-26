import logging

import httpx

from libclaim.catalog.http import CatalogHttp, Candidate, describe
from libclaim.resolve_config import load_resolve_config
from libclaim.settings import load_settings

logger = logging.getLogger(__name__)

SOURCE = "googlebooks"
ISBN_13 = "ISBN_13"


def query_for(title: str, author: str) -> str:
    query = f"intitle:{title}"
    return f"{query} inauthor:{author}" if author else query


class GoogleBooks:
    def __init__(self, http: CatalogHttp, api_key: str | None = None):
        self.http = http
        self.config = load_resolve_config().catalog
        self.api_key = load_settings().google_books_api_key if api_key is None else api_key

    async def search(self, title: str, author: str) -> list[Candidate]:
        cfg = self.config.googlebooks
        params = {"q": query_for(title, author), "maxResults": self.config.search_limit}
        headers = {cfg.key_header: self.api_key} if self.api_key else None
        try:
            body = await self.http.get_json(cfg.volumes_url, params, headers)
        except httpx.HTTPError as error:
            logger.warning("google books search failed: %s", describe(error))
            return []
        candidates = []
        for item in body.get("items", []):
            info = item.get("volumeInfo", {})
            if not info.get("title"):
                continue
            isbns = [i["identifier"] for i in info.get("industryIdentifiers", []) if i.get("type") == ISBN_13]
            candidates.append(
                Candidate(
                    source=SOURCE,
                    title=info["title"],
                    author=", ".join(info.get("authors", [])),
                    publisher=info.get("publisher", ""),
                    isbn=isbns[0] if isbns else "",
                    year=int(info["publishedDate"][:4]) if info.get("publishedDate", "")[:4].isdigit() else None,
                    url=info.get("infoLink", ""),
                    work=item.get("id", ""),
                )
            )
        return candidates
