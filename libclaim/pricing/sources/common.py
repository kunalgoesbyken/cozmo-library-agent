from libclaim.models import Evidence, NotFound
from libclaim.pricing.http import CachedResponse

OK = 200
SCHEMA_PREFIX_SEPARATOR = "/"


def evidence_of(response: CachedResponse, isbn_on_page: str = "") -> Evidence:
    return Evidence(
        url=response.final_url,
        cache_ref=response.cache_ref,
        retrieved_at=response.retrieved_at,
        isbn_on_page=isbn_on_page,
    )


def not_found(source: str, isbn: str, reason: str, response: CachedResponse | None = None, isbn_on_page: str = "") -> NotFound:
    evidence = evidence_of(response, isbn_on_page) if response else None
    return NotFound(source=source, isbn=isbn, reason=reason, evidence=evidence)


def schema_name(value: object) -> str:
    return str(value or "").rsplit(SCHEMA_PREFIX_SEPARATOR, 1)[-1]
