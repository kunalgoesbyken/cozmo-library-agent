from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from rapidfuzz import fuzz, process
from rapidfuzz.utils import default_process

from libclaim.catalog.googlebooks import GoogleBooks
from libclaim.catalog.http import Candidate, CatalogHttp, Edition, guarded
from libclaim.catalog.googlebooks import SOURCE as GOOGLE_BOOKS
from libclaim.catalog.openlibrary import SOURCE as OPEN_LIBRARY
from libclaim.catalog.openlibrary import OpenLibrary
from libclaim.resolve_config import load_resolve_config
from libclaim.vision.reading import Reading

WORD_SEPARATOR = " "


class SpineDims(BaseModel):
    height_cm: float | None = None
    thickness_cm: float | None = None


class Scored(BaseModel):
    best: Candidate
    members: list[Candidate]
    score: float
    author_matched: bool
    distinctive: int

    @property
    def sources(self) -> set[str]:
        return {member.source for member in self.members}


class Resolution(BaseModel):
    status: Literal["identified", "unidentified"]
    title: str = ""
    author: str = ""
    edition: str = ""
    isbn: str = ""
    year: int | None = None
    edition_unverified: bool = False
    id_confidence: float = 0.0
    match_score: float = 0.0
    margin: float = 0.0
    source_urls: list[str] = []
    reason: str = ""
    vlm_text: str = ""
    ocr_text: str = ""
    dims: SpineDims | None = None


@dataclass
class Catalogs:
    open_library: OpenLibrary
    google_books: GoogleBooks

    @classmethod
    def for_sweep(cls, sweep_root: Path) -> "Catalogs":
        http = CatalogHttp(Path(sweep_root) / load_resolve_config().catalog.cache_dir)
        return cls(OpenLibrary(http), GoogleBooks(http))


def tokens(text: str) -> list[str]:
    return default_process(text).split()


def query_text(text: str) -> str:
    return WORD_SEPARATOR.join(text.casefold().split())


def read_text(reading: Reading) -> str:
    return WORD_SEPARATOR.join(part for part in (reading.title_verbatim, reading.author_verbatim) if part)


def countable_title_tokens(reading: Reading) -> list[str]:
    gates = load_resolve_config().gates
    title = tokens(reading.title_verbatim)
    author = set(tokens(reading.author_verbatim))
    kept = [t for t in title if len(t) >= gates.distinctive_min_chars and t not in author]
    return kept or title


def matches_any(token: str, pool: list[str], cutoff: float) -> bool:
    return bool(pool) and process.extractOne(token, pool, scorer=fuzz.ratio, score_cutoff=cutoff) is not None


def ocr_overlap(reading: Reading, ocr_text: str) -> float:
    wanted = countable_title_tokens(reading)
    seen = tokens(ocr_text)
    if not wanted or not seen:
        return 0.0
    cutoff = load_resolve_config().gates.token_match
    return sum(1 for token in wanted if matches_any(token, seen, cutoff)) / len(wanted)


def author_matches(reading: Reading, candidate: Candidate) -> bool:
    gates = load_resolve_config().gates
    if not tokens(reading.author_verbatim) or not candidate.author:
        return False
    return fuzz.token_set_ratio(reading.author_verbatim, candidate.author, processor=default_process) >= gates.author_match


def unmatched(source: list[str], other: list[str]) -> int:
    gates = load_resolve_config().gates
    return sum(
        1 for t in source if len(t) >= gates.distinctive_min_chars and not matches_any(t, other, gates.token_match)
    )


def title_similarity(a: str, b: str) -> float:
    gates = load_resolve_config().gates
    left, right = tokens(a), tokens(b)
    base = fuzz.token_set_ratio(a, b, processor=default_process)
    return max(0.0, base - gates.extra_token_penalty * (unmatched(left, right) + unmatched(right, left)))


def score_candidate(reading: Reading, candidate: Candidate) -> float:
    gates = load_resolve_config().gates
    title = title_similarity(reading.title_verbatim, candidate.title)
    if not tokens(reading.author_verbatim):
        return title
    author = fuzz.token_set_ratio(reading.author_verbatim, candidate.author, processor=default_process)
    return (1 - gates.author_weight) * title + gates.author_weight * author


def distinctive_tokens(reading: Reading, candidate: Candidate) -> int:
    gates = load_resolve_config().gates
    wanted = tokens(reading.title_verbatim)
    return sum(
        1
        for token in set(tokens(candidate.title))
        if len(token) >= gates.distinctive_min_chars and matches_any(token, wanted, gates.token_match)
    )


def same_work(a: Candidate, b: Candidate) -> bool:
    gates = load_resolve_config().gates
    if title_similarity(a.title, b.title) < gates.agree_score:
        return False
    if not a.author or not b.author:
        return True
    return fuzz.token_set_ratio(a.author, b.author, processor=default_process) >= gates.author_match


def rank(reading: Reading, candidates: list[Candidate]) -> list[Scored]:
    groups: list[list[Candidate]] = []
    for candidate in candidates:
        group = next((g for g in groups if same_work(g[0], candidate)), None)
        if group is None:
            groups.append([candidate])
        else:
            group.append(candidate)
    scored = []
    for group in groups:
        best = max(group, key=lambda member: score_candidate(reading, member))
        scored.append(
            Scored(
                best=best,
                members=group,
                score=score_candidate(reading, best),
                author_matched=any(author_matches(reading, member) for member in group),
                distinctive=distinctive_tokens(reading, best),
            )
        )
    return sorted(scored, key=lambda s: s.score, reverse=True)


def margin_of(candidates: list[Scored]) -> float:
    if not candidates:
        return 0.0
    return candidates[0].score - (candidates[1].score if len(candidates) > 1 else 0.0)


def accept_identification(reading: Reading, ocr: tuple[str, float], candidates: list[Scored]) -> bool:
    gates = load_resolve_config().gates
    if not candidates or ocr_overlap(reading, ocr[0]) < gates.ocr_overlap_min:
        return False
    top = candidates[0]
    if top.score < gates.score_min or margin_of(candidates) < gates.margin_min:
        return False
    if top.distinctive < gates.distinctive_min_tokens and not top.author_matched:
        return False
    if len(candidates) == 1 and len(top.sources) == 1 and top.score < gates.lone_score_min:
        return False
    return len(top.sources) > 1 or top.score >= gates.confident_score


def pick_edition(reading: Reading, editions: list[Edition]) -> tuple[Edition | None, bool]:
    gates = load_resolve_config().gates
    if not tokens(reading.publisher_verbatim):
        return None, False
    scored = [
        (
            fuzz.token_set_ratio(reading.publisher_verbatim, e.publisher, processor=default_process),
            fuzz.ratio(reading.publisher_verbatim, e.publisher, processor=default_process),
            e,
        )
        for e in editions
    ]
    best = max(((a, b) for a, b, _ in scored), default=(0.0, 0.0))
    if best[0] < gates.publisher_match:
        return None, True
    top = [e for a, b, e in scored if (a, b) == best]
    if len({e.isbn for e in top}) > 1:
        return None, True
    return top[0], False


async def edition_options(reading: Reading, top: Scored, catalogs: Catalogs) -> tuple[list[Edition], bool]:
    options = [Edition(publisher=m.publisher, isbn=m.isbn, url=m.url) for m in top.members if m.publisher and m.isbn]
    library = next((m for m in top.members if m.source == OPEN_LIBRARY), None)
    if library is None or not tokens(reading.publisher_verbatim):
        return options, False
    fetched, down = await guarded(catalogs.open_library.editions(library.work), "open library editions")
    return options + fetched, down


def unidentified(base: Resolution, reason: str, candidates: list[Scored]) -> Resolution:
    score = candidates[0].score if candidates else 0.0
    return base.model_copy(update={"reason": reason, "match_score": score, "margin": margin_of(candidates)})


async def resolve(
    reading: Reading,
    ocr: tuple[str, float],
    dims: SpineDims | None,
    catalogs: Catalogs,
) -> Resolution:
    gates = load_resolve_config().gates
    base = Resolution(status="unidentified", vlm_text=read_text(reading), ocr_text=ocr[0], dims=dims)
    if reading.failure:
        return unidentified(base, "reading_failed", [])
    if not reading.legible or not tokens(reading.title_verbatim):
        return unidentified(base, "illegible", [])
    if ocr_overlap(reading, ocr[0]) < gates.ocr_overlap_min:
        return unidentified(base, "ocr_disagrees", [])
    title = query_text(reading.title_verbatim)
    author = query_text(reading.author_verbatim)
    found, unavailable = await guarded(catalogs.open_library.search(title, author), "open library search")
    ranked = rank(reading, found)
    if not accept_identification(reading, ocr, ranked):
        more, down = await guarded(catalogs.google_books.search(title, author), "google books search")
        found, unavailable = found + more, unavailable or down
        ranked = rank(reading, found)
    if not accept_identification(reading, ocr, ranked):
        if unavailable:
            return unidentified(base, "catalogue_unavailable", ranked)
        return unidentified(base, "not_confident" if ranked else "no_catalogue_match", ranked)
    top = ranked[0]
    options, editions_down = await edition_options(reading, top, catalogs)
    edition, unverified = pick_edition(reading, options)
    urls = list(dict.fromkeys(m.url for m in top.members if m.url))
    if edition is not None and edition.url not in urls:
        urls.append(edition.url)
    return base.model_copy(
        update={
            "status": "identified",
            "title": top.best.title,
            "author": top.best.author,
            "edition": edition.publisher if edition else "",
            "isbn": edition.isbn if edition else "",
            "year": next((m.year for m in top.members if m.source == GOOGLE_BOOKS and m.year), None),
            "edition_unverified": unverified or editions_down,
            "id_confidence": round(top.score / 100 * ocr_overlap(reading, ocr[0]), 3),
            "match_score": top.score,
            "margin": margin_of(ranked),
            "source_urls": urls,
        }
    )
