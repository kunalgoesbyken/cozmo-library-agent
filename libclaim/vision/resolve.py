from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from rapidfuzz import fuzz, process
from rapidfuzz.utils import default_process

from libclaim.catalog.googlebooks import GoogleBooks
from libclaim.catalog.http import Candidate, CatalogHttp, Edition, guarded
from libclaim.catalog.openlibrary import OpenLibrary
from libclaim.resolve_config import load_resolve_config
from libclaim.vision.reading import Reading, ocr_yield

WORD_SEPARATOR = " "


class Corroboration(StrEnum):
    none = ""
    ocr = "ocr"
    catalogue_only = "catalogue_only"


class SpineDims(BaseModel):
    height_cm: float | None = None
    thickness_cm: float | None = None


class Scored(BaseModel):
    best: Candidate
    members: list[Candidate]
    score: float
    author_matched: bool
    distinctive: int


class EditionChoice(BaseModel):
    edition: Edition | None = None
    equivalent: bool = False
    publisher_known: bool = False


class Resolution(BaseModel):
    status: Literal["identified", "unidentified"]
    title: str = ""
    author: str = ""
    edition: str = ""
    isbn: str = ""
    year: int | None = None
    physical_dimensions: str = ""
    edition_unverified: bool = False
    equivalent_edition: bool = False
    work_year: int | None = None
    corroboration: Corroboration = Corroboration.none
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


def seen_in(token: str, seen: list[str]) -> bool:
    gates = load_resolve_config().gates
    if matches_any(token, seen, gates.token_match):
        return True
    return len(token) >= gates.distinctive_min_chars and fuzz.partial_ratio(token, "".join(seen)) >= gates.token_match


def ocr_overlap(reading: Reading, ocr_text: str) -> float:
    wanted = countable_title_tokens(reading)
    seen = tokens(ocr_text)
    if not wanted or not seen:
        return 0.0
    return sum(1 for token in wanted if seen_in(token, seen)) / len(wanted)


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


def title_without_author(candidate: Candidate) -> str:
    author = set(tokens(candidate.author))
    return WORD_SEPARATOR.join(t for t in tokens(candidate.title) if t not in author)


def display_title(candidate: Candidate) -> str:
    gates = load_resolve_config().gates
    author = set(tokens(candidate.author))
    words = candidate.title.split()
    removed = {i for i, w in enumerate(words) if tokens(w) and set(tokens(w)) <= author}
    if not removed:
        return candidate.title
    joining = {i for r in removed for i in (r - 1, r + 1)
               if 0 <= i < len(words) and len("".join(tokens(words[i]))) < gates.distinctive_min_chars}
    kept = [w for i, w in enumerate(words) if i not in removed | joining]
    return WORD_SEPARATOR.join(kept).strip(" ,:;-") or candidate.title


def shown_title(reading: Reading, candidate: Candidate) -> str:
    bare = display_title(candidate)
    closer = title_similarity(reading.title_verbatim, bare) > title_similarity(reading.title_verbatim, candidate.title)
    return bare if closer else candidate.title


def score_candidate(reading: Reading, candidate: Candidate) -> float:
    gates = load_resolve_config().gates
    title = title_similarity(reading.title_verbatim, candidate.title)
    bare = title_without_author(candidate)
    if bare:
        title = max(title, title_similarity(reading.title_verbatim, bare))
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


def surnames(candidate: Candidate) -> set[str]:
    gates = load_resolve_config().gates
    return {t for t in tokens(candidate.author) if len(t) >= gates.distinctive_min_chars}


def same_work(a: Candidate, b: Candidate) -> bool:
    gates = load_resolve_config().gates
    if title_similarity(title_without_author(a) or a.title, title_without_author(b) or b.title) < gates.agree_score:
        return False
    if not a.author or not b.author:
        return True
    if surnames(a) & surnames(b):
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


def interchangeable(top: Scored, other: Scored) -> bool:
    gates = load_resolve_config().gates
    same_title = title_similarity(title_without_author(top.best) or top.best.title, title_without_author(other.best) or other.best.title)
    return same_title >= gates.agree_score and top.author_matched and other.author_matched


def margin_of(candidates: list[Scored]) -> float:
    if not candidates:
        return 0.0
    rivals = [c for c in candidates[1:] if not interchangeable(candidates[0], c)]
    return candidates[0].score - (rivals[0].score if rivals else 0.0)


def catalogue_agrees(candidates: list[Scored]) -> bool:
    gates = load_resolve_config().gates
    if not candidates:
        return False
    top = candidates[0]
    if top.score < gates.score_min or margin_of(candidates) < gates.margin_min:
        return False
    if top.distinctive < gates.distinctive_min_tokens and not top.author_matched:
        return False
    if len(candidates) == 1 and top.score < gates.lone_score_min:
        return False
    return top.score >= gates.confident_score


def ocr_contradicts(reading: Reading, ocr: tuple[str, float]) -> bool:
    gates = load_resolve_config().gates
    if ocr_yield(*ocr) < gates.ocr_contradiction_yield:
        return False
    seen = tokens(ocr[0])
    read = tokens(f"{read_text(reading)} {reading.publisher_verbatim}")
    wanted = [token for token in read if len(token) >= gates.distinctive_min_chars]
    return not any(seen_in(token, seen) for token in wanted)


def catalogue_strong(reading: Reading, candidates: list[Scored]) -> bool:
    gates = load_resolve_config().gates
    top = candidates[0]
    return (
        author_matches(reading, top.best)
        and top.score >= gates.catalogue_only_score
        and title_similarity(reading.title_verbatim, top.best.title) >= gates.catalogue_only_title
        and margin_of(candidates) >= gates.catalogue_only_margin
    )


def accept_identification(reading: Reading, ocr: tuple[str, float], candidates: list[Scored]) -> Corroboration:
    gates = load_resolve_config().gates
    if not catalogue_agrees(candidates):
        return Corroboration.none
    if ocr_overlap(reading, ocr[0]) >= gates.ocr_overlap_min:
        return Corroboration.ocr
    if not ocr_contradicts(reading, ocr) and catalogue_strong(reading, candidates):
        return Corroboration.catalogue_only
    return Corroboration.none


def id_confidence(reading: Reading, ocr: tuple[str, float], top: Scored, corroboration: Corroboration) -> float:
    gates = load_resolve_config().gates
    support = ocr_overlap(reading, ocr[0]) if corroboration == Corroboration.ocr else gates.catalogue_only_confidence
    return round(top.score / 100 * support, 3)


def publisher_matches(reading: Reading, editions: list[Edition]) -> list[Edition]:
    gates = load_resolve_config().gates
    if not tokens(reading.publisher_verbatim):
        return []
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
        return []
    return [e for a, b, e in scored if (a, b) == best]


def equivalent_of(title: str, editions: list[Edition]) -> Edition | None:
    cfg = load_resolve_config().editions
    same = [e for e in editions if e.isbn and (not e.title or title_similarity(e.title, title) >= cfg.title_match)]
    return max(
        same, key=lambda e: (e.physical_format.casefold() in cfg.preferred_formats, e.year or 0, e.isbn), default=None
    )


def choose_edition(reading: Reading, title: str, editions: list[Edition]) -> EditionChoice:
    matched = publisher_matches(reading, editions)
    if len({e.isbn for e in matched}) == 1:
        return EditionChoice(edition=matched[0], publisher_known=True)
    found = equivalent_of(title, matched or editions)
    return EditionChoice(edition=found, equivalent=found is not None, publisher_known=bool(matched))


async def edition_options(top: Scored, catalogs: Catalogs) -> tuple[list[Edition], bool]:
    options = [
        Edition(publisher=m.publisher, isbn=m.isbn, url=m.url, year=m.year, title=m.title)
        for m in top.members
        if m.publisher and m.isbn
    ]
    isbn = next((m.isbn for m in top.members if m.isbn), "")
    if not isbn:
        return options, False
    fetched, down = await guarded(library_editions(isbn, catalogs.open_library), "open library editions")
    return fetched + options, down


async def library_editions(isbn: str, library: OpenLibrary) -> list[Edition]:
    work = await library.work_of(isbn)
    return await library.editions(work) if work else []


def unidentified(base: Resolution, reason: str, candidates: list[Scored]) -> Resolution:
    score = candidates[0].score if candidates else 0.0
    return base.model_copy(update={"reason": reason, "match_score": score, "margin": margin_of(candidates)})


async def search_catalogs(reading: Reading, catalogs: Catalogs) -> tuple[list[Scored], bool]:
    title = query_text(reading.title_verbatim)
    author = query_text(reading.author_verbatim)
    found, unavailable = await guarded(catalogs.google_books.search(title, author), "google books search")
    return rank(reading, found), unavailable


def without_trailing_numbers(reading: Reading) -> Reading | None:
    words = reading.title_verbatim.split()
    kept = list(words)
    while len(kept) > 1 and kept[-1].isdigit():
        kept.pop()
    if kept == words:
        return None
    return reading.model_copy(update={"title_verbatim": WORD_SEPARATOR.join(kept)})


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
    supported = ocr_overlap(reading, ocr[0]) >= gates.ocr_overlap_min
    if not supported and ocr_contradicts(reading, ocr):
        return unidentified(base, "ocr_disagrees", [])
    if not supported and not tokens(reading.author_verbatim):
        return unidentified(base, "not_corroborated", [])
    ranked, unavailable = await search_catalogs(reading, catalogs)
    unnumbered = without_trailing_numbers(reading)
    if not ranked and unnumbered is not None:
        reading = unnumbered
        ranked, unavailable = await search_catalogs(reading, catalogs)
    corroboration = accept_identification(reading, ocr, ranked)
    if not corroboration:
        if unavailable:
            return unidentified(base, "catalogue_unavailable", ranked)
        if not ranked:
            return unidentified(base, "no_catalogue_match", ranked)
        return unidentified(base, "not_confident" if supported else "not_corroborated", ranked)
    top = ranked[0]
    options, editions_down = await edition_options(top, catalogs)
    choice = choose_edition(reading, top.best.title, options)
    edition = choice.edition
    urls = list(dict.fromkeys(m.url for m in top.members if m.url))
    if edition is not None and edition.url not in urls:
        urls.append(edition.url)
    unverified = edition is None and bool(tokens(reading.publisher_verbatim))
    return base.model_copy(
        update={
            "status": "identified",
            "title": shown_title(reading, top.best),
            "author": top.best.author,
            "edition": edition.publisher if choice.publisher_known else "",
            "isbn": edition.isbn if edition else "",
            "year": edition.year if edition else None,
            "physical_dimensions": edition.physical_dimensions if edition and not choice.equivalent else "",
            "edition_unverified": unverified or editions_down or choice.equivalent,
            "equivalent_edition": choice.equivalent,
            "work_year": min((m.year for m in top.members if m.year), default=None),
            "corroboration": corroboration,
            "id_confidence": id_confidence(reading, ocr, top, corroboration),
            "match_score": top.score,
            "margin": margin_of(ranked),
            "source_urls": urls,
        }
    )
