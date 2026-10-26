from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from rapidfuzz import fuzz, process
from rapidfuzz.utils import default_process

from libclaim.catalog.googlebooks import GoogleBooks
from libclaim.catalog.http import Candidate, CatalogHttp, Edition
from libclaim.catalog.openlibrary import SOURCE as OPEN_LIBRARY
from libclaim.catalog.openlibrary import OpenLibrary
from libclaim.resolve_config import load_resolve_config
from libclaim.vision.reading import Reading

WORD_SEPARATOR = " "


class SpineDims(BaseModel):
    height_cm: float | None = None
    thickness_cm: float | None = None


class Scored(BaseModel):
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


def read_text(reading: Reading) -> str:
    return WORD_SEPARATOR.join(part for part in (reading.title_verbatim, reading.author_verbatim) if part)


def ocr_overlap(reading: Reading, ocr_text: str) -> float:
    wanted = tokens(read_text(reading))
    seen = tokens(ocr_text)
    if not wanted or not seen:
        return 0.0
    cutoff = load_resolve_config().gates.token_match
    found = sum(1 for token in wanted if process.extractOne(token, seen, scorer=fuzz.ratio, score_cutoff=cutoff))
    return found / len(wanted)


def author_matches(reading: Reading, candidate: Candidate) -> bool:
    gates = load_resolve_config().gates
    if not tokens(reading.author_verbatim) or not candidate.author:
        return False
    return fuzz.token_set_ratio(reading.author_verbatim, candidate.author, processor=default_process) >= gates.author_match


def score_candidate(reading: Reading, candidate: Candidate) -> float:
    gates = load_resolve_config().gates
    title = fuzz.token_set_ratio(reading.title_verbatim, candidate.title, processor=default_process)
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
        if len(token) >= gates.distinctive_min_chars
        and wanted
        and process.extractOne(token, wanted, scorer=fuzz.ratio, score_cutoff=gates.token_match)
    )


def same_work(a: Candidate, b: Candidate) -> bool:
    gates = load_resolve_config().gates
    if fuzz.token_set_ratio(a.title, b.title, processor=default_process) < gates.agree_score:
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
    scored = [
        Scored(
            members=group,
            score=score_candidate(reading, group[0]),
            author_matched=any(author_matches(reading, member) for member in group),
            distinctive=distinctive_tokens(reading, group[0]),
        )
        for group in groups
    ]
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
    return len(top.sources) > 1 or top.score >= gates.confident_score


def pick_edition(reading: Reading, editions: list[Edition]) -> Edition | None:
    gates = load_resolve_config().gates
    if not tokens(reading.publisher_verbatim) or not editions:
        return None
    best = process.extractOne(
        reading.publisher_verbatim,
        {i: e.publisher for i, e in enumerate(editions)},
        scorer=fuzz.token_set_ratio,
        processor=default_process,
        score_cutoff=gates.publisher_match,
    )
    return editions[best[2]] if best else None


async def edition_options(reading: Reading, top: Scored, catalogs: Catalogs) -> list[Edition]:
    options = [
        Edition(publisher=m.publisher, isbn=m.isbn, url=m.url) for m in top.members if m.publisher and m.isbn
    ]
    library = next((m for m in top.members if m.source == OPEN_LIBRARY), None)
    if library is not None and tokens(reading.publisher_verbatim):
        options += await catalogs.open_library.editions(library.work)
    return options


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
    if not reading.legible or not tokens(reading.title_verbatim):
        return unidentified(base, "illegible", [])
    if ocr_overlap(reading, ocr[0]) < gates.ocr_overlap_min:
        return unidentified(base, "ocr_disagrees", [])
    title = WORD_SEPARATOR.join(tokens(reading.title_verbatim))
    author = WORD_SEPARATOR.join(tokens(reading.author_verbatim))
    found = await catalogs.open_library.search(title, author)
    ranked = rank(reading, found)
    if not accept_identification(reading, ocr, ranked):
        found += await catalogs.google_books.search(title, author)
        ranked = rank(reading, found)
    if not ranked:
        return unidentified(base, "no_catalogue_match", ranked)
    if not accept_identification(reading, ocr, ranked):
        return unidentified(base, "not_confident", ranked)
    top = ranked[0]
    edition = pick_edition(reading, await edition_options(reading, top, catalogs))
    urls = list(dict.fromkeys(m.url for m in top.members if m.url))
    if edition is not None and edition.url not in urls:
        urls.append(edition.url)
    return base.model_copy(
        update={
            "status": "identified",
            "title": top.members[0].title,
            "author": top.members[0].author,
            "edition": edition.publisher if edition else "",
            "isbn": edition.isbn if edition else "",
            "id_confidence": round(top.score / 100 * ocr_overlap(reading, ocr[0]), 3),
            "match_score": top.score,
            "margin": margin_of(ranked),
            "source_urls": urls,
        }
    )
