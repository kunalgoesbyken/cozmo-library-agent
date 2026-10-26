import asyncio
import json

import numpy as np
import pytest

from libclaim import pipeline as pipeline_module
from libclaim.llm import LlmResponse
from libclaim.models import Still
from libclaim.pipeline import SweepRun
from libclaim.stages import BookRead, BookSlot, SegmentPlan, SegmentTracks, load_pipeline_config
from libclaim.vision.boxes import Box
from libclaim.vision.reading import Reading, SpineCrop
from libclaim.catalog.http import Candidate
from libclaim.vision.resolve import Catalogs
from libclaim.vision.spines import Orientation
from tests import sweep12 as fx
from tests.test_pipeline import services

BOOKS = 30
DELAY_S = 0.05


@pytest.fixture
def sweep(tmp_path):
    return fx.copy_sweep(tmp_path / "run")


class SlowCatalogue:
    def __init__(self):
        self.now = 0
        self.peak = 0
        self.searched = asyncio.Event()

    async def search(self, title, author):
        self.searched.set()
        self.now += 1
        self.peak = max(self.peak, self.now)
        await asyncio.sleep(DELAY_S)
        self.now -= 1
        return []

    async def editions(self, work):
        return []


def slots(count):
    still = Still(seq=1, t_ms=0, path="frame.jpg", w=100, h=100, blur=0, glare=0, motion_dps=0)
    box = Box(x1=0, y1=0, x2=10, y2=50)
    return [BookSlot(id=f"b01-{i:03d}", segment=0, track=i, shelf="S1", position=i + 1,
                     orientation=Orientation.vertical, frame=still, box=box, seen=[1]) for i in range(count)]


def run_with(sweep, tmp_path, catalogue, books, llm=None):
    found = services(sweep, tmp_path, fx.Web(), llm=llm)
    found.catalogs = Catalogs(catalogue, catalogue)
    run = SweepRun(sweep, found, config=load_pipeline_config().model_copy(update={"deadline_poll_s": 0.01}))
    run.results.tracks.segments[0] = SegmentTracks(index=0, stills=[], tracks=[], rows=[], flat=[], books=books)
    return run


def read_of(index):
    return Reading(track_id=index, title_verbatim=f"Unknown Title {index}", author_verbatim="Some Author", legible=True)


async def test_books_of_a_segment_are_resolved_concurrently(sweep, tmp_path):
    catalogue, books = SlowCatalogue(), slots(BOOKS)
    run = run_with(sweep, tmp_path, catalogue, books)
    for index, book in enumerate(books):
        run.results.reading.reads[book.id] = BookRead(reading=read_of(index), ocr_text=f"Unknown Title {index}")
    await run.resolve_segment(SegmentPlan(index=0, label="S1", seqs=[]))
    assert len(run.results.resolution.resolutions) == BOOKS
    assert catalogue.peak > 1


class Library:
    def __init__(self, titles):
        self.titles = titles

    async def search(self, title, author):
        found = [t for t in self.titles if t.casefold() == title]
        return [Candidate(source=source, title=t, author="Enid Blyton", work=f"{source}{t}", url=f"https://x/{source}/{t}")
                for t in found for source in ("openlibrary", "googlebooks")]

    async def editions(self, work):
        return []


async def test_same_title_in_overlapping_boxes_counts_once_and_a_separate_copy_is_kept(sweep, tmp_path):
    books = slots(4)
    spans = [(0, 50), (25, 75), (200, 250), (400, 450)]
    books = [book.model_copy(update={"box": Box(x1=a, y1=0, x2=b, y2=300)}) for book, (a, b) in zip(books, spans)]
    titles = ["Well Done, Secret Seven", "Well Done, Secret Seven", "Well Done, Secret Seven", "Five Go to Camp"]
    run = run_with(sweep, tmp_path, Library(titles), books)
    for index, (book, title) in enumerate(zip(books, titles)):
        spine = Reading(track_id=index, title_verbatim=title, author_verbatim="Enid Blyton", legible=True)
        run.results.reading.reads[book.id] = BookRead(reading=spine, ocr_text="")
    await run.resolve_segment(SegmentPlan(index=0, label="S1", seqs=[]))
    found = run.results.resolution.resolutions
    assert [found[book.id].status for book in books] == ["identified", "unidentified", "identified", "identified"]
    assert found[books[1].id].reason == "repeated_read" and found[books[1].id].title == ""


class BatchedReads:
    def __init__(self, catalogue):
        self.catalogue = catalogue
        self.resolved_during_reading = False
        self.calls = 0

    async def generate(self, model, parts, schema):
        self.calls += 1
        if self.calls > 1:
            await asyncio.wait_for(self.catalogue.searched.wait(), timeout=5)
            self.resolved_during_reading = True
        ids = [p.removeprefix("crop_id: ") for p in parts if isinstance(p, str) and p.startswith("crop_id: ")]
        reads = [{"crop_id": i, "title": f"Unknown Title {i}", "author": "Some Author", "publisher": "",
                  "legible": True} for i in ids]
        return LlmResponse(text=json.dumps({"reads": reads}), input_tokens=1, output_tokens=1)


async def test_resolution_starts_while_later_batches_are_still_being_read(sweep, tmp_path, monkeypatch):
    catalogue, books = SlowCatalogue(), slots(BOOKS)
    client = BatchedReads(catalogue)
    crops = [SpineCrop(track_id=i, image=np.zeros((40, 10, 3), np.uint8), rotation="ROTATE_90_CLOCKWISE",
                       ocr_text="", ocr_score=0.0) for i in range(BOOKS)]
    monkeypatch.setattr(pipeline_module, "spine_crops", lambda found: crops)
    run = run_with(sweep, tmp_path, catalogue, books, llm=fx.fake_llm(sweep, tmp_path, client))
    plan = SegmentPlan(index=0, label="S1", seqs=[])
    await run.read_segment(plan)
    await run.resolve_segment(plan)
    assert client.calls == 2 and client.resolved_during_reading
    assert len(run.results.resolution.resolutions) == BOOKS
