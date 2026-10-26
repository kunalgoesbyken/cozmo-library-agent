import json
import threading
import time

import cv2
import numpy as np
from pydantic import BaseModel

from libclaim.budget import Budget
from libclaim.llm import Llm, LlmResponse
from libclaim.resolve_config import load_resolve_config
from libclaim.vision import reading as reading_module
from libclaim.vision.boxes import Box
from libclaim.vision.ocr import ocr_engine, ocr_params, ocr_text
from libclaim.vision.reading import SpineCrop, crop_spine, crop_spines, read_spines
from libclaim.vision.spines import Orientation, SpineDet, Source


class FakeClient:
    def __init__(self, reads_for):
        self.reads_for = reads_for
        self.calls = []

    async def generate(self, model, parts, schema: type[BaseModel]):
        self.calls.append(parts)
        ids = [p.removeprefix("crop_id: ") for p in parts if isinstance(p, str) and p.startswith("crop_id: ")]
        reads = [self.reads_for(i) for i in ids]
        return LlmResponse(text=json.dumps({"reads": [r for r in reads if r]}), input_tokens=10, output_tokens=5)


def llm_for(tmp_path, reads_for):
    client = FakeClient(reads_for)
    return Llm(client=client, budget=Budget(tmp_path / "spend.jsonl"), quota_path=tmp_path / "q.json"), client


def blank(width=40, height=100):
    return np.zeros((height, width, 3), np.uint8)


async def test_reads_in_batches_of_24_keyed_by_crop_id(tmp_path):
    def reads_for(crop_id):
        if crop_id == "7":
            return None
        return {"crop_id": crop_id, "title": f" T{crop_id} ", "author": "A", "publisher": "P", "legible": True}

    llm, client = llm_for(tmp_path, reads_for)
    crops = [SpineCrop(track_id=i, image=blank()) for i in range(30)]
    readings = await read_spines(crops, llm, ["gemini-3.5-flash-lite"])
    assert len(client.calls) == 2
    assert [len([p for p in call if isinstance(p, str) and p.startswith("crop_id")]) for call in client.calls] == [24, 6]
    assert "transcribe only characters you can see; empty if unsure" in client.calls[0][-1].lower()
    assert [r.track_id for r in readings] == list(range(30))
    assert readings[3].title_verbatim == "T3"
    assert readings[7].legible is False and readings[7].title_verbatim == ""


async def test_empty_title_is_not_legible(tmp_path):
    llm, _ = llm_for(tmp_path, lambda i: {"crop_id": i, "title": "", "author": "", "publisher": "", "legible": True})
    readings = await read_spines([SpineCrop(track_id=0, image=blank())], llm, ["gemini-3.5-flash-lite"])
    assert readings[0].legible is False


def text_image(text="SAPIENS"):
    image = np.full((120, 400, 3), 255, np.uint8)
    cv2.putText(image, text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    return image


def vertical_det(image):
    box = Box(x1=0, y1=0, x2=image.shape[1], y2=image.shape[0])
    return SpineDet(box_px=box, orientation=Orientation.vertical, source=Source.gemini)


def test_top_to_bottom_text_is_turned_counter_clockwise():
    shelf = cv2.rotate(text_image(), cv2.ROTATE_90_CLOCKWISE)
    crop = crop_spine(shelf, vertical_det(shelf), 3)
    assert crop.rotation == "ROTATE_90_COUNTERCLOCKWISE"
    assert crop.ocr_text == "SAPIENS"
    assert crop.image.shape[:2] == (120, 400)
    assert crop.track_id == 3


def test_bottom_to_top_text_is_turned_clockwise():
    shelf = cv2.rotate(text_image(), cv2.ROTATE_90_COUNTERCLOCKWISE)
    crop = crop_spine(shelf, vertical_det(shelf), 4)
    assert crop.rotation == "ROTATE_90_CLOCKWISE"
    assert crop.ocr_text == "SAPIENS"


def test_horizontal_spines_are_not_rotated():
    image = text_image()
    box = Box(x1=0, y1=0, x2=400, y2=120)
    det = SpineDet(box_px=box, orientation=Orientation.horizontal, source=Source.gemini)
    crop = crop_spine(image, det, 5)
    assert crop.rotation == ""
    assert crop.image.shape[:2] == (120, 400)
    assert crop.ocr_text == "SAPIENS"


async def test_chosen_rotation_is_recorded_in_the_reading(tmp_path):
    llm, _ = llm_for(tmp_path, lambda i: {"crop_id": i, "title": "T", "author": "", "publisher": "", "legible": True})
    crop = SpineCrop(track_id=2, image=blank(), rotation="ROTATE_90_CLOCKWISE")
    readings = await read_spines([crop], llm, ["gemini-3.5-flash-lite"])
    assert readings[0].rotation == "ROTATE_90_CLOCKWISE"


async def test_each_batch_is_handed_on_as_soon_as_it_is_read(tmp_path):
    llm, client = llm_for(tmp_path, lambda i: {"crop_id": i, "title": "T", "author": "", "publisher": "", "legible": True})
    seen = []
    crops = [SpineCrop(track_id=i, image=blank()) for i in range(30)]
    await read_spines(crops, llm, ["gemini-3.5-flash-lite"], lambda batch: seen.append((len(client.calls), len(batch))))
    assert seen == [(1, 24), (2, 6)]


async def test_failed_batch_keeps_other_batches(tmp_path):
    class Flaky:
        def __init__(self):
            self.calls = 0

        async def generate(self, model, parts, schema):
            self.calls += 1
            ids = [p.removeprefix("crop_id: ") for p in parts if isinstance(p, str) and p.startswith("crop_id: ")]
            if ids[0] != "0":
                return LlmResponse(text="not json", input_tokens=1, output_tokens=1)
            reads = [{"crop_id": i, "title": "T", "author": "", "publisher": "", "legible": True} for i in ids]
            return LlmResponse(text=json.dumps({"reads": reads}), input_tokens=1, output_tokens=1)

    llm = Llm(client=Flaky(), budget=Budget(tmp_path / "spend.jsonl"), quota_path=tmp_path / "q.json")
    crops = [SpineCrop(track_id=i, image=blank()) for i in range(30)]
    readings = await read_spines(crops, llm, ["gemini-3.5-flash-lite"])
    assert len(readings) == 30
    assert all(r.legible and not r.failure for r in readings[:24])
    assert all(not r.legible and r.failure == "LlmReplyInvalid" for r in readings[24:])


def test_ocr_reads_printed_text_and_engine_is_cached():
    image = np.full((120, 400, 3), 255, np.uint8)
    cv2.putText(image, "SAPIENS", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    text, score = ocr_text(image)
    assert text == "SAPIENS"
    assert score > 0.9
    assert ocr_engine() is ocr_engine()


def test_ocr_of_blank_crop_is_empty():
    assert ocr_text(np.full((50, 50, 3), 255, np.uint8)) == ("", 0.0)


class CountingOcr:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def __call__(self, crop):
        self.calls += 1
        return self.replies.pop(0)


def test_first_rotation_with_a_good_yield_skips_the_second(monkeypatch):
    fake = CountingOcr([("THE SECRET OF SKYTOP HILL ENID BLYTON", 0.95), ("NOT USED", 0.99)])
    monkeypatch.setattr(reading_module, "ocr_text", fake)
    image = blank(40, 300)
    crop = crop_spine(image, vertical_det(image), 1)
    assert fake.calls == 1
    assert crop.rotation == load_resolve_config().reading.rotations[0]


def test_low_first_yield_tries_the_other_rotation_and_keeps_the_better(monkeypatch):
    fake = CountingOcr([("11", 0.6), ("TRANSHIMALAJA SVEN HEDIN", 0.9)])
    monkeypatch.setattr(reading_module, "ocr_text", fake)
    image = blank(40, 300)
    crop = crop_spine(image, vertical_det(image), 1)
    assert fake.calls == 2
    assert crop.rotation == load_resolve_config().reading.rotations[1]
    assert crop.ocr_text == "TRANSHIMALAJA SVEN HEDIN"


def test_horizontal_spine_is_read_once(monkeypatch):
    fake = CountingOcr([("", 0.0)])
    monkeypatch.setattr(reading_module, "ocr_text", fake)
    det = SpineDet(box_px=Box(x1=0, y1=0, x2=300, y2=40), orientation=Orientation.horizontal, source=Source.gemini)
    crop_spine(blank(300, 40), det, 1)
    assert fake.calls == 1


def test_engine_takes_size_threads_and_classifier_from_config():
    cfg = load_resolve_config().ocr
    params = ocr_params()
    assert params["Det.limit_type"] == cfg.det_limit_type
    assert params["Det.limit_side_len"] == cfg.det_limit_side_len
    assert params["EngineConfig.onnxruntime.intra_op_num_threads"] == cfg.threads
    assert params["Global.use_cls"] is cfg.use_cls


def test_concurrent_crops_share_a_bounded_pool(monkeypatch):
    lock, state = threading.Lock(), {"now": 0, "peak": 0}

    def slow(crop):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.02)
        with lock:
            state["now"] -= 1
        return "", 0.0

    monkeypatch.setattr(reading_module, "ocr_text", slow)
    image = blank(40, 300)
    jobs = [(image, vertical_det(image))] * 12
    crops = crop_spines(jobs)
    assert [crop.track_id for crop in crops] == list(range(12))
    assert 1 < state["peak"] <= load_resolve_config().ocr.workers
