import json

import cv2
import numpy as np
from pydantic import BaseModel

from libclaim.budget import Budget
from libclaim.llm import Llm, LlmResponse
from libclaim.vision.boxes import Box
from libclaim.vision.ocr import ocr_engine, ocr_text
from libclaim.vision.reading import SpineCrop, crop_spine, read_spines
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


def test_vertical_spines_rotate_upright_and_horizontal_do_not():
    image = blank(200, 300)
    box = Box(x1=10, y1=20, x2=50, y2=120)
    vertical = SpineDet(box_px=box, orientation=Orientation.vertical, source=Source.gemini)
    horizontal = SpineDet(box_px=box, orientation=Orientation.horizontal, source=Source.gemini)
    assert crop_spine(image, vertical).shape[:2] == (40, 100)
    assert crop_spine(image, horizontal).shape[:2] == (100, 40)


def test_ocr_reads_printed_text_and_engine_is_cached():
    image = np.full((120, 400, 3), 255, np.uint8)
    cv2.putText(image, "SAPIENS", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    text, score = ocr_text(image)
    assert text == "SAPIENS"
    assert score > 0.9
    assert ocr_engine() is ocr_engine()


def test_ocr_of_blank_crop_is_empty():
    assert ocr_text(np.full((50, 50, 3), 255, np.uint8)) == ("", 0.0)
