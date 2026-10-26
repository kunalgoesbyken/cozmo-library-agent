import json
import warnings

import cv2
import numpy as np
import pytest
from pydantic import BaseModel

from libclaim.budget import Budget
from libclaim.llm import Llm, LlmResponse
from libclaim.models import Still
from libclaim.vision.boxes import Box, ImageUnreadable, nms
from libclaim.vision.spines import Orientation, Source, SpineDet, detect_spines, reconcile
from libclaim.vision.rfdetr import decode, rfdetr_boxes

WIDTH = 1000
HEIGHT = 400


class FakeClient:
    def __init__(self, spines):
        self.spines = spines
        self.parts = []

    async def generate(self, model, parts, schema: type[BaseModel]):
        self.parts.append(parts)
        return LlmResponse(text=json.dumps({"spines": self.spines}), input_tokens=10, output_tokens=5)


def keyframe(tmp_path):
    path = tmp_path / "kf.png"
    cv2.imwrite(str(path), np.zeros((HEIGHT, WIDTH, 3), np.uint8))
    return Still(seq=0, t_ms=0, path=str(path), w=WIDTH, h=HEIGHT, blur=1.0, glare=0.0, motion_dps=0.0)


def llm_for(tmp_path, spines):
    client = FakeClient(spines)
    llm = Llm(client=client, budget=Budget(tmp_path / "spend.jsonl"), quota_path=tmp_path / "q.json")
    return llm, client


def spine(index, box, orientation="vertical"):
    return {"image_index": index, "box_2d": box, "orientation": orientation}


def det(x1, x2, y1=0.0, y2=300.0, source=Source.gemini):
    return SpineDet(box_px=Box(x1=x1, y1=y1, x2=x2, y2=y2), orientation=Orientation.vertical, source=source)


def row_of(count, width=30.0, gap=2.0):
    return [det(i * (width + gap), i * (width + gap) + width) for i in range(count)]


async def test_detect_sends_two_tiles_and_converts_to_pixels(tmp_path):
    llm, client = llm_for(
        tmp_path,
        [spine(0, [0, 0, 1000, 100]), spine(1, [100, 500, 600, 600], "horizontal")],
    )
    dets = await detect_spines(keyframe(tmp_path), llm, ["gemini-3.5-flash-lite"])
    assert len(client.parts[0]) == 3
    first = next(d for d in dets if d.orientation == Orientation.vertical)
    second = next(d for d in dets if d.orientation == Orientation.horizontal)
    assert first.box_px.x1 == pytest.approx(0)
    assert first.box_px.y2 == pytest.approx(HEIGHT)
    tile_w = WIDTH / 1.9
    tile_x0 = tile_w * 0.9
    assert second.box_px.x1 == pytest.approx(tile_x0 + 0.5 * tile_w)
    assert second.box_px.y1 == pytest.approx(0.1 * HEIGHT)
    assert all(d.source == Source.gemini for d in dets)


async def test_detect_nms_drops_overlap_duplicates_and_bad_boxes(tmp_path):
    llm, _ = llm_for(
        tmp_path,
        [
            spine(0, [0, 800, 1000, 1000]),
            spine(0, [0, 810, 1000, 1000]),
            spine(0, [10, 10]),
            spine(5, [0, 0, 100, 100]),
            spine(0, [100, 100, 100, 200]),
        ],
    )
    dets = await detect_spines(keyframe(tmp_path), llm, ["gemini-3.5-flash-lite"])
    assert len(dets) == 1


async def test_detect_suppresses_tile_edge_clipped_duplicate(tmp_path):
    llm, _ = llm_for(tmp_path, [spine(0, [0, 836, 1000, 950]), spine(1, [0, 0, 1000, 50])])
    dets = await detect_spines(keyframe(tmp_path), llm, ["gemini-3.5-flash-lite"])
    assert len(dets) == 1
    assert dets[0].box_px.width > 55


async def test_detect_unreadable_keyframe_raises_clear_error(tmp_path):
    llm, _ = llm_for(tmp_path, [])
    still = keyframe(tmp_path).model_copy(update={"path": str(tmp_path / "missing.png")})
    with pytest.raises(ImageUnreadable):
        await detect_spines(still, llm, ["gemini-3.5-flash-lite"])


def test_nms_keeps_distinct_boxes():
    boxes = [Box(x1=0, y1=0, x2=10, y2=10), Box(x1=1, y1=0, x2=11, y2=10), Box(x1=50, y1=0, x2=60, y2=10)]
    assert nms(boxes, 0.6, 0.8) == [0, 2]


def test_reconcile_flags_twenty_percent_disagreement():
    gemini = row_of(10)
    local = [d.box_px for d in row_of(8)]
    result = reconcile({"row-1": gemini}, {"row-1": local})
    assert [e.reason for e in result.review_entries] == ["count_mismatch"]
    assert result.requery_rows == ["row-1"]
    assert len(result.rows["row-1"]) == 10


def test_reconcile_agreement_is_clean():
    gemini = row_of(10)
    local = [d.box_px for d in gemini]
    result = reconcile({"row-1": gemini}, {"row-1": local})
    assert result.review_entries == []
    assert result.requery_rows == []


def test_reconcile_flags_gap_wider_than_one_and_a_half_thicknesses():
    dets = row_of(4) + [det(4 * 32 + 60, 4 * 32 + 90)]
    result = reconcile({"row-1": dets}, {"row-1": [d.box_px for d in dets]})
    assert [e.reason for e in result.review_entries] == ["gap"]
    assert result.requery_rows == ["row-1"]


def test_reconcile_adds_rfdetr_only_boxes_and_caps_requeries():
    rows = {f"r{i}": row_of(10) for i in range(8)}
    local = {name: [d.box_px for d in dets][: 10 - (i + 2)] for i, (name, dets) in enumerate(rows.items())}
    result = reconcile(rows, local)
    assert len(result.requery_rows) == 5
    assert result.requery_rows[0] == "r7"
    extra = reconcile({"r": row_of(5)}, {"r": [d.box_px for d in row_of(5)] + [Box(x1=900, y1=0, x2=930, y2=300)]})
    assert any(d.source == Source.rfdetr for d in extra.rows["r"])


def test_reconcile_without_local_model_flags_review_once():
    result = reconcile({"row-1": row_of(5), "row-2": row_of(5)}, {"row-1": None, "row-2": None})
    assert [e.reason for e in result.review_entries] == ["local_counter_unavailable"]
    assert len(result.rows["row-1"]) == 5


def test_reconcile_shifted_duplicate_is_not_a_phantom_book():
    gemini = [det(0, 30)]
    shifted = Box(x1=10, y1=0, x2=40, y2=300)
    result = reconcile({"r": gemini}, {"r": [shifted]})
    assert len(result.rows["r"]) == 1


def test_reconcile_local_only_box_orientation_from_aspect():
    wide = Box(x1=500, y1=0, x2=700, y2=40)
    tall = Box(x1=800, y1=0, x2=830, y2=300)
    result = reconcile({"r": row_of(2)}, {"r": [d.box_px for d in row_of(2)] + [wide, tall]})
    added = {d.box_px.x1: d.orientation for d in result.rows["r"] if d.source == Source.rfdetr}
    assert added == {500: Orientation.horizontal, 800: Orientation.vertical}


def test_rfdetr_returns_none_when_model_absent(monkeypatch, tmp_path):
    monkeypatch.setenv("LIBCLAIM_RFDETR_PATH", str(tmp_path / "missing.onnx"))
    assert rfdetr_boxes(np.zeros((10, 10, 3), np.uint8)) is None


def test_rfdetr_decode_extreme_logits_do_not_warn():
    boxes = np.array([[0.5, 0.5, 0.2, 0.4]], np.float32)
    logits = np.array([[-5000.0, 5000.0]], np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert len(decode(boxes, logits, 100, 100, 0.5)) == 1


def test_rfdetr_decode_scales_centres_to_pixels():
    boxes = np.array([[0.5, 0.5, 0.2, 0.4], [0.1, 0.1, 0.1, 0.1]], np.float32)
    logits = np.array([[5.0, -5.0], [-5.0, -5.0]], np.float32)
    found = decode(boxes, logits, 200, 100, 0.5)
    assert len(found) == 1
    assert [found[0].x1, found[0].y1, found[0].x2, found[0].y2] == pytest.approx([80, 30, 120, 70])
