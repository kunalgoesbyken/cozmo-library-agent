import json

import cv2
import numpy as np
import pytest
from pydantic import BaseModel

from libclaim.budget import Budget
from libclaim.live.tools import Statement, StatementKind
from libclaim.llm import Llm, LlmResponse
from libclaim.models import Still
from libclaim.room.depth import DepthSet
from libclaim.vision.boxes import load_vision_config
from libclaim.vision.items import FrameItems, detect_items
from libclaim.vision.scale import camera_matrix

WIDTH, HEIGHT = 1000, 500
SHIFT = 80
MODELS = ["gemini-3.5-flash-lite"]


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.parts = []
        self.schemas = []

    async def generate(self, model, parts, schema: type[BaseModel]):
        self.parts.append(parts)
        self.schemas.append(schema)
        return LlmResponse(text=json.dumps({"items": self.replies.pop(0)}), input_tokens=10, output_tokens=5)


def llm_for(tmp_path, replies):
    client = FakeClient(replies)
    return Llm(client=client, budget=Budget(tmp_path / "spend.jsonl"), quota_path=tmp_path / "q.json"), client


def panned_stills(tmp_path, count, blur=None):
    rng = np.random.default_rng(0)
    scene = cv2.GaussianBlur(rng.integers(0, 256, (HEIGHT, WIDTH + SHIFT * count), dtype=np.uint8), (3, 3), 0)
    stills = []
    for i in range(count):
        path = tmp_path / f"{i:05d}_{i}.png"
        cv2.imwrite(str(path), scene[:, SHIFT * i : SHIFT * i + WIDTH])
        stills.append(
            Still(seq=i, t_ms=i * 1000, path=str(path), w=WIDTH, h=HEIGHT,
                  blur=(blur or {}).get(i, 1.0), glare=0.0, motion_dps=0.0)
        )
    return stills


def item(index, category, x1, x2, y1=100, y2=300, **extra):
    return {
        "image_index": index,
        "category": category,
        "description": extra.get("description", category),
        "brand_model_text": extra.get("brand", ""),
        "box_2d": [y1 * 1000 // HEIGHT, x1, y2 * 1000 // HEIGHT, x2],
        "is_art": extra.get("is_art", False),
        "truncated": extra.get("truncated", False),
    }


async def test_four_keyframes_per_call_with_taxonomy_enum(tmp_path):
    stills = panned_stills(tmp_path, 5)
    llm, client = llm_for(tmp_path, [[], []])
    assert await detect_items(stills, llm, MODELS) == []
    assert [len(parts) for parts in client.parts] == [9, 3]
    assert client.parts[0][0] == "image_index: 0"
    assert client.parts[1][0] == "image_index: 0"
    schema = json.dumps(client.schemas[0].model_json_schema())
    for category in ("furniture", "lighting", "artwork", "other"):
        assert f'"{category}"' in schema
    assert client.schemas[0] is FrameItems


async def test_adjacent_frames_merge_by_homography_and_keep_sharpest(tmp_path):
    stills = panned_stills(tmp_path, 2, blur={1: 5.0})
    replies = [[
        item(0, "lighting", 300, 400, brand="IKEA"),
        item(1, "lighting", 300 - SHIFT, 400 - SHIFT),
        item(1, "lighting", 700, 800),
        item(1, "furniture", 300 - SHIFT, 400 - SHIFT),
    ]]
    llm, _ = llm_for(tmp_path, replies)
    dets = await detect_items(stills, llm, MODELS)
    assert sorted(d.category for d in dets) == ["furniture", "lighting", "lighting"]
    merged = next(d for d in dets if d.seen_in == [0, 1])
    assert merged.category == "lighting"
    assert merged.frame_seq == 1
    assert merged.frame_path == stills[1].path
    assert merged.brand_model == "IKEA"
    assert merged.box_px.x1 == pytest.approx(300 - SHIFT)


async def test_frames_beyond_lookback_are_not_merged(tmp_path):
    cfg = load_vision_config().items
    last = cfg.dedup_lookback + 1
    stills = panned_stills(tmp_path, last + 1)
    replies = [[] for _ in range(0, last + 1, cfg.batch_size)]
    replies[0].append(item(0, "decor", 500, 600))
    replies[last // cfg.batch_size].append(item(last % cfg.batch_size, "decor", 500 - SHIFT * last, 600 - SHIFT * last))
    llm, _ = llm_for(tmp_path, replies)
    dets = await detect_items(stills, llm, MODELS)
    assert sorted(d.seen_in for d in dets) == [[0], [last]]


async def test_untruncated_sighting_is_preferred(tmp_path):
    stills = panned_stills(tmp_path, 2, blur={0: 9.0})
    replies = [[item(0, "electronics", 300, 400, truncated=True), item(1, "electronics", 300 - SHIFT, 400 - SHIFT)]]
    llm, _ = llm_for(tmp_path, replies)
    [det] = await detect_items(stills, llm, MODELS)
    assert det.frame_seq == 1
    assert det.truncated is False


async def test_art_needs_appraisal_unless_user_said_print(tmp_path):
    stills = panned_stills(tmp_path, 3)
    replies = [[
        item(0, "artwork", 100, 200, is_art=True),
        item(2, "decor", 600, 700, is_art=True, description="bronze figure"),
        item(2, "furniture", 300, 500),
    ]]
    said = Statement(subject="poster", kind=StatementKind.art_print, value="print", t_ms=0,
                     still_seq_from=0, still_seq_to=0)
    llm, _ = llm_for(tmp_path, replies)
    dets = await detect_items(stills, llm, MODELS, statements=[said])
    by_category = {d.category: d for d in dets}
    assert by_category["artwork"].needs_appraisal is False
    assert by_category["decor"].needs_appraisal is True
    assert by_category["decor"].is_art is True
    assert by_category["furniture"].needs_appraisal is False


async def test_print_statement_with_several_artworks_in_view_stays_appraisal(tmp_path):
    stills = panned_stills(tmp_path, 1)
    replies = [[item(0, "artwork", 100, 200, is_art=True), item(0, "artwork", 600, 700, is_art=True)]]
    said = Statement(subject="poster", kind=StatementKind.art_print, value="print", t_ms=0,
                     still_seq_from=0, still_seq_to=0)
    llm, _ = llm_for(tmp_path, replies)
    dets = await detect_items(stills, llm, MODELS, statements=[said])
    assert len(dets) == 2
    assert all(d.needs_appraisal for d in dets)
    assert all(d.review for d in dets)


async def test_original_art_statement_is_never_overridden(tmp_path):
    stills = panned_stills(tmp_path, 1)
    replies = [[item(0, "artwork", 100, 200, is_art=True)]]
    window = {"t_ms": 0, "still_seq_from": 0, "still_seq_to": 0}
    said = [
        Statement(subject="painting", kind=StatementKind.art_print, value="print", **window),
        Statement(subject="painting", kind=StatementKind.original_art, value="original", **window),
    ]
    llm, _ = llm_for(tmp_path, replies)
    [det] = await detect_items(stills, llm, MODELS, statements=said)
    assert det.needs_appraisal is True


async def test_sized_from_depth_when_available(tmp_path):
    stills = panned_stills(tmp_path, 1)
    replies = [[item(0, "lighting", 300, 400, 100, 300)]]
    llm, _ = llm_for(tmp_path, replies)
    depths = DepthSet({0: np.full((HEIGHT, WIDTH), 2.0)}, "depth_aruco", 0.7)
    [det] = await detect_items(stills, llm, MODELS, depths=depths)
    focal = camera_matrix(stills[0], load_vision_config().scale)[0, 0]
    assert det.dimensions_cm.w == pytest.approx(100 * 2.0 / focal * 100, rel=0.02)
    assert det.dimensions_cm.h == pytest.approx(200 * 2.0 / focal * 100, rel=0.02)
    assert det.dimensions_cm.d is None
    assert det.dimensions_method == "depth_aruco"


async def test_unsized_without_depth_gets_review(tmp_path):
    stills = panned_stills(tmp_path, 1)
    replies = [[item(0, "lighting", 300, 400)]]
    llm, _ = llm_for(tmp_path, replies)
    [det] = await detect_items(stills, llm, MODELS, depths=DepthSet({}, "", 0.0, ["no [ml]"]))
    assert det.dimensions_cm.w is None and det.dimensions_cm.h is None
    assert det.dimensions_method == ""
    assert det.review


async def test_out_of_range_index_and_bad_boxes_are_dropped(tmp_path):
    stills = panned_stills(tmp_path, 1)
    bad = item(0, "decor", 100, 200)
    bad["box_2d"] = [10, 10]
    replies = [[item(3, "decor", 100, 200), bad, item(0, "decor", 400, 400)]]
    llm, _ = llm_for(tmp_path, replies)
    assert await detect_items(stills, llm, MODELS) == []


async def test_no_keyframes_makes_no_call(tmp_path):
    llm, client = llm_for(tmp_path, [])
    assert await detect_items([], llm, MODELS) == []
    assert client.parts == []
