import logging
from dataclasses import dataclass

import cv2
import numpy as np
from google.genai import errors, types
from pydantic import BaseModel

from libclaim.llm import Llm, LlmReplyInvalid, QuotaExhausted, default_llm
from libclaim.resolve_config import load_resolve_config
from libclaim.settings import load_settings
from libclaim.vision.ocr import ocr_text
from libclaim.vision.spines import Orientation, SpineDet

logger = logging.getLogger(__name__)

BATCH_FAILURES = (QuotaExhausted, LlmReplyInvalid, errors.APIError)


@dataclass
class SpineCrop:
    track_id: int
    image: np.ndarray
    rotation: str = ""
    ocr_text: str = ""
    ocr_score: float = 0.0


class SpineRead(BaseModel):
    crop_id: str
    title: str
    author: str
    publisher: str
    legible: bool


class SpineReads(BaseModel):
    reads: list[SpineRead]


class Reading(BaseModel):
    track_id: int
    title_verbatim: str = ""
    author_verbatim: str = ""
    publisher_verbatim: str = ""
    legible: bool = False
    rotation: str = ""
    failure: str = ""


def ocr_yield(text: str, score: float) -> float:
    return score * len(text)


def crop_spine(image: np.ndarray, det: SpineDet, track_id: int) -> SpineCrop:
    box = det.box_px
    height, width = image.shape[:2]
    x1, x2 = (min(max(round(v), 0), width) for v in (box.x1, box.x2))
    y1, y2 = (min(max(round(v), 0), height) for v in (box.y1, box.y2))
    area = image[y1:y2, x1:x2]
    if det.orientation == Orientation.horizontal:
        text, score = ocr_text(area)
        return SpineCrop(track_id=track_id, image=area, ocr_text=text, ocr_score=score)
    best = None
    for name in load_resolve_config().reading.rotations:
        turned = cv2.rotate(area, getattr(cv2, name))
        text, score = ocr_text(turned)
        if best is None or ocr_yield(text, score) > ocr_yield(best.ocr_text, best.ocr_score):
            best = SpineCrop(track_id=track_id, image=turned, rotation=name, ocr_text=text, ocr_score=score)
    return best


def encode_crop(crop: SpineCrop) -> list[types.Part | str]:
    cfg = load_resolve_config().reading
    ok, buffer = cv2.imencode(".jpg", crop.image, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
    return [f"crop_id: {crop.track_id}", types.Part.from_bytes(data=buffer.tobytes(), mime_type=cfg.jpeg_mime)]


def reading_from(crop: SpineCrop, read: SpineRead | None) -> Reading:
    if read is None:
        return Reading(track_id=crop.track_id, rotation=crop.rotation)
    return Reading(
        track_id=crop.track_id,
        rotation=crop.rotation,
        title_verbatim=read.title.strip(),
        author_verbatim=read.author.strip(),
        publisher_verbatim=read.publisher.strip(),
        legible=read.legible and bool(read.title.strip()),
    )


async def read_spines(
    crops: list[SpineCrop], llm: Llm | None = None, models: list[str] | None = None
) -> list[Reading]:
    cfg = load_resolve_config().reading
    readings: list[Reading] = []
    for start in range(0, len(crops), cfg.batch_size):
        batch = crops[start : start + cfg.batch_size]
        parts: list = [part for crop in batch for part in encode_crop(crop)] + [cfg.prompt]
        try:
            answer = await (llm or default_llm()).structured(
                models or load_settings().vision_models, parts, SpineReads
            )
        except BATCH_FAILURES as error:
            logger.warning("spine reading batch failed: %s", type(error).__name__)
            readings += [
                Reading(track_id=crop.track_id, rotation=crop.rotation, failure=type(error).__name__) for crop in batch
            ]
            continue
        by_id = {read.crop_id: read for read in answer.reads}
        readings += [reading_from(crop, by_id.get(str(crop.track_id))) for crop in batch]
    return readings
