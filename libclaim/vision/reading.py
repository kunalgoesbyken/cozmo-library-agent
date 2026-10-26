from dataclasses import dataclass

import cv2
import numpy as np
from google.genai import types
from pydantic import BaseModel

from libclaim.llm import Llm, default_llm
from libclaim.resolve_config import load_resolve_config
from libclaim.settings import load_settings
from libclaim.vision.boxes import Box
from libclaim.vision.spines import Orientation, SpineDet


@dataclass
class SpineCrop:
    track_id: int
    image: np.ndarray


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


def upright(image: np.ndarray, orientation: Orientation) -> np.ndarray:
    if orientation == Orientation.vertical:
        return cv2.rotate(image, getattr(cv2, load_resolve_config().reading.vertical_rotation))
    return image


def crop_spine(image: np.ndarray, det: SpineDet, box: Box | None = None) -> np.ndarray:
    area = box or det.box_px
    height, width = image.shape[:2]
    x1, x2 = (min(max(round(v), 0), width) for v in (area.x1, area.x2))
    y1, y2 = (min(max(round(v), 0), height) for v in (area.y1, area.y2))
    return upright(image[y1:y2, x1:x2], det.orientation)


def encode_crop(crop: SpineCrop) -> list[types.Part | str]:
    cfg = load_resolve_config().reading
    ok, buffer = cv2.imencode(".jpg", crop.image, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality])
    return [f"crop_id: {crop.track_id}", types.Part.from_bytes(data=buffer.tobytes(), mime_type=cfg.jpeg_mime)]


def reading_from(crop: SpineCrop, read: SpineRead | None) -> Reading:
    if read is None:
        return Reading(track_id=crop.track_id)
    return Reading(
        track_id=crop.track_id,
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
        answer = await (llm or default_llm()).structured(models or load_settings().vision_models, parts, SpineReads)
        by_id = {read.crop_id: read for read in answer.reads}
        readings += [reading_from(crop, by_id.get(str(crop.track_id))) for crop in batch]
    return readings
