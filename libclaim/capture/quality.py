from enum import StrEnum

import cv2
import numpy as np
import yaml
from pydantic import BaseModel

from libclaim.settings import CONFIG_DIR


class AlertCode(StrEnum):
    blur = "blur"
    glare = "glare"
    too_fast = "too_fast"
    occluded = "occluded"
    too_far = "too_far"
    missed_region = "missed_region"


class CaptureConfig(BaseModel):
    blur_floor: float
    glare_pixel_level: int
    glare_clipped_fraction: float
    too_fast_dps: float


class Quality(BaseModel):
    blur: float
    glare: float


def load_capture_config() -> CaptureConfig:
    path = CONFIG_DIR / "capture.yaml"
    return CaptureConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def score(img: np.ndarray, cfg: CaptureConfig) -> Quality:
    gray = _gray(img)
    blur = float(cv2.Laplacian(gray.astype(np.float32) / 255.0, cv2.CV_32F).var())
    glare = float(np.count_nonzero(gray >= cfg.glare_pixel_level)) / gray.size
    return Quality(blur=blur, glare=glare)


def gate(q: Quality, motion_dps: float, cfg: CaptureConfig) -> list[AlertCode]:
    alerts = []
    if q.blur < cfg.blur_floor:
        alerts.append(AlertCode.blur)
    if q.glare > cfg.glare_clipped_fraction:
        alerts.append(AlertCode.glare)
    if motion_dps > cfg.too_fast_dps:
        alerts.append(AlertCode.too_fast)
    return alerts
