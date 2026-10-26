import os
import threading
from functools import cache
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from libclaim.settings import ROOT
from libclaim.vision.boxes import Box, VisionConfig, load_vision_config

SESSION_LOCK = threading.Lock()


def model_path(cfg: VisionConfig) -> Path:
    return ROOT / os.environ.get(cfg.rfdetr.path_env, cfg.rfdetr.model_path)


@cache
def load_session(path: Path, providers: tuple[str, ...]) -> ort.InferenceSession:
    available = set(ort.get_available_providers())
    chosen = [name for name in providers if name in available]
    return ort.InferenceSession(str(path), providers=chosen)


def preprocess(img: np.ndarray, cfg: VisionConfig) -> np.ndarray:
    size = cfg.rfdetr.input_size
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    pixels = (resized.astype(np.float32) / 255.0 - np.array(cfg.rfdetr.mean, np.float32)) / np.array(
        cfg.rfdetr.std, np.float32
    )
    return pixels.transpose(2, 0, 1)[None].astype(np.float32)


def run_model(session: ort.InferenceSession, img: np.ndarray, cfg: VisionConfig) -> tuple[np.ndarray, np.ndarray]:
    names = [output.name for output in session.get_outputs()]
    wanted = [cfg.rfdetr.boxes_output, cfg.rfdetr.logits_output]
    if not set(wanted) <= set(names) and len(names) != len(wanted):
        raise ValueError(f"unexpected RF-DETR outputs {names}, expected {wanted}")
    results = dict(zip(names, session.run(None, {session.get_inputs()[0].name: preprocess(img, cfg)})))
    if set(wanted) <= set(names):
        return results[wanted[0]], results[wanted[1]]
    first, second = results.values()
    return first, second


def decode(boxes: np.ndarray, logits: np.ndarray, width: int, height: int, threshold: float) -> list[Box]:
    cutoff = np.log(threshold / (1.0 - threshold))
    scores = logits.max(axis=1)
    return [
        Box(
            x1=(cx - w / 2) * width,
            y1=(cy - h / 2) * height,
            x2=(cx + w / 2) * width,
            y2=(cy + h / 2) * height,
        )
        for cx, cy, w, h in boxes[scores >= cutoff]
    ]


def rfdetr_boxes(img: np.ndarray) -> list[Box] | None:
    cfg = load_vision_config()
    path = model_path(cfg)
    if not path.is_file():
        return None
    with SESSION_LOCK:
        session = load_session(path, tuple(cfg.rfdetr.providers))
        boxes, logits = run_model(session, img, cfg)
    height, width = img.shape[:2]
    return decode(boxes[0], logits[0], width, height, cfg.rfdetr.threshold)
