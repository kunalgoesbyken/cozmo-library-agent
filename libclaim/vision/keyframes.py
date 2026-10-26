from enum import StrEnum

import cv2
import numpy as np
import yaml
from pydantic import BaseModel

from libclaim.models import Still
from libclaim.settings import CONFIG_DIR


class Phase(StrEnum):
    rows = "rows"
    room = "room"


class PhaseMarker(BaseModel):
    t_ms: int
    phase: Phase


class Keyframes(BaseModel):
    rows: list[Still]
    room: list[Still]


class KeyframeConfig(BaseModel):
    window_ms: int
    max_hamming: int
    hash_size: int
    dct_size: int


def load_keyframe_config() -> KeyframeConfig:
    path = CONFIG_DIR / "keyframes.yaml"
    return KeyframeConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def phash(path: str, cfg: KeyframeConfig) -> np.ndarray:
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError(f"unreadable still image: {path}")
    small = cv2.resize(gray, (cfg.dct_size, cfg.dct_size), interpolation=cv2.INTER_AREA)
    low = cv2.dct(small.astype(np.float32))[: cfg.hash_size, : cfg.hash_size]
    return low >= np.median(low)


def phase_at(t_ms: int, markers: list[PhaseMarker]) -> Phase | None:
    phase = None
    for marker in sorted(markers, key=lambda m: m.t_ms):
        if marker.t_ms <= t_ms:
            phase = marker.phase
    return phase


def select_keyframes(stills: list[Still], events: list[PhaseMarker]) -> Keyframes:
    cfg = load_keyframe_config()
    sharpest: dict[tuple[Phase, int], Still] = {}
    for still in sorted(stills, key=lambda s: s.t_ms):
        phase = phase_at(still.t_ms, events)
        if phase is None:
            continue
        key = (phase, still.t_ms // cfg.window_ms)
        if key not in sharpest or still.blur > sharpest[key].blur:
            sharpest[key] = still
    kept: dict[Phase, list[tuple[Still, np.ndarray]]] = {phase: [] for phase in Phase}
    for (phase, _), still in sorted(sharpest.items(), key=lambda item: item[1].t_ms):
        digest = phash(still.path, cfg)
        if all(np.count_nonzero(digest != other) > cfg.max_hamming for _, other in kept[phase]):
            kept[phase].append((still, digest))
    return Keyframes(
        rows=[still for still, _ in kept[Phase.rows]],
        room=[still for still, _ in kept[Phase.room]],
    )
