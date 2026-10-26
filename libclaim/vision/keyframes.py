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


class KeyframeStream:
    def __init__(self, cfg: KeyframeConfig):
        self.cfg = cfg
        self.window: int | None = None
        self.best: Still | None = None
        self.digests: list[np.ndarray] = []

    def add(self, still: Still) -> list[Still]:
        window = still.t_ms // self.cfg.window_ms
        out = self.flush() if self.window is not None and window != self.window else []
        if self.best is None or still.blur > self.best.blur:
            self.best = still
        self.window = window
        return out

    def flush(self) -> list[Still]:
        best, self.best, self.window = self.best, None, None
        if best is None:
            return []
        digest = phash(best.path, self.cfg)
        if any(np.count_nonzero(digest != other) <= self.cfg.max_hamming for other in self.digests):
            return []
        self.digests.append(digest)
        return [best]


def select_keyframes(stills: list[Still], events: list[PhaseMarker]) -> Keyframes:
    cfg = load_keyframe_config()
    streams = {phase: KeyframeStream(cfg) for phase in Phase}
    kept: dict[Phase, list[Still]] = {phase: [] for phase in Phase}
    for still in sorted(stills, key=lambda s: s.t_ms):
        phase = phase_at(still.t_ms, events)
        if phase is not None:
            kept[phase] += streams[phase].add(still)
    for phase, stream in streams.items():
        kept[phase] += stream.flush()
    return Keyframes(rows=kept[Phase.rows], room=kept[Phase.room])
