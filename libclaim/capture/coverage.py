from collections.abc import Iterable

from libclaim.capture.quality import CaptureConfig
from libclaim.models import Still


class CoverageMap:
    def __init__(self, cfg: CaptureConfig):
        self.cfg = cfg
        self.placed: dict[int, tuple[str, bool]] = {}

    def record(self, region: str, still: Still) -> None:
        self.placed[still.seq] = (region, still.blur >= self.cfg.blur_floor)

    def assign(self, seq: int, region: str) -> None:
        if seq in self.placed:
            self.placed[seq] = (region, self.placed[seq][1])

    @property
    def seen(self) -> set[str]:
        return {region for region, _ in self.placed.values()}

    @property
    def sharp(self) -> dict[str, list[int]]:
        found: dict[str, list[int]] = {}
        for seq, (region, is_sharp) in self.placed.items():
            if is_sharp:
                found.setdefault(region, []).append(seq)
        return found

    def missed_regions(self, expected: Iterable[str] = ()) -> list[str]:
        return sorted((self.seen | set(expected)) - self.sharp.keys())
