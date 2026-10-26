from collections.abc import Iterable

from libclaim.capture.quality import CaptureConfig
from libclaim.models import Still


class CoverageMap:
    def __init__(self, cfg: CaptureConfig):
        self.cfg = cfg
        self.seen: set[str] = set()
        self.sharp: dict[str, list[int]] = {}

    def record(self, region: str, still: Still) -> None:
        self.seen.add(region)
        if still.blur >= self.cfg.blur_floor:
            self.sharp.setdefault(region, []).append(still.seq)

    def missed_regions(self, expected: Iterable[str] = ()) -> list[str]:
        return sorted((self.seen | set(expected)) - self.sharp.keys())
