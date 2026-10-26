from pathlib import Path

import cv2
import numpy as np

from libclaim.packet.models import Locator
from libclaim.stages import CropConfig
from libclaim.sweep import SweepDir
from libclaim.vision.boxes import Box


class Cropper:
    def __init__(self, sweep: SweepDir, cfg: CropConfig):
        self.sweep = sweep
        self.cfg = cfg
        self.frames: dict[str, np.ndarray | None] = {}

    def frame(self, path: str) -> np.ndarray | None:
        name = Path(path).name
        if name not in self.frames:
            self.frames[name] = cv2.imread(str(self.sweep.frames / name)) if path else None
        return self.frames[name]

    def locate(self, key: str, path: str, box: Box) -> Locator:
        image = self.frame(path)
        if image is None:
            return Locator()
        height, width = image.shape[:2]
        x1, y1 = max(int(box.x1), 0), max(int(box.y1), 0)
        x2, y2 = min(round(box.x2), width), min(round(box.y2), height)
        if x2 <= x1 or y2 <= y1:
            return Locator()
        return Locator(crop_ref=self.write(key, image, (x1, y1, x2, y2)), box=[x1, y1, x2, y2],
                       frame_size=[width, height])

    def write(self, key: str, image: np.ndarray, box: tuple[int, int, int, int]) -> str:
        x1, y1, x2, y2 = box
        pad_x, pad_y = round((x2 - x1) * self.cfg.margin_frac), round((y2 - y1) * self.cfg.margin_frac)
        height, width = image.shape[:2]
        crop = image[max(y1 - pad_y, 0) : min(y2 + pad_y, height), max(x1 - pad_x, 0) : min(x2 + pad_x, width)]
        longest = max(crop.shape[:2])
        if longest > self.cfg.max_side_px:
            factor = self.cfg.max_side_px / longest
            crop = cv2.resize(crop, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        target = self.sweep.frames / self.cfg.dir / f"{key}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), crop, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality])
        return target.relative_to(self.sweep.root).as_posix()
