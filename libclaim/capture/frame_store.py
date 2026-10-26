import cv2
import numpy as np
from pydantic import BaseModel

from libclaim.capture.quality import CaptureConfig, score
from libclaim.models import Still
from libclaim.sweep import SweepDir


class Motion(BaseModel):
    rot_dps: float = 0.0
    acc: float = 0.0


class FrameHeader(BaseModel):
    seq: int
    t_ms: int
    w: int
    h: int
    motion: Motion = Motion()
    orient: str | None = None
    pose: list[float] | None = None
    intrinsics: dict[str, float] | None = None
    depth_path: str | None = None


class FrameStore:
    def __init__(self, sweep: SweepDir, cfg: CaptureConfig):
        self.sweep = sweep
        self.cfg = cfg

    def add(self, header: FrameHeader, jpeg: bytes) -> Still:
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("undecodable jpeg")
        quality = score(img, self.cfg)
        path = self.sweep.frame_path(header.seq, header.t_ms)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(jpeg)
        return Still(
            seq=header.seq,
            t_ms=header.t_ms,
            path=str(path),
            w=header.w,
            h=header.h,
            blur=quality.blur,
            glare=quality.glare,
            motion_dps=header.motion.rot_dps,
            pose=header.pose,
            intrinsics=header.intrinsics,
            depth_path=header.depth_path,
        )
