import yaml
from pydantic import BaseModel

from libclaim.settings import CONFIG_DIR


class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)


class TileConfig(BaseModel):
    cols: int
    rows: int
    overlap: float


class RfdetrConfig(BaseModel):
    model_path: str
    path_env: str
    input_size: int
    threshold: float
    mean: list[float]
    std: list[float]
    providers: list[str]


class ReconcileConfig(BaseModel):
    match_iou: float
    count_diff_fraction: float
    gap_thickness_factor: float
    max_requery_rows: int


class TrackConfig(BaseModel):
    orb_features: int
    match_ratio: float
    ransac_reproj_px: float
    min_inliers: int
    track_iou: float


class VisionConfig(BaseModel):
    tiles: TileConfig
    jpeg_quality: int
    jpeg_mime: str
    box_scale: int
    nms_iou: float
    prompt: str
    rfdetr: RfdetrConfig
    reconcile: ReconcileConfig
    tracks: TrackConfig


def load_vision_config() -> VisionConfig:
    path = CONFIG_DIR / "vision.yaml"
    return VisionConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def iou(a: Box, b: Box) -> float:
    inter = Box(
        x1=max(a.x1, b.x1), y1=max(a.y1, b.y1), x2=min(a.x2, b.x2), y2=min(a.y2, b.y2)
    ).area
    union = a.area + b.area - inter
    return inter / union if union else 0.0


def nms(boxes: list[Box], threshold: float) -> list[int]:
    kept: list[int] = []
    for index in sorted(range(len(boxes)), key=lambda i: boxes[i].area, reverse=True):
        if all(iou(boxes[index], boxes[other]) <= threshold for other in kept):
            kept.append(index)
    return sorted(kept)
