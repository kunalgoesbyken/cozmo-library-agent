from functools import cache

import cv2
import numpy as np
import yaml
from pydantic import BaseModel, Field

from libclaim.settings import CONFIG_DIR

PRIOR_CONFIDENCE_CAP = 0.4


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
    boxes_output: str
    logits_output: str
    providers: list[str]


class ReconcileConfig(BaseModel):
    match_iou: float
    count_diff_fraction: float
    gap_thickness_factor: float
    max_requery_rows: int
    phantom_containment: float
    vertical_min_aspect: float
    sweep_label: str


class TrackConfig(BaseModel):
    orb_features: int
    match_ratio: float
    ransac_reproj_px: float
    min_inliers: int
    track_iou: float
    edge_px: float
    cut_containment: float
    noise_min_views: int
    noise_max_rate: float
    anchor_lookback: int
    det_min: float
    det_max: float
    area_ratio_min: float
    area_ratio_max: float


class ScaleConfig(BaseModel):
    aruco_dictionary: str
    corner_refinement: str
    min_marker_px: float
    aruco_confidence: float
    chained_confidence_factor: float
    prior_confidence: float = Field(le=PRIOR_CONFIDENCE_CAP)
    spine_recess_cm: float
    box_height_trim: float
    focal_key: str
    fallback_long_side_fov_deg: float
    shared_gap_factor: float
    cross_check_tolerance: float
    length_units_cm: dict[str, float]


class MaskConfig(BaseModel):
    model_id: str
    band: float
    height_tolerance: float
    min_width_ratio: float
    max_width_ratio: float


class MarkerConfig(BaseModel):
    print_size_cm: float
    check_line_cm: float
    line_width_cm: float
    page_cm: tuple[float, float]
    margin_cm: float
    gap_cm: float
    label_height_cm: float
    dpi: int


class ItemConfig(BaseModel):
    batch_size: int
    prompt: str
    art_categories: list[str]
    dedup_iou: float
    dedup_lookback: int
    depth_core: float
    min_depth_confidence: float


class VisionConfig(BaseModel):
    tiles: TileConfig
    jpeg_quality: int
    jpeg_mime: str
    box_scale: int
    nms_iou: float
    nms_containment: float
    prompt: str
    rfdetr: RfdetrConfig
    reconcile: ReconcileConfig
    tracks: TrackConfig
    scale: ScaleConfig
    masks: MaskConfig
    markers: MarkerConfig
    items: ItemConfig


class ImageUnreadable(ValueError):
    pass


@cache
def load_vision_config() -> VisionConfig:
    path = CONFIG_DIR / "vision.yaml"
    return VisionConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def read_image(path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    img = cv2.imread(path, flags)
    if img is None:
        raise ImageUnreadable(f"cannot read image {path}")
    return img


def intersection(a: Box, b: Box) -> float:
    return Box(x1=max(a.x1, b.x1), y1=max(a.y1, b.y1), x2=min(a.x2, b.x2), y2=min(a.y2, b.y2)).area


def iou(a: Box, b: Box) -> float:
    inter = intersection(a, b)
    union = a.area + b.area - inter
    return inter / union if union else 0.0


def containment(a: Box, b: Box) -> float:
    smaller = min(a.area, b.area)
    return intersection(a, b) / smaller if smaller else 0.0


def nms(boxes: list[Box], iou_threshold: float, containment_threshold: float) -> list[int]:
    kept: list[int] = []
    for index in sorted(range(len(boxes)), key=lambda i: boxes[i].area, reverse=True):
        if all(
            iou(boxes[index], boxes[other]) <= iou_threshold
            and containment(boxes[index], boxes[other]) < containment_threshold
            for other in kept
        ):
            kept.append(index)
    return sorted(kept)
