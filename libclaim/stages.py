from functools import cache

import yaml
from pydantic import BaseModel, Field

from libclaim.live.tools import Statement
from libclaim.models import Still
from libclaim.packet.models import SecondCountry, Severity
from libclaim.pricing.books import BookPrices
from libclaim.pricing.items import ItemPrice
from libclaim.room.geometry import RoomGeometry
from libclaim.settings import CONFIG_DIR
from libclaim.vision.boxes import Box
from libclaim.vision.items import ItemDet
from libclaim.vision.reading import Reading
from libclaim.vision.resolve import Resolution
from libclaim.vision.scale import SpineDims
from libclaim.vision.spines import Orientation, SpineDet
from libclaim.vision.spines import ReviewEntry as ReconcileNote
from libclaim.vision.tracks import BookTrack

STAGES = [
    "keyframes", "spines", "tracks", "scale", "reading", "resolution",
    "room", "items", "appraisal", "pricing", "packet",
]
STREAM_STAGES = STAGES[:2]
SEGMENT_STAGES = STAGES[2:6]
POST_STAGES = STAGES[6:]
STILLS = "stills"
FAILURES = "failures"
VISION_CUTOFF = "vision_cutoff"


class SummaryText(BaseModel):
    books: str
    book_value: str
    items: str
    room: str
    no_room: str
    review: str


class Readback(BaseModel):
    instruction: str


class PipelineConfig(BaseModel):
    deadline_s: float
    vision_cutoff_s: float
    deadline_poll_s: float
    row_min_spines: int
    row_min_spine_height_frac: float
    segment_max_frames: int
    revisit_tails: int
    statement_subject_match: float
    stage_aliases: dict[str, str]
    packet_name: str
    llm_cache_dir: str
    runs_url: str
    cm_decimals: int
    severity: dict[str, Severity]
    money_format: str
    area_format: str
    summary: SummaryText
    readback: Readback


@cache
def load_pipeline_config() -> PipelineConfig:
    path = CONFIG_DIR / "pipeline.yaml"
    return PipelineConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def item_key(index: int) -> str:
    return f"i_{index + 1:03d}"


def book_key(segment: int, index: int) -> str:
    return f"b{segment + 1:02d}-{index + 1:03d}"


class StillLog(BaseModel):
    stills: list[Still] = Field(default_factory=list)


class KeyframesStage(BaseModel):
    keyframes: list[Still] = Field(default_factory=list)


class FrameSpines(BaseModel):
    still: Still
    rows: bool
    gemini: list[SpineDet] = Field(default_factory=list)
    local: list[Box] | None = None
    dets: list[SpineDet] = Field(default_factory=list)
    notes: list[ReconcileNote] = Field(default_factory=list)
    flagged: bool = False
    requeried: bool = False
    failure: str = ""


class SegmentPlan(BaseModel):
    index: int
    label: str
    unit: int | None = None
    seqs: list[int]
    carry: list[int] = Field(default_factory=list)


class SpinesStage(BaseModel):
    frames: list[FrameSpines] = Field(default_factory=list)
    segments: list[SegmentPlan] = Field(default_factory=list)


class BookSlot(BaseModel):
    id: str
    segment: int
    track: int
    shelf: str
    position: int
    orientation: Orientation
    frame: Still
    box: Box
    seen: list[int]


class SegmentTracks(BaseModel):
    index: int
    stills: list[Still]
    tracks: list[BookTrack]
    rows: list[list[int]]
    flat: list[int]
    books: list[BookSlot]


class TracksStage(BaseModel):
    segments: dict[int, SegmentTracks] = Field(default_factory=dict)


class ScaleStage(BaseModel):
    dims: dict[str, SpineDims] = Field(default_factory=dict)
    planes: dict[str, Box] = Field(default_factory=dict)


class BookRead(BaseModel):
    reading: Reading
    ocr_text: str = ""
    ocr_score: float = 0.0


class ReadingStage(BaseModel):
    reads: dict[str, BookRead] = Field(default_factory=dict)


class ResolutionStage(BaseModel):
    resolutions: dict[str, Resolution] = Field(default_factory=dict)


class ItemsStage(BaseModel):
    items: list[ItemDet] = Field(default_factory=list)
    failure: str = ""
    review: list[str] = Field(default_factory=list)


class BookAppraisal(BaseModel):
    statements: list[Statement] = Field(default_factory=list)
    excluded: str = ""


class AppraisalStage(BaseModel):
    books: dict[str, BookAppraisal] = Field(default_factory=dict)
    review: list[str] = Field(default_factory=list)


class StageFailure(BaseModel):
    stage: str
    ref: str
    error: str


class FailureLog(BaseModel):
    items: list[StageFailure] = Field(default_factory=list)


class PricedItem(BaseModel):
    listing: str = ""
    price: ItemPrice


class PricingStage(BaseModel):
    books: dict[str, BookPrices] = Field(default_factory=dict)
    items: dict[str, PricedItem] = Field(default_factory=dict)
    second_country: SecondCountry | None = None


class Results(BaseModel):
    stills: StillLog = Field(default_factory=StillLog)
    keyframes: KeyframesStage = Field(default_factory=KeyframesStage)
    spines: SpinesStage = Field(default_factory=SpinesStage)
    tracks: TracksStage = Field(default_factory=TracksStage)
    scale: ScaleStage = Field(default_factory=ScaleStage)
    reading: ReadingStage = Field(default_factory=ReadingStage)
    resolution: ResolutionStage = Field(default_factory=ResolutionStage)
    room: RoomGeometry | None = None
    items: ItemsStage | None = None
    appraisal: AppraisalStage = Field(default_factory=AppraisalStage)
    pricing: PricingStage = Field(default_factory=PricingStage)
    failures: FailureLog = Field(default_factory=FailureLog)
