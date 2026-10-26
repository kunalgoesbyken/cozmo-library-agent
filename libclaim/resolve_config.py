from functools import cache

import yaml
from pydantic import BaseModel

from libclaim.settings import CONFIG_DIR


class ReadingConfig(BaseModel):
    batch_size: int
    jpeg_quality: int
    jpeg_mime: str
    rotations: list[str]
    prompt: str


class OcrConfig(BaseModel):
    use_dml: bool
    log_level: str
    use_cls: bool
    det_limit_type: str
    det_limit_side_len: int
    threads: int
    inter_threads: int
    workers: int
    min_yield: float


class GateConfig(BaseModel):
    ocr_overlap_min: float
    ocr_contradiction_yield: float
    catalogue_only_score: float
    catalogue_only_title: float
    catalogue_only_margin: float
    catalogue_only_confidence: float
    repeat_iou: float
    token_match: float
    score_min: float
    margin_min: float
    distinctive_min_tokens: int
    distinctive_min_chars: int
    author_match: float
    author_weight: float
    confident_score: float
    lone_score_min: float
    extra_token_penalty: float
    agree_score: float
    publisher_match: float


class EditionConfig(BaseModel):
    title_match: float
    preferred_formats: list[str]


class OpenLibraryConfig(BaseModel):
    search_url: str
    editions_url: str
    site_url: str
    search_fields: str


class GoogleBooksConfig(BaseModel):
    volumes_url: str
    key_header: str


class CatalogConfig(BaseModel):
    cache_dir: str
    timeout_seconds: float
    requests_per_second: float
    search_limit: int
    editions_limit: int
    openlibrary: OpenLibraryConfig
    googlebooks: GoogleBooksConfig


class ResolveConfig(BaseModel):
    reading: ReadingConfig
    ocr: OcrConfig
    gates: GateConfig
    editions: EditionConfig
    catalog: CatalogConfig


@cache
def load_resolve_config() -> ResolveConfig:
    path = CONFIG_DIR / "resolve.yaml"
    return ResolveConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
