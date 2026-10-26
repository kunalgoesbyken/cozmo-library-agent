from functools import cache

import yaml
from pydantic import BaseModel

from libclaim.settings import CONFIG_DIR


class ReadingConfig(BaseModel):
    batch_size: int
    jpeg_quality: int
    jpeg_mime: str
    vertical_rotation: str
    prompt: str


class OcrConfig(BaseModel):
    use_dml: bool
    log_level: str


class GateConfig(BaseModel):
    ocr_overlap_min: float
    token_match: float
    score_min: float
    margin_min: float
    distinctive_min_tokens: int
    distinctive_min_chars: int
    author_match: float
    author_weight: float
    confident_score: float
    agree_score: float
    publisher_match: float


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
    catalog: CatalogConfig


@cache
def load_resolve_config() -> ResolveConfig:
    path = CONFIG_DIR / "resolve.yaml"
    return ResolveConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
