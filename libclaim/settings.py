import os
from decimal import Decimal
from pathlib import Path

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
APP_NAME = "libclaim"
APP_VERSION = "0.1"


class Locale(BaseModel):
    country: str
    currency: str
    new_sources: list[str]
    used_sources: list[str]
    appraisal_threshold: Decimal
    antiquarian_year: int


class Settings(BaseModel):
    gemini_api_key: str = ""
    google_books_api_key: str = ""
    live_model: str = ""
    vision_models: list[str] = []
    tiebreak_model: str = ""
    checker_model: str = ""
    live_voice: str = ""
    default_country: str = "IN"
    second_country: str = "GB"
    aruco_marker_size_cm: float | None = None
    http_contact_email: str = ""
    hf_token: str = ""
    roboflow_api_key: str = ""
    port: int = 8000
    access_token: str = ""

    @property
    def user_agent(self) -> str:
        agent = f"{APP_NAME}/{APP_VERSION}"
        if self.http_contact_email:
            agent += f" (+mailto:{self.http_contact_email})"
        return agent


def load_locale(country: str) -> Locale:
    available = {path.stem.upper(): path for path in (CONFIG_DIR / "locales").glob("*.yaml")}
    path = available.get(country.upper())
    if path is None:
        raise KeyError(country)
    return Locale.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def load_settings(env_file: Path = ROOT / ".env") -> Settings:
    values = {**dotenv_values(env_file), **os.environ}
    fields = {}
    for name in Settings.model_fields:
        raw = values.get(name.upper())
        if raw:
            fields[name] = raw.split(",") if name == "vision_models" else raw
    return Settings.model_validate(fields)
