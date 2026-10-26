from collections.abc import Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from libclaim.live.bridge import LiveConfig
from libclaim.live.session import ToolSpec
from libclaim.models import Still
from libclaim.settings import load_locale
from libclaim.sweep import SweepDir

STATEMENTS_STAGE = "statements"
EXCLUSIONS_STAGE = "exclusions"
LOCALE_STAGE = "locale"
MONITOR_NAME = "start_capture_monitor"


class StatementKind(StrEnum):
    first_edition = "first_edition"
    signed = "signed"
    original_art = "original_art"
    other = "other"


class Statement(BaseModel):
    subject: str
    kind: StatementKind
    value: str
    t_ms: int
    still_seq_from: int | None = None
    still_seq_to: int | None = None

    @property
    def needs_appraisal(self) -> bool:
        return self.kind != StatementKind.other


class StatementLog(BaseModel):
    items: list[Statement] = Field(default_factory=list)


class Exclusion(BaseModel):
    description: str
    shelf_id: str | None = None
    t_ms: int


class ExclusionLog(BaseModel):
    items: list[Exclusion] = Field(default_factory=list)


class LocaleChoice(BaseModel):
    country: str
    currency: str


SPECS = [
    ToolSpec(
        name="record_user_statement",
        description="Record something the user said about an item they are showing, such as a first edition, a signed copy or an original artwork.",
        parameters={
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "kind": {"type": "string", "enum": [kind.value for kind in StatementKind]},
                "value": {"type": "string"},
            },
            "required": ["subject", "kind", "value"],
        },
    ),
    ToolSpec(
        name="exclude_region",
        description="Exclude a shelf or region the user says is not theirs or should be skipped.",
        parameters={
            "type": "object",
            "properties": {"description": {"type": "string"}, "shelf_id": {"type": "string"}},
            "required": ["description"],
        },
    ),
    ToolSpec(
        name="set_locale",
        description="Set the country and currency used for the valuation.",
        parameters={
            "type": "object",
            "properties": {"country": {"type": "string"}, "currency": {"type": "string"}},
            "required": ["country", "currency"],
        },
    ),
    ToolSpec(name="get_sweep_status", description="Return the counts captured so far in this sweep."),
    ToolSpec(name="end_sweep", description="Finish the sweep when the user says they are done."),
    ToolSpec(
        name=MONITOR_NAME,
        description="Start the capture monitor that streams inventory events into the conversation.",
        non_blocking=True,
    ),
]


class LiveTools:
    monitor_name = MONITOR_NAME

    def __init__(
        self,
        sweep: SweepDir,
        config: LiveConfig,
        stills: Callable[[], list[Still]],
        now_ms: Callable[[], int],
        on_end_sweep: Callable[[], None],
    ):
        self.sweep = sweep
        self.config = config
        self.stills = stills
        self.now_ms = now_ms
        self.on_end_sweep = on_end_sweep
        self.handlers: dict[str, Callable[..., dict[str, Any]]] = {
            "record_user_statement": self.record_user_statement,
            "exclude_region": self.exclude_region,
            "set_locale": self.set_locale,
            "get_sweep_status": self.get_sweep_status,
            "end_sweep": self.end_sweep,
            MONITOR_NAME: self.start_capture_monitor,
        }

    async def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        handler = self.handlers.get(name)
        if handler is None:
            return {"error": f"unknown tool {name}"}
        try:
            return handler(**args)
        except (TypeError, ValueError, KeyError, ValidationError) as error:
            return {"error": f"{name} rejected: {error}"}

    def read_log(self, stage: str, cls: type[BaseModel]) -> Any:
        path = self.sweep.stages / f"{stage}.json"
        return self.sweep.read_stage(stage, cls) if path.is_file() else cls()

    def record_user_statement(self, subject: str, kind: str, value: str) -> dict[str, Any]:
        t_ms = self.now_ms()
        window = [
            still.seq
            for still in self.stills()
            if t_ms - self.config.statement_window_ms <= still.t_ms <= t_ms
        ]
        statement = Statement(
            subject=subject,
            kind=StatementKind(kind),
            value=value,
            t_ms=t_ms,
            still_seq_from=min(window) if window else None,
            still_seq_to=max(window) if window else None,
        )
        log = self.read_log(STATEMENTS_STAGE, StatementLog)
        log.items.append(statement)
        self.sweep.write_stage(STATEMENTS_STAGE, log)
        return {"recorded": True, "needs_appraisal": statement.needs_appraisal}

    def exclude_region(self, description: str, shelf_id: str | None = None) -> dict[str, Any]:
        log = self.read_log(EXCLUSIONS_STAGE, ExclusionLog)
        log.items.append(Exclusion(description=description, shelf_id=shelf_id, t_ms=self.now_ms()))
        self.sweep.write_stage(EXCLUSIONS_STAGE, log)
        return {"excluded": True}

    def set_locale(self, country: str, currency: str) -> dict[str, Any]:
        locale = load_locale(country)
        choice = LocaleChoice(country=locale.country, currency=currency)
        self.sweep.write_stage(LOCALE_STAGE, choice)
        return {"country": choice.country, "currency": choice.currency}

    def get_sweep_status(self) -> dict[str, Any]:
        frames = self.sweep.frames
        return {
            "frames": sum(1 for _ in frames.glob("*.jpg")) if frames.is_dir() else 0,
            "statements": len(self.read_log(STATEMENTS_STAGE, StatementLog).items),
            "exclusions": len(self.read_log(EXCLUSIONS_STAGE, ExclusionLog).items),
        }

    def end_sweep(self) -> dict[str, Any]:
        self.on_end_sweep()
        return {"ended": True}

    def start_capture_monitor(self) -> dict[str, Any]:
        return {"monitoring": True}
