from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, TypeAdapter

M = TypeVar("M", bound=BaseModel)

LEDGER_NAME = "_ledger"


class LedgerEntry(BaseModel):
    stage: str
    seconds: float
    tokens: int
    usd_at_paid_rates: float


class Ledger:
    _adapter = TypeAdapter(list[LedgerEntry])

    def __init__(self, path: Path):
        self.path = path
        self.entries: list[LedgerEntry] = (
            self._adapter.validate_json(path.read_bytes()) if path.is_file() else []
        )

    def add(self, stage: str, seconds: float, tokens: int, usd_at_paid_rates: float) -> None:
        self.entries.append(
            LedgerEntry(
                stage=stage,
                seconds=seconds,
                tokens=tokens,
                usd_at_paid_rates=usd_at_paid_rates,
            )
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(self._adapter.dump_json(self.entries, indent=2))

    def total_usd(self) -> float:
        return sum(entry.usd_at_paid_rates for entry in self.entries)


class SweepDir:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.stages = self.root / "stages"
        self.frames = self.root / "frames"
        self.ledger = Ledger(self.stages / f"{LEDGER_NAME}.json")

    def frame_path(self, seq: int, t_ms: int) -> Path:
        return self.frames / f"{seq:05d}_{t_ms}.jpg"

    def write_stage(self, name: str, model: BaseModel) -> Path:
        path = self.stages / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
        return path

    def read_stage(self, name: str, cls: type[M]) -> M:
        return cls.model_validate_json((self.stages / f"{name}.json").read_text(encoding="utf-8"))
