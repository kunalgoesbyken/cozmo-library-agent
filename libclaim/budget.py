import json
from pathlib import Path

import yaml

from libclaim.settings import CONFIG_DIR, ROOT

PRICES_PATH = CONFIG_DIR / "model_prices.yaml"
DEFAULT_LEDGER = ROOT / "runs" / "spend.jsonl"
DEFAULT_CAP_USD = 2.00
TOKENS_PER_PRICE_UNIT = 1_000_000


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, path: Path = DEFAULT_LEDGER, cap_usd: float = DEFAULT_CAP_USD):
        self.path = Path(path)
        self.cap_usd = cap_usd
        self.pending = 0.0
        self.models = yaml.safe_load(PRICES_PATH.read_text(encoding="utf-8"))["models"]
        rates = [rate for table in self.models.values() for rate in table.values()]
        self.worst = {
            "input": max(rate["input"] for rate in rates),
            "output": max(rate["output"] for rate in rates),
        }

    def rates(self, model: str, modality: str) -> dict[str, float]:
        return self.models.get(model, {}).get(modality, self.worst)

    def record(self, model: str, input_tokens: int, output_tokens: int, modality: str = "text") -> float:
        rate = self.rates(model, modality)
        usd = (input_tokens * rate["input"] + output_tokens * rate["output"]) / TOKENS_PER_PRICE_UNIT
        entry = {
            "model": model,
            "modality": modality,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "usd": usd,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        return usd

    def total_usd(self) -> float:
        if not self.path.is_file():
            return 0.0
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return sum(json.loads(line)["usd"] for line in lines if line)

    def check(self) -> None:
        total = self.total_usd() + self.pending
        if total >= self.cap_usd:
            raise BudgetExceeded(f"spend {total:.4f} USD reached cap {self.cap_usd:.2f} USD")

    def reserve(self, usd: float) -> None:
        total = self.total_usd() + self.pending
        if total + usd > self.cap_usd:
            raise BudgetExceeded(f"spend {total:.4f} USD plus reserve {usd:.4f} USD exceeds cap {self.cap_usd:.2f} USD")
        self.pending += usd

    def release(self, usd: float) -> None:
        self.pending = max(0.0, self.pending - usd)
