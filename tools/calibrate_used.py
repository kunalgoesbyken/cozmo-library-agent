import asyncio
import csv
import json
import statistics
import sys
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel

from libclaim.models import PriceQuote
from libclaim.pricing.config import PricingConfig, load_pricing_config
from libclaim.pricing.http import PriceHttp
from libclaim.pricing.sources.bookchor import Bookchor
from libclaim.pricing.sources.pustakkosh import Pustakkosh
from libclaim.settings import CONFIG_DIR, ROOT

NEW = "new"
USED = "used"
PRECISION = Decimal("0.001")
SOURCES = {"bookchor": Bookchor, "pustakkosh": Pustakkosh}


class Pair(BaseModel):
    isbn: str
    source: str
    new: Decimal
    used: Decimal
    new_url: str
    used_url: str
    new_cache_ref: str
    used_cache_ref: str


async def collect_pairs(sources: list, isbns: list[str]) -> list[Pair]:
    pairs = []
    for isbn in isbns:
        for source in sources:
            new, used = await source.quote(isbn, NEW), await source.quote(isbn, USED)
            if isinstance(new, PriceQuote) and isinstance(used, PriceQuote):
                pairs.append(
                    Pair(
                        isbn=isbn,
                        source=new.source,
                        new=new.amount,
                        used=used.amount,
                        new_url=new.url,
                        used_url=used.url,
                        new_cache_ref=new.cache_ref,
                        used_cache_ref=used.cache_ref,
                    )
                )
    return pairs


def build_factors(pairs: list[Pair], cfg: PricingConfig) -> dict[str, Decimal]:
    old = cfg.condition_factors[cfg.books.factor_version]
    grade = cfg.calibration.grade
    median = statistics.median(pair.used / pair.new for pair in pairs).quantize(PRECISION)
    scaled = {name: (median * value / old[grade]).quantize(PRECISION) for name, value in old.items()}
    return scaled | {NEW: old[NEW], grade: median}


def held_out(path: Path, column: str) -> set[str]:
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        return {row[column] for row in csv.DictReader(handle)}


def choose_isbns(isbns: list[str], cfg: PricingConfig, ground_truth: Path) -> list[str]:
    sources = len(cfg.calibration.sources)
    limit = cfg.calibration.max_requests // sources - 1
    excluded = held_out(ground_truth, cfg.calibration.isbn_column)
    return [isbn for isbn in dict.fromkeys(isbns) if isbn not in excluded][:limit]


async def calibrate(isbns: list[str], sources: list, cfg: PricingConfig, out_dir: Path) -> Path:
    pairs = await collect_pairs(sources, isbns)
    if len(pairs) < cfg.calibration.min_pairs:
        raise ValueError(f"{len(pairs)} pairs found, {cfg.calibration.min_pairs} needed")
    snapshot = {
        "version": cfg.books.factor_version,
        "grade": cfg.calibration.grade,
        "condition_factors": {name: str(value) for name, value in build_factors(pairs, cfg).items()},
        "pairs": [pair.model_dump(mode="json") for pair in pairs],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"condition_factors.{cfg.books.factor_version}.json"
    path.write_text(json.dumps(snapshot, indent=1), encoding="utf-8")
    return path


async def main(isbn_file: Path, cache_dir: Path) -> None:
    cfg = load_pricing_config()
    http = PriceHttp(cache_dir, config=cfg)
    sources = [SOURCES[name](http, cfg) for name in cfg.calibration.sources]
    isbns = choose_isbns(isbn_file.read_text(encoding="utf-8").split(), cfg, ROOT / cfg.calibration.ground_truth_csv)
    print(await calibrate(isbns, sources, cfg, CONFIG_DIR / cfg.calibration.snapshot_dir))


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]), Path(sys.argv[2])))
