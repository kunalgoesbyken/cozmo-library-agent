import asyncio
import csv
import json
import statistics
import sys
from datetime import UTC, date, datetime
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
ISBN13_LENGTH = 13
ISBN10_LENGTH = 10
ISBN13_PREFIX = "978"
ISBN13_PREFIXES = ("978", "979")
ISBN10_CHECK_MODULUS = 11
ISBN10_X = 10
ISBN13_WEIGHTS = (1, 3)
ISBN13_CHECK_MODULUS = 10
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
    old = cfg.condition_factors
    grade = cfg.calibration.grade
    median = statistics.median(pair.used / pair.new for pair in pairs).quantize(PRECISION)
    scaled = {name: (median * value / old[grade]).quantize(PRECISION) for name, value in old.items()}
    return scaled | {NEW: old[NEW], grade: median}


class RequestCapExceeded(RuntimeError):
    pass


def isbn13_check(first12: str) -> str:
    total = sum(int(digit) * ISBN13_WEIGHTS[index % 2] for index, digit in enumerate(first12))
    return str((ISBN13_CHECK_MODULUS - total % ISBN13_CHECK_MODULUS) % ISBN13_CHECK_MODULUS)


def normalize_isbn(text: str) -> str | None:
    raw = "".join(char for char in text.upper() if char.isalnum())
    if len(raw) == ISBN13_LENGTH and raw.isdigit() and raw.startswith(ISBN13_PREFIXES):
        return raw if raw[-1] == isbn13_check(raw[:-1]) else None
    body, check = raw[:-1], raw[-1:]
    if len(raw) != ISBN10_LENGTH or not body.isdigit() or not (check.isdigit() or check == "X"):
        return None
    total = sum((ISBN10_LENGTH - index) * int(digit) for index, digit in enumerate(body))
    total += ISBN10_X if check == "X" else int(check)
    if total % ISBN10_CHECK_MODULUS:
        return None
    return ISBN13_PREFIX + body + isbn13_check(ISBN13_PREFIX + body)


def held_out(path: Path, column: str) -> set[str]:
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if column not in (reader.fieldnames or []):
            raise ValueError(f"{path.name} has no '{column}' column")
        return {isbn for row in reader if (isbn := normalize_isbn(row[column]))}


def sample_size(cfg: PricingConfig) -> int:
    sources = len(cfg.calibration.sources)
    size = (cfg.calibration.max_requests - sources) // sources
    if size * sources < cfg.calibration.min_pairs:
        raise ValueError(
            f"max_requests {cfg.calibration.max_requests} cannot reach min_pairs {cfg.calibration.min_pairs} with {sources} sources"
        )
    return size


def choose_isbns(isbns: list[str], cfg: PricingConfig, ground_truth: Path) -> list[str]:
    size = sample_size(cfg)
    excluded = held_out(ground_truth, cfg.calibration.isbn_column)
    wanted = dict.fromkeys(isbn for text in isbns if (isbn := normalize_isbn(text)) and isbn not in excluded)
    return list(wanted)[:size]


def cap_requests(http: PriceHttp, limit: int) -> None:
    sent: list[str] = []

    async def count(request) -> None:
        if len(sent) >= limit:
            raise RequestCapExceeded(f"request cap of {limit} reached")
        sent.append(str(request.url))

    http.client.event_hooks["request"].append(count)


async def calibrate(isbns: list[str], sources: list, cfg: PricingConfig, out_dir: Path, on: date) -> Path:
    pairs = await collect_pairs(sources, isbns)
    if len(pairs) < cfg.calibration.min_pairs:
        raise ValueError(f"{len(pairs)} pairs found, {cfg.calibration.min_pairs} needed")
    snapshot = {
        "version": on.isoformat(),
        "base_version": cfg.books.factor_version,
        "grade": cfg.calibration.grade,
        "condition_factors": {name: str(value) for name, value in build_factors(pairs, cfg).items()},
        "pairs": [pair.model_dump(mode="json") for pair in pairs],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"condition_factors.{on.isoformat()}.json"
    path.write_text(json.dumps(snapshot, indent=1), encoding="utf-8")
    return path


async def main(isbn_file: Path, cache_dir: Path) -> None:
    cfg = load_pricing_config()
    http = PriceHttp(cache_dir, config=cfg)
    cap_requests(http, cfg.calibration.max_requests)
    sources = [SOURCES[name](http, cfg) for name in cfg.calibration.sources]
    try:
        isbns = choose_isbns(isbn_file.read_text(encoding="utf-8").split(), cfg, ROOT / cfg.calibration.ground_truth_csv)
        path = await calibrate(isbns, sources, cfg, CONFIG_DIR / cfg.calibration.snapshot_dir, datetime.now(UTC).date())
    except (ValueError, RequestCapExceeded) as error:
        sys.exit(f"calibration failed: {error}")
    print(path)
    print(f"set books.factors_snapshot to {path.stem} in pricing.yaml to adopt it")


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]), Path(sys.argv[2])))
