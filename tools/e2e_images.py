import argparse
import asyncio
import json
import shutil
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from libclaim.budget import Budget
from libclaim.capture.quality import load_capture_config
from libclaim.packet.models import ClaimPacket
from libclaim.pipeline import SweepRun, services_for, still_from_file
from libclaim.settings import ROOT, load_settings
from libclaim.stages import load_pipeline_config
from libclaim.sweep import SweepDir

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MS_PER_S = 1000
DEFAULT_SPACING_MS = 2000
DEFAULT_CAP_USD = 0.20


def hand_counts(path: Path | None) -> dict[str, int]:
    if path is None:
        return {}
    return {photo["name"]: photo["hand_count"] for photo in json.loads(path.read_text(encoding="utf-8"))["photos"]}


def stage_images(images: Path, sweep: SweepDir, spacing_ms: int) -> dict[int, str]:
    names: dict[int, str] = {}
    start_ms = int(time.time() * MS_PER_S)
    sweep.frames.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in images.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    for seq, path in enumerate(files):
        target = sweep.frame_path(seq, start_ms + seq * spacing_ms)
        if path.suffix.lower() == ".jpg":
            shutil.copyfile(path, target)
        else:
            write_jpeg(path, target)
        names[seq] = path.stem
    return names


def write_jpeg(source: Path, target: Path) -> None:
    from PIL import Image

    Image.open(source).convert("RGB").save(target, "JPEG", quality=95)


async def run(images: Path, out: Path, spacing_ms: int, cap_usd: float, deadline_s: float | None) -> tuple[ClaimPacket, dict[int, str], float]:
    sweep = SweepDir(out)
    names = stage_images(images, sweep, spacing_ms)
    budget = Budget()
    start = budget.total_usd()
    budget.cap_usd = start + cap_usd
    settings = load_settings()
    cfg = load_pipeline_config()
    if deadline_s is not None:
        cfg = cfg.model_copy(update={"deadline_s": deadline_s, "vision_cutoff_s": deadline_s})
    run_ = SweepRun(sweep, services_for(sweep, settings, budget), config=cfg, close_services=True)
    capture = load_capture_config()
    for path in sorted(sweep.frames.glob("*.jpg")):
        run_.add_still(still_from_file(path, capture))
    packet = await run_.finish()
    return packet, names, budget.total_usd() - start


def per_photo(packet: ClaimPacket, names: dict[int, str]) -> Counter:
    counts: Counter = Counter()
    for book in packet.books:
        seq = int(Path(book.frame_ref).stem.split("_")[0])
        counts[names.get(seq, f"seq{seq}")] += 1
    return counts


def summarise(packet: ClaimPacket, names: dict[int, str], hand: dict[str, int], spent: float) -> str:
    counts = per_photo(packet, names)
    lines = ["photo: counted vs hand"]
    for name in sorted(set(names.values())):
        lines.append(f"  {name}: {counts.get(name, 0)} vs {hand.get(name, '?')}")
    totals = packet.totals
    lines.append(
        f"books {totals.book_count}: identified {totals.books_identified}, unidentified {totals.books_unidentified}, "
        f"needs_appraisal {totals.books_needs_appraisal}"
    )
    priced = [book for book in packet.books if book.replacement_cost.amount is not None]
    used = [book for book in packet.books if book.used_value.amount is not None]
    lines.append(f"replacement priced {len(priced)}, used value {len(used)}")
    lines.append("price sources: " + json.dumps(dict(Counter(book.replacement_cost.source for book in priced))))
    lines.append(f"items {len(packet.items)}, review entries {len(packet.review_queue)}")
    lines.append(f"cost {json.dumps(packet.costs.model_dump(mode='json'))}")
    lines.append(f"shared spend ledger moved by {spent:.4f} USD (includes other processes)")
    lines.append("latency_s " + json.dumps(packet.latency_s))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python tools/e2e_images.py")
    parser.add_argument("images", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--hand-counts", type=Path)
    parser.add_argument("--spacing-ms", type=int, default=DEFAULT_SPACING_MS)
    parser.add_argument("--deadline-s", type=float)
    parser.add_argument("--cap-usd", type=float, default=DEFAULT_CAP_USD)
    args = parser.parse_args(argv)
    out = args.out or ROOT / "runs" / f"e2e-{datetime.now():%Y%m%d-%H%M%S}"
    packet, names, spent = asyncio.run(run(args.images, out, args.spacing_ms, args.cap_usd, args.deadline_s))
    print(out / load_pipeline_config().packet_name)
    print(summarise(packet, names, hand_counts(args.hand_counts), spent))


if __name__ == "__main__":
    main()
