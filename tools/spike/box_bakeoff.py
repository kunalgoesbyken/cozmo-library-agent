import argparse
import io
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
from google import genai
from google.genai import types
from PIL import Image, ImageDraw
from pydantic import BaseModel

from libclaim.budget import Budget
from libclaim.settings import ROOT, load_settings

SPIKE_DIR = Path(__file__).resolve().parent
SCRATCH = ROOT / "_scratch"
SHELF_DIR = SCRATCH / "shelf"
MODEL_PATH = SCRATCH / "models" / "rfdetr-nano-bookshelf.onnx"
MODEL_URL = "https://huggingface.co/Geraldine/rf-detr-nano-bookshelf/resolve/main/onnx/rfdetr-nano-bookshelf.onnx"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
SPEND_CAP_USD = 0.15
GEMINI_SPACING_SECONDS = {"gemini-3.8-flash": 13.0}
DEFAULT_SPACING_SECONDS = 7.0
LOCAL_INPUT_SIZE = 384
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PROMPT = (
    "This photo shows books standing on a shelf. Return one entry per visible book spine, "
    "including partially visible and horizontally stacked books, as a 2D bounding box "
    "[ymin, xmin, ymax, xmax] normalised to 0-1000."
)


class Spine(BaseModel):
    box_2d: list[int]


class Spines(BaseModel):
    spines: list[Spine]


def http_get(url: str, user_agent: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request) as response:
        return response.read()


def commons_info(title: str, width: int, user_agent: str) -> dict:
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "iiurlwidth": width,
        "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist",
    }
    payload = json.loads(http_get(f"{COMMONS_API}?{urllib.parse.urlencode(params)}", user_agent))
    page = next(iter(payload["query"]["pages"].values()))
    return page["imageinfo"][0]


def fetch(config: dict, user_agent: str) -> None:
    SHELF_DIR.mkdir(parents=True, exist_ok=True)
    sources = {}
    for photo in config["photos"]:
        info = commons_info(photo["title"], config["width"], user_agent)
        (SHELF_DIR / f"{photo['name']}.jpg").write_bytes(http_get(info["thumburl"], user_agent))
        meta = info["extmetadata"]
        sources[photo["name"]] = {
            "title": photo["title"],
            "page": info["descriptionurl"],
            "license": meta["LicenseShortName"]["value"],
            "license_url": meta.get("LicenseUrl", {}).get("value"),
            "artist": meta.get("Artist", {}).get("value"),
        }
        time.sleep(1.0)
    (SHELF_DIR / "sources.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
    if not MODEL_PATH.is_file():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        MODEL_PATH.write_bytes(http_get(MODEL_URL, user_agent))
    print(json.dumps(sources, indent=2))


def iou(a: list[float], b: list[float]) -> float:
    top, left = max(a[0], b[0]), max(a[1], b[1])
    bottom, right = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, bottom - top) * max(0.0, right - left)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


class LocalDetector:
    def __init__(self, threshold: float):
        self.threshold = threshold
        self.session = ort.InferenceSession(str(MODEL_PATH), providers=["DmlExecutionProvider", "CPUExecutionProvider"])
        self.providers = self.session.get_providers()
        self.input_name = self.session.get_inputs()[0].name

    def prepare(self, image: Image.Image) -> np.ndarray:
        resized = image.convert("RGB").resize((LOCAL_INPUT_SIZE, LOCAL_INPUT_SIZE), Image.BILINEAR)
        pixels = (np.asarray(resized, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return pixels.transpose(2, 0, 1)[None].astype(np.float32)

    def detect(self, image: Image.Image) -> tuple[list[list[float]], float]:
        tensor = self.prepare(image)
        self.session.run(None, {self.input_name: tensor})
        start = time.perf_counter()
        boxes, logits = self.session.run(None, {self.input_name: tensor})
        elapsed = time.perf_counter() - start
        scores = (1.0 / (1.0 + np.exp(-logits[0]))).max(axis=1)
        keep = scores >= self.threshold
        centred = boxes[0][keep]
        found = [
            [
                (cy - h / 2) * 1000,
                (cx - w / 2) * 1000,
                (cy + h / 2) * 1000,
                (cx + w / 2) * 1000,
            ]
            for cx, cy, w, h in centred
        ]
        return found, elapsed


def gemini_boxes(client: genai.Client, budget: Budget, model: str, jpeg: bytes) -> tuple[list[list[float]], float, dict]:
    budget.check()
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=Spines,
        temperature=0.0,
    )
    contents = [types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), PROMPT]
    start = time.perf_counter()
    response = client.models.generate_content(model=model, contents=contents, config=config)
    elapsed = time.perf_counter() - start
    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count or 0
    output_tokens = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
    usd = budget.record(model, input_tokens, output_tokens)
    found = [[float(v) for v in spine.box_2d] for spine in response.parsed.spines if len(spine.box_2d) == 4]
    return found, elapsed, {"input_tokens": input_tokens, "output_tokens": output_tokens, "usd": usd}


def overlay(image: Image.Image, boxes: list[list[float]], path: Path) -> None:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    for ymin, xmin, ymax, xmax in boxes:
        draw.rectangle([xmin * width / 1000, ymin * height / 1000, xmax * width / 1000, ymax * height / 1000], outline=(255, 0, 0), width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=85)


def agreement(boxes: list[list[float]], reference: list[list[float]]) -> int:
    return sum(1 for box in boxes if any(iou(box, other) >= 0.5 for other in reference))


def run(config: dict, models: list[str], threshold: float, only: str | None) -> None:
    settings = load_settings()
    budget = Budget(cap_usd=SPEND_CAP_USD)
    client = genai.Client(api_key=settings.gemini_api_key)
    detector = LocalDetector(threshold)
    results = []
    for photo in config["photos"]:
        if only and photo["name"] != only:
            continue
        path = SHELF_DIR / f"{photo['name']}.jpg"
        image = Image.open(path)
        jpeg = path.read_bytes()
        local_boxes, local_seconds = detector.detect(image)
        overlay(image, local_boxes, SHELF_DIR / "out" / f"{photo['name']}-rfdetr.jpg")
        results.append(
            {
                "photo": photo["name"],
                "detector": "rfdetr-nano-bookshelf",
                "providers": detector.providers,
                "hand_count": photo["hand_count"],
                "count": len(local_boxes),
                "seconds": round(local_seconds, 4),
            }
        )
        print(results[-1], flush=True)
        for model in models:
            try:
                boxes, seconds, cost = gemini_boxes(client, budget, model, jpeg)
            except Exception as error:
                results.append({"photo": photo["name"], "detector": model, "error": f"{type(error).__name__}: {str(error)[:300]}"})
                print(results[-1], flush=True)
                time.sleep(GEMINI_SPACING_SECONDS.get(model, DEFAULT_SPACING_SECONDS))
                continue
            overlay(image, boxes, SHELF_DIR / "out" / f"{photo['name']}-{model}.jpg")
            results.append(
                {
                    "photo": photo["name"],
                    "detector": model,
                    "hand_count": photo["hand_count"],
                    "count": len(boxes),
                    "seconds": round(seconds, 2),
                    "agree_with_rfdetr_iou50": agreement(boxes, local_boxes),
                    **cost,
                }
            )
            print(results[-1], flush=True)
            time.sleep(GEMINI_SPACING_SECONDS.get(model, DEFAULT_SPACING_SECONDS))
    (SHELF_DIR / "out" / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"ledger total usd {budget.total_usd():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["fetch", "run"])
    parser.add_argument("--models", default="gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-3.8-flash")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--only")
    args = parser.parse_args()
    config = json.loads((SPIKE_DIR / "box_photos.json").read_text(encoding="utf-8"))
    if args.command == "fetch":
        fetch(config, load_settings().user_agent)
    else:
        run(config, args.models.split(","), args.threshold, args.only)


if __name__ == "__main__":
    sys.exit(main())
