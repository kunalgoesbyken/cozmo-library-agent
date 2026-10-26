import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cv2
import httpx
import numpy as np
from pydantic import BaseModel

from libclaim.budget import Budget
from libclaim.catalog.googlebooks import GoogleBooks
from libclaim.catalog.http import CatalogHttp
from libclaim.catalog.openlibrary import OpenLibrary
from libclaim.live.tools import LocaleChoice, Statement, StatementKind, StatementLog
from libclaim.llm import Llm, LlmResponse, load_llm_config
from libclaim.pricing.http import PriceHttp
from libclaim.resolve_config import load_resolve_config
from libclaim.room.depth import DepthUnavailable
from libclaim.settings import Settings
from libclaim.sweep import SweepDir
from libclaim.vision.boxes import Box, load_vision_config
from libclaim.vision.items import FrameItems
from libclaim.vision.reading import SpineReads
from libclaim.vision.resolve import Catalogs
from libclaim.vision.spines import TileSpines
from tests.pricing.conftest import FIXTURES as PRICE_FIXTURES
from tests.pricing.conftest import ROUTES, fast_config, public_resolver

ROOT = Path(__file__).resolve().parent / "fixtures" / "sweep12"
WIDTH, HEIGHT = 960, 540
T0_MS = 1761460000000
STEP_MS = 2000
ROW_SHIFT = 300
ROOM_SHIFT = 400
SHELF_Y = 430
SHELF_PITCH = 200
SLOT = 60
SLOT_MARGIN = 20
GAP = 6
MARKER_PX = 120
MARKER_BORDER = 16
MARKER_AT = (140, 290)
COLOUR_TOL = 34
MIN_SIDE = 12
SPINE_NOISE = 6
JPEG_QUALITY = 85
BOX_SCALE = 1000
MARKER_CM = 16.0
TOKENS_IN, TOKENS_OUT = 120, 40
REQUERY_FRAME = 3
LOCAL_MISSES = 2
FX_RATES = {("GBP", "INR"): 110.25, ("INR", "GBP"): 0.0091}
MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
TIEBREAK = "gemini-3.8-flash"
SETTINGS = Settings(vision_models=MODELS, tiebreak_model=TIEBREAK, default_country="IN", second_country="GB",
                    aruco_marker_size_cm=MARKER_CM)


@dataclass(frozen=True)
class Book:
    key: str
    width: int
    height: int
    title: str = ""
    author: str = ""
    publisher: str = ""
    spine_publisher: str | None = None
    legible: bool = True
    ocr: str | None = None
    isbn: str = ""
    year: int = 2015
    dims: str = ""

    @property
    def spine(self) -> dict:
        publisher = self.publisher if self.spine_publisher is None else self.spine_publisher
        return {"title": self.title.upper() if self.legible else "", "author": self.author.upper() if self.legible else "",
                "publisher": publisher.upper() if self.legible else "", "legible": self.legible}

    @property
    def ocr_text(self) -> str:
        if self.ocr is not None:
            return self.ocr
        spine = self.spine
        return " ".join(part for part in (spine["title"], spine["author"], spine["publisher"]) if part)


@dataclass(frozen=True)
class Strip:
    seed: int
    marker: int | None
    start: int
    shelves: list[list[Book]]


@dataclass(frozen=True)
class Thing:
    key: str
    category: str
    description: str
    span: tuple[int, int, int, int]
    is_art: bool = False
    brand: str = ""


STRIPS = [
    Strip(seed=11, marker=1, start=800, shelves=[[
        Book("sapiens", 34, 170, "Sapiens", "Yuval Noah Harari", "Vintage", isbn="9780099590088",
             dims="19.8 x 12.9 x 3 centimeters"),
        Book("einty", 30, 176, "Everything I Never Told You", "Celeste Ng", "Penguin", isbn="9780143127550"),
        Book("wings", 26, 160, "Wings of Fire", "A P J Abdul Kalam", "Universities Press", isbn="9788173711466"),
        Book("hobbit", 36, 182, "The Hobbit", "J R R Tolkien", "HarperCollins",
             spine_publisher="HarperCollins Signed Edition", isbn="9780261103344"),
        Book("blank_a", 28, 150, legible=False),
        Book("pride", 30, 172, "Pride and Prejudice", "Jane Austen", "Penguin Classics", isbn="9780141439518"),
    ]]),
    Strip(seed=12, marker=None, start=340, shelves=[[
        Book("midnight", 38, 185, "Midnight's Children", "Salman Rushdie", "Vintage", isbn="9780099578512"),
        Book("small", 32, 170, "The God of Small Things", "Arundhati Roy", "Fourth Estate", spine_publisher="",
             isbn="9780006550686"),
        Book("blank_b", 30, 158, legible=False),
        Book("atomic", 34, 166, "Atomic Habits", "James Clear", "Random House", isbn="9781847941831"),
        Book("clean", 160, 32, "Clean Code", "Robert C Martin", "Pearson", isbn="9780132350884"),
    ]]),
    Strip(seed=13, marker=None, start=340, shelves=[
        [
            Book("gatsby", 30, 174, "The Great Gatsby", "F Scott Fitzgerald", "Scribner", isbn="9780743273565"),
            Book("dune", 40, 180, "Dune", "Frank Herbert", "Hodder", isbn="9781473233959"),
            Book("ulysses", 40, 178, "Ulysses", "James Joyce", "Penguin", ocr="", isbn="9780141182803"),
            Book("alchemist", 28, 168, "The Alchemist", "Paulo Coelho", "HarperOne", isbn="9780062315007"),
        ],
        [
            Book("gitanjali", 26, 162, "Gitanjali", "Rabindranath Tagore", "Rupa", isbn="9788171670062"),
            Book("blank_c", 34, 170, legible=False),
            Book("malgudi", 30, 176, "Malgudi Days", "R K Narayan", "Indian Thought", isbn="9788185986173"),
        ],
    ]),
    Strip(seed=14, marker=2, start=800, shelves=[[
        Book("tiger", 32, 178, "The White Tiger", "Aravind Adiga", "HarperCollins", isbn="9788172238476"),
        Book("train", 30, 170, "Train to Pakistan", "Khushwant Singh", "Penguin", isbn="9780143065883"),
        Book("guide", 34, 176, "The Guide", "R K Narayan", "Indian Thought", isbn="9788185986005"),
    ]]),
]
THINGS = [
    Thing("bookcase", "furniture", "wooden bookcase with five shelves", (420, 120, 760, 480)),
    Thing("lamp", "lighting", "brass table lamp", (1000, 220, 1140, 420)),
    Thing("art", "artwork", "framed painting of a harbour", (1400, 100, 1700, 300), is_art=True),
    Thing("coffee", "appliance", "black drip coffee maker", (1850, 300, 2050, 460), brand="Philips"),
]
ROW_FRAMES = 2
ROOM_FRAMES = 4
STATEMENTS = [
    Statement(subject="The Great Gatsby", kind=StatementKind.first_edition, value="first edition",
              t_ms=T0_MS + 6 * STEP_MS, still_seq_from=5, still_seq_to=6),
    Statement(subject="the harbour painting", kind=StatementKind.art_print, value="it is a print",
              t_ms=T0_MS + 12 * STEP_MS, still_seq_from=11, still_seq_to=12),
]
BOOKS = [book for strip in STRIPS for shelf in strip.shelves for book in shelf]
SHELF_LABELS = {"U1-1", "S2", "S3.1", "S3.2", "U2-1"}


def palette(count: int, value: int, offset: float) -> list[tuple[int, int, int]]:
    hues = np.array([[[int((i + offset) * 180 / count) % 180, 255, value] for i in range(count)]], np.uint8)
    return [tuple(int(c) for c in colour) for colour in cv2.cvtColor(hues, cv2.COLOR_HSV2BGR)[0]]


BOOK_COLOURS = dict(zip((book.key for book in BOOKS), palette(len(BOOKS), 230, 0.0)))
THING_COLOURS = dict(zip((thing.key for thing in THINGS), palette(len(THINGS), 140, 0.5)))


def background(seed: int, width: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gray = cv2.GaussianBlur(rng.integers(30, 220, (HEIGHT, width), dtype=np.uint8), (5, 5), 0)
    return np.dstack([gray] * 3)


def paint(canvas: np.ndarray, colour: tuple[int, int, int], x1: int, y1: int, x2: int, y2: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    noise = rng.integers(-SPINE_NOISE, SPINE_NOISE + 1, (y2 - y1, x2 - x1, 1))
    canvas[y1:y2, x1:x2] = np.clip(np.array(colour) + noise, 0, 255).astype(np.uint8)


def marker(canvas: np.ndarray, marker_id: int) -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, load_vision_config().scale.aruco_dictionary))
    x, y = MARKER_AT
    canvas[y - MARKER_BORDER : y + MARKER_PX + MARKER_BORDER, x - MARKER_BORDER : x + MARKER_PX + MARKER_BORDER] = 255
    canvas[y : y + MARKER_PX, x : x + MARKER_PX] = cv2.aruco.generateImageMarker(dictionary, marker_id, MARKER_PX)[:, :, None]


def strip_panorama(strip: Strip) -> np.ndarray:
    canvas = background(strip.seed, WIDTH + ROW_SHIFT * (ROW_FRAMES - 1))
    for level, shelf in enumerate(strip.shelves):
        bottom = SHELF_Y - (len(strip.shelves) - 1 - level) * SHELF_PITCH
        x = strip.start
        for number, book in enumerate(shelf):
            paint(canvas, BOOK_COLOURS[book.key], x, bottom - book.height, x + book.width, bottom,
                  strip.seed * 100 + level * 10 + number)
            x += book.width + GAP
    if strip.marker is not None:
        marker(canvas, strip.marker)
    return canvas


def room_panorama() -> np.ndarray:
    canvas = background(15, WIDTH + ROOM_SHIFT * (ROOM_FRAMES - 1))
    for number, thing in enumerate(THINGS):
        paint(canvas, THING_COLOURS[thing.key], *thing.span, 1500 + number)
    return canvas


def frames() -> list[np.ndarray]:
    views = []
    for strip in STRIPS:
        panorama = strip_panorama(strip)
        views += [panorama[:, i * ROW_SHIFT : i * ROW_SHIFT + WIDTH] for i in range(ROW_FRAMES)]
    room = room_panorama()
    views += [room[:, i * ROOM_SHIFT : i * ROOM_SHIFT + WIDTH] for i in range(ROOM_FRAMES)]
    return views


def build(root: Path = ROOT) -> None:
    sweep = SweepDir(root)
    shutil.rmtree(root, ignore_errors=True)
    sweep.frames.mkdir(parents=True)
    for seq, view in enumerate(frames(), 1):
        path = sweep.frame_path(seq, T0_MS + seq * STEP_MS)
        cv2.imwrite(str(path), view, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    sweep.write_stage("statements", StatementLog(items=STATEMENTS))
    sweep.write_stage("locale", LocaleChoice(country="IN", currency="INR"))


@dataclass(frozen=True)
class Pan:
    seed: int
    slots: dict[int, Book]
    marker_slot: int | None = None
    marker: int = 7


def pan_panorama(pan: Pan, width: int) -> np.ndarray:
    canvas = background(pan.seed, width)
    for slot, book in pan.slots.items():
        x = slot * SLOT + SLOT_MARGIN
        paint(canvas, BOOK_COLOURS[book.key], x, SHELF_Y - book.height, x + book.width, SHELF_Y, pan.seed * 100 + slot)
    if pan.marker_slot is not None:
        x, y = pan.marker_slot * SLOT + SLOT_MARGIN + MARKER_BORDER, MARKER_AT[1]
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, load_vision_config().scale.aruco_dictionary))
        canvas[y - MARKER_BORDER : y + MARKER_PX + MARKER_BORDER, x - MARKER_BORDER : x + MARKER_PX + MARKER_BORDER] = 255
        canvas[y : y + MARKER_PX, x : x + MARKER_PX] = cv2.aruco.generateImageMarker(dictionary, pan.marker, MARKER_PX)[:, :, None]
    return canvas


def write_pans(root: Path, views: list[tuple[Pan, int]]) -> SweepDir:
    sweep = SweepDir(root)
    sweep.frames.mkdir(parents=True)
    width = WIDTH + max(offset for _, offset in views)
    panoramas = {pan.seed: pan_panorama(pan, width) for pan, _ in views}
    for seq, (pan, offset) in enumerate(views, 1):
        view = panoramas[pan.seed][:, offset : offset + WIDTH]
        cv2.imwrite(str(sweep.frame_path(seq, T0_MS + seq * STEP_MS)), view, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    sweep.write_stage("locale", LocaleChoice(country="IN", currency="INR"))
    return sweep


def upright(count: int, start: int = 0) -> list[Book]:
    return [book for book in BOOKS if book.width <= SLOT - SLOT_MARGIN][start : start + count]


def copy_sweep(target: Path) -> SweepDir:
    shutil.copytree(ROOT, target)
    return SweepDir(target)


def decode(part) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(part.inline_data.data, np.uint8), cv2.IMREAD_COLOR)


def blobs(image: np.ndarray, colours: dict[str, tuple[int, int, int]]) -> list[tuple[str, Box]]:
    found = []
    for key, colour in colours.items():
        centre = np.array(colour, np.int16)
        lower, upper = np.clip(centre - COLOUR_TOL, 0, 255), np.clip(centre + COLOUR_TOL, 0, 255)
        mask = cv2.inRange(image, lower.astype(np.uint8), upper.astype(np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        for x, y, w, h, _ in stats[1:count]:
            if w >= MIN_SIDE and h >= MIN_SIDE:
                found.append((key, Box(x1=float(x), y1=float(y), x2=float(x + w), y2=float(y + h))))
    return sorted(found, key=lambda pair: (pair[1].x1, pair[1].y1))


def box_2d(box: Box, image: np.ndarray) -> list[int]:
    height, width = image.shape[:2]
    return [round(box.y1 * BOX_SCALE / height), round(box.x1 * BOX_SCALE / width),
            round(box.y2 * BOX_SCALE / height), round(box.x2 * BOX_SCALE / width)]


def nearest(colour: np.ndarray, colours: dict[str, tuple[int, int, int]]) -> str:
    return min(colours, key=lambda key: float(np.abs(colour - np.array(colours[key])).sum()))


def crop_book(crop: np.ndarray) -> Book:
    key = nearest(np.median(crop.reshape(-1, 3), axis=0), BOOK_COLOURS)
    return next(book for book in BOOKS if book.key == key)


def spines_reply(parts: list) -> dict:
    spines = []
    for index, part in enumerate(p for p in parts if not isinstance(p, str)):
        image = decode(part)
        for _, box in blobs(image, BOOK_COLOURS):
            orientation = "vertical" if box.height >= box.width else "horizontal"
            spines.append({"image_index": index, "box_2d": box_2d(box, image), "orientation": orientation})
    return {"spines": spines}


def reads_reply(parts: list) -> dict:
    reads = []
    for label, part in zip(parts[0::2], parts[1::2]):
        reads.append({"crop_id": label.split(": ")[1], **crop_book(decode(part)).spine})
    return {"reads": reads}


def items_reply(parts: list) -> dict:
    found = []
    images = [p for p in parts if not isinstance(p, str)]
    for index, part in enumerate(images):
        image = decode(part)
        height, width = image.shape[:2]
        for key, box in blobs(image, THING_COLOURS):
            thing = next(t for t in THINGS if t.key == key)
            truncated = box.x1 <= 1 or box.y1 <= 1 or box.x2 >= width - 1 or box.y2 >= height - 1
            found.append({"image_index": index, "category": thing.category, "description": thing.description,
                          "brand_model_text": thing.brand, "box_2d": box_2d(box, image), "is_art": thing.is_art,
                          "truncated": truncated})
    return {"items": found}


REPLIES = {TileSpines: spines_reply, SpineReads: reads_reply, FrameItems: items_reply}


@dataclass
class FakeVision:
    calls: list[tuple[str, str]] = field(default_factory=list)

    async def generate(self, model: str, parts: list, schema: type[BaseModel]) -> LlmResponse:
        self.calls.append((model, schema.__name__))
        reply = REPLIES[schema](parts)
        return LlmResponse(text=json.dumps(reply), input_tokens=TOKENS_IN, output_tokens=TOKENS_OUT)


def fake_ocr(crop: np.ndarray) -> tuple[str, float]:
    text = crop_book(crop).ocr_text
    return (text, 0.95) if text else ("", 0.0)


def fake_local(image: np.ndarray) -> list[Box]:
    return [box for _, box in blobs(image, BOOK_COLOURS)]


def local_with_miss(sweep: SweepDir):
    target = cv2.imread(str(sorted(sweep.frames.glob("*.jpg"))[REQUERY_FRAME - 1]))

    def local(image: np.ndarray) -> list[Box]:
        boxes = fake_local(image)
        return boxes[LOCAL_MISSES:] if np.array_equal(image, target) else boxes

    return local


def no_depth(still) -> np.ndarray:
    raise DepthUnavailable("metric room depth needs the [ml] extra (torch, transformers)")


def fast_llm_config():
    config = load_llm_config()
    models = {name: limits.model_copy(update={"rpm": 10_000}) for name, limits in config.models.items()}
    return config.model_copy(update={"models": models})


async def no_sleep(seconds: float) -> None:
    return None


def fake_llm(sweep: SweepDir, tmp_path: Path, client=None) -> Llm:
    return Llm(client=client or FakeVision(), budget=Budget(tmp_path / "spend.jsonl"), ledger=sweep.ledger,
               quota_path=tmp_path / "quota.json", config=fast_llm_config(), sleep=no_sleep)


def catalog_docs(title: str) -> list[dict]:
    wanted = " ".join(title.casefold().split())
    return [
        {"key": f"/works/OL{index}W", "title": book.title, "author_name": [book.author], "first_publish_year": book.year}
        for index, book in enumerate(BOOKS)
        if book.title and " ".join(book.title.casefold().split()) == wanted
    ]


def catalog_editions(work: str) -> list[dict]:
    book = BOOKS[int(work.removeprefix("OL").removesuffix("W"))]
    entry = {"key": f"/books/{work}M", "publishers": [book.publisher], "isbn_13": [book.isbn],
             "publish_date": str(book.year)}
    return [entry | ({"physical_dimensions": book.dims} if book.dims else {})]


def fx_rows(query: dict[str, list[str]]) -> list[dict]:
    base, quote, day = query["base"][0], query["quotes"][0], query["date"][0]
    return [{"base": base, "quote": quote, "rate": FX_RATES[base, quote], "date": day}]


class Web:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.requests: list[str] = []

    def dynamic(self, url: str) -> object | None:
        parts = urlsplit(url)
        query = parse_qs(parts.query)
        catalog = load_resolve_config().catalog
        if url.startswith(catalog.openlibrary.search_url):
            return {"docs": catalog_docs(query["title"][0])}
        if parts.path.endswith("/editions.json"):
            return {"entries": catalog_editions(parts.path.split("/")[2])}
        if url.startswith(catalog.googlebooks.volumes_url):
            return {"items": []}
        if url.startswith(fast_config().sources.fx.url):
            return fx_rows(query)
        return None

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        if self.fail:
            raise httpx.ConnectError("network disabled in this test")
        body = self.dynamic(url)
        if body is not None:
            return httpx.Response(200, content=json.dumps(body).encode(), headers={"content-type": "application/json"})
        route = ROUTES.get(url)
        if route is None:
            return httpx.Response(404, content=b"")
        headers = {"content-type": route.get("type", "text/html")}
        if "location" in route:
            headers["location"] = route["location"]
        content = (PRICE_FIXTURES / route["file"]).read_bytes() if "file" in route else b""
        return httpx.Response(route["status"], content=content, headers=headers)


def client_for(web: Web, **extra) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(web.handler), headers={"User-Agent": "test"}, **extra)


def catalogs_for(sweep: SweepDir, web: Web) -> Catalogs:
    catalog = load_resolve_config().catalog.model_copy(update={"requests_per_second": 10_000})
    http = CatalogHttp(sweep.root / catalog.cache_dir, client=client_for(web, follow_redirects=True), config=catalog)
    return Catalogs(OpenLibrary(http), GoogleBooks(http, api_key=""))


def price_http_for(sweep: SweepDir, web: Web) -> PriceHttp:
    cache = sweep.root / load_resolve_config().catalog.cache_dir
    return PriceHttp(cache, client=client_for(web), config=fast_config(), resolver=public_resolver)


if __name__ == "__main__":
    build()
