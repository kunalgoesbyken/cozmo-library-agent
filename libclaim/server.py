import asyncio
import contextlib
import json
import logging
import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse
from uuid import uuid4

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from pydantic import BaseModel, ValidationError
from starlette.websockets import WebSocketDisconnect

from libclaim.budget import Budget
from libclaim.capture.coverage import CoverageMap
from libclaim.capture.frame_store import FrameHeader, FrameStore
from libclaim.capture.quality import AlertCode, CaptureConfig, Quality, gate, load_capture_config
from libclaim.live.bridge import LiveBridge, LiveConfig, load_live_config
from libclaim.live.genai_session import GenaiFactory
from libclaim.live.monitor import Alert, Monitor
from libclaim.live.session import SessionClosed
from libclaim.live.tools import SPECS, LiveTools
from libclaim.models import Still
from libclaim.settings import CONFIG_DIR, ROOT, Settings, load_settings
from libclaim.sweep import SweepDir

SERVER_CONFIG_PATH = CONFIG_DIR / "server.yaml"
WEB_DIST = ROOT / "web" / "dist"
RUNS_DIR = ROOT / "runs"
KIND_MIC = 0x01
KIND_LIVE_FRAME = 0x02
KIND_STILL = 0x03
KIND_AGENT_AUDIO = 0x81
HEADER_LENGTH_BYTES = 4
PHONE_ROLES = {"user": "user", "model": "agent"}
TOKEN_PARAM = "t"
WS_POLICY_VIOLATION = 1008
SOF_MARKERS = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}
STANDALONE_MARKERS = frozenset(range(0xD0, 0xD8)) | {0x01}
TOKEN_BYTES = 32

logger = logging.getLogger(__name__)

Emit = Callable[[dict[str, Any]], Awaitable[None]]
EndHook = Callable[[SweepDir, Emit], Awaitable[None]]


class AlertText(BaseModel):
    severity: str
    say: str


class ServerConfig(BaseModel):
    host: str
    max_header_bytes: int
    max_mic_bytes: int
    max_frame_bytes: int
    max_image_pixels: int
    max_stills_per_sweep: int
    max_sweep_bytes: int
    ws_max_size_bytes: int
    max_sessions: int
    hello_timeout_seconds: float
    state_poll_seconds: float
    unassigned_region: str
    report_files: list[str]
    public_dirs: list[str]
    alerts: dict[AlertCode, AlertText]
    errors: dict[str, str]


def load_server_config(path: Path = SERVER_CONFIG_PATH) -> ServerConfig:
    return ServerConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


class Hello(BaseModel):
    token: str = ""
    t_ms: int | None = None


class Control(BaseModel):
    action: str


class Rejected(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class Bridge(Protocol):
    session: Any

    async def run(self) -> None: ...

    async def send_audio(self, data: bytes) -> None: ...

    async def send_frame(self, data: bytes) -> None: ...

    def end_sweep(self) -> None: ...

    async def inject(self, event: dict[str, Any], scheduling: Any) -> None: ...

    async def close(self) -> None: ...


@dataclass(frozen=True)
class Hooks:
    sweep: SweepDir
    on_audio: Callable[[bytes], Awaitable[None]]
    on_transcript: Callable[[str, str], Awaitable[None]]
    on_interrupted: Callable[[], Awaitable[None]]
    on_end_sweep: Callable[[], None]
    now_ms: Callable[[], int]
    stills: Callable[[], list[Still]]
    emit: Emit


BridgeFactory = Callable[[Hooks], Bridge]


class Slots:
    def __init__(self, limit: int):
        self.limit = limit
        self.active = 0

    def acquire(self) -> bool:
        if self.active >= self.limit:
            return False
        self.active += 1
        return True

    def release(self) -> None:
        self.active -= 1


async def no_end_hook(sweep: SweepDir, emit: Emit) -> None:
    return None


def live_bridge_factory(settings: Settings, budget: Budget | None = None) -> BridgeFactory:
    config = load_live_config()
    budget = budget or Budget()
    sessions = GenaiFactory(
        genai.Client(api_key=settings.gemini_api_key), settings.live_model, config, SPECS, settings.live_voice
    )

    def make(hooks: Hooks) -> Bridge:
        tools = LiveTools(hooks.sweep, config, hooks.stills, hooks.now_ms, hooks.on_end_sweep)
        return LiveBridge(
            hooks.sweep,
            tools,
            hooks.on_audio,
            hooks.on_transcript,
            hooks.on_interrupted,
            factory=sessions,
            budget=budget,
            model=settings.live_model,
            config=config,
        )

    return make


def parse_binary(data: bytes, max_header_bytes: int) -> tuple[int, dict[str, Any], bytes]:
    if len(data) <= HEADER_LENGTH_BYTES:
        raise ValueError("frame too short")
    size = int.from_bytes(data[1 : 1 + HEADER_LENGTH_BYTES], "big")
    start = 1 + HEADER_LENGTH_BYTES
    if size > max_header_bytes or start + size > len(data):
        raise ValueError("bad header length")
    header = json.loads(data[start : start + size])
    if not isinstance(header, dict):
        raise ValueError("header is not an object")
    return data[0], header, data[start + size :]


def pack_binary(kind: int, header: dict[str, Any], payload: bytes) -> bytes:
    body = json.dumps(header).encode()
    return bytes([kind]) + len(body).to_bytes(HEADER_LENGTH_BYTES, "big") + body + payload


def jpeg_size(data: bytes) -> tuple[int, int]:
    if data[:2] != b"\xff\xd8":
        raise ValueError("not a jpeg")
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            raise ValueError("bad jpeg marker")
        marker = data[i + 1]
        if marker == 0xFF:
            i += 1
        elif marker in STANDALONE_MARKERS:
            i += 2
        elif marker in SOF_MARKERS:
            if i + 9 > len(data):
                break
            return int.from_bytes(data[i + 7 : i + 9], "big"), int.from_bytes(data[i + 5 : i + 7], "big")
        else:
            i += 2 + int.from_bytes(data[i + 2 : i + 4], "big")
    raise ValueError("jpeg size not found")


def check_image(data: bytes, max_pixels: int, expected: tuple[int, int] | None = None) -> None:
    width, height = jpeg_size(data)
    if width <= 0 or height <= 0 or width * height > max_pixels:
        raise ValueError("image dimensions out of range")
    if expected is not None and expected != (width, height):
        raise ValueError("header dimensions do not match image")


def token_matches(given: str | None, expected: str) -> bool:
    return given is not None and secrets.compare_digest(given.encode(), expected.encode())


def same_origin(origin: str | None, host: str | None) -> bool:
    return origin is None or (host is not None and urlparse(origin).netloc == host)


class Connection:
    def __init__(
        self,
        websocket: WebSocket,
        settings: Settings,
        token: str,
        slots: Slots,
        background: set[asyncio.Task],
        bridge_factory: BridgeFactory,
        on_end: EndHook,
        runs_dir: Path,
        config: ServerConfig,
        capture: CaptureConfig,
        live: LiveConfig,
    ):
        self.websocket = websocket
        self.settings = settings
        self.token = token
        self.slots = slots
        self.background = background
        self.bridge_factory = bridge_factory
        self.on_end = on_end
        self.runs_dir = runs_dir
        self.config = config
        self.capture = capture
        self.live = live
        self.send_lock = asyncio.Lock()
        self.slot_held = False
        self.sweep: SweepDir | None = None
        self.sweep_id = ""
        self.bridge: Bridge | None = None
        self.store: FrameStore | None = None
        self.monitor: Monitor | None = None
        self.coverage = CoverageMap(capture)
        self.stills: list[Still] = []
        self.still_bytes = 0
        self.coverage_sent: dict[str, str] = {}
        self.clock_offset_ms = 0
        self.audio_seq = 0
        self.muted = False
        self.ending = False
        self.closed_sent = False
        self.close_code = 1000
        self.bridge_task: asyncio.Task | None = None
        self.watcher_task: asyncio.Task | None = None
        self.receive_task: asyncio.Task | None = None

    def now_ms(self) -> int:
        return int(time.time() * 1000) - self.clock_offset_ms

    async def send_json(self, message: dict[str, Any]) -> None:
        async with self.send_lock:
            with contextlib.suppress(WebSocketDisconnect, RuntimeError):
                await self.websocket.send_text(json.dumps(message))

    async def send_audio_to_phone(self, data: bytes) -> None:
        self.audio_seq += 1
        async with self.send_lock:
            with contextlib.suppress(WebSocketDisconnect, RuntimeError):
                await self.websocket.send_bytes(pack_binary(KIND_AGENT_AUDIO, {"seq": self.audio_seq}, data))

    async def send_transcript(self, speaker: str, text: str) -> None:
        role = PHONE_ROLES.get(speaker, speaker)
        await self.send_json({"type": "transcript", "role": role, "text": text, "final": False, "t_ms": self.now_ms()})

    async def send_interrupted(self) -> None:
        await self.send_json({"type": "interrupted"})

    async def send_error(self, code: str) -> None:
        await self.send_json({"type": "error", "code": code, "say": self.config.errors[code]})

    async def send_session(self, state: str, reason: str | None = None) -> None:
        message = {
            "type": "session",
            "state": state,
            "model": self.settings.live_model,
            "sweep_id": self.sweep_id,
            "live_fps": self.live.frame_max_fps,
        }
        if reason:
            message["reason"] = reason
        if state == "closed":
            self.closed_sent = True
        await self.send_json(message)

    async def run(self) -> None:
        self.receive_task = asyncio.create_task(self.receive_loop())
        try:
            await self.receive_task
        except asyncio.CancelledError:
            if not self.receive_task.cancelled():
                raise
        except Exception:
            logger.exception("sweep connection failed")
        finally:
            await self.shutdown()

    async def receive_loop(self) -> None:
        try:
            while True:
                timeout = None if self.bridge else self.config.hello_timeout_seconds
                try:
                    message = await asyncio.wait_for(self.websocket.receive(), timeout)
                except TimeoutError:
                    raise Rejected("hello_timeout")
                if message["type"] == "websocket.disconnect":
                    return
                if message.get("bytes") is not None:
                    await self.handle_binary(message["bytes"])
                elif message.get("text") is not None:
                    await self.handle_text(message["text"])
        except Rejected as rejected:
            self.close_code = WS_POLICY_VIOLATION
            await self.send_session("closed", rejected.reason)

    async def handle_text(self, text: str) -> None:
        try:
            payload = json.loads(text)
            kind = payload["type"]
            if kind == "hello":
                await self.hello(Hello.model_validate(payload))
            elif kind == "control" and self.bridge is not None:
                await self.control(Control.model_validate(payload))
        except (ValueError, KeyError, TypeError, ValidationError):
            logger.warning("dropped malformed text message")

    async def hello(self, hello: Hello) -> None:
        if self.bridge is not None:
            return
        if not token_matches(hello.token, self.token):
            raise Rejected("unauthorized")
        if not self.slots.acquire():
            raise Rejected("busy")
        self.slot_held = True
        if hello.t_ms is not None:
            self.clock_offset_ms = int(time.time() * 1000) - hello.t_ms
            logger.info("phone clock offset %d ms", self.clock_offset_ms)
        self.sweep_id = uuid4().hex
        self.sweep = SweepDir(self.runs_dir / self.sweep_id)
        self.store = FrameStore(self.sweep, self.capture)
        hooks = Hooks(
            sweep=self.sweep,
            on_audio=self.send_audio_to_phone,
            on_transcript=self.send_transcript,
            on_interrupted=self.send_interrupted,
            on_end_sweep=self.finish,
            now_ms=self.now_ms,
            stills=lambda: self.stills,
            emit=self.send_json,
        )
        try:
            bridge = self.bridge_factory(hooks)
        except Exception:
            logger.exception("bridge could not be created")
            raise Rejected("bridge_unavailable")
        self.bridge = bridge
        self.monitor = Monitor(bridge.inject, self.live)
        await self.send_session("connecting")
        self.bridge_task = asyncio.create_task(bridge.run())
        self.bridge_task.add_done_callback(self.bridge_stopped)
        self.watcher_task = asyncio.create_task(self.watch_state())

    def bridge_stopped(self, task: asyncio.Task) -> None:
        if self.receive_task and not self.receive_task.done():
            self.receive_task.cancel()

    async def watch_state(self) -> None:
        last = ""
        was_live = False
        while True:
            if self.bridge.session is not None:
                state, was_live = "live", True
            else:
                state = "reconnecting" if was_live else "connecting"
            if state != last:
                last = state
                await self.send_session(state)
            await asyncio.sleep(self.config.state_poll_seconds)

    async def control(self, control: Control) -> None:
        if control.action == "mute":
            self.muted = True
        elif control.action == "unmute":
            self.muted = False
        elif control.action == "end_sweep":
            self.finish()

    def finish(self) -> None:
        if self.ending:
            return
        self.ending = True
        task = asyncio.create_task(self.process())
        self.background.add(task)
        task.add_done_callback(self.background.discard)

    async def process(self) -> None:
        if self.watcher_task:
            self.watcher_task.cancel()
        self.bridge.end_sweep()
        await self.send_session("processing")
        try:
            await self.on_end(self.sweep, self.send_json)
        except Exception:
            logger.exception("end of sweep processing failed")
            await self.send_session("closed", "processing_failed")
            return
        await self.send_session("summary")

    async def handle_binary(self, data: bytes) -> None:
        if self.bridge is None:
            return
        try:
            kind, header, payload = parse_binary(data, self.config.max_header_bytes)
            if kind == KIND_MIC:
                await self.mic(payload)
            elif kind == KIND_LIVE_FRAME:
                await self.live_frame(payload)
            elif kind == KIND_STILL:
                await self.still(FrameHeader.model_validate(header), payload)
        except (ValueError, ValidationError):
            logger.warning("dropped malformed binary frame")

    async def mic(self, payload: bytes) -> None:
        if self.muted or self.ending or len(payload) > self.config.max_mic_bytes:
            return
        await self.bridge.send_audio(payload)

    async def live_frame(self, payload: bytes) -> None:
        if len(payload) > self.config.max_frame_bytes:
            return
        check_image(payload, self.config.max_image_pixels)
        await self.bridge.send_frame(payload)

    async def still(self, header: FrameHeader, payload: bytes) -> None:
        if self.ending or len(payload) > self.config.max_frame_bytes:
            return
        if (
            len(self.stills) >= self.config.max_stills_per_sweep
            or self.still_bytes + len(payload) > self.config.max_sweep_bytes
        ):
            await self.send_error("limit")
            return
        check_image(payload, self.config.max_image_pixels, (header.w, header.h))
        try:
            still = await asyncio.to_thread(self.store.add, header, payload)
        except OSError:
            logger.exception("could not store still")
            await self.send_error("storage")
            return
        self.stills.append(still)
        self.still_bytes += len(payload)
        region = self.config.unassigned_region
        self.coverage.record(region, still)
        quality = Quality(blur=still.blur, glare=still.glare)
        for code in gate(quality, still.motion_dps, self.capture):
            await self.alert(code, region)
        await self.send_coverage()

    async def alert(self, code: AlertCode, region: str) -> None:
        try:
            spoken = await self.monitor.push(Alert(code, region, self.monitor.clock()))
        except (LookupError, SessionClosed):
            spoken = True
        if spoken:
            text = self.config.alerts[code]
            await self.send_json(
                {"type": "capture_alert", "code": code.value, "region": region, "severity": text.severity, "say": text.say}
            )

    async def send_coverage(self) -> None:
        states = {region: "sharp" if region in self.coverage.sharp else "partial" for region in self.coverage.seen}
        if states != self.coverage_sent:
            self.coverage_sent = states
            await self.send_json(
                {"type": "coverage", "regions": [{"id": region, "state": state} for region, state in sorted(states.items())]}
            )

    async def shutdown(self) -> None:
        if self.slot_held:
            self.slots.release()
        if self.watcher_task:
            self.watcher_task.cancel()
        if self.bridge is not None:
            with contextlib.suppress(Exception):
                await self.bridge.close()
        pending = [task for task in (self.watcher_task, self.bridge_task) if task]
        await asyncio.gather(*pending, return_exceptions=True)
        if not self.closed_sent:
            await self.send_session("closed")
        with contextlib.suppress(RuntimeError):
            await self.websocket.close(code=self.close_code)


def create_app(
    settings: Settings,
    bridge_factory: BridgeFactory,
    on_end: EndHook = no_end_hook,
    *,
    runs_dir: Path = RUNS_DIR,
    web_dist: Path = WEB_DIST,
    config: ServerConfig | None = None,
    capture: CaptureConfig | None = None,
    live: LiveConfig | None = None,
) -> FastAPI:
    config = config or load_server_config()
    capture = capture or load_capture_config()
    live = live or load_live_config()
    runs_dir.mkdir(parents=True, exist_ok=True)
    token = settings.access_token or secrets.token_urlsafe(TOKEN_BYTES)
    slots = Slots(config.max_sessions)
    background: set[asyncio.Task] = set()
    app = FastAPI()
    app.state.access_token = token

    @app.websocket("/ws/sweep")
    async def sweep_socket(websocket: WebSocket) -> None:
        if not same_origin(websocket.headers.get("origin"), websocket.headers.get("host")):
            await websocket.close(code=WS_POLICY_VIOLATION)
            return
        await websocket.accept()
        await Connection(
            websocket, settings, token, slots, background, bridge_factory, on_end, runs_dir, config, capture, live
        ).run()

    @app.get("/runs/{sweep_id}/{name:path}")
    async def run_file(sweep_id: str, name: str, request: Request) -> FileResponse:
        given = request.query_params.get(TOKEN_PARAM) or request.cookies.get(TOKEN_PARAM)
        if not token_matches(given, token):
            raise HTTPException(status_code=401)
        parts = Path(name).parts
        allowed = name in config.report_files or (len(parts) > 1 and parts[0] in config.public_dirs)
        root = (runs_dir / sweep_id).resolve()
        path = (root / name).resolve()
        if (
            not allowed
            or root.parent != runs_dir.resolve()
            or not path.is_relative_to(root)
            or not path.is_file()
        ):
            raise HTTPException(status_code=404)
        response = FileResponse(path)
        response.set_cookie(TOKEN_PARAM, token, httponly=True, samesite="strict", secure=request.url.scheme == "https")
        return response

    if web_dist.is_dir():
        app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")
    return app


def main() -> None:
    settings = load_settings()
    config = load_server_config()
    app = create_app(settings, live_bridge_factory(settings, Budget(RUNS_DIR / "spend.jsonl")), config=config)
    print(f"Open http://localhost:{settings.port}/?{TOKEN_PARAM}={app.state.access_token}", flush=True)
    uvicorn.run(app, host=config.host, port=settings.port, ws_max_size=config.ws_max_size_bytes)


if __name__ == "__main__":
    main()
