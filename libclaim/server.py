import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import uvicorn
import yaml
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from google import genai
from pydantic import BaseModel, Field, ValidationError
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
RUNS_DIR = Path("runs")
KIND_MIC = 0x01
KIND_LIVE_FRAME = 0x02
KIND_STILL = 0x03
KIND_AGENT_AUDIO = 0x81
HEADER_LENGTH_BYTES = 4
PHONE_ROLES = {"user": "user", "model": "agent"}
SWEEP_ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
SESSION_ID_LENGTH = 12

logger = logging.getLogger(__name__)

Emit = Callable[[dict[str, Any]], Awaitable[None]]
EndHook = Callable[[SweepDir, Emit], Awaitable[None]]


class AlertText(BaseModel):
    severity: str
    say: str


class ServerConfig(BaseModel):
    max_header_bytes: int
    max_mic_bytes: int
    max_frame_bytes: int
    state_poll_seconds: float
    unassigned_region: str
    alerts: dict[AlertCode, AlertText]


def load_server_config(path: Path = SERVER_CONFIG_PATH) -> ServerConfig:
    return ServerConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


class Hello(BaseModel):
    sweep_id: str | None = Field(default=None, pattern=SWEEP_ID_PATTERN)
    t_ms: int | None = None


class Control(BaseModel):
    action: str


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


class Connection:
    def __init__(
        self,
        websocket: WebSocket,
        settings: Settings,
        bridge_factory: BridgeFactory,
        on_end: EndHook,
        runs_dir: Path,
        config: ServerConfig,
        capture: CaptureConfig,
        live: LiveConfig,
    ):
        self.websocket = websocket
        self.settings = settings
        self.bridge_factory = bridge_factory
        self.on_end = on_end
        self.runs_dir = runs_dir
        self.config = config
        self.capture = capture
        self.live = live
        self.send_lock = asyncio.Lock()
        self.sweep: SweepDir | None = None
        self.sweep_id = ""
        self.bridge: Bridge | None = None
        self.store: FrameStore | None = None
        self.monitor: Monitor | None = None
        self.coverage = CoverageMap(capture)
        self.stills: list[Still] = []
        self.coverage_sent: dict[str, str] = {}
        self.clock_offset_ms = 0
        self.audio_seq = 0
        self.muted = False
        self.ending = False
        self.bridge_task: asyncio.Task | None = None
        self.watcher_task: asyncio.Task | None = None
        self.end_task: asyncio.Task | None = None
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

    async def send_session(self, state: str) -> None:
        await self.send_json(
            {
                "type": "session",
                "state": state,
                "model": self.settings.live_model,
                "sweep_id": self.sweep_id,
                "live_fps": self.live.frame_max_fps,
            }
        )

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
        while True:
            message = await self.websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
            if message.get("bytes") is not None:
                await self.handle_binary(message["bytes"])
            elif message.get("text") is not None:
                await self.handle_text(message["text"])

    async def handle_text(self, text: str) -> None:
        try:
            payload = json.loads(text)
            kind = payload["type"]
            if kind == "hello":
                await self.hello(Hello.model_validate(payload))
            elif kind == "control":
                await self.control(Control.model_validate(payload))
        except (ValueError, KeyError, TypeError, ValidationError):
            logger.warning("dropped malformed text message")

    async def hello(self, hello: Hello) -> None:
        if self.bridge is not None:
            return
        if hello.t_ms is not None:
            self.clock_offset_ms = int(time.time() * 1000) - hello.t_ms
            logger.info("phone clock offset %d ms", self.clock_offset_ms)
        self.sweep_id = hello.sweep_id or uuid4().hex[:SESSION_ID_LENGTH]
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
        self.bridge = self.bridge_factory(hooks)
        self.monitor = Monitor(self.bridge.inject, self.live)
        await self.send_session("connecting")
        self.bridge_task = asyncio.create_task(self.bridge.run())
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
        elif control.action == "end_sweep" and self.bridge is not None:
            self.finish()

    def finish(self) -> None:
        if self.ending:
            return
        self.ending = True
        self.end_task = asyncio.create_task(self.process())

    async def process(self) -> None:
        if self.watcher_task:
            self.watcher_task.cancel()
        self.bridge.end_sweep()
        await self.send_session("processing")
        try:
            await self.on_end(self.sweep, self.send_json)
        except Exception:
            logger.exception("end of sweep processing failed")
            await self.send_session("closed")
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
        if len(payload) <= self.config.max_frame_bytes:
            await self.bridge.send_frame(payload)

    async def still(self, header: FrameHeader, payload: bytes) -> None:
        if self.ending or len(payload) > self.config.max_frame_bytes:
            return
        still = await asyncio.to_thread(self.store.add, header, payload)
        self.stills.append(still)
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
        tasks = [task for task in (self.watcher_task, self.end_task) if task]
        for task in tasks:
            task.cancel()
        if self.bridge is not None:
            with contextlib.suppress(Exception):
                await self.bridge.close()
        if self.bridge_task:
            tasks.append(self.bridge_task)
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.send_session("closed")
        with contextlib.suppress(RuntimeError):
            await self.websocket.close()


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
    app = FastAPI()

    @app.websocket("/ws/sweep")
    async def sweep_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        await Connection(websocket, settings, bridge_factory, on_end, runs_dir, config, capture, live).run()

    app.mount("/runs", StaticFiles(directory=runs_dir), name="runs")
    if web_dist.is_dir():
        app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")
    return app


def main() -> None:
    settings = load_settings()
    app = create_app(settings, live_bridge_factory(settings))
    uvicorn.run(app, host="0.0.0.0", port=settings.port)


if __name__ == "__main__":
    main()
