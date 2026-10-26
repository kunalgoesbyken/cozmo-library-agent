import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

import yaml
from pydantic import BaseModel

from libclaim.budget import Budget, BudgetExceeded
from libclaim.live import session as events
from libclaim.settings import CONFIG_DIR
from libclaim.sweep import SweepDir

LIVE_CONFIG_PATH = CONFIG_DIR / "live.yaml"


class LiveConfig(BaseModel):
    compression_enabled: bool
    compression_trigger_tokens: int
    compression_target_tokens: int
    frame_max_fps: float
    frame_mime: str
    audio_mime: str
    media_resolution: str
    kickoff_text: str
    backoff_base_seconds: float
    backoff_max_seconds: float
    max_retries: int
    ledger_stage: str
    statement_window_ms: int
    monitor_min_gap_seconds: float
    monitor_region_cooldown_seconds: float
    alert_ttl_seconds: float


def load_live_config(path=LIVE_CONFIG_PATH) -> LiveConfig:
    return LiveConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


class LiveBridge:
    def __init__(
        self,
        sweep: SweepDir,
        tools: events.Tools,
        on_audio: Callable[[bytes], Awaitable[None]],
        on_transcript: Callable[[str, str], Awaitable[None]],
        on_interrupted: Callable[[], Awaitable[None]],
        *,
        factory: events.SessionFactory,
        budget: Budget,
        model: str,
        config: LiveConfig | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.sweep = sweep
        self.tools = tools
        self.on_audio = on_audio
        self.on_transcript = on_transcript
        self.on_interrupted = on_interrupted
        self.factory = factory
        self.budget = budget
        self.model = model
        self.config = config or load_live_config()
        self.sleep = sleep
        self.clock = clock
        self.session: events.LiveSession | None = None
        self.handle: str | None = None
        self.monitor_call_id: str | None = None
        self.sweep_ended = False
        self.closed = False
        self.exceeded: BudgetExceeded | None = None
        self.last_frame_at: float | None = None
        self.last_usage_at = clock()
        self.kicked_off = False
        self.attempts = 0
        self.task: asyncio.Task | None = None
        self.done = asyncio.Event()

    async def run(self) -> None:
        self.task = asyncio.current_task()
        try:
            while not self.closed:
                await self.connect_and_listen()
        except asyncio.CancelledError:
            if not self.closed:
                raise
        finally:
            self.session = None
            self.done.set()

    async def connect_and_listen(self) -> None:
        try:
            self.budget.check()
            async with self.factory(self.handle) as session:
                self.session = session
                await self.kickoff(session)
                async for event in session.receive():
                    self.attempts = 0
                    if isinstance(event, events.GoAway):
                        return
                    await self.handle_event(session, event)
                raise events.SessionClosed()
        except BudgetExceeded as error:
            self.exceeded = error
            raise
        except (events.RateLimited, events.SessionClosed):
            await self.back_off()
        finally:
            self.session = None

    async def back_off(self) -> None:
        if self.attempts >= self.config.max_retries:
            raise
        delay = min(self.config.backoff_base_seconds * 2**self.attempts, self.config.backoff_max_seconds)
        self.attempts += 1
        await self.sleep(delay)

    async def kickoff(self, session: events.LiveSession) -> None:
        if not self.kicked_off:
            self.kicked_off = True
            await session.send_text(self.config.kickoff_text)

    async def handle_event(self, session: events.LiveSession, event: events.Event) -> None:
        match event:
            case events.Audio(data):
                await self.on_audio(data)
            case events.Transcript(speaker, text):
                await self.on_transcript(speaker, text)
            case events.Interrupted():
                await self.on_interrupted()
            case events.Handle(value):
                self.handle = value
            case events.Usage():
                self.record_usage(event)
            case events.TurnComplete():
                self.budget.check()
            case events.ToolCall():
                await self.answer_tool(session, event)

    def record_usage(self, usage: events.Usage) -> None:
        usd = sum(self.budget.record(self.model, tokens, 0, modality) for modality, tokens in usage.input.items())
        usd += sum(self.budget.record(self.model, 0, tokens, modality) for modality, tokens in usage.output.items())
        now = self.clock()
        tokens = sum(usage.input.values()) + sum(usage.output.values())
        self.sweep.ledger.add(self.config.ledger_stage, now - self.last_usage_at, tokens, usd)
        self.last_usage_at = now
        self.budget.check()

    async def answer_tool(self, session: events.LiveSession, call: events.ToolCall) -> None:
        result = await self.tools.call(call.name, call.args)
        if call.name == self.tools.monitor_name:
            self.monitor_call_id = call.id
            await session.send_tool_response(call.id, call.name, {"output": result}, True, events.Scheduling.SILENT)
        else:
            await session.send_tool_response(call.id, call.name, {"output": result}, False, events.Scheduling.WHEN_IDLE)

    def ready_session(self) -> events.LiveSession | None:
        if self.exceeded:
            raise self.exceeded
        return self.session

    async def send_audio(self, data: bytes) -> None:
        session = self.ready_session()
        if session:
            await session.send_audio(data)

    async def send_frame(self, data: bytes) -> None:
        session = self.ready_session()
        if not session or self.sweep_ended:
            return
        now = self.clock()
        if self.last_frame_at is not None and now - self.last_frame_at < 1 / self.config.frame_max_fps:
            return
        self.last_frame_at = now
        await session.send_video(data)

    def end_sweep(self) -> None:
        self.sweep_ended = True

    async def inject(self, event: dict[str, Any], scheduling: events.Scheduling) -> None:
        session = self.ready_session()
        if not session or not self.monitor_call_id:
            raise LookupError("monitor tool has not been called in the live session")
        await session.send_tool_response(
            self.monitor_call_id, self.tools.monitor_name, {"output": event}, True, scheduling
        )

    async def close(self) -> None:
        self.closed = True
        if self.task and not self.done.is_set():
            self.task.cancel()
            await self.done.wait()
