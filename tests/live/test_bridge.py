import asyncio
from contextlib import asynccontextmanager

import pytest

from libclaim.budget import Budget, BudgetExceeded
from libclaim.live.bridge import LiveBridge, load_live_config
from libclaim.live.session import (
    Audio,
    GoAway,
    Handle,
    Interrupted,
    RateLimited,
    Scheduling,
    ToolCall,
    Transcript,
    TurnComplete,
    Usage,
)
from libclaim.sweep import SweepDir

MODEL = "gemini-3.8-live"


class FakeSession:
    def __init__(self, script):
        self.script = script
        self.audio = []
        self.video = []
        self.text = []
        self.tool_responses = []
        self.closed = asyncio.Event()

    async def send_audio(self, data):
        self.audio.append(data)

    async def send_video(self, data):
        self.video.append(data)

    async def send_text(self, text):
        self.text.append(text)

    async def send_tool_response(self, call_id, name, response, will_continue, scheduling):
        self.tool_responses.append((call_id, name, response, will_continue, scheduling))

    async def receive(self):
        for item in self.script:
            if isinstance(item, Exception):
                raise item
            yield item
        await self.closed.wait()


class FakeFactory:
    def __init__(self, *scripts):
        self.scripts = list(scripts)
        self.handles = []
        self.sessions = []

    def __call__(self, handle):
        self.handles.append(handle)
        session = FakeSession(self.scripts[len(self.sessions)])
        self.sessions.append(session)

        @asynccontextmanager
        async def connection():
            try:
                yield session
            finally:
                session.closed.set()

        return connection()


class FakeTools:
    monitor_name = "capture_monitor"

    def __init__(self):
        self.calls = []

    async def call(self, name, args):
        self.calls.append((name, args))
        return {"ok": name}


class Harness:
    def __init__(self, tmp_path, factory, **overrides):
        self.factory = factory
        self.sleeps = []
        self.now = 0.0
        self.audio = []
        self.transcripts = []
        self.interrupts = 0
        self.tools = FakeTools()
        self.sweep = SweepDir(tmp_path / "sweep")
        self.budget = overrides.pop("budget", None) or Budget(tmp_path / "spend.jsonl")
        config = load_live_config().model_copy(update=overrides)

        async def sleep(seconds):
            self.sleeps.append(seconds)

        async def on_audio(data):
            self.audio.append(data)

        async def on_transcript(speaker, text):
            self.transcripts.append((speaker, text))

        async def on_interrupted():
            self.interrupts += 1

        self.bridge = LiveBridge(
            self.sweep,
            self.tools,
            on_audio,
            on_transcript,
            on_interrupted,
            factory=factory,
            budget=self.budget,
            model=MODEL,
            config=config,
            sleep=sleep,
            clock=lambda: self.now,
        )
        self.task = None

    def start(self):
        self.task = asyncio.create_task(self.bridge.run())

    async def until(self, predicate):
        for _ in range(2000):
            if predicate():
                return
            if self.task.done():
                self.task.result()
            await asyncio.sleep(0)
        raise AssertionError("condition never met")

    async def stop(self):
        await self.bridge.close()
        await self.task


async def test_goaway_reconnects_with_last_handle_and_delivers_audio(tmp_path):
    factory = FakeFactory(
        [Handle("h1"), Audio(b"before"), Handle("h2"), GoAway()],
        [Audio(b"after")],
    )
    harness = Harness(tmp_path, factory)
    harness.start()
    await harness.until(lambda: harness.audio == [b"before", b"after"])
    await harness.stop()
    assert factory.handles == [None, "h2"]
    assert harness.sleeps == []


async def test_429_backs_off_exponentially_then_reconnects(tmp_path):
    factory = FakeFactory(
        [Handle("h1"), RateLimited()],
        [RateLimited()],
        [RateLimited()],
        [Audio(b"back")],
    )
    harness = Harness(tmp_path, factory)
    harness.start()
    await harness.until(lambda: harness.audio == [b"back"])
    await harness.stop()
    assert harness.sleeps == [1.0, 2.0, 4.0]
    assert factory.handles == [None, "h1", "h1", "h1"]


async def test_backoff_resets_after_a_healthy_event(tmp_path):
    factory = FakeFactory(
        [RateLimited()],
        [Audio(b"ok"), RateLimited()],
        [Audio(b"again")],
    )
    harness = Harness(tmp_path, factory)
    harness.start()
    await harness.until(lambda: harness.audio == [b"ok", b"again"])
    await harness.stop()
    assert harness.sleeps == [1.0, 1.0]


async def test_gives_up_after_max_retries(tmp_path):
    factory = FakeFactory([RateLimited()], [RateLimited()], [RateLimited()])
    harness = Harness(tmp_path, factory, max_retries=2)
    harness.start()
    with pytest.raises(RateLimited):
        await harness.task
    assert harness.sleeps == [1.0, 2.0]


async def test_backoff_is_capped(tmp_path):
    factory = FakeFactory([RateLimited()], [RateLimited()], [RateLimited()], [Audio(b"x")])
    harness = Harness(tmp_path, factory, backoff_max_seconds=1.5)
    harness.start()
    await harness.until(lambda: harness.audio == [b"x"])
    await harness.stop()
    assert harness.sleeps == [1.0, 1.5, 1.5]


async def test_interrupted_is_forwarded_exactly_once(tmp_path):
    factory = FakeFactory([Audio(b"a"), Interrupted(), Audio(b"b"), TurnComplete()])
    harness = Harness(tmp_path, factory)
    harness.start()
    await harness.until(lambda: harness.audio == [b"a", b"b"])
    await harness.stop()
    assert harness.interrupts == 1


async def test_transcripts_are_forwarded(tmp_path):
    factory = FakeFactory([Transcript("user", "hello"), Transcript("model", "hi")])
    harness = Harness(tmp_path, factory)
    harness.start()
    await harness.until(lambda: len(harness.transcripts) == 2)
    await harness.stop()
    assert harness.transcripts == [("user", "hello"), ("model", "hi")]


async def test_frames_are_rate_limited_and_stop_after_end_sweep(tmp_path):
    factory = FakeFactory([])
    harness = Harness(tmp_path, factory)
    harness.start()
    await harness.until(lambda: factory.sessions)
    session = factory.sessions[0]
    await harness.bridge.send_frame(b"f1")
    harness.now = 0.4
    await harness.bridge.send_frame(b"f2")
    harness.now = 1.1
    await harness.bridge.send_frame(b"f3")
    harness.bridge.end_sweep()
    harness.now = 5.0
    await harness.bridge.send_frame(b"f4")
    await harness.bridge.send_audio(b"still-audio")
    await harness.stop()
    assert session.video == [b"f1", b"f3"]
    assert session.audio == [b"still-audio"]


async def test_usage_is_written_to_budget_and_ledger(tmp_path):
    usage = Usage(input={"text": 1000, "video": 2000}, output={"audio": 500})
    factory = FakeFactory([usage])
    harness = Harness(tmp_path, factory)
    harness.start()
    await harness.until(lambda: harness.sweep.ledger.entries)
    await harness.stop()
    expected = (1000 * 0.50 + 2000 * 1.00 + 500 * 12.00) / 1_000_000
    entry = harness.sweep.ledger.entries[0]
    assert entry.stage == "live"
    assert entry.tokens == 3500
    assert entry.usd_at_paid_rates == pytest.approx(expected)
    assert harness.budget.total_usd() == pytest.approx(expected)


async def test_budget_cap_stops_sending_and_raises(tmp_path):
    budget = Budget(tmp_path / "spend.jsonl", cap_usd=0.001)
    factory = FakeFactory([Usage(input={"audio": 10_000}), Audio(b"late")])
    harness = Harness(tmp_path, factory, budget=budget)
    harness.start()
    with pytest.raises(BudgetExceeded):
        await harness.task
    session = factory.sessions[0]
    with pytest.raises(BudgetExceeded):
        await harness.bridge.send_audio(b"x")
    with pytest.raises(BudgetExceeded):
        await harness.bridge.send_frame(b"x")
    assert session.audio == []
    assert session.video == []
    assert session.closed.is_set()
    assert harness.audio == []
    assert len(factory.sessions) == 1


async def test_budget_checked_before_connect(tmp_path):
    budget = Budget(tmp_path / "spend.jsonl", cap_usd=0.001)
    budget.record(MODEL, 10_000, 0, "audio")
    factory = FakeFactory([Audio(b"x")])
    harness = Harness(tmp_path, factory, budget=budget)
    harness.start()
    with pytest.raises(BudgetExceeded):
        await harness.task
    assert factory.sessions == []


async def test_kickoff_text_is_sent_once_across_reconnects(tmp_path):
    factory = FakeFactory([GoAway()], [Audio(b"x")])
    harness = Harness(tmp_path, factory)
    harness.start()
    await harness.until(lambda: harness.audio == [b"x"])
    await harness.stop()
    assert factory.sessions[0].text == ["Begin the sweep now."]
    assert factory.sessions[1].text == []


async def test_inject_streams_into_the_monitor_call(tmp_path):
    factory = FakeFactory([ToolCall("c1", "capture_monitor", {"unit": "A"}), ToolCall("c2", "lookup", {"q": 1})])
    harness = Harness(tmp_path, factory)
    harness.start()
    await harness.until(lambda: len(factory.sessions[0].tool_responses) == 2 if factory.sessions else False)
    await harness.bridge.inject({"note": "blur"}, Scheduling.INTERRUPT)
    await harness.stop()
    responses = factory.sessions[0].tool_responses
    assert responses[0] == ("c1", "capture_monitor", {"output": {"ok": "capture_monitor"}}, True, Scheduling.SILENT)
    assert responses[1] == ("c2", "lookup", {"output": {"ok": "lookup"}}, False, Scheduling.WHEN_IDLE)
    assert responses[2] == ("c1", "capture_monitor", {"output": {"note": "blur"}}, True, Scheduling.INTERRUPT)


async def test_inject_before_monitor_call_is_an_error(tmp_path):
    factory = FakeFactory([])
    harness = Harness(tmp_path, factory)
    harness.start()
    await harness.until(lambda: factory.sessions)
    with pytest.raises(LookupError):
        await harness.bridge.inject({"note": "x"}, Scheduling.SILENT)
    await harness.stop()
