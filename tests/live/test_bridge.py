import asyncio
import json
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
    SessionClosed,
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
        self.fail_sends = False

    async def send_audio(self, data):
        if self.fail_sends:
            raise SessionClosed()
        self.audio.append(data)

    async def send_video(self, data):
        if self.fail_sends:
            raise SessionClosed()
        self.video.append(data)

    async def send_text(self, text):
        self.text.append(text)

    async def send_tool_response(self, call_id, name, response, will_continue, scheduling):
        self.tool_responses.append((call_id, name, response, will_continue, scheduling))

    async def receive(self):
        for item in self.script:
            if isinstance(item, Exception):
                raise item
            if isinstance(item, asyncio.Event):
                await item.wait()
            elif callable(item):
                item()
            else:
                yield item
        await self.closed.wait()


class FakeFactory:
    def __init__(self, *scripts):
        self.scripts = list(scripts)
        self.handles = []
        self.sessions = []

    def __call__(self, handle):
        self.handles.append(handle)
        script = self.scripts[len(self.handles) - 1]
        if isinstance(script, Exception):

            @asynccontextmanager
            async def refused():
                raise script
                yield

            return refused()
        session = FakeSession(script)
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
        self.sleep_gate = None
        self.fail_audio = False
        self.tools = FakeTools()
        self.sweep = SweepDir(tmp_path / "sweep")
        self.budget = overrides.pop("budget", None) or Budget(tmp_path / "spend.jsonl")
        overrides.setdefault("min_healthy_seconds", 0.0)
        config = load_live_config().model_copy(update=overrides)

        async def sleep(seconds):
            self.sleeps.append(seconds)
            if self.sleep_gate:
                await self.sleep_gate.wait()

        async def on_audio(data):
            if self.fail_audio:
                raise ValueError("sink broke")
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
    with pytest.raises(RateLimited):
        await harness.bridge.send_audio(b"x")
    await harness.bridge.close()


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
    await harness.until(lambda: harness.bridge.session)
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
    await harness.until(lambda: factory.sessions and len(factory.sessions[0].tool_responses) == 2)
    await harness.bridge.inject({"note": "blur"}, Scheduling.INTERRUPT)
    await harness.stop()
    responses = factory.sessions[0].tool_responses
    assert responses[0] == ("c1", "capture_monitor", {"output": {"ok": "capture_monitor"}}, True, Scheduling.SILENT)
    assert responses[1] == ("c2", "lookup", {"output": {"ok": "lookup"}}, False, Scheduling.WHEN_IDLE)
    assert responses[2] == ("c1", "capture_monitor", {"output": {"note": "blur"}}, True, Scheduling.INTERRUPT)


async def test_inject_before_the_monitor_call_goes_out_as_a_text_turn(tmp_path):
    factory = FakeFactory([])
    harness = Harness(tmp_path, factory)
    harness.start()
    await harness.until(lambda: harness.bridge.session)
    await harness.bridge.inject({"readback": "I counted 3 books"}, Scheduling.WHEN_IDLE)
    await harness.stop()
    [text] = factory.sessions[0].text[1:]
    assert json.loads(text) == {"readback": "I counted 3 books"}
    assert factory.sessions[0].tool_responses == []


async def test_silent_inject_before_the_monitor_call_is_dropped(tmp_path):
    factory = FakeFactory([])
    harness = Harness(tmp_path, factory)
    harness.start()
    await harness.until(lambda: harness.bridge.session)
    await harness.bridge.inject({"note": "x"}, Scheduling.SILENT)
    await harness.stop()
    assert factory.sessions[0].text[1:] == []


async def test_inject_without_a_session_is_an_error(tmp_path):
    harness = Harness(tmp_path, FakeFactory([]))
    with pytest.raises(LookupError):
        await harness.bridge.inject({"note": "x"}, Scheduling.WHEN_IDLE)


async def test_goaway_straight_after_connect_is_backed_off_and_capped(tmp_path):
    factory = FakeFactory(*[[GoAway()] for _ in range(4)])
    harness = Harness(tmp_path, factory, min_healthy_seconds=5.0, max_retries=3)
    harness.start()
    with pytest.raises(SessionClosed):
        await harness.task
    assert harness.sleeps == [1.0, 2.0, 4.0]
    assert len(factory.sessions) == 4


async def test_session_that_emits_one_event_then_drops_does_not_reset_backoff(tmp_path):
    factory = FakeFactory(*[[Audio(b"x"), RateLimited()] for _ in range(4)])
    harness = Harness(tmp_path, factory, min_healthy_seconds=5.0, max_retries=3)
    harness.start()
    with pytest.raises(RateLimited):
        await harness.task
    assert harness.sleeps == [1.0, 2.0, 4.0]


async def test_healthy_session_resets_backoff_and_goaway_reconnects_at_once(tmp_path):
    def age():
        harness.now += 10.0

    factory = FakeFactory([RateLimited()], [age, Audio(b"a"), GoAway()], [Audio(b"b")])
    harness = Harness(tmp_path, factory, min_healthy_seconds=5.0)
    harness.start()
    await harness.until(lambda: harness.audio == [b"a", b"b"])
    await harness.stop()
    assert harness.sleeps == [1.0]


async def test_connect_errors_are_retried_under_the_same_cap(tmp_path):
    factory = FakeFactory(TimeoutError(), ConnectionResetError(), [Audio(b"ok")])
    harness = Harness(tmp_path, factory)
    harness.start()
    await harness.until(lambda: harness.audio == [b"ok"])
    await harness.stop()
    assert harness.sleeps == [1.0, 2.0]


async def test_connect_errors_surface_when_retries_run_out(tmp_path):
    factory = FakeFactory(OSError("down"), OSError("down"))
    harness = Harness(tmp_path, factory, max_retries=1)
    harness.start()
    with pytest.raises(OSError):
        await harness.task
    await harness.bridge.close()


async def test_callback_failure_surfaces_and_close_still_works(tmp_path):
    factory = FakeFactory([Audio(b"x")])
    harness = Harness(tmp_path, factory)
    harness.fail_audio = True
    harness.start()
    with pytest.raises(ValueError):
        await harness.task
    await harness.bridge.close()


async def test_send_racing_a_reconnect_is_buffered_not_raised(tmp_path):
    gate = asyncio.Event()
    factory = FakeFactory([gate, GoAway()], [])
    harness = Harness(tmp_path, factory)
    harness.start()
    await harness.until(lambda: harness.bridge.session)
    factory.sessions[0].fail_sends = True
    await harness.bridge.send_audio(b"raced")
    await harness.bridge.send_frame(b"frame")
    gate.set()
    await harness.until(lambda: len(factory.sessions) == 2 and factory.sessions[1].audio)
    await harness.stop()
    assert factory.sessions[1].audio == [b"raced"]
    assert factory.sessions[1].video == []


async def test_audio_during_backoff_is_buffered_and_flushed_but_frames_are_dropped(tmp_path):
    factory = FakeFactory([RateLimited()], [])
    harness = Harness(tmp_path, factory, audio_buffer_max_chunks=2)
    harness.sleep_gate = asyncio.Event()
    harness.start()
    await harness.until(lambda: harness.sleeps == [1.0])
    for chunk in (b"a", b"b", b"c"):
        await harness.bridge.send_audio(chunk)
    await harness.bridge.send_frame(b"f")
    harness.sleep_gate.set()
    await harness.until(lambda: len(factory.sessions) == 2 and factory.sessions[1].audio)
    await harness.stop()
    assert factory.sessions[1].audio == [b"b", b"c"]
    assert factory.sessions[1].video == []


async def test_stale_buffered_audio_is_discarded(tmp_path):
    factory = FakeFactory([RateLimited()], [])
    harness = Harness(tmp_path, factory, audio_buffer_max_age_seconds=10.0)
    harness.sleep_gate = asyncio.Event()
    harness.start()
    await harness.until(lambda: harness.sleeps == [1.0])
    await harness.bridge.send_audio(b"old")
    harness.now = 11.0
    await harness.bridge.send_audio(b"new")
    harness.sleep_gate.set()
    await harness.until(lambda: len(factory.sessions) == 2 and factory.sessions[1].audio)
    await harness.stop()
    assert factory.sessions[1].audio == [b"new"]


async def test_sends_check_the_budget_each_time(tmp_path):
    budget = Budget(tmp_path / "spend.jsonl", cap_usd=0.001)
    factory = FakeFactory([])
    harness = Harness(tmp_path, factory, budget=budget)
    harness.start()
    await harness.until(lambda: harness.bridge.session)
    budget.record(MODEL, 10_000, 0, "audio")
    with pytest.raises(BudgetExceeded):
        await harness.bridge.send_audio(b"x")
    await harness.stop()
    assert factory.sessions[0].audio == []


async def test_each_usage_report_is_recorded_additively(tmp_path):
    factory = FakeFactory([Usage(input={"text": 100}), Usage(input={"text": 100})])
    harness = Harness(tmp_path, factory)
    harness.start()
    await harness.until(lambda: len(harness.sweep.ledger.entries) == 2)
    await harness.stop()
    assert harness.budget.total_usd() == pytest.approx(2 * 100 * 0.50 / 1_000_000)


async def test_kickoff_failure_is_retried_on_next_connection(tmp_path):
    factory = FakeFactory([], [])
    harness = Harness(tmp_path, factory)
    original = FakeSession.send_text

    async def failing(self, text):
        if len(factory.sessions) == 1:
            raise SessionClosed()
        await original(self, text)

    FakeSession.send_text = failing
    try:
        harness.start()
        await harness.until(lambda: len(factory.sessions) == 2 and factory.sessions[1].text)
        await harness.stop()
    finally:
        FakeSession.send_text = original
    assert factory.sessions[1].text == ["Begin the sweep now."]


async def test_rate_limited_sends_are_buffered_or_dropped_and_never_propagate(tmp_path):
    gate = asyncio.Event()
    factory = FakeFactory([gate, GoAway()], [])
    harness = Harness(tmp_path, factory)
    harness.start()
    await harness.until(lambda: harness.bridge.session)
    first = factory.sessions[0]

    async def limited(data):
        raise RateLimited()

    first.send_audio = limited
    first.send_video = limited
    await harness.bridge.send_audio(b"a")
    await harness.bridge.send_frame(b"f")
    gate.set()
    await harness.until(lambda: len(factory.sessions) == 2 and factory.sessions[1].audio)
    await harness.stop()
    assert factory.sessions[1].audio == [b"a"]
    assert factory.sessions[1].video == []


async def test_oserror_from_a_callback_is_surfaced_not_retried(tmp_path):
    factory = FakeFactory([Audio(b"x")], [Audio(b"y")])
    harness = Harness(tmp_path, factory)

    async def broken(data):
        raise ConnectionResetError("sink")

    harness.bridge.on_audio = broken
    harness.start()
    with pytest.raises(ConnectionResetError):
        await harness.task
    assert harness.sleeps == []
    assert len(factory.sessions) == 1
    assert isinstance(harness.bridge.failure, ConnectionResetError)
    await harness.bridge.close()
