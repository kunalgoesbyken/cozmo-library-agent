import argparse
import asyncio
import io
import json
import os
import subprocess
import tempfile
import time
import wave
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image

from libclaim.budget import Budget
from libclaim.settings import ROOT, load_settings

SCRATCH = ROOT / "_scratch"
SHELF_IMAGE = SCRATCH / "shelf" / "german.jpg"
OUT_DIR = SCRATCH / "live_soak"
SPEND_CAP_USD = 0.15
SOAK_CAP_USD = 0.10
RESUME_RESERVE_USD = 0.015
SAMPLE_RATE = 16000
CHUNK_SECONDS = 0.1
CHUNK_BYTES = int(SAMPLE_RATE * CHUNK_SECONDS) * 2
FRAME_INTERVAL_SECONDS = 2.0
FRAME_LONG_SIDE = 768
PAN_STEPS = 12
FILLER_ENTRIES = 900
PAN_WINDOW = 0.6
REACTION_WINDOW_SECONDS = 10.0
TRIGGER_TOKENS = 12000
TARGET_TOKENS = 6000
TOOL_NAME = "capture_monitor"
SYSTEM_INSTRUCTION = (
    "You are a quiet library inventory coach watching a live camera of a bookshelf. "
    f"At the start of the session call {TOOL_NAME} exactly once and never again. "
    "Answer every question in one short sentence. Do not speak unless asked or told something new."
)
SPOKEN_PROMPTS = {
    "what_do_you_see": "What do you see on the shelf right now?",
    "how_many_logged": "How many books have we logged so far?",
}
INTERRUPT_PROBE_TEXT = "Describe in at least eight sentences every book you can see, one sentence per book."
INTERRUPT_NOTE = "Note: unit A row 2 has been logged."
TOOL_PAYLOADS = {
    "SILENT": {"logged_books": 14, "unreadable": 3, "note": "state only, do not speak"},
    "WHEN_IDLE": {"coaching": "Frames on unit A row 3 are blurry. Tell the user in one sentence to slow down."},
    "INTERRUPT": {"urgent": "Tell the user in one sentence to stop and hold still right now."},
    "CLOSE": {"done": True},
}
MODALITY_BY_MEDIA = {"AUDIO": "audio", "VIDEO": "video", "IMAGE": "video", "TEXT": "text"}


def synthesize(text: str) -> bytes:
    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / "prompt.wav"
        script = (
            "Add-Type -AssemblyName System.Speech;"
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
            "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo("
            f"{SAMPLE_RATE}, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono);"
            f"$s.SetOutputToWaveFile('{target}', $f);"
            "$s.Speak($env:SPIKE_TEXT);"
            "$s.Dispose()"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            check=True,
            env={**os.environ, "SPIKE_TEXT": text},
        )
        with wave.open(str(target), "rb") as handle:
            return handle.readframes(handle.getnframes())


def prepare_frames() -> list[bytes]:
    image = Image.open(SHELF_IMAGE).convert("RGB")
    width, height = image.size
    window = int(width * PAN_WINDOW)
    frames = []
    for step in range(PAN_STEPS):
        left = (width - window) * step // (PAN_STEPS - 1)
        frame = image.crop((left, 0, left + window, height))
        frame.thumbnail((FRAME_LONG_SIDE, FRAME_LONG_SIDE))
        buffer = io.BytesIO()
        frame.save(buffer, format="JPEG", quality=80)
        frames.append(buffer.getvalue())
    return frames


class Soak:
    def __init__(self, model: str, seconds: float, scenario: str, client: genai.Client, budget: Budget):
        self.model = model
        self.seconds = seconds
        self.scenario = scenario
        self.client = client
        self.budget = budget
        self.started = time.monotonic()
        self.spent_start = budget.total_usd()
        self.cap = min(SOAK_CAP_USD, SPEND_CAP_USD - self.spent_start - RESUME_RESERVE_USD)
        self.events: list[dict] = []
        self.pending_audio = bytearray()
        self.tool_call_id: str | None = None
        self.handle: str | None = None
        self.first_audio_after_probe = asyncio.Event()
        self.probe_armed = False
        self.over_budget = asyncio.Event()

    def now(self) -> float:
        return round(time.monotonic() - self.started, 2)

    def log(self, kind: str, **detail) -> None:
        self.events.append({"t": self.now(), "kind": kind, **detail})

    def config(self, handle: str | None) -> types.LiveConnectConfig:
        declaration = types.FunctionDeclaration(
            name=TOOL_NAME,
            description="Start the capture monitor that streams inventory events into the conversation.",
            behavior=types.Behavior.NON_BLOCKING,
        )
        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=SYSTEM_INSTRUCTION,
            tools=[types.Tool(function_declarations=[declaration])],
            session_resumption=types.SessionResumptionConfig(handle=handle),
            context_window_compression=types.ContextWindowCompressionConfig(
                trigger_tokens=TRIGGER_TOKENS,
                sliding_window=types.SlidingWindow(target_tokens=TARGET_TOKENS),
            ),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
        )

    def spend(self) -> float:
        return self.budget.total_usd() - self.spent_start

    def record_usage(self, usage: types.UsageMetadata) -> None:
        for detail in usage.prompt_tokens_details or []:
            modality = MODALITY_BY_MEDIA.get(detail.modality.name, "text")
            self.budget.record(self.model, detail.token_count or 0, 0, modality)
        for detail in usage.response_tokens_details or []:
            modality = MODALITY_BY_MEDIA.get(detail.modality.name, "text")
            self.budget.record(self.model, 0, detail.token_count or 0, modality)
        if self.spend() >= self.cap:
            self.over_budget.set()

    async def send_video(self, session) -> None:
        frames = prepare_frames()
        index = 0
        while True:
            frame = frames[index % len(frames)]
            index += 1
            await session.send_realtime_input(video=types.Blob(data=frame, mime_type="image/jpeg"))
            self.log("video_frame", bytes=len(frame))
            await asyncio.sleep(FRAME_INTERVAL_SECONDS)

    async def send_audio(self, session) -> None:
        silence = bytes(CHUNK_BYTES)
        next_tick = time.monotonic()
        while True:
            if self.pending_audio:
                chunk = bytes(self.pending_audio[:CHUNK_BYTES]).ljust(CHUNK_BYTES, b"\x00")
                del self.pending_audio[:CHUNK_BYTES]
            else:
                chunk = silence
            await session.send_realtime_input(audio=types.Blob(data=chunk, mime_type=f"audio/pcm;rate={SAMPLE_RATE}"))
            next_tick += CHUNK_SECONDS
            await asyncio.sleep(max(0.0, next_tick - time.monotonic()))

    async def receive(self, session) -> None:
        while True:
            async for message in session.receive():
                self.handle_message(message)

    def handle_message(self, message: types.LiveServerMessage) -> None:
        if message.setup_complete:
            self.log("setup_complete")
        if message.usage_metadata:
            usage = message.usage_metadata
            self.log(
                "usage",
                prompt=usage.prompt_token_count,
                response=usage.response_token_count,
                thoughts=usage.thoughts_token_count,
                total=usage.total_token_count,
                prompt_details={d.modality.name: d.token_count for d in usage.prompt_tokens_details or []},
                spend_usd=round(self.spend(), 5),
            )
            self.record_usage(usage)
        if message.go_away:
            self.log("go_away", time_left=str(message.go_away.time_left))
        if message.session_resumption_update:
            update = message.session_resumption_update
            self.log("resumption_update", resumable=update.resumable, has_handle=bool(update.new_handle))
            if update.resumable and update.new_handle:
                self.handle = update.new_handle
        if message.tool_call:
            for call in message.tool_call.function_calls:
                self.log("tool_call", name=call.name, id=call.id)
                if call.name == TOOL_NAME:
                    self.tool_call_id = call.id
        if message.tool_call_cancellation:
            self.log("tool_call_cancellation", ids=list(message.tool_call_cancellation.ids or []))
        content = message.server_content
        if content:
            if content.model_turn:
                audio = sum(len(part.inline_data.data) for part in content.model_turn.parts or [] if part.inline_data)
                if audio:
                    self.log("model_audio", bytes=audio)
                    if self.probe_armed:
                        self.first_audio_after_probe.set()
            if content.input_transcription and content.input_transcription.text:
                self.log("input_text", text=content.input_transcription.text)
            if content.output_transcription and content.output_transcription.text:
                self.log("output_text", text=content.output_transcription.text)
            if content.interrupted:
                self.log("interrupted")
            if content.generation_complete:
                self.log("generation_complete")
            if content.turn_complete:
                self.log("turn_complete")

    def mark(self, name: str) -> None:
        self.log("marker", name=name)

    async def speak(self, key: str, pcm: bytes) -> None:
        self.mark(f"spoken:{key}")
        self.pending_audio.extend(pcm)

    async def stream_tool(self, session, name: str, scheduling: types.FunctionResponseScheduling, will_continue: bool) -> None:
        if not self.tool_call_id:
            self.log("tool_skipped", name=name, reason="model never called the tool")
            return
        self.mark(f"tool:{name}")
        response = types.FunctionResponse(
            id=self.tool_call_id,
            name=TOOL_NAME,
            response={"output": TOOL_PAYLOADS[name]},
            will_continue=will_continue,
            scheduling=scheduling,
        )
        await session.send_tool_response(function_responses=[response])

    async def interrupt_probe(self, session) -> None:
        self.mark("client_content_probe:start")
        self.probe_armed = True
        await session.send_realtime_input(text=INTERRUPT_PROBE_TEXT)
        try:
            await asyncio.wait_for(self.first_audio_after_probe.wait(), timeout=REACTION_WINDOW_SECONDS)
        except asyncio.TimeoutError:
            self.log("client_content_probe", result="model never started speaking")
            return
        await asyncio.sleep(1.0)
        self.mark("client_content_probe:send_turn_complete_false")
        await session.send_client_content(
            turns=types.Content(role="user", parts=[types.Part(text=INTERRUPT_NOTE)]),
            turn_complete=False,
        )

    async def flood_context(self, session) -> None:
        self.mark("compression_probe:flood")
        entries = [f"Inventory log entry {index}: unit A row {index % 5} position {index}." for index in range(FILLER_ENTRIES)]
        await session.send_realtime_input(text="Here is the inventory log, just remember it. " + " ".join(entries))

    async def timeline(self, session, pcm: dict[str, bytes]) -> None:
        begin = (0.02, lambda: session.send_realtime_input(text="Begin the sweep now."))
        if self.scenario == "interrupt":
            await self.follow(
                [
                    begin,
                    (0.20, lambda: self.interrupt_probe(session)),
                ]
            )
            return
        if self.scenario == "compress":
            await self.follow(
                [
                    begin,
                    (0.20, lambda: self.flood_context(session)),
                    (0.40, lambda: session.send_realtime_input(text="In one sentence, how many log entries did I just give you?")),
                    (0.70, lambda: session.send_realtime_input(text="In one sentence, what was the last log entry?")),
                ]
            )
            return
        plan = [
            (0.02, lambda: session.send_realtime_input(text="Begin the sweep now.")),
            (0.12, lambda: self.speak("what_do_you_see", pcm["what_do_you_see"])),
            (0.25, lambda: self.stream_tool(session, "SILENT", types.FunctionResponseScheduling.SILENT, True)),
            (0.35, lambda: self.speak("how_many_logged", pcm["how_many_logged"])),
            (0.50, lambda: self.stream_tool(session, "WHEN_IDLE", types.FunctionResponseScheduling.WHEN_IDLE, True)),
            (0.65, lambda: self.stream_tool(session, "INTERRUPT", types.FunctionResponseScheduling.INTERRUPT, True)),
            (0.78, lambda: self.interrupt_probe(session)),
            (0.95, lambda: self.stream_tool(session, "CLOSE", types.FunctionResponseScheduling.SILENT, False)),
        ]
        await self.follow(plan)

    async def follow(self, plan) -> None:
        for fraction, action in plan:
            await asyncio.sleep(max(0.0, fraction * self.seconds - (time.monotonic() - self.started)))
            await action()

    async def run(self) -> None:
        pcm = {key: synthesize(text) for key, text in SPOKEN_PROMPTS.items()}
        self.budget.check()
        async with self.client.aio.live.connect(model=self.model, config=self.config(None)) as session:
            self.log("connected")
            tasks = [
                asyncio.create_task(self.send_video(session)),
                asyncio.create_task(self.send_audio(session)),
                asyncio.create_task(self.receive(session)),
                asyncio.create_task(self.timeline(session, pcm)),
            ]
            try:
                await asyncio.wait_for(self.over_budget.wait(), timeout=self.seconds)
                self.log("stopped_over_budget")
            except asyncio.TimeoutError:
                pass
            finally:
                for task in tasks:
                    task.cancel()
                done = await asyncio.gather(*tasks, return_exceptions=True)
                for outcome in done:
                    if isinstance(outcome, Exception) and not isinstance(outcome, asyncio.CancelledError):
                        self.log("task_error", error=f"{type(outcome).__name__}: {str(outcome)[:600]}")

    async def resume_check(self) -> None:
        if not self.handle:
            self.log("resume_check", result="no resumption handle was received")
            return
        self.budget.check()
        async with self.client.aio.live.connect(model=self.model, config=self.config(self.handle)) as session:
            self.mark("resume:connected")
            await session.send_realtime_input(text="In one short sentence, what did you last see on the shelf?")
            try:
                await asyncio.wait_for(self.drain_one_turn(session), timeout=20)
            except asyncio.TimeoutError:
                self.log("resume_check", result="no turn within 20 s")

    async def drain_one_turn(self, session) -> None:
        async for message in session.receive():
            self.handle_message(message)


def summarise(events: list[dict], seconds: float) -> dict:
    usage = [e for e in events if e["kind"] == "usage"]
    prompts = [e["prompt"] for e in usage if e["prompt"] is not None]
    drops = [(a["t"], a["prompt"], b["prompt"]) for a, b in zip(usage, usage[1:]) if a["prompt"] and b["prompt"] and b["prompt"] < a["prompt"]]
    reactions = {}
    markers = [e for e in events if e["kind"] == "marker"]
    for marker in markers:
        window = [e for e in events if marker["t"] <= e["t"] <= marker["t"] + REACTION_WINDOW_SECONDS]
        reactions[f"{marker['t']}:{marker['name']}"] = {
            "model_audio_chunks": sum(1 for e in window if e["kind"] == "model_audio"),
            "interrupted": any(e["kind"] == "interrupted" for e in window),
            "output_text": "".join(e["text"] for e in window if e["kind"] == "output_text")[:200],
        }
    return {
        "turns_with_usage": len(usage),
        "prompt_tokens_min": min(prompts, default=None),
        "prompt_tokens_max": max(prompts, default=None),
        "prompt_tokens_last": prompts[-1] if prompts else None,
        "prompt_drops": drops,
        "go_away": [e for e in events if e["kind"] == "go_away"],
        "tool_calls": [e for e in events if e["kind"] == "tool_call"],
        "errors": [e for e in events if e["kind"] == "task_error"],
        "billed_prompt_tokens_per_minute": round(sum(prompts) / (seconds / 60), 1) if prompts else None,
        "reactions": reactions,
    }


async def main_async(seconds: float, resume: bool, scenario: str) -> None:
    settings = load_settings()
    budget = Budget(cap_usd=SPEND_CAP_USD)
    client = genai.Client(api_key=settings.gemini_api_key)
    soak = Soak(settings.live_model, seconds, scenario, client, budget)
    try:
        await soak.run()
        if resume and budget.total_usd() + RESUME_RESERVE_USD < SPEND_CAP_USD:
            await soak.resume_check()
    except Exception as error:
        soak.log("fatal", error=f"{type(error).__name__}: {str(error)[:800]}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (OUT_DIR / f"{stamp}.json").write_text(json.dumps(soak.events, indent=1), encoding="utf-8")
    summary = summarise(soak.events, seconds)
    summary["spent_usd"] = round(soak.spend(), 5)
    summary["fatal"] = [e for e in soak.events if e["kind"] == "fatal"]
    print(json.dumps(summary, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=240.0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--scenario", choices=["soak", "interrupt", "compress"], default="soak")
    args = parser.parse_args()
    asyncio.run(main_async(args.seconds, not args.no_resume, args.scenario))


if __name__ == "__main__":
    main()
