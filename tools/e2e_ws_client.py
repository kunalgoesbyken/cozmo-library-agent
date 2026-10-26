import argparse
import asyncio
import io
import json
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

from PIL import Image, ImageFilter
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from libclaim.server import KIND_AGENT_AUDIO, KIND_LIVE_FRAME, KIND_MIC, KIND_STILL, pack_binary, parse_binary
from libclaim.settings import ROOT

TOKEN_ENV = "E2E_TOKEN"
SAMPLE_RATE = 16000
MIC_FRAME_SECONDS = 0.04
MIC_FRAME_BYTES = int(SAMPLE_RATE * MIC_FRAME_SECONDS) * 2
LIVE_LONG_SIDE = 768
STILL_WIDTH = 1280
STILL_FPS = 2.0
LIVE_FPS = 1.0
PAN_STEPS = 14
PAN_WINDOW = 0.62
BLUR_RADIUS = 14
BLURRED_STILLS = 4
SHELF_DIR = ROOT / "_scratch" / "shelf"
SHELF_PHOTOS = ["german.jpg", "library.jpg", "blyton.jpg"]
AGENT_SILENCE_SECONDS = 2.0
SPEECH = {
    "locale": "Hello, I am in India, the currency is rupees, I am starting the sweep now.",
    "statement": "That book is a first edition.",
    "long": "Please explain the whole sweep process to me in detail, step by step.",
    "skip": "Skip this shelf, those are not mine.",
    "done": "I am done.",
}
USER_AGENT = "e2e-ws-client"
OVERSIZED_BYTES = 12_000_000


def synthesize(text: str) -> bytes:
    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / "speech.wav"
        script = (
            "Add-Type -AssemblyName System.Speech;"
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
            "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo("
            f"{SAMPLE_RATE}, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono);"
            f"$s.SetOutputToWaveFile('{target}', $f);"
            "$s.Speak($env:E2E_TEXT);"
            "$s.Dispose()"
        )
        subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True, env={**os.environ, "E2E_TEXT": text})
        with wave.open(str(target), "rb") as handle:
            return handle.readframes(handle.getnframes())


def jpeg_bytes(image: Image.Image, quality: int) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def pan_crops() -> list[Image.Image]:
    crops = []
    for name in SHELF_PHOTOS:
        image = Image.open(SHELF_DIR / name).convert("RGB")
        width, height = image.size
        window = int(width * PAN_WINDOW)
        for step in range(PAN_STEPS):
            left = (width - window) * step // (PAN_STEPS - 1)
            crops.append(image.crop((left, 0, left + window, height)))
    return crops


def resized(image: Image.Image, width: int | None = None, long_side: int | None = None) -> Image.Image:
    copy = image.copy()
    if width:
        copy = copy.resize((width, round(copy.height * width / copy.width)))
    else:
        copy.thumbnail((long_side, long_side))
    return copy


class Transcript:
    def __init__(self, path: Path):
        self.path = path
        self.started = time.monotonic()
        self.counts: dict[str, int] = {}
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("w", encoding="utf-8")

    def now(self) -> float:
        return round(time.monotonic() - self.started, 2)

    def log(self, kind: str, **detail) -> None:
        self.counts[kind] = self.counts.get(kind, 0) + 1
        self.handle.write(json.dumps({"t": self.now(), "kind": kind, **detail}) + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


class Phone:
    def __init__(self, url: str, token: str, origin: str | None, transcript: Transcript, runs_dir: Path):
        self.url = url
        self.token = token
        self.origin = origin
        self.transcript = transcript
        self.runs_dir = runs_dir
        self.ws = None
        self.sweep_id = ""
        self.mic = bytearray()
        self.last_audio_at = 0.0
        self.audio_frames = 0
        self.audio_bytes = 0
        self.interrupted_at: list[float] = []
        self.audio_times: list[float] = []
        self.states: list[str] = []
        self.packet: dict | None = None
        self.packet_event = asyncio.Event()
        self.processing_event = asyncio.Event()
        self.closed_event = asyncio.Event()
        self.close_code: int | None = None
        self.agent_text: list[tuple[float, str]] = []
        self.user_text: list[tuple[float, str]] = []
        self.origin_ms = int(time.time() * 1000)
        self.blur_done = False
        self.fast_done = False
        self.seq = {KIND_MIC: 0, KIND_LIVE_FRAME: 0, KIND_STILL: 0}

    def t_ms(self) -> int:
        return int(time.time() * 1000)

    async def open(self) -> None:
        headers = {"Origin": self.origin} if self.origin else None
        self.ws = await connect(self.url, additional_headers=headers, max_size=None, open_timeout=20)

    async def hello(self, token: str | None = None) -> None:
        await self.ws.send(
            json.dumps({"type": "hello", "token": self.token if token is None else token, "t_ms": self.t_ms(), "device": {"ua": USER_AGENT}})
        )

    def next_seq(self, kind: int) -> int:
        self.seq[kind] += 1
        return self.seq[kind]

    async def send_mic_forever(self) -> None:
        next_tick = time.monotonic()
        silence = bytes(MIC_FRAME_BYTES)
        while True:
            if self.mic:
                chunk = bytes(self.mic[:MIC_FRAME_BYTES]).ljust(MIC_FRAME_BYTES, b"\x00")
                del self.mic[:MIC_FRAME_BYTES]
            else:
                chunk = silence
            await self.ws.send(pack_binary(KIND_MIC, {"seq": self.next_seq(KIND_MIC), "t_ms": self.t_ms()}, chunk))
            next_tick += MIC_FRAME_SECONDS
            await asyncio.sleep(max(0.0, next_tick - time.monotonic()))

    async def send_images_forever(self, crops: list[Image.Image], blur_at: float, fast_at: float) -> None:
        started = time.monotonic()
        index = 0
        live_every = round(STILL_FPS / LIVE_FPS)
        blurred_left = 0
        while True:
            elapsed = time.monotonic() - started
            crop = crops[index % len(crops)]
            motion = {"rot_dps": 4.0, "acc": 0.1}
            if elapsed >= blur_at and blurred_left == 0 and not self.blur_done:
                blurred_left = BLURRED_STILLS
                self.blur_done = True
                self.transcript.log("blurred_stills_start")
            if elapsed >= fast_at and not self.fast_done:
                motion = {"rot_dps": 240.0, "acc": 3.0}
                self.fast_done = True
                self.transcript.log("fast_motion_still")
            still = resized(crop, width=STILL_WIDTH)
            if blurred_left:
                still = still.filter(ImageFilter.GaussianBlur(BLUR_RADIUS))
                blurred_left -= 1
            data = jpeg_bytes(still, 85)
            header = {"seq": self.next_seq(KIND_STILL), "t_ms": self.t_ms(), "w": still.width, "h": still.height, "motion": motion, "orient": "portrait"}
            await self.ws.send(pack_binary(KIND_STILL, header, data))
            if index % live_every == 0:
                live = resized(crop, long_side=LIVE_LONG_SIDE)
                live_header = {"seq": self.next_seq(KIND_LIVE_FRAME), "t_ms": self.t_ms(), "w": live.width, "h": live.height}
                await self.ws.send(pack_binary(KIND_LIVE_FRAME, live_header, jpeg_bytes(live, 75)))
            index += 1
            await asyncio.sleep(1 / STILL_FPS)

    async def receive_forever(self) -> None:
        try:
            async for message in self.ws:
                if isinstance(message, bytes):
                    kind, header, payload = parse_binary(message, 4096)
                    if kind == KIND_AGENT_AUDIO:
                        self.audio_frames += 1
                        self.audio_bytes += len(payload)
                        self.last_audio_at = time.monotonic()
                        self.audio_times.append(self.transcript.now())
                        if self.audio_frames <= 3 or self.audio_frames % 50 == 0:
                            self.transcript.log("agent_audio", seq=header.get("seq"), bytes=len(payload), total_frames=self.audio_frames)
                    continue
                self.handle_text(json.loads(message))
        except ConnectionClosed as closed:
            self.close_code = closed.rcvd.code if closed.rcvd else None
        finally:
            self.closed_event.set()

    def handle_text(self, event: dict) -> None:
        kind = event.get("type")
        self.transcript.log("recv_" + str(kind), **{k: v for k, v in event.items() if k not in ("type", "token")})
        if kind == "session":
            self.states.append(event["state"])
            self.sweep_id = event.get("sweep_id") or self.sweep_id
            if event["state"] == "processing":
                self.processing_event.set()
        elif kind == "interrupted":
            self.interrupted_at.append(self.transcript.now())
        elif kind == "transcript":
            target = self.agent_text if event["role"] == "agent" else self.user_text
            target.append((self.transcript.now(), event["text"]))
        elif kind == "packet_ready":
            self.packet = event
            self.packet_event.set()

    def speak(self, key: str, pcm: dict[str, bytes]) -> None:
        self.transcript.log("user_speech", key=key, text=SPEECH[key], seconds=round(len(pcm[key]) / 2 / SAMPLE_RATE, 2))
        self.mic.extend(pcm[key])

    def agent_idle(self) -> bool:
        return self.last_audio_at > 0 and time.monotonic() - self.last_audio_at >= AGENT_SILENCE_SECONDS and not self.mic

    async def wait_until(self, predicate, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            if self.closed_event.is_set():
                return False
            await asyncio.sleep(0.2)
        return False

    async def close(self) -> None:
        if self.ws is not None:
            await self.ws.close()


def http_status(url: str) -> int:
    request = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def stage_summary(sweep_dir: Path) -> dict:
    stages = sweep_dir / "stages"
    found = {}
    for name in ("locale", "statements", "exclusions", "device", "stills"):
        path = stages / f"{name}.json"
        found[name] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    if found["stills"]:
        found["stills"] = {"count": len(found["stills"].get("stills", []))}
    return found


async def sweep(args, transcript: Transcript) -> dict:
    token = os.environ[TOKEN_ENV]
    base = f"http://{args.host}:{args.port}"
    phone = Phone(f"ws://{args.host}:{args.port}/ws/sweep", token, base, transcript, Path(args.runs_dir))
    pcm = {key: synthesize(text) for key, text in SPEECH.items()}
    crops = pan_crops()
    await phone.open()
    await phone.hello()
    tasks = [
        asyncio.create_task(phone.receive_forever()),
        asyncio.create_task(phone.send_mic_forever()),
        asyncio.create_task(phone.send_images_forever(crops, args.blur_at, args.fast_at)),
    ]
    results: dict = {}
    started = time.monotonic()
    try:
        results["greeted"] = await phone.wait_until(lambda: phone.audio_frames > 0, 40)
        results["greeting_latency_s"] = transcript.now() if results["greeted"] else None
        await phone.wait_until(phone.agent_idle, 40)
        phone.speak("locale", pcm)
        await phone.wait_until(lambda: time.monotonic() - started >= args.statement_at and phone.agent_idle(), args.statement_at + 30)
        phone.speak("statement", pcm)
        await phone.wait_until(lambda: time.monotonic() - started >= args.long_at and phone.agent_idle(), args.long_at + 40)
        before = phone.audio_frames
        phone.speak("long", pcm)
        await phone.wait_until(lambda: phone.audio_frames - before >= args.talking_frames, 30)
        results["agent_talking_before_skip"] = phone.audio_frames - before >= args.talking_frames
        frames_before_skip = phone.audio_frames
        skip_at = transcript.now()
        phone.speak("skip", pcm)
        results["interrupted"] = await phone.wait_until(lambda: any(t >= skip_at for t in phone.interrupted_at), 15)
        await asyncio.sleep(3)
        stop = max(phone.interrupted_at, default=skip_at)
        results["audio_frames_in_1s_after_interrupt"] = sum(1 for t in phone.audio_times if stop < t <= stop + 1.0)
        results["audio_frames_in_1s_before_interrupt"] = sum(1 for t in phone.audio_times if stop - 1.0 <= t < stop)
        results["frames_after_skip_sent"] = phone.audio_frames - frames_before_skip
        await phone.wait_until(lambda: time.monotonic() - started >= args.duration, args.duration + 30)
        await phone.wait_until(phone.agent_idle, 40)
        phone.speak("done", pcm)
        results["agent_ended_sweep"] = await phone.wait_until(phone.processing_event.is_set, args.done_wait)
        if not results["agent_ended_sweep"]:
            transcript.log("control_end_sweep")
            await phone.ws.send(json.dumps({"type": "control", "action": "end_sweep"}))
            await phone.wait_until(phone.processing_event.is_set, 15)
        results["packet_ready"] = await phone.wait_until(phone.packet_event.is_set, args.pipeline_wait)
        ready_at = transcript.now()
        if phone.packet:
            token_query = f"?t={token}"
            urls = {name: base + phone.packet[name] if phone.packet[name].startswith("/") else phone.packet[name] for name in ("json", "report")}
            results["url_with_token"] = {name: http_status(url + token_query) for name, url in urls.items()}
            results["url_without_token"] = {name: http_status(url) for name, url in urls.items()}
            frames_before_readback = phone.audio_frames
            results["readback_spoken"] = await phone.wait_until(lambda: phone.audio_frames - frames_before_readback >= 5, args.readback_wait)
            results["readback_audio_frames"] = phone.audio_frames - frames_before_readback
        results["ready_at"] = ready_at
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await phone.close()
    results["elapsed_s"] = round(time.monotonic() - started, 1)
    results["session_states"] = phone.states
    results["event_counts"] = dict(transcript.counts)
    results["agent_audio_frames"] = phone.audio_frames
    results["agent_text"] = " ".join(text for _, text in phone.agent_text)[:600]
    results["user_text"] = " ".join(text for _, text in phone.user_text)[:600]
    results["close_code"] = phone.close_code
    if phone.sweep_id:
        results["stages"] = stage_summary(Path(args.runs_dir) / phone.sweep_id)
        results["packet_json_exists"] = (Path(args.runs_dir) / phone.sweep_id / "claim_packet.json").is_file()
        results["report_html_exists"] = (Path(args.runs_dir) / phone.sweep_id / "report.html").is_file()
    return results


async def negative(args, transcript: Transcript) -> dict:
    token = os.environ[TOKEN_ENV]
    base = f"http://{args.host}:{args.port}"
    url = f"ws://{args.host}:{args.port}/ws/sweep"
    results: dict = {}

    async def attempt(name: str, origin: str | None, hello_token: str | None, payload=None) -> Phone:
        phone = Phone(url, token, origin, transcript, Path(args.runs_dir))
        outcome: dict = {}
        try:
            await phone.open()
            receiver = asyncio.create_task(phone.receive_forever())
            if payload is not None:
                await phone.ws.send(payload)
            else:
                await phone.hello(hello_token)
            await asyncio.wait_for(phone.closed_event.wait(), 15)
            await receiver
            outcome = {"close_code": phone.close_code, "states": phone.states}
        except ConnectionClosed as closed:
            outcome = {"close_code": closed.rcvd.code if closed.rcvd else None, "during": "handshake_or_send"}
        except Exception as error:
            outcome = {"error": type(error).__name__, "detail": str(error)[:200]}
        results[name] = outcome
        transcript.log("negative", name=name, **outcome)
        return phone

    await attempt("wrong_token", base, "x" * 40)
    await attempt("empty_token", base, "")
    await attempt("wrong_origin", "https://evil.example.com", None)
    await attempt("oversized_preauth_text", base, None, payload=json.dumps({"type": "hello", "token": "y" * 5000}))
    await attempt("binary_before_hello", base, None, payload=bytes(100))
    await attempt("garbage_json", base, None, payload="not json")
    first = Phone(url, token, base, transcript, Path(args.runs_dir))
    await first.open()
    first_receiver = asyncio.create_task(first.receive_forever())
    await first.hello()
    await first.wait_until(lambda: "live" in first.states or "connecting" in first.states, 15)
    second = await attempt("second_concurrent_connection", base, None)
    results["first_connection_survived_second"] = not first.closed_event.is_set()
    await first.ws.send(pack_binary(KIND_MIC, {"seq": 1, "t_ms": first.t_ms()}, bytes(OVERSIZED_BYTES)))
    await first.wait_until(first.closed_event.is_set, 15)
    results["oversized_binary_close_code"] = first.close_code
    await first_receiver
    healthy = Phone(url, token, base, transcript, Path(args.runs_dir))
    await healthy.open()
    receiver = asyncio.create_task(healthy.receive_forever())
    await healthy.hello()
    await healthy.wait_until(lambda: bool(healthy.states), 15)
    results["server_accepts_after_negatives"] = bool(healthy.states) and healthy.states[0] != "closed"
    await healthy.close()
    await receiver
    return results


async def abrupt(args, transcript: Transcript) -> dict:
    token = os.environ[TOKEN_ENV]
    base = f"http://{args.host}:{args.port}"
    phone = Phone(f"ws://{args.host}:{args.port}/ws/sweep", token, base, transcript, Path(args.runs_dir))
    crops = pan_crops()
    await phone.open()
    await phone.hello()
    receiver = asyncio.create_task(phone.receive_forever())
    sender = asyncio.create_task(phone.send_images_forever(crops, 10_000, 10_000))
    await phone.wait_until(lambda: "live" in phone.states, 30)
    await asyncio.sleep(args.abrupt_seconds)
    sweep_id = phone.sweep_id
    sender.cancel()
    phone.ws.transport.abort()
    await asyncio.gather(sender, receiver, return_exceptions=True)
    transcript.log("abrupt_disconnect", sweep_id=sweep_id)
    packet = Path(args.runs_dir) / sweep_id / "claim_packet.json"
    deadline = time.monotonic() + args.pipeline_wait
    while time.monotonic() < deadline and not packet.is_file():
        await asyncio.sleep(2)
    return {
        "sweep_id": sweep_id,
        "packet_written_after_abort": packet.is_file(),
        "report_written_after_abort": (packet.parent / "report.html").is_file(),
        "waited_s": round(args.pipeline_wait - max(0.0, deadline - time.monotonic()), 1),
    }


SCENARIOS = {"sweep": sweep, "negative": negative, "abrupt": abrupt}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", choices=sorted(SCENARIOS))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--runs-dir", default=str(ROOT / "runs"))
    parser.add_argument("--transcript", default=str(ROOT / "_scratch" / "e2e" / "transcript.jsonl"))
    parser.add_argument("--duration", type=float, default=100.0)
    parser.add_argument("--statement-at", type=float, default=30.0)
    parser.add_argument("--long-at", type=float, default=50.0)
    parser.add_argument("--blur-at", type=float, default=75.0)
    parser.add_argument("--fast-at", type=float, default=85.0)
    parser.add_argument("--talking-frames", type=int, default=15)
    parser.add_argument("--done-wait", type=float, default=20.0)
    parser.add_argument("--pipeline-wait", type=float, default=420.0)
    parser.add_argument("--readback-wait", type=float, default=40.0)
    parser.add_argument("--abrupt-seconds", type=float, default=12.0)
    args = parser.parse_args()
    transcript = Transcript(Path(args.transcript))
    try:
        results = asyncio.run(SCENARIOS[args.scenario](args, transcript))
    finally:
        transcript.close()
    print(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    main()
