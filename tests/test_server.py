import asyncio
import json

import cv2
import numpy as np
from fastapi.testclient import TestClient

from libclaim.settings import Settings
from libclaim.server import KIND_LIVE_FRAME, KIND_MIC, KIND_STILL, create_app, pack_binary

SWEEP_ID = "sweep_test"
MAX_MESSAGES = 30


class FakeBridge:
    def __init__(self, hooks, script):
        self.hooks = hooks
        self.script = script
        self.session = object()
        self.audio = []
        self.frames = []
        self.injected = []
        self.ended = False
        self.closed = asyncio.Event()

    async def run(self):
        for step in self.script:
            await step(self.hooks)
        await self.closed.wait()

    async def send_audio(self, data):
        self.audio.append(data)

    async def send_frame(self, data):
        self.frames.append(data)

    def end_sweep(self):
        self.ended = True

    async def inject(self, event, scheduling):
        self.injected.append(event)

    async def close(self):
        self.closed.set()


class Harness:
    def __init__(self, tmp_path, script=(), end_calls=None):
        self.bridges = []
        self.end_calls = [] if end_calls is None else end_calls

        def factory(hooks):
            bridge = FakeBridge(hooks, script)
            self.bridges.append(bridge)
            return bridge

        async def on_end(sweep, emit):
            self.end_calls.append(sweep.root)
            await emit({"type": "packet_ready", "json": "/runs/x/claim_packet.json", "report": "/runs/x/report.html"})

        self.runs = tmp_path / "runs"
        dist = tmp_path / "dist"
        dist.mkdir()
        (dist / "index.html").write_text("<html>ok</html>", encoding="utf-8")
        self.app = create_app(Settings(live_model="fake-live"), factory, on_end, runs_dir=self.runs, web_dist=dist)


def hello(ws):
    ws.send_text(json.dumps({"type": "hello", "sweep_id": SWEEP_ID, "t_ms": 1}))


def read_until(ws, kind):
    for _ in range(MAX_MESSAGES):
        message = ws.receive_json()
        if message["type"] == kind:
            return message
    raise AssertionError(f"no {kind} message")


def jpeg(img):
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def checkerboard(size=256, square=16):
    cells = (np.indices((size, size)) // square).sum(axis=0) % 2
    return (cells * 190 + 30).astype(np.uint8)


def still_frame(img, seq=1, rot_dps=0.0):
    header = {"seq": seq, "t_ms": 1000 + seq, "w": 256, "h": 256, "motion": {"rot_dps": rot_dps, "acc": 0.0}}
    return pack_binary(KIND_STILL, header, jpeg(img))


def test_hello_reports_session_states(tmp_path):
    h = Harness(tmp_path)
    with TestClient(h.app) as client, client.websocket_connect("/ws/sweep") as ws:
        hello(ws)
        first = ws.receive_json()
        assert first["type"] == "session" and first["state"] == "connecting"
        assert first["model"] == "fake-live" and first["sweep_id"] == SWEEP_ID
        assert first["live_fps"] > 0
        assert read_until(ws, "session")["state"] == "live"


def test_blurred_still_is_stored_and_alerts(tmp_path):
    h = Harness(tmp_path)
    blurred = cv2.GaussianBlur(checkerboard(), (0, 0), 5)
    with TestClient(h.app) as client, client.websocket_connect("/ws/sweep") as ws:
        hello(ws)
        ws.send_bytes(still_frame(blurred))
        alert = read_until(ws, "capture_alert")
        assert alert["code"] == "blur" and alert["severity"] and alert["say"]
        coverage = read_until(ws, "coverage")
        assert coverage["regions"][0]["state"] == "partial"
    assert len(list((h.runs / SWEEP_ID / "frames").glob("*.jpg"))) == 1
    assert h.bridges[0].injected[0]["alert"] == "blur"


def test_sharp_still_marks_region_sharp_without_alert(tmp_path):
    h = Harness(tmp_path)
    with TestClient(h.app) as client, client.websocket_connect("/ws/sweep") as ws:
        hello(ws)
        ws.send_bytes(still_frame(checkerboard()))
        assert read_until(ws, "coverage")["regions"][0]["state"] == "sharp"
    assert h.bridges[0].injected == []


def test_mic_and_live_frames_reach_bridge(tmp_path):
    h = Harness(tmp_path)
    pcm = bytes(1280)
    with TestClient(h.app) as client, client.websocket_connect("/ws/sweep") as ws:
        hello(ws)
        ws.send_bytes(pack_binary(KIND_MIC, {"seq": 1, "t_ms": 1}, pcm))
        ws.send_bytes(pack_binary(KIND_LIVE_FRAME, {"seq": 1, "t_ms": 1, "w": 4, "h": 4}, b"jpegbytes"))
        ws.send_text(json.dumps({"type": "control", "action": "end_sweep"}))
        read_until(ws, "session")
        read_until(ws, "packet_ready")
    assert h.bridges[0].audio == [pcm]
    assert h.bridges[0].frames == [b"jpegbytes"]


def test_muted_mic_is_dropped(tmp_path):
    h = Harness(tmp_path)
    with TestClient(h.app) as client, client.websocket_connect("/ws/sweep") as ws:
        hello(ws)
        ws.send_text(json.dumps({"type": "control", "action": "mute"}))
        ws.send_bytes(pack_binary(KIND_MIC, {"seq": 1, "t_ms": 1}, bytes(1280)))
        ws.send_text(json.dumps({"type": "control", "action": "unmute"}))
        ws.send_bytes(pack_binary(KIND_MIC, {"seq": 2, "t_ms": 2}, b"after"))
        ws.send_text(json.dumps({"type": "control", "action": "end_sweep"}))
        read_until(ws, "packet_ready")
    assert h.bridges[0].audio == [b"after"]


def test_bridge_events_are_forwarded(tmp_path):
    async def speak(hooks):
        await hooks.on_audio(b"\x01\x02")
        await hooks.on_transcript("model", "hello there")
        await hooks.on_interrupted()
        await hooks.emit({"type": "inventory.upsert", "item": {"id": "b_1"}})

    h = Harness(tmp_path, script=[speak])
    with TestClient(h.app) as client, client.websocket_connect("/ws/sweep") as ws:
        hello(ws)
        messages = []
        audio = None
        while len(messages) < 3:
            frame = ws.receive()
            if frame.get("bytes") is not None:
                audio = frame["bytes"]
            else:
                message = json.loads(frame["text"])
                if message["type"] in ("transcript", "interrupted", "inventory.upsert"):
                    messages.append(message)
    assert [m["type"] for m in messages] == ["transcript", "interrupted", "inventory.upsert"]
    assert messages[0]["role"] == "agent" and messages[0]["text"] == "hello there"
    assert audio[0] == 0x81 and audio.endswith(b"\x01\x02")


def test_end_sweep_control_runs_hook(tmp_path):
    h = Harness(tmp_path)
    with TestClient(h.app) as client, client.websocket_connect("/ws/sweep") as ws:
        hello(ws)
        ws.send_text(json.dumps({"type": "control", "action": "end_sweep"}))
        states = []
        while True:
            message = ws.receive_json()
            if message["type"] == "session":
                states.append(message["state"])
            if "summary" in states:
                break
    assert "processing" in states and states.index("processing") < states.index("summary")
    assert h.end_calls == [h.runs / SWEEP_ID]
    assert h.bridges[0].ended


def test_agent_end_sweep_tool_runs_hook_once(tmp_path):
    async def agent_ends(hooks):
        hooks.on_end_sweep()
        hooks.on_end_sweep()

    h = Harness(tmp_path, script=[agent_ends])
    with TestClient(h.app) as client, client.websocket_connect("/ws/sweep") as ws:
        hello(ws)
        read_until(ws, "packet_ready")
        read_until(ws, "session")
    assert len(h.end_calls) == 1


def test_bad_input_does_not_crash_socket(tmp_path):
    h = Harness(tmp_path)
    bad = [
        bytes([0x7F]) + (2).to_bytes(4, "big") + b"{}",
        bytes([KIND_MIC]) + (0xFFFFFFFF).to_bytes(4, "big") + b"{}",
        bytes([KIND_MIC]) + (50).to_bytes(4, "big") + b"{}",
        bytes([KIND_MIC]),
        b"",
        pack_binary(KIND_STILL, {"seq": "x"}, b"notajpeg"),
        pack_binary(KIND_STILL, {"seq": 1, "t_ms": 1, "w": 1, "h": 1}, b"notajpeg"),
        bytes([KIND_MIC]) + (3).to_bytes(4, "big") + b"[1]",
        bytes([KIND_MIC]) + (3).to_bytes(4, "big") + b"zzz",
    ]
    with TestClient(h.app) as client, client.websocket_connect("/ws/sweep") as ws:
        ws.send_bytes(pack_binary(KIND_MIC, {"seq": 0, "t_ms": 0}, b"beforehello"))
        hello(ws)
        for frame in bad:
            ws.send_bytes(frame)
        for text in ("not json", "[]", '{"type":"nope"}', '{"type":"control"}', '{"type":"hello","sweep_id":"../x"}'):
            ws.send_text(text)
        ws.send_bytes(pack_binary(KIND_MIC, {"seq": 1, "t_ms": 1}, b"good"))
        ws.send_text(json.dumps({"type": "control", "action": "end_sweep"}))
        read_until(ws, "packet_ready")
    assert h.bridges[0].audio == [b"good"]


def test_disconnect_closes_bridge(tmp_path):
    h = Harness(tmp_path)
    with TestClient(h.app) as client:
        with client.websocket_connect("/ws/sweep") as ws:
            hello(ws)
            read_until(ws, "session")
        client.portal.call(asyncio.sleep, 0.2)
    assert h.bridges[0].closed.is_set()


def test_disconnect_without_hello_is_clean(tmp_path):
    h = Harness(tmp_path)
    with TestClient(h.app) as client:
        with client.websocket_connect("/ws/sweep"):
            pass
    assert h.bridges == []


def test_serves_web_and_runs(tmp_path):
    h = Harness(tmp_path)
    (h.runs / "abc").mkdir(parents=True)
    (h.runs / "abc" / "report.html").write_text("report", encoding="utf-8")
    with TestClient(h.app) as client:
        index = client.get("/")
        assert index.status_code == 200 and "ok" in index.text
        assert client.get("/runs/abc/report.html").text == "report"
        assert client.get("/runs/abc/missing").status_code == 404
