import asyncio
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from libclaim.server import KIND_LIVE_FRAME, KIND_MIC, KIND_STILL, create_app, load_server_config, pack_binary
from libclaim.settings import Settings

TOKEN = "secret-token-0123456789abcdef"
MAX_MESSAGES = 30
SECRET_SPEND = "secret spend"


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
    def __init__(self, tmp_path, script=(), end_delay=0.0, public_origin="", **limits):
        self.bridges = []
        self.end_calls = []
        self.fail_factory = False

        def factory(hooks):
            if self.fail_factory:
                raise RuntimeError("no bridge")
            bridge = FakeBridge(hooks, script)
            self.bridges.append(bridge)
            return bridge

        async def on_end(sweep, emit):
            await asyncio.sleep(end_delay)
            self.end_calls.append(sweep.root)
            await emit({"type": "packet_ready", "json": "/runs/x/claim_packet.json", "report": "/runs/x/report.html"})

        self.runs = tmp_path / "runs"
        dist = tmp_path / "dist"
        dist.mkdir()
        (dist / "index.html").write_text("<html>ok</html>", encoding="utf-8")
        config = load_server_config().model_copy(update=limits)
        self.app = create_app(
            Settings(live_model="fake-live", access_token=TOKEN, public_origin=public_origin),
            factory,
            on_end,
            runs_dir=self.runs,
            web_dist=dist,
            config=config,
        )


def hello(ws, token=TOKEN, **extra):
    ws.send_text(json.dumps({"type": "hello", "token": token, "t_ms": 1, **extra}))
    return ws.receive_json()


def end_sweep(ws):
    ws.send_text(json.dumps({"type": "control", "action": "end_sweep"}))


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
        first = hello(ws)
        assert first["type"] == "session" and first["state"] == "connecting"
        assert first["model"] == "fake-live" and len(first["sweep_id"]) == 32
        assert first["live_fps"] > 0
        assert read_until(ws, "session")["state"] == "live"


def test_blurred_still_is_stored_and_alerts(tmp_path):
    h = Harness(tmp_path)
    blurred = cv2.GaussianBlur(checkerboard(), (0, 0), 5)
    with TestClient(h.app) as client, client.websocket_connect("/ws/sweep") as ws:
        sweep_id = hello(ws)["sweep_id"]
        ws.send_bytes(still_frame(blurred))
        alert = read_until(ws, "capture_alert")
        assert alert["code"] == "blur" and alert["severity"] and alert["say"]
        coverage = read_until(ws, "coverage")
        assert coverage["regions"][0]["state"] == "partial"
    assert len(list((h.runs / sweep_id / "frames").glob("*.jpg"))) == 1
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
    live = jpeg(checkerboard())
    with TestClient(h.app) as client, client.websocket_connect("/ws/sweep") as ws:
        hello(ws)
        ws.send_bytes(pack_binary(KIND_MIC, {"seq": 1, "t_ms": 1}, pcm))
        ws.send_bytes(pack_binary(KIND_LIVE_FRAME, {"seq": 1, "t_ms": 1, "w": 256, "h": 256}, live))
        end_sweep(ws)
        read_until(ws, "packet_ready")
    assert h.bridges[0].audio == [pcm]
    assert h.bridges[0].frames == [live]


def test_muted_mic_is_dropped(tmp_path):
    h = Harness(tmp_path)
    with TestClient(h.app) as client, client.websocket_connect("/ws/sweep") as ws:
        hello(ws)
        ws.send_text(json.dumps({"type": "control", "action": "mute"}))
        ws.send_bytes(pack_binary(KIND_MIC, {"seq": 1, "t_ms": 1}, bytes(1280)))
        ws.send_text(json.dumps({"type": "control", "action": "unmute"}))
        ws.send_bytes(pack_binary(KIND_MIC, {"seq": 2, "t_ms": 2}, b"after"))
        end_sweep(ws)
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
        sweep_id = hello(ws)["sweep_id"]
        end_sweep(ws)
        states = []
        while "summary" not in states:
            message = ws.receive_json()
            if message["type"] == "session":
                states.append(message["state"])
    assert "processing" in states and states.index("processing") < states.index("summary")
    assert h.end_calls == [h.runs / sweep_id]
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
        pack_binary(KIND_LIVE_FRAME, {"seq": 1}, b"notajpeg"),
        bytes([KIND_MIC]) + (3).to_bytes(4, "big") + b"[1]",
        bytes([KIND_MIC]) + (3).to_bytes(4, "big") + b"zzz",
    ]
    with TestClient(h.app) as client, client.websocket_connect("/ws/sweep") as ws:
        hello(ws)
        for frame in bad:
            ws.send_bytes(frame)
        for text in ("not json", "[]", '{"type":"nope"}', '{"type":"control"}', '{"type":"hello","token":5}'):
            ws.send_text(text)
        ws.send_bytes(pack_binary(KIND_MIC, {"seq": 1, "t_ms": 1}, b"good"))
        end_sweep(ws)
        read_until(ws, "packet_ready")
    assert h.bridges[0].audio == [b"good"]


def test_oversized_payloads_and_header_are_dropped(tmp_path):
    h = Harness(tmp_path, max_mic_bytes=8, max_frame_bytes=64, max_header_bytes=30)
    with TestClient(h.app) as client, client.websocket_connect("/ws/sweep") as ws:
        hello(ws)
        ws.send_bytes(pack_binary(KIND_MIC, {"seq": 1, "t_ms": 1}, bytes(9)))
        ws.send_bytes(pack_binary(KIND_LIVE_FRAME, {"seq": 1}, bytes(65)))
        ws.send_bytes(pack_binary(KIND_MIC, {"pad": "x" * 40}, b"hdr"))
        ws.send_bytes(pack_binary(KIND_MIC, {"seq": 1}, b"ok"))
        end_sweep(ws)
        read_until(ws, "packet_ready")
    assert h.bridges[0].audio == [b"ok"]
    assert h.bridges[0].frames == []


def test_disconnect_closes_bridge(tmp_path):
    h = Harness(tmp_path)
    with TestClient(h.app) as client:
        with client.websocket_connect("/ws/sweep") as ws:
            hello(ws)
        client.portal.call(asyncio.sleep, 0.2)
    assert h.bridges[0].closed.is_set()


def test_disconnect_without_hello_is_clean(tmp_path):
    h = Harness(tmp_path)
    with TestClient(h.app) as client:
        with client.websocket_connect("/ws/sweep"):
            pass
    assert h.bridges == []


def test_disconnect_after_end_sweep_still_completes_hook(tmp_path):
    h = Harness(tmp_path, end_delay=0.3)
    with TestClient(h.app) as client:
        with client.websocket_connect("/ws/sweep") as ws:
            hello(ws)
            end_sweep(ws)
            read_until(ws, "session")
        assert h.end_calls == []
        client.portal.call(asyncio.sleep, 0.8)
    assert len(h.end_calls) == 1


def test_serves_web_index(tmp_path):
    h = Harness(tmp_path)
    with TestClient(h.app) as client:
        index = client.get("/")
        assert index.status_code == 200 and "ok" in index.text


def make_sweep(h, files):
    sweep = h.runs / ("a" * 32)
    for name, text in files.items():
        (sweep / name).parent.mkdir(parents=True, exist_ok=True)
        (sweep / name).write_text(text, encoding="utf-8")
    (h.runs / "spend.jsonl").write_text(SECRET_SPEND, encoding="utf-8")
    (h.runs / "_quota.json").write_text("quota", encoding="utf-8")
    return sweep.name


def test_runs_serves_only_scoped_files_with_token(tmp_path):
    h = Harness(tmp_path)
    sid = make_sweep(h, {"report.html": "report", "frames/00001_1.jpg": "jpg", "stages/_ledger.json": "ledger"})
    with TestClient(h.app) as client:
        assert client.get(f"/runs/{sid}/report.html").status_code == 401
        assert client.get(f"/runs/{sid}/report.html?t=wrong").status_code == 401
        ok = client.get(f"/runs/{sid}/report.html?t={TOKEN}")
        assert ok.status_code == 200 and ok.text == "report"
        assert client.get(f"/runs/{sid}/frames/00001_1.jpg").text == "jpg"
        assert client.get(f"/runs/{sid}/stages/_ledger.json?t={TOKEN}").status_code == 404
        assert client.get(f"/runs/{sid}/missing.html?t={TOKEN}").status_code == 404


@pytest.mark.parametrize(
    "path",
    [
        "/runs/spend.jsonl",
        "/runs/_quota.json",
        "/runs/{sid}/../spend.jsonl",
        "/runs/{sid}/frames/../../spend.jsonl",
        "/runs/{sid}/frames/%2e%2e/%2e%2e/spend.jsonl",
        "/runs/{sid}/frames/../stages/_ledger.json",
        "/runs/{sid}/frames/%2e%2e/stages/_ledger.json",
        "/runs/{sid}/frames/..%2fstages/_ledger.json",
        "/runs/%2e%2e/spend.jsonl",
        "/runs/../spend.jsonl",
        "/runs/..%2f/spend.jsonl",
    ],
)
def test_runs_traversal_and_global_files_not_served(tmp_path, path):
    h = Harness(tmp_path)
    sid = make_sweep(h, {"report.html": "report", "frames/a.jpg": "jpg", "stages/_ledger.json": "ledger"})
    with TestClient(h.app) as client:
        response = client.get(path.format(sid=sid) + f"?t={TOKEN}")
        assert response.status_code == 404
        assert SECRET_SPEND not in response.text and response.text != "quota" and response.text != "ledger"


def test_ws_rejects_bad_and_missing_token(tmp_path):
    h = Harness(tmp_path)
    with TestClient(h.app) as client:
        for token in ("", "wrong"):
            with client.websocket_connect("/ws/sweep") as ws:
                reply = hello(ws, token=token)
                assert reply["state"] == "closed" and reply["reason"] == "unauthorized"
                with pytest.raises(WebSocketDisconnect) as closed:
                    ws.receive_json()
                assert closed.value.code == 1008
    assert h.bridges == []


@pytest.mark.parametrize(
    "headers",
    [
        {"origin": "https://evil.example"},
        {"origin": "https://evil-trycloudflare.com"},
        {"origin": "https://x.trycloudflare.com.evil.example"},
        {"origin": "https://evil.example", "x-forwarded-host": "other.example"},
    ],
)
def test_ws_rejects_hostile_origin(tmp_path, headers):
    h = Harness(tmp_path)
    with TestClient(h.app) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws/sweep", headers=headers):
                pass


@pytest.mark.parametrize(
    "headers",
    [
        {"origin": "http://testserver"},
        {"origin": "https://quiet-name-1.trycloudflare.com", "host": "localhost:8000"},
        {"origin": "https://my.tunnel.example", "host": "localhost:8000", "x-forwarded-host": "my.tunnel.example"},
        {},
    ],
)
def test_ws_allows_same_suffix_forwarded_and_non_browser_origins(tmp_path, headers):
    h = Harness(tmp_path, trust_forwarded_host=True)
    with TestClient(h.app) as client, client.websocket_connect("/ws/sweep", headers=headers) as ws:
        assert hello(ws)["state"] == "connecting"


def test_forwarded_host_is_untrusted_by_default(tmp_path):
    h = Harness(tmp_path)
    headers = {"origin": "https://my.tunnel.example", "host": "localhost:8000", "x-forwarded-host": "my.tunnel.example"}
    with TestClient(h.app) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws/sweep", headers=headers):
                pass


def test_public_origin_accepts_only_exact_origin(tmp_path):
    h = Harness(tmp_path, public_origin="https://abc.trycloudflare.com", trust_forwarded_host=True)
    with TestClient(h.app) as client:
        with client.websocket_connect("/ws/sweep", headers={"origin": "https://abc.trycloudflare.com"}) as ws:
            assert hello(ws)["state"] == "connecting"
        client.portal.call(asyncio.sleep, 0.2)
        for headers in (
            {"origin": "https://other.trycloudflare.com"},
            {"origin": "https://other.example", "x-forwarded-host": "other.example"},
        ):
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect("/ws/sweep", headers=headers):
                    pass
        with client.websocket_connect("/ws/sweep", headers={"origin": "http://testserver"}) as ws:
            assert hello(ws, token=TOKEN)["state"] == "connecting"


def test_short_access_token_is_replaced_by_random(tmp_path):
    h = Harness(tmp_path)
    short = create_app(Settings(access_token="short"), lambda hooks: None, runs_dir=h.runs, web_dist=tmp_path / "none")
    assert short.state.access_token != "short" and len(short.state.access_token) >= 24


def test_responses_carry_no_referrer_policy_and_secure_cookie(tmp_path):
    h = Harness(tmp_path)
    sid = make_sweep(h, {"report.html": "report"})
    with TestClient(h.app) as client:
        assert client.get("/").headers["referrer-policy"] == "no-referrer"
        plain = client.get(f"/runs/{sid}/report.html?t={TOKEN}")
        assert "secure" not in plain.headers["set-cookie"].lower()
        proxied = client.get(f"/runs/{sid}/report.html?t={TOKEN}", headers={"x-forwarded-proto": "https"})
        assert "secure" in proxied.headers["set-cookie"].lower()


def test_second_concurrent_session_is_refused_and_slot_is_freed(tmp_path):
    h = Harness(tmp_path)
    with TestClient(h.app) as client:
        with client.websocket_connect("/ws/sweep") as first:
            assert hello(first)["state"] == "connecting"
            with client.websocket_connect("/ws/sweep") as second:
                reply = hello(second)
                assert reply["state"] == "closed" and reply["reason"] == "busy"
        client.portal.call(asyncio.sleep, 0.2)
        with client.websocket_connect("/ws/sweep") as third:
            assert hello(third)["state"] == "connecting"


def test_hello_timeout_closes_unauthenticated_socket(tmp_path):
    h = Harness(tmp_path, hello_timeout_seconds=0.1)
    with TestClient(h.app) as client, client.websocket_connect("/ws/sweep") as ws:
        assert ws.receive_json()["reason"] == "hello_timeout"


def test_pre_auth_deadline_is_total_not_per_message(tmp_path):
    h = Harness(tmp_path, hello_timeout_seconds=0.3)
    with TestClient(h.app) as client, client.websocket_connect("/ws/sweep") as ws:
        assert ws.receive_json()["reason"] == "hello_timeout"


def test_non_hello_message_before_auth_closes_socket(tmp_path):
    h = Harness(tmp_path)
    junk = ['{"type":"control","action":"mute"}', "not json", "[]", "x" * 5000]
    with TestClient(h.app) as client:
        for text in junk:
            with client.websocket_connect("/ws/sweep") as ws:
                ws.send_text(text)
                assert ws.receive_json()["reason"] == "unauthorized"
        with client.websocket_connect("/ws/sweep") as ws:
            ws.send_bytes(pack_binary(KIND_MIC, {"seq": 1}, b"pcm"))
            assert ws.receive_json()["reason"] == "unauthorized"
    assert h.bridges == []


def test_unauthenticated_socket_cap(tmp_path):
    h = Harness(tmp_path, max_unauthenticated=1)
    with TestClient(h.app) as client:
        with client.websocket_connect("/ws/sweep"):
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect("/ws/sweep"):
                    pass
        client.portal.call(asyncio.sleep, 0.2)
        with client.websocket_connect("/ws/sweep") as ws:
            assert hello(ws)["state"] == "connecting"


def test_per_client_preauth_limit(tmp_path):
    h = Harness(tmp_path, max_unauthenticated=10, max_unauthenticated_per_client=1)
    with TestClient(h.app) as client:
        with client.websocket_connect("/ws/sweep"):
            with pytest.raises(WebSocketDisconnect):
                with client.websocket_connect("/ws/sweep"):
                    pass


def test_repeated_failed_connects_do_not_exhaust_pool(tmp_path):
    h = Harness(tmp_path, max_unauthenticated=1, max_unauthenticated_per_client=1)
    with TestClient(h.app) as client:
        for _ in range(8):
            with client.websocket_connect("/ws/sweep") as ws:
                assert hello(ws, token="wrong")["reason"] == "unauthorized"
            client.portal.call(asyncio.sleep, 0.05)
        with client.websocket_connect("/ws/sweep") as ws:
            assert hello(ws)["state"] == "connecting"


def test_preauth_slot_released_when_constructor_raises(tmp_path, monkeypatch):
    h = Harness(tmp_path, max_unauthenticated=1, max_unauthenticated_per_client=1)

    def boom(*args, **kwargs):
        raise RuntimeError("constructor failed")

    with TestClient(h.app, raise_server_exceptions=False) as client:
        monkeypatch.setattr("libclaim.server.Connection", boom)
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/sweep"):
                pass
        monkeypatch.undo()
        client.portal.call(asyncio.sleep, 0.1)
        with client.websocket_connect("/ws/sweep") as ws:
            assert hello(ws)["state"] == "connecting"


def test_preauth_slot_released_when_accept_raises(tmp_path, monkeypatch):
    h = Harness(tmp_path, max_unauthenticated=1, max_unauthenticated_per_client=1)

    async def boom(self, *args, **kwargs):
        raise RuntimeError("accept failed")

    with TestClient(h.app, raise_server_exceptions=False) as client:
        monkeypatch.setattr("starlette.websockets.WebSocket.accept", boom)
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/sweep"):
                pass
        monkeypatch.undo()
        client.portal.call(asyncio.sleep, 0.1)
        with client.websocket_connect("/ws/sweep") as ws:
            assert hello(ws)["state"] == "connecting"


def test_forged_sweep_id_is_ignored(tmp_path):
    h = Harness(tmp_path)
    with TestClient(h.app) as client, client.websocket_connect("/ws/sweep") as ws:
        sweep_id = hello(ws, sweep_id="../../evil")["sweep_id"]
    assert sweep_id != "../../evil" and len(sweep_id) == 32
    assert not (tmp_path / "evil").exists()


def test_bridge_factory_failure_reports_reason(tmp_path):
    h = Harness(tmp_path)
    h.fail_factory = True
    with TestClient(h.app) as client:
        with client.websocket_connect("/ws/sweep") as ws:
            reply = hello(ws)
            assert reply["state"] == "closed" and reply["reason"] == "bridge_unavailable"
        assert not any(h.runs.iterdir())
        client.portal.call(asyncio.sleep, 0.2)
        h.fail_factory = False
        with client.websocket_connect("/ws/sweep") as ws:
            assert hello(ws)["state"] == "connecting"


def test_oversized_dimensions_are_rejected(tmp_path):
    h = Harness(tmp_path, max_image_pixels=1000)
    with TestClient(h.app) as client, client.websocket_connect("/ws/sweep") as ws:
        sweep_id = hello(ws)["sweep_id"]
        ws.send_bytes(still_frame(checkerboard()))
        ws.send_bytes(pack_binary(KIND_LIVE_FRAME, {"seq": 1, "t_ms": 1, "w": 256, "h": 256}, jpeg(checkerboard())))
        end_sweep(ws)
        read_until(ws, "packet_ready")
    assert not (h.runs / sweep_id / "frames").exists()
    assert h.bridges[0].frames == []


def test_header_dimensions_must_match_image(tmp_path):
    h = Harness(tmp_path)
    header = {"seq": 1, "t_ms": 1, "w": 10, "h": 10}
    with TestClient(h.app) as client, client.websocket_connect("/ws/sweep") as ws:
        sweep_id = hello(ws)["sweep_id"]
        ws.send_bytes(pack_binary(KIND_STILL, header, jpeg(checkerboard())))
        end_sweep(ws)
        read_until(ws, "packet_ready")
    assert not (h.runs / sweep_id / "frames").exists()


def test_still_count_limit_sends_error(tmp_path):
    h = Harness(tmp_path, max_stills_per_sweep=1)
    with TestClient(h.app) as client, client.websocket_connect("/ws/sweep") as ws:
        sweep_id = hello(ws)["sweep_id"]
        ws.send_bytes(still_frame(checkerboard(), seq=1))
        ws.send_bytes(still_frame(checkerboard(), seq=2))
        assert read_until(ws, "error")["code"] == "limit"
    assert len(list((h.runs / sweep_id / "frames").glob("*.jpg"))) == 1


def test_byte_limit_sends_error(tmp_path):
    h = Harness(tmp_path, max_sweep_bytes=10)
    with TestClient(h.app) as client, client.websocket_connect("/ws/sweep") as ws:
        hello(ws)
        ws.send_bytes(still_frame(checkerboard()))
        assert read_until(ws, "error")["code"] == "limit"


def test_storage_failure_sends_error_and_keeps_socket(tmp_path, monkeypatch):
    def broken(self, header, jpeg):
        raise OSError("disk full")

    monkeypatch.setattr("libclaim.capture.frame_store.FrameStore.add", broken)
    h = Harness(tmp_path)
    with TestClient(h.app) as client, client.websocket_connect("/ws/sweep") as ws:
        hello(ws)
        ws.send_bytes(still_frame(checkerboard()))
        assert read_until(ws, "error")["code"] == "storage"
        ws.send_bytes(pack_binary(KIND_MIC, {"seq": 1, "t_ms": 1}, b"still-alive"))
        end_sweep(ws)
        read_until(ws, "packet_ready")
    assert h.bridges[0].audio == [b"still-alive"]
