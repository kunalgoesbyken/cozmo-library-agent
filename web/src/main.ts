import { AgentPlayback, MicCapture } from "./audio";
import { Camera, Loop, enableMotion, stillFps } from "./capture";
import { Ui } from "./ui";
import { KIND_LIVE_FRAME, KIND_MIC, KIND_STILL, SweepSocket, type ServerMessage } from "./ws";

const accessToken = new URLSearchParams(location.search).get("t") ?? "";
const ui = new Ui(accessToken);
const camera = new Camera(ui.video);
const mic = new MicCapture();
const playback = new AgentPlayback();

let playing = false;
let ending = false;

const liveLoop = new Loop(async () => {
  const shot = await camera.liveFrame();
  if (shot && !socket.congested) {
    socket.sendBinary(KIND_LIVE_FRAME, { w: shot.w, h: shot.h }, await shot.blob.arrayBuffer());
  }
});

const stillLoop = new Loop(async () => {
  if (socket.congested) {
    return;
  }
  const shot = await camera.still();
  if (shot) {
    socket.sendBinary(
      KIND_STILL,
      { w: shot.w, h: shot.h, motion: shot.motion, orient: shot.orient },
      await shot.blob.arrayBuffer(),
    );
  }
});

function onSession(message: ServerMessage): void {
  const state = String(message.state);
  ui.setConnection(state, state === "live" ? "ok" : state === "closed" ? "idle" : "warn");
  if (state === "connecting" && !liveLoop.running) {
    liveLoop.start(Number(message.live_fps) || 1);
    stillLoop.start(stillFps());
  }
  if (state === "processing" || state === "summary" || state === "closed") {
    stopCapture();
  }
  if (message.reason) {
    ui.showStatus(`Session ended: ${String(message.reason).replaceAll("_", " ")}`);
  }
  if (state === "summary") {
    ui.showStatus("Sweep complete.");
  }
}

function stopCapture(): void {
  liveLoop.stop();
  stillLoop.stop();
  mic.stop();
  camera.stop();
}

const socket = new SweepSocket({
  onJson(message) {
    if (message.type === "session") {
      onSession(message);
    } else if (message.type === "interrupted") {
      playback.flush();
      ui.endCaption();
    } else {
      ui.handle(message);
    }
  },
  onAudio(pcm) {
    playback.push(pcm);
  },
  onConnection(state) {
    if (state === "connecting") {
      ui.setConnection("connecting", "warn");
    } else if (state === "closed") {
      ui.setConnection("disconnected", "idle");
      stopCapture();
      playback.stop();
      ui.start.disabled = false;
      ui.setRunning(false);
    }
  },
});

async function begin(): Promise<void> {
  ui.start.disabled = true;
  ending = false;
  ui.end.disabled = false;
  mic.unlock();
  playback.unlock();
  const motion = enableMotion();
  try {
    const stream = await camera.start();
    await Promise.all([
      mic.start(stream, (pcm) => {
        if (!socket.congested) {
          socket.sendBinary(KIND_MIC, {}, pcm);
        }
      }),
      playback.start((isPlaying) => {
        if (isPlaying !== playing) {
          playing = isPlaying;
          socket.sendJson({ type: "playback", state: isPlaying ? "playing" : "idle" });
        }
      }),
      motion,
    ]);
  } catch (error) {
    ui.start.disabled = false;
    ui.toast(error instanceof Error ? error.message : "Could not start camera and microphone", "urgent");
    stopCapture();
    return;
  }
  ui.setRunning(true);
  socket.connect(accessToken);
}

function finish(): void {
  if (ending) {
    return;
  }
  ending = true;
  ui.setEnding();
  socket.sendJson({ type: "control", action: "end_sweep" });
}

ui.start.addEventListener("click", () => void begin());
ui.end.addEventListener("click", finish);
