const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");

const URL_ARG = process.argv[2];
const SWEEP_SECONDS = Number(process.argv[3] || 115);
const OUT = path.join(__dirname, process.argv[4] || "live");
const CAMERA = path.join(__dirname, "camera.mjpeg");
const VOICE = path.join(__dirname, "user_voice.wav");
const VIEW = { width: 430, height: 800 };
const TITLE = "Library sweep";

const APP_INIT = () => {
  const gum = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
  navigator.mediaDevices.getUserMedia = async (constraints) => {
    const stream = await gum(constraints);
    if (stream.getAudioTracks().length) {
      const monitor = new AudioContext();
      monitor.createMediaStreamSource(new MediaStream(stream.getAudioTracks())).connect(monitor.destination);
      window.__monitor = monitor;
    }
    return stream;
  };
  const banner = () => {
    if (document.getElementById("demo-note") || !document.body) return;
    const note = document.createElement("div");
    note.id = "demo-note";
    note.textContent = "Demo run: a fake camera pans over public shelf photos (Wikimedia Commons, CC BY-SA 4.0: Hoary; Roman Eisele). The user's voice is text-to-speech fed live into the microphone. No physical shelf.";
    note.style.cssText = "position:fixed;left:12px;right:12px;top:46px;z-index:50;padding:6px 10px;border-radius:10px;background:rgba(0,0,0,.72);color:#fff;font:600 11px/1.35 system-ui;text-align:center;pointer-events:none";
    document.body.append(note);
  };
  document.addEventListener("DOMContentLoaded", banner);
  setInterval(banner, 1000);
};

(async () => {
  fs.rmSync(OUT, { recursive: true, force: true });
  fs.mkdirSync(OUT, { recursive: true });
  const file = fs.openSync(path.join(OUT, "take.webm"), "w");
  const browser = await chromium.launch({
    headless: false,
    executablePath: process.env.CHROMIUM_PATH || undefined,
    args: [
      "--use-fake-device-for-media-stream",
      `--use-file-for-fake-video-capture=${CAMERA}`,
      `--use-file-for-fake-audio-capture=${VOICE}%noloop`,
      "--autoplay-policy=no-user-gesture-required",
      "--window-size=450,980",
      "--window-position=0,0",
      `--auto-select-tab-capture-source-by-title=${TITLE}`,
      "--auto-accept-this-tab-capture",
    ],
  });
  const context = await browser.newContext({ viewport: null, permissions: ["camera", "microphone"] });
  await context.exposeBinding("__chunk", (_, b64) => fs.writeSync(file, Buffer.from(b64, "base64")));
  const app = await context.newPage();
  await app.addInitScript(APP_INIT);
  const started = Date.now();
  const log = (m) => console.log(`[${((Date.now() - started) / 1000).toFixed(1)}s] ${m}`);
  app.on("websocket", (ws) => ws.on("framereceived", (f) => {
    if (typeof f.payload !== "string") return;
    try {
      const m = JSON.parse(f.payload);
      if (["session", "packet_ready", "error", "capture_alert"].includes(m.type)) log(`ws ${m.type} ${JSON.stringify(m).slice(0, 150)}`);
      if (m.type === "transcript" && m.role === "agent") process.stdout.write(m.text);
    } catch {}
  }));
  await app.goto(URL_ARG);

  const recorder = await context.newPage();
  await recorder.goto(new URL("/recorder-blank", URL_ARG).toString());
  await recorder.evaluate(() => {
    document.title = "recorder";
    const button = document.createElement("button");
    button.id = "rec";
    button.textContent = "record";
    document.body.append(button);
    button.onclick = async () => {
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: { frameRate: 30 },
        audio: { suppressLocalAudioPlayback: false, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
        preferCurrentTab: false,
        selfBrowserSurface: "exclude",
      });
      const rec = new MediaRecorder(stream, { mimeType: "video/webm;codecs=vp9,opus", videoBitsPerSecond: 3_000_000, audioBitsPerSecond: 160_000 });
      rec.ondataavailable = async (e) => {
        if (!e.data.size) return;
        const bytes = new Uint8Array(await e.data.arrayBuffer());
        let bin = "";
        for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
        await window.__chunk(btoa(bin));
      };
      rec.onstop = () => { window.__stopped = true; };
      rec.start(1000);
      window.__rec = rec;
      window.__tracks = stream.getTracks().map((t) => `${t.kind}:${t.label}`);
    };
  });
  await recorder.click("#rec");
  await recorder.waitForFunction(() => window.__rec, null, { timeout: 20000 });
  log(`recording ${await recorder.evaluate(() => window.__tracks.join(", "))}`);
  await app.bringToFront();
  await app.waitForTimeout(4000);

  log("start");
  await app.click("#start");
  const sweepEnd = Date.now() + SWEEP_SECONDS * 1000;
  let opened = false;
  while (Date.now() < sweepEnd) {
    await app.waitForTimeout(1000);
    if (!opened && Date.now() > started + 70000) {
      await app.click("#sheet-toggle").catch(() => {});
      opened = true;
      await app.waitForTimeout(4000);
      await app.click("#sheet-toggle").catch(() => {});
    }
    if ((await app.evaluate(() => document.getElementById("app").dataset.phase)) !== "live") break;
  }
  if ((await app.evaluate(() => document.getElementById("app").dataset.phase)) === "live") {
    log("end sweep");
    await app.click("#end", { force: true });
  }
  await app.locator("#result a").waitFor({ timeout: 420000 });
  log("packet ready");
  await app.waitForTimeout(22000);
  log("open report");
  await app.click("#result a");
  await app.waitForLoadState("load");
  const glide = async (prefix, pause) => {
    const top = await app.evaluate((p) => {
      const node = [...document.querySelectorAll("h2")].find((h) => h.textContent.startsWith(p));
      return node ? node.getBoundingClientRect().top + window.scrollY - 12 : null;
    }, prefix);
    if (top === null) return;
    const from = await app.evaluate(() => window.scrollY);
    for (let i = 1; i <= 24; i++) {
      await app.evaluate((y) => window.scrollTo(0, y), from + ((top - from) * i) / 24);
      await app.waitForTimeout(60);
    }
    await app.waitForTimeout(pause);
  };
  await app.waitForTimeout(3000);
  await glide("Totals", 6000);
  await glide("Books", 3000);
  for (let i = 0; i < 26; i++) {
    await app.evaluate(() => window.scrollBy(0, 220));
    await app.waitForTimeout(900);
  }
  await glide("Second country", 3000);
  await glide("Costs and latency", 7000);
  log("stop");
  await recorder.evaluate(() => window.__rec.stop());
  await recorder.waitForFunction(() => window.__stopped, null, { timeout: 30000 });
  await recorder.waitForTimeout(3000);
  fs.closeSync(file);
  await browser.close();
  log("done");
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
