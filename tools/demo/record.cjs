const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");

const URL = process.argv[2];
const SWEEP_SECONDS = Number(process.argv[3] || 112);
const OUT = path.join(__dirname, process.argv[4] || "out");
const CAMERA = path.join(__dirname, "camera.mjpeg");
const VOICE = path.join(__dirname, "user_voice.wav");
const VIEW = { width: 540, height: 1080 };

const TAP = () => {
  window.__taps = [];
  const original = AudioNode.prototype.connect;
  AudioNode.prototype.connect = function (target, ...rest) {
    if (target instanceof AudioDestinationNode) {
      const ctx = this.context;
      if (!ctx.__tap) {
        ctx.__tap = ctx.createMediaStreamDestination();
        const entry = { start: Date.now(), chunks: [], rate: ctx.sampleRate };
        const recorder = new MediaRecorder(ctx.__tap.stream, { mimeType: "audio/webm;codecs=opus" });
        recorder.ondataavailable = (e) => e.data.size && entry.chunks.push(e.data);
        recorder.start(1000);
        entry.recorder = recorder;
        window.__taps.push(entry);
      }
      original.call(this, ctx.__tap);
    }
    return original.call(this, target, ...rest);
  };
  const gum = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
  navigator.mediaDevices.getUserMedia = async (c) => {
    const stream = await gum(c);
    window.__gumAt = Date.now();
    return stream;
  };
  const banner = () => {
    if (document.getElementById("demo-note") || !document.body) return;
    const note = document.createElement("div");
    note.id = "demo-note";
    note.textContent = "Demo run: a fake camera pans over public shelf photos (Wikimedia Commons, CC BY-SA 4.0: Hoary; Roman Eisele). User voice is text-to-speech. No physical shelf.";
    note.style.cssText = "position:fixed;left:12px;right:12px;bottom:calc(env(safe-area-inset-bottom) + 6px);z-index:50;padding:6px 10px;border-radius:10px;background:rgba(0,0,0,.72);color:#fff;font:600 11px/1.35 system-ui;text-align:center;pointer-events:none";
    document.body.append(note);
  };
  document.addEventListener("DOMContentLoaded", banner);
  setInterval(banner, 1000);
};

async function collectAudio(page) {
  const taps = await page.evaluate(async () => {
    const out = [];
    for (const tap of window.__taps || []) {
      await new Promise((resolve) => {
        tap.recorder.onstop = resolve;
        tap.recorder.stop();
      });
      const blob = new Blob(tap.chunks, { type: "audio/webm" });
      const bytes = new Uint8Array(await blob.arrayBuffer());
      let binary = "";
      for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
      out.push({ start: tap.start, rate: tap.rate, data: btoa(binary) });
    }
    return { taps: out, gumAt: window.__gumAt || null };
  });
  taps.taps.forEach((tap, i) => fs.writeFileSync(path.join(OUT, `tap${i}.webm`), Buffer.from(tap.data, "base64")));
  return { taps: taps.taps.map(({ start, rate }, i) => ({ file: `tap${i}.webm`, start, rate })), gumAt: taps.gumAt };
}

(async () => {
  fs.rmSync(OUT, { recursive: true, force: true });
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch({
    headless: false,
    executablePath: process.env.CHROMIUM_PATH || undefined,
    args: [
      "--use-fake-ui-for-media-stream",
      "--use-fake-device-for-media-stream",
      `--use-file-for-fake-video-capture=${CAMERA}`,
      `--use-file-for-fake-audio-capture=${VOICE}%noloop`,
      "--autoplay-policy=no-user-gesture-required",
    ],
  });
  const context = await browser.newContext({ viewport: VIEW, recordVideo: { dir: OUT, size: VIEW }, permissions: ["camera", "microphone"] });
  await context.addInitScript(TAP);
  const videoStart = Date.now();
  const page = await context.newPage();
  const log = (m) => console.log(`[${((Date.now() - videoStart) / 1000).toFixed(1)}s] ${m}`);
  page.on("console", (m) => m.type() === "error" && log(`console: ${m.text()}`));
  page.on("websocket", (ws) => {
    ws.on("framereceived", (f) => {
      if (typeof f.payload !== "string") return;
      try {
        const m = JSON.parse(f.payload);
        if (["session", "packet_ready", "error", "pipeline.progress", "capture_alert"].includes(m.type)) log(`ws ${m.type} ${JSON.stringify(m).slice(0, 160)}`);
        if (m.type === "transcript" && m.role === "agent") process.stdout.write(m.text);
      } catch {}
    });
  });
  await page.goto(URL);
  await page.waitForTimeout(4000);
  log("start");
  await page.click("#start");
  const sweepEnd = Date.now() + SWEEP_SECONDS * 1000;
  let opened = false;
  while (Date.now() < sweepEnd) {
    await page.waitForTimeout(1000);
    if (!opened && Date.now() > videoStart + 80000) {
      await page.click("#sheet-toggle").catch(() => {});
      opened = true;
      await page.waitForTimeout(4000);
      await page.click("#sheet-toggle").catch(() => {});
    }
    if (await page.locator("#progress").isVisible()) break;
  }
  if ((await page.evaluate(() => document.getElementById("app").dataset.phase)) === "live") {
    log("end sweep");
    await page.click("#end", { force: true });
  }
  await page.locator("#result a").waitFor({ timeout: 420000 });
  log("packet ready");
  await page.waitForTimeout(20000);
  const audio = await collectAudio(page);
  fs.writeFileSync(path.join(OUT, "timing.json"), JSON.stringify({ videoStart, ...audio }, null, 2));
  log("open report");
  await page.click("#result a");
  await page.waitForLoadState("load");
  await page.waitForTimeout(3000);
  const glide = async (selector, pause) => {
    const top = await page.evaluate((sel) => {
      const node = [...document.querySelectorAll("h2")].find((h) => h.textContent.startsWith(sel));
      return node ? node.getBoundingClientRect().top + window.scrollY - 12 : null;
    }, selector);
    if (top === null) return;
    const from = await page.evaluate(() => window.scrollY);
    const steps = 24;
    for (let i = 1; i <= steps; i++) {
      await page.evaluate((y) => window.scrollTo(0, y), from + ((top - from) * i) / steps);
      await page.waitForTimeout(60);
    }
    await page.waitForTimeout(pause);
  };
  await page.waitForTimeout(3000);
  await glide("Totals", 6000);
  await glide("Books", 3000);
  for (let i = 0; i < 26; i++) {
    await page.evaluate(() => window.scrollBy(0, 220));
    await page.waitForTimeout(900);
  }
  await glide("Second country", 3000);
  await glide("Costs and latency", 7000);
  await page.waitForTimeout(2000);
  log("done");
  const video = page.video();
  await context.close();
  fs.copyFileSync(await video.path(), path.join(OUT, "screen.webm"));
  await browser.close();
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
