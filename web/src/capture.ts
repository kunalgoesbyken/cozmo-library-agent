const LIVE_LONG_SIDE = 768;
const LIVE_QUALITY = 0.7;
const STILL_LONG_SIDE = 1920;
const STILL_QUALITY = 0.85;
const TUNNEL_STILL_FPS = 2;
const LOCAL_STILL_FPS = 4;
const LOCAL_HOSTS = ["localhost", "127.0.0.1", "[::1]"];

export type Motion = { rot_dps: number; acc: number };

export type StillShot = { blob: Blob; w: number; h: number; motion: Motion; orient: string };

export type LiveShot = { blob: Blob; w: number; h: number };

type MotionPermissionEvent = typeof DeviceMotionEvent & { requestPermission?: () => Promise<string> };

export function stillFps(): number {
  return LOCAL_HOSTS.includes(location.hostname) ? LOCAL_STILL_FPS : TUNNEL_STILL_FPS;
}

export async function enableMotion(): Promise<boolean> {
  if (typeof DeviceMotionEvent === "undefined") {
    return false;
  }
  const request = (DeviceMotionEvent as MotionPermissionEvent).requestPermission;
  if (!request) {
    return true;
  }
  try {
    return (await request.call(DeviceMotionEvent)) === "granted";
  } catch {
    return false;
  }
}

function magnitude(a: number | null, b: number | null, c: number | null): number {
  return Math.hypot(a ?? 0, b ?? 0, c ?? 0);
}

export class Camera {
  readonly video: HTMLVideoElement;
  stream: MediaStream | null = null;
  private canvas = document.createElement("canvas");
  private peak: Motion = { rot_dps: 0, acc: 0 };
  private wakeLock: WakeLockSentinel | null = null;
  private onMotion = (event: DeviceMotionEvent) => {
    const rate = event.rotationRate;
    const rot = magnitude(rate?.alpha ?? null, rate?.beta ?? null, rate?.gamma ?? null);
    const accel = event.acceleration;
    const acc = magnitude(accel?.x ?? null, accel?.y ?? null, accel?.z ?? null);
    this.peak = { rot_dps: Math.max(this.peak.rot_dps, rot), acc: Math.max(this.peak.acc, acc) };
  };
  private onVisible = () => {
    if (document.visibilityState === "visible" && this.stream) {
      void this.holdScreen();
    }
  };

  constructor(video: HTMLVideoElement) {
    this.video = video;
  }

  async start(): Promise<MediaStream> {
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1080 } },
      audio: { echoCancellation: true, noiseSuppression: true },
    });
    this.video.srcObject = this.stream;
    await this.video.play();
    window.addEventListener("devicemotion", this.onMotion);
    document.addEventListener("visibilitychange", this.onVisible);
    await this.holdScreen();
    return this.stream;
  }

  private async holdScreen(): Promise<void> {
    if (!("wakeLock" in navigator)) {
      return;
    }
    try {
      this.wakeLock = await navigator.wakeLock.request("screen");
    } catch {
      this.wakeLock = null;
    }
  }

  stop(): void {
    window.removeEventListener("devicemotion", this.onMotion);
    document.removeEventListener("visibilitychange", this.onVisible);
    void this.wakeLock?.release();
    this.wakeLock = null;
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    this.video.srcObject = null;
  }

  private async encode(longSide: number, quality: number): Promise<{ blob: Blob; w: number; h: number } | null> {
    const sourceW = this.video.videoWidth;
    const sourceH = this.video.videoHeight;
    if (!sourceW || !sourceH) {
      return null;
    }
    const scale = Math.min(1, longSide / Math.max(sourceW, sourceH));
    const w = Math.round(sourceW * scale);
    const h = Math.round(sourceH * scale);
    this.canvas.width = w;
    this.canvas.height = h;
    this.canvas.getContext("2d")!.drawImage(this.video, 0, 0, w, h);
    const blob = await new Promise<Blob | null>((resolve) => this.canvas.toBlob(resolve, "image/jpeg", quality));
    return blob ? { blob, w, h } : null;
  }

  liveFrame(): Promise<LiveShot | null> {
    return this.encode(LIVE_LONG_SIDE, LIVE_QUALITY);
  }

  async still(): Promise<StillShot | null> {
    const shot = await this.encode(STILL_LONG_SIDE, STILL_QUALITY);
    if (!shot) {
      return null;
    }
    const motion = this.peak;
    this.peak = { rot_dps: 0, acc: 0 };
    return { ...shot, motion, orient: screen.orientation?.type ?? "" };
  }
}

export class Loop {
  private timer: number | null = null;
  private busy = false;

  constructor(private readonly task: () => Promise<void>) {}

  get running(): boolean {
    return this.timer !== null;
  }

  start(fps: number): void {
    this.stop();
    this.timer = window.setInterval(async () => {
      if (this.busy) {
        return;
      }
      this.busy = true;
      try {
        await this.task();
      } finally {
        this.busy = false;
      }
    }, 1000 / fps);
  }

  stop(): void {
    if (this.timer !== null) {
      window.clearInterval(this.timer);
      this.timer = null;
    }
  }
}
