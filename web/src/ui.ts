import type { ServerMessage } from "./ws";

const TOAST_MS = 5000;
const MAX_TOASTS = 2;
const CAPTION_MS = 7000;
const PIPELINE_STAGES = ["keyframes", "spines", "tracks", "scale", "reading", "resolution", "room", "items", "appraisal", "pricing", "packet"];
const STAGE_LABELS: Record<string, string> = {
  keyframes: "Picking the sharpest frames",
  spines: "Finding every spine",
  tracks: "Matching spines across frames",
  scale: "Measuring size",
  reading: "Reading titles",
  resolution: "Matching books to catalogues",
  room: "Estimating the room",
  items: "Looking for other items",
  appraisal: "Appraising",
  pricing: "Checking prices",
  packet: "Writing the report",
};

function el<T extends HTMLElement>(id: string): T {
  return document.getElementById(id) as T;
}

type Item = { id: string; kind?: string; label?: string | null; shelf?: string; status?: string; confidence?: number };
type Phase = "intro" | "live" | "processing" | "done";

export class Ui {
  readonly start = el<HTMLButtonElement>("start");
  readonly end = el<HTMLButtonElement>("end");
  readonly video = el<HTMLVideoElement>("preview");
  private readonly app = el<HTMLElement>("app");
  private readonly connection = el<HTMLElement>("connection");
  private readonly timer = el<HTMLElement>("timer");
  private readonly inventory = el<HTMLUListElement>("inventory");
  private readonly count = el<HTMLElement>("count");
  private readonly latest = el<HTMLElement>("latest");
  private readonly sheet = el<HTMLElement>("sheet");
  private readonly coverage = el<HTMLElement>("coverage");
  private readonly toasts = el<HTMLElement>("toasts");
  private readonly transcript = el<HTMLElement>("transcript");
  private readonly progress = el<HTMLElement>("progress");
  private readonly progressTitle = el<HTMLElement>("progress-title");
  private readonly progressStep = el<HTMLElement>("progress-step");
  private readonly barFill = el<HTMLElement>("bar-fill");
  private readonly result = el<HTMLElement>("result");
  private readonly items = new Map<string, HTMLLIElement>();
  private lastRole = "";
  private clearCaption = 0;
  private startedAt = 0;
  private tick = 0;

  constructor(private readonly token: string) {
    el<HTMLButtonElement>("sheet-toggle").addEventListener("click", () => this.toggleSheet());
  }

  get phase(): Phase {
    return this.app.dataset.phase as Phase;
  }

  private setPhase(phase: Phase): void {
    this.app.dataset.phase = phase;
    el("intro").hidden = phase !== "intro";
    this.progress.hidden = phase !== "processing" && phase !== "done";
  }

  private toggleSheet(): void {
    const open = this.sheet.dataset.open !== "true";
    this.sheet.dataset.open = String(open);
    el("sheet-toggle").setAttribute("aria-expanded", String(open));
  }

  setConnection(label: string, tone: "ok" | "warn" | "idle"): void {
    this.connection.querySelector("b")!.textContent = label.charAt(0).toUpperCase() + label.slice(1);
    this.connection.dataset.tone = tone;
  }

  setRunning(running: boolean): void {
    this.end.hidden = !running;
    if (running) {
      this.setPhase("live");
      this.startedAt = Date.now();
      this.timer.hidden = false;
      this.renderTimer();
      this.tick = window.setInterval(() => this.renderTimer(), 1000);
    } else {
      window.clearInterval(this.tick);
      this.timer.hidden = true;
      if (this.phase === "live") {
        this.setPhase("intro");
      }
    }
  }

  private renderTimer(): void {
    const seconds = Math.floor((Date.now() - this.startedAt) / 1000);
    this.timer.textContent = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
  }

  setEnding(): void {
    this.end.disabled = true;
    this.end.textContent = "Ending";
  }

  toast(text: string, severity = "info"): void {
    const node = document.createElement("div");
    node.className = "toast";
    node.dataset.severity = severity;
    node.textContent = text;
    this.toasts.append(node);
    while (this.toasts.children.length > MAX_TOASTS) {
      this.toasts.firstElementChild?.remove();
    }
    window.setTimeout(() => node.remove(), TOAST_MS);
  }

  upsert(item: Item): void {
    let row = this.items.get(item.id);
    if (!row) {
      row = document.createElement("li");
      this.items.set(item.id, row);
      this.inventory.prepend(row);
    }
    row.dataset.status = item.status ?? "";
    const label = item.label ?? (item.kind === "item" ? "Unidentified item" : "Unidentified book");
    const text = `${item.shelf ? item.shelf + " " : ""}${label}`;
    row.replaceChildren(document.createTextNode(text));
    if (typeof item.confidence === "number") {
      const confidence = document.createElement("span");
      confidence.className = "conf";
      confidence.textContent = `${Math.round(item.confidence * 100)}%`;
      row.append(confidence);
    }
    this.count.textContent = String(this.items.size);
    el("progress-count").textContent = String(this.items.size);
    this.latest.textContent = text;
  }

  setCoverage(regions: { id: string; state: string }[]): void {
    this.coverage.replaceChildren(
      ...regions.map((region) => {
        const chip = document.createElement("span");
        chip.className = "chip";
        chip.dataset.state = region.state;
        chip.textContent = region.id;
        return chip;
      }),
    );
  }

  caption(role: string, text: string): void {
    if (role !== this.lastRole) {
      this.lastRole = role;
      this.transcript.dataset.role = role;
      this.transcript.textContent = "";
    }
    this.transcript.textContent += text;
    window.clearTimeout(this.clearCaption);
    this.clearCaption = window.setTimeout(() => this.transcript.replaceChildren(), CAPTION_MS);
  }

  endCaption(): void {
    this.lastRole = "";
  }

  showProcessing(): void {
    if (this.phase === "processing" || this.phase === "done") {
      return;
    }
    this.setPhase("processing");
    this.progressTitle.textContent = "Building your report";
    this.result.replaceChildren();
  }

  showResult(report: string): void {
    this.setPhase("done");
    this.progressTitle.textContent = "Report ready";
    this.progressStep.textContent = "Every count, title and price in the report links back to its frame or source.";
    this.setBar(1);
    const link = document.createElement("a");
    link.href = `${report}?t=${encodeURIComponent(this.token)}`;
    link.textContent = "Open report";
    this.result.replaceChildren(link);
  }

  showStatus(text: string): void {
    if (this.phase === "processing" || this.phase === "done") {
      this.progressStep.textContent = text;
      return;
    }
    this.toast(text, "urgent");
  }

  private setBar(fraction: number): void {
    const clamped = Math.max(0, Math.min(1, fraction));
    this.barFill.style.transform = `scaleX(${clamped})`;
    this.barFill.parentElement?.setAttribute("aria-valuenow", String(Math.round(clamped * 100)));
  }

  private showStage(stage: string, done: number, total: number): void {
    if (this.phase !== "processing") {
      return;
    }
    const index = Math.max(0, PIPELINE_STAGES.indexOf(stage));
    const within = total > 0 ? done / total : 0;
    this.progressStep.textContent = STAGE_LABELS[stage] ?? stage;
    this.setBar((index + within) / PIPELINE_STAGES.length);
  }

  handle(message: ServerMessage): boolean {
    switch (message.type) {
      case "inventory.upsert":
        this.upsert(message.item as Item);
        return true;
      case "capture_alert":
        this.toast(String(message.say), String(message.severity));
        return true;
      case "coverage":
        this.setCoverage(message.regions as { id: string; state: string }[]);
        return true;
      case "transcript":
        this.caption(String(message.role), String(message.text));
        return true;
      case "error":
        this.toast(String(message.say), "urgent");
        return true;
      case "question":
        if (message.state === "asked") {
          this.toast(String(message.text), "info");
        }
        return true;
      case "pipeline.progress":
        this.showStage(String(message.stage), Number(message.done), Number(message.total));
        return true;
      case "packet_ready":
        this.showResult(String(message.report));
        return true;
      default:
        return false;
    }
  }
}
