import type { ServerMessage } from "./ws";

const TOAST_MS = 4000;
const MAX_TOASTS = 3;

function el<T extends HTMLElement>(id: string): T {
  return document.getElementById(id) as T;
}

type Item = { id: string; kind?: string; label?: string | null; shelf?: string; status?: string; confidence?: number };

export class Ui {
  readonly start = el<HTMLButtonElement>("start");
  readonly end = el<HTMLButtonElement>("end");
  readonly video = el<HTMLVideoElement>("preview");
  private readonly connection = el<HTMLElement>("connection");
  private readonly inventory = el<HTMLUListElement>("inventory");
  private readonly count = el<HTMLElement>("count");
  private readonly coverage = el<HTMLElement>("coverage");
  private readonly toasts = el<HTMLElement>("toasts");
  private readonly transcript = el<HTMLElement>("transcript");
  private readonly result = el<HTMLElement>("result");
  private readonly items = new Map<string, HTMLLIElement>();
  private lastRole = "";
  private lastLine: HTMLElement | null = null;

  setConnection(label: string, tone: "ok" | "warn" | "idle"): void {
    this.connection.textContent = label;
    this.connection.dataset.tone = tone;
  }

  setRunning(running: boolean): void {
    this.start.hidden = running;
    this.end.hidden = !running;
  }

  setEnding(): void {
    this.end.disabled = true;
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
    const confidence = typeof item.confidence === "number" ? ` ${Math.round(item.confidence * 100)}%` : "";
    row.textContent = `${item.shelf ? item.shelf + " " : ""}${label}${confidence}`;
    this.count.textContent = String(this.items.size);
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
    if (role !== this.lastRole || !this.lastLine) {
      this.lastRole = role;
      this.lastLine = document.createElement("p");
      this.lastLine.dataset.role = role;
      this.transcript.append(this.lastLine);
    }
    this.lastLine.textContent += text;
    this.transcript.scrollTop = this.transcript.scrollHeight;
  }

  endCaption(): void {
    this.lastRole = "";
    this.lastLine = null;
  }

  showResult(report: string): void {
    const link = document.createElement("a");
    link.href = report;
    link.textContent = "Open report";
    this.result.replaceChildren(link);
  }

  showStatus(text: string): void {
    this.result.textContent = text;
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
      case "question":
        if (message.state === "asked") {
          this.toast(String(message.text), "info");
        }
        return true;
      case "pipeline.progress":
        this.showStatus(`${message.stage}: ${message.done}/${message.total}`);
        return true;
      case "packet_ready":
        this.showResult(String(message.report));
        return true;
      default:
        return false;
    }
  }
}
