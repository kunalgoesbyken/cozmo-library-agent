export const KIND_MIC = 0x01;
export const KIND_LIVE_FRAME = 0x02;
export const KIND_STILL = 0x03;
export const KIND_AGENT_AUDIO = 0x81;

const HEADER_LENGTH_BYTES = 4;
const MAX_BUFFERED_BYTES = 1_500_000;

export type ServerMessage = { type: string; [key: string]: unknown };

export type Handlers = {
  onJson: (message: ServerMessage) => void;
  onAudio: (pcm: ArrayBuffer) => void;
  onConnection: (state: "connecting" | "open" | "closed") => void;
};

export function timestampMs(): number {
  return Math.round(performance.timeOrigin + performance.now());
}

function pack(kind: number, header: object, payload: ArrayBuffer | Uint8Array): Uint8Array {
  const head = new TextEncoder().encode(JSON.stringify(header));
  const body = payload instanceof Uint8Array ? payload : new Uint8Array(payload);
  const out = new Uint8Array(1 + HEADER_LENGTH_BYTES + head.length + body.length);
  out[0] = kind;
  new DataView(out.buffer).setUint32(1, head.length, false);
  out.set(head, 1 + HEADER_LENGTH_BYTES);
  out.set(body, 1 + HEADER_LENGTH_BYTES + head.length);
  return out;
}

function unpack(data: ArrayBuffer): { kind: number; header: Record<string, unknown>; payload: ArrayBuffer } {
  const view = new DataView(data);
  const size = view.getUint32(1, false);
  const start = 1 + HEADER_LENGTH_BYTES;
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(data, start, size)));
  return { kind: view.getUint8(0), header, payload: data.slice(start + size) };
}

export class SweepSocket {
  private socket: WebSocket | null = null;
  private seq = new Map<number, number>();

  constructor(private readonly handlers: Handlers) {}

  connect(token: string): void {
    const scheme = location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${scheme}://${location.host}/ws/sweep`);
    socket.binaryType = "arraybuffer";
    this.socket = socket;
    this.handlers.onConnection("connecting");
    socket.onopen = () => {
      this.handlers.onConnection("open");
      this.sendJson({ type: "hello", token, t_ms: timestampMs(), device: { ua: navigator.userAgent } });
    };
    socket.onclose = () => this.handlers.onConnection("closed");
    socket.onmessage = (event) => {
      if (typeof event.data === "string") {
        this.handlers.onJson(JSON.parse(event.data));
        return;
      }
      const frame = unpack(event.data as ArrayBuffer);
      if (frame.kind === KIND_AGENT_AUDIO) {
        this.handlers.onAudio(frame.payload);
      }
    };
  }

  get isOpen(): boolean {
    return this.socket?.readyState === WebSocket.OPEN;
  }

  get congested(): boolean {
    return (this.socket?.bufferedAmount ?? 0) > MAX_BUFFERED_BYTES;
  }

  sendJson(message: object): void {
    if (this.isOpen) {
      this.socket!.send(JSON.stringify(message));
    }
  }

  sendBinary(kind: number, header: Record<string, unknown>, payload: ArrayBuffer | Uint8Array): void {
    if (!this.isOpen) {
      return;
    }
    const seq = (this.seq.get(kind) ?? 0) + 1;
    this.seq.set(kind, seq);
    this.socket!.send(pack(kind, { seq, t_ms: timestampMs(), ...header }, payload));
  }

  close(): void {
    this.socket?.close();
  }
}
