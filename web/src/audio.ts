const MIC_RATE = 16000;
const AGENT_RATE = 24000;
const MIC_CHUNK_SAMPLES = 640;

const micWorkletSource = `
class MicEncoder extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.target = options.processorOptions.targetRate;
    this.chunk = options.processorOptions.chunkSamples;
    this.step = sampleRate / this.target;
    this.position = 0;
    this.previous = 0;
    this.out = new Int16Array(this.chunk);
    this.filled = 0;
  }
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;
    for (let i = 0; i < channel.length; i++) {
      const current = channel[i];
      while (this.position < 1) {
        const value = this.previous + (current - this.previous) * this.position;
        const clamped = Math.max(-1, Math.min(1, value));
        this.out[this.filled++] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
        if (this.filled === this.chunk) {
          this.port.postMessage(this.out.buffer.slice(0));
          this.filled = 0;
        }
        this.position += this.step;
      }
      this.position -= 1;
      this.previous = current;
    }
    return true;
  }
}
registerProcessor("mic-encoder", MicEncoder);
`;

const playerWorkletSource = `
class Player extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.step = options.processorOptions.sourceRate / sampleRate;
    this.queue = [];
    this.offset = 0;
    this.playing = false;
    this.port.onmessage = (event) => {
      if (event.data.flush) {
        this.queue = [];
        this.offset = 0;
      } else {
        this.queue.push(event.data.samples);
      }
    };
  }
  process(inputs, outputs) {
    const out = outputs[0][0];
    let produced = 0;
    while (produced < out.length && this.queue.length) {
      const head = this.queue[0];
      const index = Math.floor(this.offset);
      if (index >= head.length) {
        this.offset -= head.length;
        this.queue.shift();
        continue;
      }
      out[produced++] = head[index] / 32768;
      this.offset += this.step;
    }
    while (produced < out.length) out[produced++] = 0;
    const playing = this.queue.length > 0;
    if (playing !== this.playing) {
      this.playing = playing;
      this.port.postMessage({ playing });
    }
    return true;
  }
}
registerProcessor("player", Player);
`;

async function loadWorklet(context: AudioContext, source: string): Promise<void> {
  const url = URL.createObjectURL(new Blob([source], { type: "text/javascript" }));
  try {
    await context.audioWorklet.addModule(url);
  } finally {
    URL.revokeObjectURL(url);
  }
}

export class MicCapture {
  private context: AudioContext | null = null;

  unlock(): void {
    this.context = new AudioContext({ sampleRate: MIC_RATE });
    void this.context.resume();
  }

  async start(stream: MediaStream, onChunk: (pcm: ArrayBuffer) => void): Promise<void> {
    const context = this.context!;
    await context.resume();
    await loadWorklet(context, micWorkletSource);
    const node = new AudioWorkletNode(context, "mic-encoder", {
      processorOptions: { targetRate: MIC_RATE, chunkSamples: MIC_CHUNK_SAMPLES },
    });
    node.port.onmessage = (event) => onChunk(event.data as ArrayBuffer);
    context.createMediaStreamSource(stream).connect(node);
  }

  stop(): void {
    void this.context?.close();
    this.context = null;
  }
}

export class AgentPlayback {
  private context: AudioContext | null = null;
  private node: AudioWorkletNode | null = null;

  unlock(): void {
    this.context = new AudioContext({ sampleRate: AGENT_RATE });
    void this.context.resume();
  }

  async start(onState: (playing: boolean) => void): Promise<void> {
    const context = this.context!;
    await context.resume();
    await loadWorklet(context, playerWorkletSource);
    const node = new AudioWorkletNode(context, "player", {
      numberOfInputs: 0,
      outputChannelCount: [1],
      processorOptions: { sourceRate: AGENT_RATE },
    });
    node.port.onmessage = (event) => onState(Boolean(event.data.playing));
    node.connect(context.destination);
    this.node = node;
  }

  push(pcm: ArrayBuffer): void {
    const samples = new Int16Array(pcm.byteLength >> 1);
    new Uint8Array(samples.buffer).set(new Uint8Array(pcm, 0, samples.length * 2));
    this.node?.port.postMessage({ samples }, [samples.buffer]);
  }

  flush(): void {
    this.node?.port.postMessage({ flush: true });
  }

  stop(): void {
    void this.context?.close();
    this.context = null;
    this.node = null;
  }
}
