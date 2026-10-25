# 04 · Live agent stack (state of the art, 25 Oct 2026)

Scope: the live voice and camera sweep (brief capability 1, 20% of the score). It covers the real-time conversation, capture coaching, live inventory push, voice corrections, and the hand-off to the background pipeline. Vision and measurement are covered in `02-vision-measurement.md`, and the reference app in `01-reference-app.md`.

Method: I downloaded the official docs pages on 2026-10-25 and grepped the raw HTML and markdown, so the model IDs and field names below are copied from the source, not recalled. Each section has its own citations, and they are collected at the end. Some numbers are my own estimates, not documented facts. Those are marked **[est]**. Docs that contradict each other are marked **[conflict]**.

---

## 0. TL;DR decisions

| Decision | Choice | Why |
|---|---|---|
| Live model | `gemini-3.8-live` (stable, Sep 2026) | Native audio-to-audio. Takes video frames, and async (NON_BLOCKING) tools are on by default. Proactive audio is always on. The reference app uses it too. |
| Topology | Phone ⇄ **our FastAPI backend** ⇄ Gemini Live (server-to-server) | The backend needs to see every frame and event anyway: full-res stills, the tool loop, inventory state, frame_ref timestamps. It also keeps the API key off the phone. The direct-browser path with ephemeral tokens would mean duplicating the tool plumbing. |
| Session config | `contextWindowCompression` **must** be on; `sessionResumption` on; GoAway reconnect | Audio+video sessions are **capped at 2 min without compression**, and a sweep is up to 3 min plus the greeting. Connections also last only ~10 min, while the greeting + sweep + ≤5 min processing + readback timeline comes to ~9 min. |
| Two video streams | Live model: 1 fps at ~768 px, MEDIA_RESOLUTION_LOW/MEDIUM. Pipeline: 2–4 fps stills at ≥1280 px sent to the backend | The Live API caps input at 1 fps. The live model only coaches. Spine reading happens offline on the full-res stills. |
| Pushing detections into the conversation | A long-running NON_BLOCKING "capture monitor" tool that streams `FunctionResponse`s with `willContinue=true`, using `scheduling` = SILENT, WHEN_IDLE or INTERRUPT | This is the documented way to inject results without breaking the turn. `sendClientContent` is the fallback. |
| Phone access | **cloudflared quick tunnel** for the demo and the review call. mkcert + LAN IP as an optional low-latency dev setup | No account needed, real HTTPS, and works on any network the evaluators bring. |
| Demo recording | Android built-in recorder with "device audio + microphone", **plus** an in-app MediaRecorder backup track. Do a 20 s audio test first | Chrome may route call-mode audio where playback capture can't reach it (see §5). |

---

## 1. Gemini Live API (Oct 2026)

### 1.1 Models

From the models page and the 3.8 Live model card (last updated 2026-10-19):

| Model ID | Status | Notes |
|---|---|---|
| `gemini-3.8-live` | **Stable, default** | "Default option for most low-latency voice agent experiences without reasoning delays". Interleaved reasoning; `thinkingLevel` is **not supported** and must be omitted. 131,072 input / 65,536 output tokens. |
| `gemini-3.8-live-extended-thinking` | Stable | Background reasoning with spoken "fillers". Tracks `interactionStatus` (IN_PROGRESS/IDLE) instead of `turnComplete`. **Only** NON_BLOCKING tools, no scheduling. Wrong fit here: we want snappy coaching, not deliberation. |
| `gemini-3.1-flash-live-preview` | Legacy preview | Synchronous tools only, so the model stays silent until the tool response comes back. Avoid. |
| `gemini-2.5-flash-native-audio-preview-12-2025` | Access-limited | "limiting access to the 2.5 models to users who have actively used them in the past". On Vertex it is `gemini-live-2.5-flash-native-audio` (GA). |
| `gemini-3.5-live-translate-preview`, `gemini-3.5-transcribe-live` | Specialised | Not conversational. |

**Half-cascade is gone.** There is no half-cascade model in the current models list. The 2.0/2.5 "half-cascade" IDs (`gemini-2.0-flash-live-001`, `gemini-live-2.5-flash-preview`) appear nowhere in the current docs. All Live models are native audio and output **AUDIO only**. Text comes from `outputAudioTranscription`.

What changed in 3.8 Live versus 3.1 (model card "Migrating" section). Every item matters for us:
- Async function calling (`behavior: NON_BLOCKING`) is now the **default**. `BLOCKING` is still available. Scheduling `SILENT` / `WHEN_IDLE` / `INTERRUPT` is supported.
- `send_client_content` works throughout the session with explicit `user`/`model` roles. `turn_complete=true` **unconditionally interrupts** generation. "If you send content without turn_complete, the server waits for subsequent messages before responding."
- **Proactive audio is permanently on.** Setting `proactive_audio:false` is an error. The model can choose not to answer speech that isn't relevant, which helps when the user is muttering titles to themself. You are billed for input the whole time it listens.
- Affective dialog has been **removed**. Sending `enable_affective_dialog` is an error.
- **Turn coverage** defaults to `TURN_INCLUDES_AUDIO_ACTIVITY_AND_ALL_VIDEO`: every frame you send goes into context. "Only send frames when needed to manage context and cost."

Sources: https://ai.google.dev/gemini-api/docs/models, https://ai.google.dev/gemini-api/docs/models/gemini-3.8-live, https://ai.google.dev/gemini-api/docs/live-api/capabilities, https://ai.google.dev/gemini-api/docs/live-api/thinking

### 1.2 Media formats, video rate and resolution

- Audio in: raw 16-bit PCM, little-endian, 16 kHz (`audio/pcm;rate=16000`). Other rates are resampled if the MIME type declares them. Audio out: 16-bit PCM at **24 kHz**.
- Video: "individual images (e.g., JPEG or PNG) at a specific frame rate (**max 1 frame per second**)". The overview table says "JPEG <= 1FPS". Vertex says "images/video (JPEG 1FPS)".
- Resolution: there is no hard pixel limit in the Gemini API docs. ADK's audio/video guide recommends **768×768**, JPEG, ~1 fps. LiveKit's sampler resizes to 1024×1024 by default.
- `mediaResolution` (session-level in Live) sets tokens per frame. For the Gemini 3/4 family, the media-resolution table gives video frames 70 tokens at LOW/MEDIUM and **280 at HIGH** ("required only when … reading dense text (OCR)"). Audio is 25 tokens/s whatever the setting.
- Implication: the live model watching at 1 fps / 70 tokens can tell "that's blurry" or "you skipped the left unit". It **cannot** reliably read 2 cm spines. Spine reading belongs to the offline stills pipeline.

Sources: https://ai.google.dev/gemini-api/docs/live-api, https://ai.google.dev/gemini-api/docs/media-resolution, https://adk.dev/live/audio-video/, https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/live-api

### 1.3 Session limits, compression and resumption (the critical part)

Verbatim from the session-management page (updated 2026-09-15). The best-practices page and ADK's limits table repeat it:

> "Without compression, audio-only sessions are limited to 15 minutes, and **audio-video sessions are limited to 2 minutes**. Exceeding these limits will terminate the session … you can use context window compression to extend sessions to an unlimited amount of time. The lifetime of a connection is limited as well, to **around 10 minutes**."

So the historical 2-minute audio+video limit **still applies in Oct 2026**. A sweep of up to 3 minutes plus a greeting **will be killed mid-sweep** unless compression is enabled. I read the reference app's code and found it sets neither `context_window_compression` nor `session_resumption`. Instead it replays the transcript on reconnect with `history_config.initial_history_in_client_content`. With the camera on, that design would hit the 2-minute wall. I did not run it to confirm (see `_reference/.../live_demo/live_tools.py` and `server.py`).

**contextWindowCompression** (from the API reference, `ContextWindowCompressionConfig`):
- `triggerTokens` (int64): "number of tokens (before running a turn) required to trigger a context window compression". The default is 80% of the model's context window.
- `slidingWindow.targetTokens` (int64): tokens kept after compression. The default is `trigger_tokens/2`. "The resulting context will always begin at the start of a USER role turn. System instructions and any `prefixTurns` will always remain."
- Each compression causes "a temporary latency increase", so don't trigger it often. Best-practices gives the example trigger 25,000 / window 8,000 as a cost cap.
- **Recommendation:** `triggerTokens ≈ 40000`, `targetTokens ≈ 16000` **[est]**. At about 25 audio tokens/s plus 70 tokens per frame at 1 fps, that is roughly 95 tok/s, so compression fires at most once or twice in a 4–5 minute session. Inventory state lives in the backend and gets re-asserted through SILENT tool responses (§3.3), so whatever the sliding window drops can be recovered.

**sessionResumption**: set `sessionResumption: {handle: null}` on the first connect. Store `sessionResumptionUpdate.newHandle` whenever `resumable` is true. Reconnect with `{handle}`. The handle is valid for **2 h** after the session ends. `resumable` is false while the model is generating or executing function calls, so resume from the last good handle.

**GoAway**: `goAway.timeLeft` warns that the connection will close soon (as ABORTED). When it arrives, open a new connection with the latest handle and switch the audio pipe over.

**Timeline risk:** greeting (~0.5 min) + sweep (≤3 min) + background processing (≤5 min) + spoken summary (~1 min) ≈ 9.5 min, right at the ~10-minute connection lifetime. Mitigations:
1. Stop sending video when the sweep ends.
2. Handle GoAway with resumption.
3. Or close the live connection after the sweep and open a fresh one with the handle (or a seeded summary) for the readback.

Sources: https://ai.google.dev/gemini-api/docs/live-api/session-management, https://ai.google.dev/api/live, https://ai.google.dev/gemini-api/docs/live-api/best-practices

### 1.4 Function calling, including async tools

- Declare functions in the setup. The client has to answer with `sendToolResponse({functionResponses:[{id,name,response}]})`. "The Live API doesn't support automatic tool response handling."
- On `gemini-3.8-live`, NON_BLOCKING is the default, so the model keeps talking while the tool runs. You can mark individual tools `BLOCKING`. That suits fast lookups like `get_sweep_status`.
- `FunctionResponse` fields (API reference, `FunctionResponse`):
  - `scheduling`: `SILENT` ("only add the result to the conversation context, do not interrupt or trigger generation"), `WHEN_IDLE` ("prompt to generate output without interrupting ongoing generation", **the default**), or `INTERRUPT` ("interrupt ongoing generation and prompt to generate output").
  - `willContinue`: "Signals that function call continues, and more responses will be returned, **turning the function call into a generator**. Only applicable to NON_BLOCKING … To avoid triggering the generation and finish the function call, additionally set scheduling to SILENT."
- `toolCallCancellation.ids`: sent when the user barges in on a turn that issued calls. Cancel those tasks, and undo side effects if possible.
- Google Search grounding works in Live. Code execution, URL context and Maps do not.
- **[conflict]** The Tools page table (updated 2026-09-15) still lists async calling as supported on "Gemini 2.5 Flash Live Preview" only. The 3.8 model card and the capabilities page (newer) say NON_BLOCKING is the default on 3.8. Trust the model card, but test it on day one.
- **[conflict]** The Tools-page samples put `"scheduling"` *inside* the `response` dict. The API reference defines `scheduling` and `willContinue` as **top-level** `FunctionResponse` fields. The reference app uses the top-level form: `types.FunctionResponse(id=…, name=…, response=…, scheduling=…)`. Use that.

Sources: https://ai.google.dev/gemini-api/docs/live-api/tools, https://ai.google.dev/api/generate-content (FunctionResponse, Scheduling), https://ai.google.dev/api/live

### 1.5 Barge-in, VAD and interruption

- Automatic VAD is on by default. With `realtimeInputConfig.activityHandling = START_OF_ACTIVITY_INTERRUPTS` (the default), user speech cuts the model off. "The ongoing generation is canceled and discarded. Only the information already sent to the client is retained in the session history." The server sends `serverContent.interrupted: true` and cancels pending function calls.
- The client **must flush its playback queue immediately** when `interrupted` arrives. Best-practices says the same. With a backend proxy, forward an `interrupted` event to the phone, which clears its AudioWorklet ring buffer.
- Tuning (`automaticActivityDetection`): `startOfSpeechSensitivity`, `endOfSpeechSensitivity`, `prefixPaddingMs`, and `silenceDurationMs` (500–800 ms recommended; the server default is ~800 ms). Below 200 ms, utterances get fragmented.
- "Hybrid VAD": keep server VAD, and send `audioStreamEnd` from a client-side VAD for faster finalization. Optional.
- Send audio in 20–40 ms chunks (best practices; ADK says 50–100 ms is fine). Don't buffer around 1 s.
- Phone-specific: the agent plays through the phone speaker, about 20 cm from the mic. Use `getUserMedia({audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true}})` so the agent doesn't barge in on itself. Test this early. Self-interruption is the most common live-demo failure **[est]**.

Sources: https://ai.google.dev/gemini-api/docs/live-api/capabilities, https://ai.google.dev/api/live, https://ai.google.dev/gemini-api/docs/live-api/best-practices

### 1.6 Transcription

- `inputAudioTranscription: {}` and `outputAudioTranscription: {}` produce `serverContent.inputTranscription.text` / `outputTranscription.text` as incremental chunks. A single server event can carry several parts at once (audio and transcript), so iterate over all of them.
- `AudioTranscriptionConfig` accepts `custom_vocabulary`. The reference app uses it. Feed in locale words and any author or publisher names the pipeline has already read, so corrections like "that's the Folio Society edition" transcribe correctly.
- Billing: transcription text is charged **at the text output rate** on top of audio.
- Use the transcripts for the UI captions, the claim audit trail ("user stated: first edition" with a timestamp), and for tying corrections to frames (§3.4).

### 1.7 Ephemeral tokens (browser direct) versus a backend proxy

- Ephemeral tokens are created server-side with `client.authTokens.create({config:{uses:1, expireTime, newSessionExpireTime, liveConnectConstraints:{model, config}}})`. The defaults are 1 minute to *start* a session and 30 minutes of messages. `liveConnectConstraints` locks the model and config, so system instructions stay server-side. Tokens are v1beta only, and outside the SDK they go in `access_token=` on `…BidiGenerateContentConstrained`. Resumption reconnects (every ~10 min) can reuse the same token even with `uses:1`.
- The docs say client-to-server "generally offers better performance" because it skips a hop.
- **Why we still proxy:** on the LAN the extra hop costs a few milliseconds. Through a tunnel it is roughly 20–80 ms **[est]**, which is small next to model latency. In return the backend becomes the single place that:
  - receives every frame (for frame_ref),
  - executes tools and updates inventory,
  - injects detections,
  - records transcripts with timestamps,
  - enforces rate limits.

  The direct path would need a second phone-to-backend channel *plus* tool forwarding. That means more moving parts for no scoring gain.

Sources: https://ai.google.dev/gemini-api/docs/live-api/ephemeral-tokens, https://ai.google.dev/gemini-api/docs/live-api/get-started-websocket, https://ai.google.dev/gemini-api/docs/live-api

### 1.8 SDKs

- **Python `google-genai` 2.33.0** (PyPI, 2026-10-23, requires Python ≥3.10). Live entry point: `client.aio.live.connect(model=..., config=types.LiveConnectConfig(...))`. Methods: `send_realtime_input(audio=|video=|text=|audio_stream_end=)`, `send_client_content(turns=, turn_complete=)`, `send_tool_response(function_responses=)`, and `receive()`. The reference app uses this stack (FastAPI + google-genai).
- **JS `@google/genai` 2.31.0** (npm; `engines.node >=20`; ships a `./web` export). Live entry point: `ai.live.connect({model, config, callbacks:{onopen,onmessage,onerror,onclose}})`, with `sendRealtimeInput`, `sendClientContent` and `sendToolResponse`. Only needed for the direct-browser variant.
- Raw WSS endpoint: `wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key=…`

### 1.9 Google ADK bidi-streaming (`run_live`)

- **`google-adk` 2.13.0** (PyPI, 2026-10-23, Python ≥3.11). `runner.run_live(..., live_request_queue=LiveRequestQueue(), run_config=RunConfig(...))` yields events. The queue methods are `send_realtime(Blob)`, `send_content(...)`, the activity signals, and `close()`.
- ADK handles tool execution automatically, plus automatic session resumption, persistent sessions and multi-agent transfer. `RunConfig` carries `session_resumption`, `context_window_compression`, `input/output_audio_transcription`, `realtime_input_config`, `speech_config`, `save_live_blob`, and `max_llm_calls` (default 500).
- Gotcha from the ADK docs: in Python, `streaming_mode=StreamingMode.BIDI` "has no effect and fails silently". Calling `run_live()` is what makes the session bidirectional.
- Two very relevant features:
  - `tool.response_scheduling = types.FunctionResponseScheduling.WHEN_IDLE` (ADK ≥2.4).
  - **Streaming tools**: async-generator tools that `yield` updates. With an `input_stream: LiveRequestQueue` parameter, ADK feeds the tool the live video frames ("react to … a person appearing in a video frame"). This is the capture-monitor pattern of §3.3, packaged.
- Caveat: ADK's live "Supported models" page still lists only 2.5 Flash Live and 3.1 Flash Live. `LlmAgent.DEFAULT_LIVE_MODEL` is `gemini-live-2.5-flash-native-audio`. 3.8 should work as a model string through google-genai, but that is **unverified in ADK docs**.
- Verdict: ADK is a reasonable choice. The raw google-genai session is about 300 lines in the reference app and gives exact control over `willContinue` and scheduling, which we need for the coaching loop. **Recommend raw google-genai.** ADK can stay for the offline background agents if `01-reference-app.md` keeps them.

Sources: https://adk.dev/live/, https://adk.dev/live/models/, https://adk.dev/live/configuration/, https://adk.dev/live/tools/, https://adk.dev/live/audio-video/

### 1.10 Pricing (Gemini API paid tier, pricing page)

`gemini-3.8-live` / `-extended-thinking` / `3.1-flash-live-preview` share one price row:

| | per 1M tokens | per minute |
|---|---|---|
| Input text | $0.75 | |
| Input audio | $3.00 | **$0.005/min** |
| Input image/video | $1.00 | **$0.002/min** |
| Output text (incl. thinking) | $4.50 | |
| Output audio | $12.00 | **$0.018/min** |

The free tier is "free of charge", but its data is "used to improve our products". Billing rules (best practices):
- **Compounding.** "The API charges you per turn for all tokens present in the session context window". Past tokens are re-billed every turn up to the compression window.
- Transcription text is billed at the text output rate.
- Proactive audio (always on in 3.8) bills input "the entire time the Live API is listening".

**Cost per sweep [est]** (4.5 min connected, ~3 min with video, ~1.5 min of agent speech, ~30 model turns):
- Streaming floor: 4.5×$0.005 + 3×$0.002 + 1.5×$0.018 ≈ **$0.055**.
- With compounding: the context grows at ~95 tok/s and averages ~12k tokens over 30 turns. That is about 360k re-billed input tokens at a blended ~$1.5/M, so **~$0.40–0.55** in the worst case. A tighter compression window (e.g. trigger 25k) cuts this.
- Measure the real number from `usageMetadata` and report it, since the brief requires cost per sweep.

Source: https://ai.google.dev/gemini-api/docs/pricing, https://ai.google.dev/gemini-api/docs/live-api/best-practices

---

## 2. Alternatives

### 2.1 OpenAI Realtime API

- Models (developers.openai.com, Oct 2026):
  - **`gpt-realtime-2.1`**: "reasoning model with tool use". Inputs are text, audio and **image**. 128k context, 32k max output.
  - **`gpt-realtime-2.1-mini`**: also takes image.
  - `gpt-realtime-2`, `gpt-realtime-1.5`, `gpt-realtime`, `gpt-realtime-mini`.
  - Newest: **`gpt-live-1`**, a full-duplex voice model on `v1/live/sessions`, $0.05/min, with reasoning and tools "delegated" to a backend Responses agent. Its card says "**Unsupported modalities: image, video**", so it can't watch the camera at all.
- **Images, not video.** Each image is sent as a `conversation.item.create` user message with an `input_image` data URL. There is no video track. LiveKit notes "OpenAI Realtime API sends each video frame as an image message in the conversation context. Each frame consumes input tokens and counts against the context window." At 1 fps for 3 minutes, 180 frames × ~800 tokens **[est]** comes to ~144k tokens, which exceeds the 128k context and triggers truncation. In practice you send one frame every 2–3 s, or only on demand.
- Browser transport: WebRTC. The server mints `POST /v1/realtime/client_secrets` (ephemeral `ek_…`), or uses the "unified interface" via `POST /v1/realtime/calls` with an SDP exchange. Events travel over the data channel. A "sideband" server WebSocket can attach to the same session. Max session length is **60 minutes**.
- Injecting context: either `conversation.item.create` (a system or user item, then `response.create`), or **out-of-band** `response.create` with `conversation:"none"` and custom `input`, which handles side classification without polluting state.
- Pricing (per 1M tokens), with audio at 10 tok/s in and 20 tok/s out (realtime-costs guide):

  | | audio in | audio out | image in | cached audio |
  |---|---|---|---|---|
  | `gpt-realtime-2.1` | $32 | $64 | $5 | $0.40 |
  | `gpt-realtime-2.1-mini` | $10 | $20 | $0.80 | $0.30 |

- **Cost per sweep [est]**: about **$1.0** for 2.1 (90 images ≈ $0.36, audio ≈ $0.19, cached context re-billing ≈ $0.5) and about **$0.30** for mini.

Sources: https://developers.openai.com/api/docs/models/gpt-realtime-2.1, https://developers.openai.com/api/docs/models/gpt-realtime-2.1-mini, https://developers.openai.com/api/docs/models/gpt-live-1, https://developers.openai.com/api/docs/guides/realtime, https://developers.openai.com/api/docs/guides/realtime-conversations, https://developers.openai.com/api/docs/guides/realtime-webrtc, https://developers.openai.com/api/docs/guides/realtime-costs, https://developers.openai.com/api/docs/guides/realtime-server-controls

### 2.2 LiveKit Agents

- `livekit-agents` 1.8.9 and `livekit-plugins-google` 1.8.9 (Python 3.10–3.14). LiveKit Inference serves `google/gemini-3.8-live` (alias `google/gemini-live`) and `-extended-thinking`. The plugin also accepts `model="gemini-3.8-live"`, with a `media_resolution` option.
- Video: set `RoomOptions(video_input=True)`. By default it samples **1 fps while the user speaks and one frame every 3 s otherwise**, resized to 1024×1024 JPEG. Override with `video_sampler`. "Video input is passive and has no effect on turn detection."
- Pros: production WebRTC (better than WebSocket on lossy mobile networks), turn detection, and swappable STT/LLM/TTS pipelines.
- Cons: it needs a LiveKit server. That means LiveKit Cloud (another account and dependency) or `livekit-server` self-hosted. Full-res frame capture still has to be added next to the WebRTC video track, which is compressed and adaptive in resolution, so it is not a good source for OCR stills.

Sources: https://docs.livekit.io/agents/models/realtime/gemini.md, https://docs.livekit.io/agents/multimodality/vision/video.md

### 2.3 Pipecat

- `pipecat-ai` 1.14.0. `GeminiLiveLLMService` defaults to `models/gemini-3.8-live`, and there is an async-tool example (`realtime-gemini-live-async-tool.py`). `SmallWebRTCTransport` does peer-to-peer WebRTC with **no media server**, using aiortc and opencv-headless (`pipecat-ai[webrtc-video]`), so it is Windows-friendly. You can also build a cascade (STT → vision LLM → TTS) with frame sampling.
- Cons: another abstraction layer to explain line by line in the review call. Over a tunnel the WebRTC ICE path may need a TURN server, whereas WebSocket over cloudflared just works.

Sources: https://docs.pipecat.ai/api-reference/server/services/s2s/gemini-live.md, https://docs.pipecat.ai/api-reference/server/services/transport/small-webrtc.md, https://docs.pipecat.ai/pipecat/features/gemini-live.md

### 2.4 Comparison

| | Gemini 3.8 Live (raw, WS proxy) | OpenAI gpt-realtime-2.1 (WebRTC) | LiveKit + Gemini Live | Pipecat cascade (STT + vision LLM + TTS) |
|---|---|---|---|---|
| Video handling | Native frame stream, 1 fps, 70–280 tok/frame | Discrete images in context; context fills fast | Native (Gemini) via sampler | Periodic stills to a vision LLM; no "watching" |
| Voice-to-voice latency [est, measure] | ~0.6–1.2 s | ~0.5–1.0 s | Same as Gemini, plus WebRTC gains on bad Wi-Fi | ~1.2–2.5 s |
| Barge-in | Built in (`interrupted`) | Built in | Built in, plus a turn detector | Framework VAD |
| Async tools / push | NON_BLOCKING + `willContinue` + scheduling | Items + `response.create`, out-of-band responses | Plugin passes through | Full control (you own the LLM loop) |
| Cost / sweep [est] | **$0.06–0.55** | ~$1.0 (mini ~$0.30) | Gemini cost + LiveKit Cloud minutes | ~$0.10–0.30 |
| Complexity on Win11, no Docker | Low: FastAPI + 1 WS | Medium: WebRTC + ephemeral keys + sideband | Medium–high: LiveKit server/cloud | Medium–high |
| Matches reference | Yes (same model) | No | Partly | No |

**Pick: Gemini 3.8 Live behind a FastAPI WebSocket proxy.** It has the cheapest video, native frame streaming, the documented async-push primitives, the fewest moving parts, and it stays close to the reference, which the brief tells us to start from.

---

## 3. Recommended architecture pattern

### 3.1 Split the roles

```
                 ┌──────────── live loop (latency-critical, cheap) ───────────┐
 phone mic 16k ──► backend ──► Gemini 3.8 Live ──► 24k audio ──► phone speaker
 phone 1 fps ≤768px ─┘          ▲  tool calls / SILENT|WHEN_IDLE|INTERRUPT
                                │
 phone stills 2–4 fps ≥1280px ──► backend frame store (disk, frame_ref = file+ts)
                                │       ├─ quality scorer (CV, every frame, <20 ms)
                                │       ├─ coverage tracker (shelf/region map)
                                │       └─ fast spine/item detector (sampled, ~1 Hz)
                                │                 │
                                └──── capture monitor ◄─┘  (debounced, prioritized)
                                         │
                                         └──► UI events (inventory.upsert, capture_alert)
 sweep end ──► background agents: identify → price → measure → review → packet (≤5 min)
```

- **Live model.** Persona and conversation, coaching phrasing, short questions ("original or print?"), taking corrections, and the final readback. It **never** produces titles, counts, dimensions or prices as facts. The system instruction says so, and tool schemas only accept user statements. (That covers the brief's "keep a model from inventing facts when money is attached" and "totals computed in code".)
- **Backend.** The single source of truth for frames, inventory, coverage, alerts and corrections. Everything the live model says about the inventory comes from tool responses.

### 3.2 Live capture-quality signals (backend, per still)

All thresholds are **relative to the session's own running statistics** (z-score against the rolling median), not fixed constants, so they carry over to an unseen room and phone:
- **Blur:** variance of the Laplacian on a downscaled grayscale crop of the shelf region (OpenCV). Flag when it is below about the 25th percentile of this session's frames and below the level where the spine detector's text confidence drops. Calibrate both against ground truth.
- **Glare:** the fraction of near-saturated, low-saturation pixels (HSV) inside detected spine boxes. A cluster of them means "tilt the phone".
- **Motion:** `DeviceMotionEvent.rotationRate` sent from the phone (iOS needs `requestPermission()`) plus frame-to-frame optical-flow magnitude. Use these for "slow down".
- **Occlusion / too far:** the median detected spine height in pixels falls below the minimum the OCR needs, or a large foreground blob covers part of the frame.
- **Missed shelves:** a coverage map built from detected shelf units and rows, where each row is "seen sharp" or not. When the user pans past a unit that has unread rows, raise `missed_region` *while they are still near it* (INTERRUPT priority).

### 3.3 Pushing server-side detections into the live session

Ranked options, all documented:

1. **Generator tool (primary).** Declare `start_capture_monitor()` as NON_BLOCKING and tell the model in the system instruction to call it once, right after the greeting. The backend keeps that call ID open and sends `FunctionResponse(id, name, response={...}, will_continue=True, scheduling=X)` for each event:
   - `SILENT`: inventory facts ("logged 14 books on unit A row 2, 3 unreadable"). These go into context so the model can answer "how many so far?" without speaking. They also re-assert state after compression.
   - `WHEN_IDLE`: coaching that can wait for a pause ("frames on unit A row 3 are blurry: ask them to slow down and hold for a second").
   - `INTERRUPT`: urgent, time-bound items (the user is leaving a unit with unread rows, or an art piece needs the print-or-original question before they walk away).
   - On reconnect or resume, the model must call it again (or ADK's streaming tool does this for you). Send a final response with `will_continue=False, scheduling=SILENT` to close it.
   - **[verify day 1]** That `willContinue` streams are honoured in Live on 3.8. They are documented in the generic FunctionResponse reference and in ADK streaming tools. If they are not, fall back to option 2.
2. **Short-lived NON_BLOCKING tool per event.** The model calls `get_capture_feedback()` periodically, or reacts to it. This is clumsier, because the model has to poll.
3. **`sendClientContent` with a role-user text** like `"[capture-monitor] unit A row 3 blurry"`:
   - `turnComplete=false` adds context, and per the 3.8 model card the server waits rather than responding.
   - `turnComplete=true` forces a response but **unconditionally interrupts**. Use it only for the end-of-sweep hand-off ("The sweep is over; tell the user processing has started").
   - **[conflict]** The API reference says any ClientContent "will interrupt any current model generation", while the 3.8 card says only `turn_complete=true` does. Test it.
4. **`sendRealtimeInput({text})`.** It is treated like user activity, and ordering against audio is "not guaranteed". Avoid it for system notes.
5. You cannot change system instructions mid-session. Setup is the first message only.

Anti-nagging rules (backend):
- At most 1 spoken coaching item per ~8 s.
- Dedupe by region.
- Never coach while the user is speaking (track VAD via `inputTranscription` arrival).
- Drop stale alerts once the user has moved on (the coverage map says the region is now covered).

### 3.4 Voice corrections ("that is a first edition", "skip that shelf")

Live-model tools (NON_BLOCKING, fast, return an ack):

| Tool | Args | Backend effect |
|---|---|---|
| `record_user_statement` | `subject` ("current_view" or item id), `kind` (edition, signed, print_or_original, ownership, condition, other), `value` (verbatim) | Ties the statement to the frames from [t−4 s, t], where t is when the utterance transcript arrived. Marks matching items. `edition: first`, `signed` and `original` go to **needs_appraisal**, never auto-priced. Stored as evidence: "user statement + timestamp". |
| `exclude_region` | `description`, optional `shelf_id` | Marks a coverage region as excluded, so its items go to `excluded_from_totals` with the reason "user: not mine". |
| `set_locale` | `country`, `currency` (ISO codes) | Locale for pricing. Confirmed in the greeting. |
| `get_sweep_status` (BLOCKING ok) | none | Counts, coverage gaps and pending questions, so the model doesn't make them up. |
| `end_sweep` | none | Stops the stills and starts the background agents. The UI button does the same. |

The live model sees only 1 fps, so it may get "current view" wrong. The backend resolves the actual item from the frames and timestamp and confirms back ("noted, *The Hobbit* on unit B row 1 is a first edition, flagged for appraisal"). Note the name in that confirmation comes from a SILENT tool response, not from the live model reading the spine.

### 3.5 Post-sweep

`end_sweep` puts the session into processing state and stops video, which also slows context growth. Background stages push `pipeline.progress` to the UI. When the packet is ready, the backend sends a `FunctionResponse` on the still-open monitor call with `scheduling=INTERRUPT` (or a `sendClientContent` with turnComplete=true), containing the **code-computed** summary numbers. The model reads them back. If GoAway arrives or the connection has passed 8 minutes, reconnect with the resumption handle first.

---

## 4. Phone access (HTTPS) and camera capture

### 4.1 HTTPS options

`getUserMedia` "can only be used in secure contexts… HTTPS or file:///… or localhost". Over plain-HTTP LAN, `navigator.mediaDevices` is `undefined` (MDN).

| Option | How | Pros | Cons |
|---|---|---|---|
| **cloudflared quick tunnel** (2026.10.3) | `winget install --id Cloudflare.cloudflared` then `cloudflared tunnel --url http://localhost:8000` gives `https://<random>.trycloudflare.com` | No account or domain needed; a valid public cert; works on whatever network the evaluators' phone uses | No uptime guarantee; **200 in-flight requests max** (429 after); **no SSE** (use WebSockets, which work in practice); URL changes on every run; adds WAN latency, and uplink of the stills goes via Cloudflare |
| **ngrok free** | `ngrok http 8000` gives a `*.ngrok-free.app` dev domain | One static dev domain, no endpoint timeout | Interstitial warning page on all HTML browser traffic (one tap, then a 7-day cookie); **1 GB/month data out**, 20k HTTP requests/month, account required |
| **mkcert + LAN IP** (v1.4.4, 2022, stable) | `mkcert -install`, `mkcert 192.168.1.23 localhost`, then uvicorn `--ssl-keyfile/--ssl-certfile` | Lowest latency; frames never leave the LAN | Root CA must be installed on the phone (iOS: install profile **and** enable full trust; Android: user CA, which Chrome accepts); IP changes with the network; won't work on the evaluators' network without redoing it |
| Chrome flag | `chrome://flags/#unsafely-treat-insecure-origin-as-secure` = `http://192.168.x.x:8000` on Android Chrome | Zero setup for solo dev | Android Chrome only; not reproducible from the README |

**Recommendation:** make cloudflared the README default, since a stranger can run it within the 15-minute target. Document mkcert as the low-latency LAN option. Keep all realtime traffic on **one WebSocket** to stay far below the 200 in-flight limit. Send stills as binary WS frames, not one POST per frame.

Sources: https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/, https://ngrok.com/docs/pricing-limits/free-plan-limits, https://github.com/FiloSottile/mkcert, https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia

### 4.2 Rear camera, resolution and full-res stills

```js
const stream = await navigator.mediaDevices.getUserMedia({
  audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
  video: { facingMode: { ideal: "environment" }, width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 30 } }
});
const track = stream.getVideoTracks()[0];
const actual = track.getSettings();
```

- Use `facingMode: {ideal:"environment"}` rather than `exact`. `exact` throws `OverconstrainedError` on laptops and in desktop testing.
- `ideal` width and height have "gravity". Always read `track.getSettings()` back and log the real resolution in the packet's `sweep.device`, along with `track.getCapabilities()` (focus, zoom, torch where supported) for diagnosis.
- Stills from the live track: draw the `<video>` element onto a canvas sized to `videoWidth × videoHeight`, then `canvas.toBlob(cb, "image/jpeg", 0.9)`. Or use `ImageCapture.grabFrame()`, which returns an ImageBitmap without the `<video>` element. Browser support from MDN browser-compat-data 8.1.7:
  - `ImageCapture`: Chrome Android 59+, Safari iOS 18.4+, Firefox Android none.
  - `grabFrame`: iOS 26+.
  - `takePhoto`: Chrome Android 60+, iOS 18.4+.
  - `MediaStreamTrackProcessor`: Chrome Android 94+, iOS 18+.
- `ImageCapture.takePhoto()` returns full **sensor** resolution (often 12 MP), but each call is slow and can re-trigger focus or exposure **[est; test]**. Use it for rare "hero" stills such as a paused shelf close-up, not for 2–4 fps.
- Run two encoders off the one track: a 1 fps 768 px JPEG at q≈0.7 for Gemini, and a 2–4 fps 1280–1920 px JPEG at q≈0.85–0.9 for the pipeline. Encode in `OffscreenCanvas.convertToBlob` (iOS 16.4+) to keep the main thread free.
- Uplink budget: 3 fps × ~200 KB ≈ 600 KB/s ≈ 5 Mbps **[est]**. That is fine on LAN Wi-Fi. Over a tunnel, drop to 2 fps and adapt to WS `bufferedAmount`.
- Other phone requirements:
  - `navigator.wakeLock.request("screen")` (iOS 18.4+) so the screen doesn't sleep mid-sweep.
  - `AudioWorklet` for mic capture (resample to 16 kHz PCM16) and for playback (a 24 kHz ring buffer you can flush on `interrupted`).
  - iOS Safari may lower playback volume or route audio to the earpiece while the mic is active. **Develop and demo on Android Chrome**, and treat iOS as best-effort.

---

## 5. Recording the one-take demo (agent voice audible)

- **Android built-in recorder** (Quick Settings → Screen record, Android 11+): choose the audio option that includes **device audio *and* microphone**. Risk: Android playback capture only takes audio with usage `USAGE_MEDIA`, `USAGE_GAME` or `USAGE_UNKNOWN` (Android AV-capture docs). While a page holds the mic with echo cancellation, Chrome likely plays audio in voice-communication mode **[unverified]**, so device audio alone may record silence for the agent. With microphone enabled, the agent voice is picked up acoustically from the speaker. **Do a 20 s test recording before the real take.**
- **scrcpy 5.0.1 on the Windows PC** (USB, or Wi-Fi via `adb tcpip`): `scrcpy --record=demo.mp4 --audio-dup` (implies `--audio-source=playback`, Android 13+) keeps sound on the phone while recording on the PC. Apps "can opt out", which carries the same call-mode risk as above. `--audio-source=output` captures everything but **mutes the phone**, so it is unusable while walking. Use `--audio-source=mic` as a fallback.
- **iOS Control Center recorder**: long-press Record, then turn on Microphone. Apple notes "Some apps might not allow you to record audio." Test whether Safari's agent audio comes through.
- **In-app backup (recommended in addition):** a `MediaRecorder` on a `MediaStream` that combines the camera track with a WebAudio mix (mic + agent output via `MediaStreamAudioDestinationNode`). Upload it in chunks to the backend. This doesn't capture the inventory UI, but it gives an exact audio and video record for the failure log and frame_ref audits. Don't submit a server-composited video as "the demo", because the brief bans editing and stitching.
- **Second-camera option:** film over the shoulder with another phone. This captures the screen, the speaker audio and the room in one honest take, and the evaluators see the real space.

Sources: https://developer.android.com/media/platform/av-capture, https://github.com/Genymobile/scrcpy/blob/master/doc/audio.md, https://support.apple.com/en-us/102653, https://support.google.com/android/answer/9075928

---

## 6. Minimal concrete architecture

### 6.1 Components

1. **Phone web app** (Vite + plain TypeScript, served by the backend as static files):
   - capture: getUserMedia, an AudioWorklet mic encoder, a frame scheduler, DeviceMotion
   - playback: AudioWorklet at 24 kHz with flush
   - UI: live inventory list, coverage strip, alert toasts, captions, End-sweep button
   - optional MediaRecorder backup
2. **Backend** (Python 3.12, FastAPI + uvicorn, one process):
   - `ws_gateway`: phone WS ⇄ session actor
   - `live_bridge`: a google-genai Live session with compression, resumption and GoAway handling, the tool router and the capture-monitor generator
   - `frame_store`: writes every still to `runs/<sweep_id>/frames/<seq>_<t_ms>.jpg` plus a JSONL index; this is the source of frame_ref
   - `quality`: OpenCV blur, glare and motion scores (synchronous, cheap)
   - `live_detect`: sampled fast detector for spines, items and shelf rows, plus the coverage map (async, bounded concurrency)
   - `inventory`: the authoritative state; emits `inventory.upsert`
   - `pipeline`: post-sweep identify → price → measure → review → packet (doc 02 / doc 01)
3. **Gemini Live**: `gemini-3.8-live` over WSS, server to server.
4. **Tunnel**: cloudflared quick tunnel to `localhost:8000`.

### 6.2 Phone ⇄ backend protocol (one WSS, `/ws/sweep`)

Binary frames carry a 1-byte kind, a 4-byte big-endian header length, a JSON header, then the payload:

| kind | dir | header | payload |
|---|---|---|---|
| `0x01` mic | ↑ | `{seq, t_ms}` | PCM16 LE mono 16 kHz, 40 ms (1280 bytes) |
| `0x02` live frame | ↑ | `{seq, t_ms, w, h}` | JPEG ≤768 px, 1 fps (forwarded to Gemini) |
| `0x03` still | ↑ | `{seq, t_ms, w, h, motion:{rot_dps, acc}, orient}` | JPEG 1280–1920 px, 2–4 fps (frame store) |
| `0x81` agent audio | ↓ | `{seq}` | PCM16 LE mono 24 kHz |

`t_ms` = `performance.timeOrigin + performance.now()`, and the backend logs the clock offset at `hello`.

JSON text messages:

```
↑ {"type":"hello","sweep_id":"…","device":{"ua":"…","track":{…getSettings()},"caps":{…}},"locale_hint":"en-IN"}
↑ {"type":"control","action":"start_sweep|end_sweep|mute|unmute"}
↑ {"type":"playback","state":"playing|idle"}
↓ {"type":"session","state":"connecting|live|reconnecting|processing|summary|closed","model":"gemini-3.8-live"}
↓ {"type":"interrupted"}
↓ {"type":"transcript","role":"user|agent","text":"…","final":false,"t_ms":…}
↓ {"type":"inventory.upsert","item":{"id":"b_0042","kind":"book|unidentified_book|item","label":"…|null","shelf":"A-3","status":"detected|identified|needs_appraisal|excluded","confidence":0.0,"frame_ref":"frames/0193_1729876543210.jpg"}}
↓ {"type":"capture_alert","code":"blur|glare|too_fast|occluded|too_far|missed_region","region":"A-3","severity":"info|warn|urgent","say":"…"}
↓ {"type":"coverage","regions":[{"id":"A-3","state":"sharp|partial|missing|excluded"}]}
↓ {"type":"question","id":"q_7","ref":"i_0005","text":"Is that portrait an original or a print?","state":"asked|answered"}
↓ {"type":"pipeline.progress","stage":"identify|price|measure|review|packet","done":12,"total":80}
↓ {"type":"packet_ready","json":"/runs/<id>/claim_packet.json","report":"/runs/<id>/report.html"}
```

### 6.3 Backend ⇄ Gemini Live

Setup (`types.LiveConnectConfig`):
- `response_modalities=["AUDIO"]`
- `system_instruction` = persona + sweep rules + "never state titles, prices or measurements unless they arrive in a tool response"
- `speech_config.voice_config.prebuilt_voice_config.voice_name` (env var)
- `input_audio_transcription={custom_vocabulary:[…]}`, `output_audio_transcription={}`
- `context_window_compression={trigger_tokens:40000, sliding_window:{target_tokens:16000}}`
- `session_resumption={handle: last_or_None}`
- `media_resolution=MEDIA_RESOLUTION_LOW` (raise to MEDIUM/HIGH only if coaching quality needs it)
- `realtime_input_config={automatic_activity_detection:{silence_duration_ms:600}, activity_handling:START_OF_ACTIVITY_INTERRUPTS}`
- `tools=[{function_declarations:[start_capture_monitor, record_user_statement, exclude_region, set_locale, get_sweep_status(behavior=BLOCKING), end_sweep]}]`

Do **not** send `thinking_config`, `enable_affective_dialog` or `proactivity` (3.8 rejects them).

Upstream: `send_realtime_input(audio=Blob(pcm,"audio/pcm;rate=16000"))`, `send_realtime_input(video=Blob(jpeg,"image/jpeg"))` at ≤1 fps, `send_tool_response(function_responses=[FunctionResponse(id,name,response,will_continue,scheduling)])`, and `send_client_content(turns=…, turn_complete=…)` (end-of-sweep and readback only).

Downstream handlers: `server_content.model_turn.parts[].inline_data` (audio, so forward 0x81), `server_content.input_transcription` / `output_transcription`, `server_content.interrupted` (forward and drop queued audio), `turn_complete`, `generation_complete`, `tool_call.function_calls` (route to an asyncio task), `tool_call_cancellation.ids` (cancel the task), `go_away.time_left` (reconnect), `session_resumption_update` (store the handle), and `usage_metadata` (accumulate for cost per sweep).

Latency instrumentation, logged per turn:
- t(user speech end ≈ last inputTranscription chunk) → t(first agent audio chunk)
- t(still captured) → t(capture_alert emitted)
- t(detection) → t(inventory.upsert on screen)

These figures feed the brief's "latency per stage".

### 6.4 Versions to pin (checked on PyPI and npm, 2026-10-25)

Python 3.12 (`requirements.txt`). All have Windows cp312, abi3 or pure wheels:
```
google-genai==2.33.0
fastapi==0.162.0
uvicorn[standard]==0.54.0
websockets==17.3
pydantic==2.14.1
numpy==2.5.4
opencv-python-headless==5.0.0.93
pillow==13.0.0
python-multipart==0.0.32
```
Optional: `google-adk==2.13.0` (≥3.11) if the background agents use ADK. `pipecat-ai==1.14.0` and `livekit-agents==1.8.9` were evaluated but are not recommended.

Node 26 (frontend build only):
```
vite@8.3.5                (engines: ^20.19 || >=22.12)
typescript (latest 5.x/6.x at install time)
@google/genai@2.31.0      (only for a direct-browser variant; engines node >=20)
```

Tools: `cloudflared 2026.10.3`, `mkcert v1.4.4`, `scrcpy v5.0.1` (for recording).

---

## 7. Open risks and day-1 verification checklist

1. 3.8 Live honours `will_continue=True` streams with per-response `scheduling`. If not, use the per-event tool or `sendClientContent(turnComplete=false)` fallback.
2. `sendClientContent(turnComplete=false)` does not interrupt speech on 3.8 **[conflict]** between the API reference and the model card.
3. The session survives more than 2 minutes with the camera on, once compression is enabled. Script a 4-minute soak test with 1 fps frames.
4. The agent does not barge in on itself through the phone speaker (AEC). If it does, raise `startOfSpeechSensitivity` to LOW, or use push-to-talk as a fallback.
5. Screen-recording audio captures the agent voice (20 s test).
6. cloudflared WS stays up for 10+ minutes with 3 fps stills. Watch `bufferedAmount`, and send keepalive pings (Cloudflare's documented proxy idle timeouts are 125 s for read and 900 s for idle).
7. Measured cost per sweep from `usageMetadata` against the [est] figures above.

---

## Sources

Google / Gemini:
- https://ai.google.dev/gemini-api/docs/live-api (overview, specs; updated 2026-10-06)
- https://ai.google.dev/gemini-api/docs/live-api/capabilities (model comparison, VAD, transcription, media resolution, limits)
- https://ai.google.dev/gemini-api/docs/live-api/session-management (2-minute / 15-minute / 10-minute limits, compression, resumption, GoAway)
- https://ai.google.dev/gemini-api/docs/live-api/tools (function calling, async, scheduling)
- https://ai.google.dev/gemini-api/docs/live-api/ephemeral-tokens
- https://ai.google.dev/gemini-api/docs/live-api/get-started-websocket
- https://ai.google.dev/gemini-api/docs/live-api/best-practices (billing, chunking, interruption)
- https://ai.google.dev/gemini-api/docs/live-api/thinking
- https://ai.google.dev/gemini-api/docs/models and https://ai.google.dev/gemini-api/docs/models/gemini-3.8-live
- https://ai.google.dev/gemini-api/docs/media-resolution
- https://ai.google.dev/gemini-api/docs/pricing
- https://ai.google.dev/api/live (ContextWindowCompressionConfig, SlidingWindow, ActivityHandling, TurnCoverage, ToolCallCancellation, SessionResumptionUpdate, GoAway)
- https://ai.google.dev/api/generate-content (FunctionResponse.willContinue, Scheduling)
- https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/live-api
- https://adk.dev/live/, https://adk.dev/live/models/, https://adk.dev/live/configuration/, https://adk.dev/live/tools/, https://adk.dev/live/audio-video/

OpenAI:
- https://developers.openai.com/api/docs/guides/realtime
- https://developers.openai.com/api/docs/guides/realtime-conversations (image input, out-of-band responses, 60-minute max)
- https://developers.openai.com/api/docs/guides/realtime-webrtc
- https://developers.openai.com/api/docs/guides/realtime-costs (tokens per second of audio, truncation)
- https://developers.openai.com/api/docs/guides/realtime-server-controls
- https://developers.openai.com/api/docs/models/gpt-realtime-2.1, https://developers.openai.com/api/docs/models/gpt-realtime-2.1-mini, https://developers.openai.com/api/docs/models/gpt-live-1

Frameworks:
- https://docs.livekit.io/agents/models/realtime/gemini.md, https://docs.livekit.io/agents/multimodality/vision/video.md
- https://docs.pipecat.ai/api-reference/server/services/s2s/gemini-live.md, https://docs.pipecat.ai/api-reference/server/services/transport/small-webrtc.md

Phone, tunnels, recording:
- https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
- MDN browser-compat-data 8.1.7 (2026-10-22), via https://unpkg.com/@mdn/browser-compat-data/data.json
- https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/
- https://developers.cloudflare.com/fundamentals/reference/connection-limits/
- https://ngrok.com/docs/pricing-limits/free-plan-limits
- https://github.com/FiloSottile/mkcert
- https://developer.android.com/media/platform/av-capture
- https://github.com/Genymobile/scrcpy/blob/master/doc/audio.md
- https://support.apple.com/en-us/102653
- https://support.google.com/android/answer/9075928

Package versions: pypi.org/pypi/<pkg>/json and registry.npmjs.org/<pkg>/latest, queried 2026-10-25.
