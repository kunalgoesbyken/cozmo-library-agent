# Spike results (Gate G0)

Run on 2026-10-26. Scripts: `tools/spike/live_soak.py`, `tools/spike/box_bakeoff.py`, `tools/spike/camera-probe.html`. Run with `PYTHONPATH=. .venv/Scripts/python.exe tools/spike/<script>.py` (the package is not installed). Raw logs and overlay images stay in gitignored `_scratch/`.

## Verdicts

| Item | Result |
|---|---|
| Capture path | **Path B** (iPhone, Safari, no WebXR): a plain `getUserMedia` page, stills by canvas or ImageCapture where Safari offers it. `camera-probe.html` replaces the planned xr-probe. |
| Live session with video + audio | Works on `gemini-3.8-live` with compression and resumption configured. No GoAway, no 429, no disconnect in 4 minutes of 0.5 fps video + 16 kHz audio. |
| Injection method that worked | NON_BLOCKING `capture_monitor` tool whose response streams with `will_continue=True` and a per-response `scheduling` (SILENT / WHEN_IDLE / INTERRUPT). All three were accepted and behaved as documented. |
| `send_client_content(turn_complete=False)` | Does **not** interrupt. Sent 1 s into a spoken answer, the model finished the whole answer (95 audio chunks after the send, no `interrupted` flag). |
| Resumption | Works. A fresh connection with the stored handle remembered the shelf contents. |
| Compression at 12000 / 6000 | **Not verified.** See "Blocked" below. |
| Box detector | All four detectors land within about 10% of the hand count. RF-DETR is 100-5000x faster than the LLMs and free. |
| Reference app | Boots and serves. Has no compression or resumption. Camera and 2-minute behaviour need a human. |

## Live soak (`gemini-3.8-live`)

Config: `response_modalities=["AUDIO"]`, `context_window_compression(trigger_tokens=12000, sliding_window.target_tokens=6000)`, `session_resumption`, input and output transcription, one NON_BLOCKING function. Frames 768 px JPEG every 2 s (panned crops of a shelf photo so frames differ), PCM 16 kHz in 100 ms chunks (silence plus two SAPI-synthesised prompts). google-genai 2.33.0.

Four-minute run (7 usage reports, one of them on the resumed connection):

| t (s) | prompt tokens | text / audio / image |
|---|---|---|
| 6 | 1,052 | 422 / 222 / 329 |
| 44 | 3,662 | 518 / 277 / 2,569 |
| 91 | 9,340 | 677 / 637 / 7,175 |
| 125 | 9,471 | 712 / 719 / 7,175 |
| 159 | 9,622 | 743 / 823 / 7,175 |
| 189 | 9,926 | 434 / 222 / 8,225 |
| 251 (resumed) | 5,617 | 440 / 222 / 4,340 |

- A frame costs about 264 tokens. At 0.5 fps that is up to about 8k tokens per minute, so the 12k trigger is reached after roughly 90-100 s of video in the context.
- Average billed prompt tokens: 12.2k per minute (sum of per-turn `prompt_token_count` over turns).
- Spend: USD 0.057 for the 4-minute run including the resume check, about USD 0.014 per minute with 7 turns. Real use has more turns, so this is a floor. The research estimate of USD 0.40-0.55 per sweep assumed about 30 turns.
- Image tokens stayed at 7,175 between t=91 and t=159 even though frames kept arriving. Turns triggered by a tool response did not appear to pick up new frames; the count grew again on later turns. Treat this as an observation, not an established rule.
- Around t=189 text and audio history shrank back to the first-turn values (434 / 222), the model called `capture_monitor` a second time (the first call had been dropped from its context), and a text prompt sent at t=187 got no answer. Max prompt was 9,926 (below the 12,000 trigger), so this is probably not the documented compression. Cause unknown. Consequence for the build: the model may forget it already started the monitor, so the backend should tolerate a repeated call and re-assert state.
- No GoAway in 4.5 minutes. No 429.

Scheduling, as observed (model speech within 10 s of the tool response):

| scheduling | Result |
|---|---|
| SILENT | No audio. Later "how many logged?" was answered "fourteen", so the payload is in context. |
| WHEN_IDLE | Model spoke the coaching sentence. |
| INTERRUPT | Model spoke the urgent sentence. The `interrupted` flag did not fire because nothing was being said at that moment; an overlap case is untested. |
| final response, `will_continue=False`, SILENT | Accepted, no audio. |

### Blocked: compression firing

Two attempts to push context past 12k cheaply with a 13k-token text flood (one `send_client_content` list of alternating turns, one big `send_realtime_input(text=...)`) left the session silent: no usage report, no answer to later questions, and a 1007 "invalid argument" close at shutdown. A third earlier attempt (several `send_client_content` user turns) did the same. I stopped to protect the budget. Needs a human or a later task: a real 3-4 minute run with a user turn each ~20 s so the context crosses 12k, then check that `prompt_token_count` falls toward 6k and that GoAway/resumption continue to work. The scenarios `interrupt` and `compress` in `live_soak.py` reproduce the probes (`compress` currently reproduces the failure).

## Box bake-off

Photos (Wikimedia Commons, thumbnails at 1920 px width, stored in `_scratch/shelf/`, not committed):

| name | file | licence | author |
|---|---|---|---|
| blyton | https://commons.wikimedia.org/wiki/File:Bookshelf_of_Enid_Blyton_works.jpg | CC BY 3.0 | Blytonite |
| german | https://commons.wikimedia.org/wiki/File:Part_of_a_bookshelf_with_mixed_books,_mostly_in_German_(1).jpg | CC BY-SA 4.0 | Roman Eisele |
| library | https://commons.wikimedia.org/wiki/File:Part_of_a_bookshelf_containing_books_by_Latin_poets_and_historians_(1).jpg | CC BY-SA 4.0 | Roman Eisele |

Hand counts (read from the images; blyton is the least certain, plus or minus 2):
- blyton 55: 43 upright (11 + 21 Famous Five + 5 small + 6) plus 12 lying or stacked.
- german 22: 20 upright, 1 partial at the right edge, 1 flat book on top.
- library 19: includes the two partial books at the edges and one fore-edge-out book.

| photo (hand) | RF-DETR nano (thr 0.5) | gemini-3.5-flash-lite | gemini-3.1-flash-lite | gemini-3.8-flash |
|---|---|---|---|---|
| blyton (55) | 51 | 49 | 48 | 52 |
| german (22) | 21 | 22 | 22 | 21 |
| library (19) | 19 | 19 | 19 | 19 |
| blyton boxes agreeing with RF-DETR at IoU 0.5 | - | 46 | 46 | 50 |

Latency (one call each):

| | blyton | german | library |
|---|---|---|---|
| RF-DETR (DirectML, warm, model run only) | 22 ms | 114 ms | 23 ms |
| gemini-3.5-flash-lite | 6.2 s | 3.8 s | 3.7 s |
| gemini-3.1-flash-lite | 4.9 s | 3.1 s | 3.2 s |
| gemini-3.8-flash | 24.7 s | 15.5 s | 11.0 s |

Cost per call: lite models about USD 0.0004-0.0008, 3.8-flash USD 0.005-0.011 (thinking tokens dominate: 1.4-3.5k output tokens).

Overlays showed the same pattern for all detectors: tight boxes on upright books, merged or double boxes in the dense Famous Five block and the lying stacks. The LLM boxes on the Blyton thin spines overlap each other more than RF-DETR's. RF-DETR at threshold 0.3 gave 62 on blyton (over-count), 0.7 gave 37 (under-count); 0.5 is the better default on this set. `onnxruntime` selected `DmlExecutionProvider` first; CPU fallback is in the provider list. Weights `Geraldine/rf-detr-nano-bookshelf` (Apache-2.0 per its card) downloaded into `_scratch/models/`; input 384x384, ImageNet normalisation, outputs `dets` (cx, cy, w, h normalised) and `labels` logits; score is the max sigmoid.

## Reference app (`insurance_claim_live_agent_team`)

- Own venv `_scratch/refapp-venv`, `pip install -r requirements.txt` succeeded (google-adk 2.13.0, google-genai 2.33.0, fastapi 0.162.0, uvicorn 0.54.0).
- Started with the key in the environment only (`GOOGLE_API_KEY`, `GOOGLE_GENAI_USE_VERTEXAI=False`), `python -m uvicorn live_demo.server:app --host 127.0.0.1 --port 4177`. `GET /` returned 200 (6,188 bytes), `/app.js` 200, `/openapi.json` 200. Stopped afterwards.
- Source check: `live_tools.py::build_live_config` sets no `context_window_compression` and no `session_resumption`, and nothing in `live_demo/*.py` handles GoAway. Its `realtime_input_config` uses `START_OF_ACTIVITY_INTERRUPTS`. So the "dies at 2 minutes with camera on" claim cannot be avoided by this app as shipped; the README "threw away" section can state that compression, resumption and GoAway handling were added.
- Camera / 2-minute behaviour: **needs human run**.

## Surprises

- `httpx` gets HTTP 403 from the Wikimedia API (with any User-Agent, even a plain curl one); `urllib` and `curl` get 200. The bake-off script fetches with `urllib`.
- The Commons search returned a "library" photo that was an auction slip and a stack photo too dense to hand-count; photos were chosen by viewing them.
- A text flood over `send_client_content` or realtime text silently stalls the Live session instead of erroring until close.
- 3.8-flash (thinking) is 3-5x slower than the lite models and gains at most a few boxes.

## Spend

| step | USD |
|---|---|
| box bake-off (12 calls: 9 Gemini plus local) | 0.0255 |
| live probe 45 s | 0.0138 |
| live soak 4 min + resume | 0.0566 |
| interrupt probe run | 0.0132 |
| two failed compression runs | about 0.0015 |
| **ledger total (`runs/spend.jsonl`)** | **0.1106** of the 0.15 cap |

`gemini-3.8-flash` used exactly 3 calls.

## Needs human

1. Reference app with the camera on for 3+ minutes: does it die at 2 minutes.
2. `camera-probe.html` on the iPhone over HTTPS (cloudflared): record `getSettings()`, still-grab times (canvas vs ImageCapture, ImageCapture is not expected on iOS Safari), wake-lock support, AudioWorklet round-trip.
3. Compression firing at 12000/6000 with a long real session (see Blocked above), and the INTERRUPT scheduling while the model is mid-sentence.
4. Real spine photos of the user's own shelf to repeat the bake-off.
