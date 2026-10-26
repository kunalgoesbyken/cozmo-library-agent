
## Live server and voice

Run on 2026-10-26 against the real server (real genai bridge, real pipeline wiring, no `[ml]` extra so every keyframe went to Gemini vision), driven by `tools/e2e_ws_client.py`. Task spend about USD 0.201 of the 0.20 cap (live 6-minute sweep about 0.137, short abrupt-disconnect run and three short voice probes the rest).

Worked:
- Token, Origin, preauth size, binary-before-hello, garbage JSON, second concurrent connection and a 12 MB binary message are all rejected with 1008 (403 for a bad Origin at the handshake, 1009 for the oversized message). The first session survived the second connection, and the server accepted a new session afterwards.
- Greeting audio and transcripts arrive (first audio about 10 s after connect, 0x81 frames), user speech is transcribed, `interrupted` arrives about 0.4 s after barge-in and agent audio stops within that second (0 frames in the following second against 19 before it).
- A 6-minute session produced `session` states connecting, live, processing, summary with no crash; `end_sweep` called by the agent from the spoken "I am done" ran the pipeline: `pipeline.progress`, `inventory.upsert`, `packet_ready`; `claim_packet.json` and `report.html` exist; both URLs return 200 with `?t=` and 401 without it. The readback was injected and spoken ("I counted 158 books").
- A deliberately blurred still produced one `capture_alert` blur, a `coverage` message arrived, and the agent relayed the blur.
- An abrupt disconnect (socket aborted, no `end_sweep`) still finished the packet (about 3 minutes for 20 stills).

Fixed in this task:
- `set_locale` was called by the model with country names ("India") and failed with a generic error that the model then ignored while saying it had noted the country. The tool schema now asks for ISO 3166-1 and ISO 4217 codes, and the prompt tells the model to call it only after the user states the country. Live check: first call is `IN` / `INR`.
- `record_user_statement` was never called for "that book is a first edition". The prompt now says to call it for first edition, signed or original claims. Live check: called once with kind `first_edition`.
- Tool calls were invisible in the server log; `LiveTools.call` now logs name, arguments and result at INFO.

Open, not fixed:
- Exclusion `shelf_id` is invented by the model (`current_shelf`) and the exclusion matches by time window only; shelf labels from the pipeline (`S3.1`) are never given to the model.
- The first `session` message is sent twice with state `connecting` (hello and the state watcher); harmless but noisy.
- The pipeline after `end_sweep` took about 240 s for 150 stills (reading stage dominated, no local detector). The phone sits in `processing` for four minutes; there is no per-book progress beyond the stage counters.
- 344 `inventory.upsert` events for 158 books: every book is announced during resolution and again at publish.
- All coverage is attributed to one region `sweep`, so the coverage map never shows more than one entry.
- `live hook failed: LookupError` is logged as a warning when the pipeline injects before the model has called `start_capture_monitor` (short sessions); the injection is dropped, as designed, but the readback in such a session is lost.
- The bridge raises `BudgetExceeded` from `send_audio` into the connection handler when the cap is hit, which closes the session without a `closed` reason the phone can show.

Needs a human (iPhone and Safari): microphone capture and AudioWorklet, 24 kHz playback and the flush on `interrupted`, wake lock, camera permission, tunnel Origin and cookie behaviour on `https`, and a real GoAway/reconnect (cannot be forced from here; no GoAway occurred in 6 minutes).
