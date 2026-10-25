# Library Contents Claim Agent: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A web app (laptop browser or phone browser) running a live voice and vision agent. In one sweep of a home library it directs the capture, and within 5 minutes of the sweep ending it produces a traceable `claim_packet.json` and an HTML report. The packet covers counted, read, measured and priced books, priced non-book items, and the room's floor and wall areas.

**Architecture:**
- The browser talks to one FastAPI backend over a single WebSocket, and the backend talks to `gemini-3.8-live`.
- The live model only converses and coaches.
- The backend saves every full-resolution still as evidence and runs separate, file-producing stages on those stills: keyframes → spines → tracks → scale → reading → resolution → room → items → appraisal → pricing → packet.
- Every number comes from code over saved frames and cached HTTP responses. No language model writes a number.

**Cost constraint:** only the user's Gemini API key (free tier). Everything else is free, needs no card, and is either an official API or allowed by robots.txt and the site's terms. Cost per sweep is $0; the ledger also reports a "would cost at paid rates" figure.

**Spec:** `docs/brief.txt`. Research it relies on:
- `research/01-reference-app.md`, `research/02-vision-measurement.md`, `research/04-live-agent-stack.md`;
- **`research/05-free-pricing.md` (pricing, supersedes 03)**;
- **`research/06-free-stack.md` (free tiers and local models)**.

**Deadline:** 2026-10-28 21:03 IST. Reply in Gmail thread `1a173b21681bc485`.

## Models and services (all free)

| Job | Choice | Free-tier limit | Fallback |
|---|---|---|---|
| Live voice + camera conversation | `gemini-3.8-live` | Free; ~65K input TPM; ~10 min per connection | Chrome Web Speech STT + `speechSynthesis` with a Gemma text brain |
| Spine boxes, spine transcription, item detection | `gemini-3.5-flash-lite` + `gemini-3.1-flash-lite`, alternated | 10 RPM / 500 RPD each | `gemini-3.8-flash` tie-breaks (5 RPM / 20 RPD, at most 5 calls per sweep) |
| Independent spine counter + live gap check | Local RF-DETR-nano bookshelf ONNX (`Geraldine/rf-detr-nano-bookshelf`, Apache-2.0) on DirectML | Local, 42–73 ms/frame | Gemini boxes only, flagged |
| Spine masks for edge-accurate thickness | Local SAM 2.1-small, fp32, box-prompted | Local | Box edges + shared-boundary method |
| OCR corroboration | RapidOCR (onnxruntime, CPU) | Local, 46–75 ms/crop | Read stays unconfirmed, so the book is `unidentified` |
| Room depth (Path B) | Depth Anything V2 Metric-Indoor-Small, fp32, rescaled by ArUco | Local | Review entry, lower confidence |
| Catalogue | Open Library (keyless, identified User-Agent) → Google Books (free key, only on a miss) | 3 req/s; about 1,000/day | `unidentified` |
| India new price | Bookchor ISBN page JSON-LD `NewCondition` → Pustakkosh Store API → Bookswagon (flag `bookswagon.enabled`, **default false**: its terms ban harvesting) | 1 req/s per host | Converted from the second country, labelled, or excluded |
| India used price | Bookchor `UsedCondition` → Pustakkosh `_Used` → **derived**: sourced new × versioned condition factor, labelled | same | Excluded |
| Second country | **GB/GBP**: Penguin UK RRP (exact ISBN) → converted from IN new at the ECB rate, labelled | same | Excluded |
| Item price ranges (IN) | Robots-allowed category listings: IKEA India, Urban Ladder, Vijay Sales, Nilkamal; P25–P75 when there are at least 5 matches, each bound a real URL | same | `needs_appraisal` with the evidence rows |
| FX | Frankfurter v2 `providers=ECB` | keyless | ECB SDMX |
| Checker agent (stretch) | `gemma-4-31b-it` | 15 RPM (unverified) | Skip |
| Phone HTTPS | cloudflared quick tunnel (no account) | n/a | localhost.run |

**Excluded on purpose** (paid, needs a card, or forbidden by terms): SerpApi, fal, Replicate, Cloud Vision, ISBNdb, Anthropic API, Amazon/Flipkart scraping, AbeBooks/B&N scraping, and Gemini Search grounding used to find pages to scrape. The README states why.

**Web app.** Development runs on the laptop at `http://localhost:8000` with the webcam. The real sweep opens the same page in iPhone Safari through the cloudflared URL (HTTPS is required for the camera). iOS Safari has no WebXR AR, so **Path B (getUserMedia + ArUco + local depth) is the decided path**. WebXR (Path A) stays as a progressive enhancement that activates only on an ARCore phone.

## Global Constraints

- Spend cap: total Gemini spend for the whole project stays under USD 1. Every Gemini call goes through `libclaim/llm/budget.py`, which logs token usage to `runs/spend.jsonl` and refuses calls once the ledger, priced at paid-tier rates, reaches USD 0.80. Only the single `GEMINI_API_KEY` is used.
- Every output contract field in `docs/brief.txt` is kept. Fields may be added but never dropped. An empty field means "unknown". A filled field with no evidence is a defect.
- Every `frame_ref` is a path relative to the sweep dir (`frames/000193_1729876543210.jpg`), and the file exists.
- Totals are computed in `libclaim/packet/totals.py` from the lines.
- No price comes from a model. Every price has `source`, `url`, `retrieved_at` and `cache_ref`, which points to the saved raw response plus the parsed JSON-LD snippet in `runs/<id>/cache/`.
- Not-found is explicit evidence, never "no number parsed". The ISBN must match on the page, or the line is recorded as `equivalent_edition` and sent to review.
- No catalogue, price list or room dimension for the test space appears in code or config. Item category → listing URL config is generic per country, not per room.
- Country and currency are a setting (`config/locales/<CC>.yaml`), with IN/INR primary and GB/GBP second.
- Art, signed copies, first editions and rare or antiquarian books are never auto-priced; they become `needs_appraisal`.
- Unreadable spines become `unidentified`, with dimensions and no title.
- Never ask the user to scan barcodes, type ISBNs, pull books out or photograph items one by one.
- Rate limits are respected in code: per-model RPM limiters, a daily counter in `runs/_quota.json`, 1 req/s per retail host, and an identified User-Agent whose contact comes from `.env` (never hardcoded).
- Tests and replays never call live APIs; they use recorded fixtures and the cache.
- Delivery: runs from the README in under 15 minutes, `.env.example` included, no secrets in git.
- Coding standard: no comments, no dead code, KISS/DRY/YAGNI, no regex over model output (structured output schemas only).
- No AI attribution lines in commits. AI tools used are disclosed once in the README, as the brief asks.
- The ML extra installs into its own venv, `.venv-ml` (about 3 GB, gitignored).

## Assumptions (stated, then moved on)

| Ambiguity | Decision |
|---|---|
| `wall_area_m2` gross or net | Gross. `wall_area_net_m2` and `openings[]` are added fields. |
| Shelf run | Sum of vertical spine thicknesses plus the horizontal extent of each flat stack, per row. |
| Confidently-wrong denominator | Wrong among `identified` divided by all `identified`. Title accuracy is correct identifications divided by legible ground-truth spines. |
| Used value with no listing | `derived_from_new`: a sourced new price × `condition_factors.v1` (calibrated from observed Bookchor/Pustakkosh used/new pairs, kept separate from the 15 hand-checked books), labelled and sub-totalled separately. Switch: `used_value_policy`. |
| Default condition | `good`, overridable by voice. |
| Appraisal threshold | ₹5,000 / £60 used, `antiquarian_year` 1950, per locale. |
| Second country | GB. Only 3 of 10 books have a free, compliant local GB print price; the rest are converted and labelled. Reported honestly. |
| Hand-check "same source type" | Hand-check each of the 15 books on the retailer the packet cites, on the same day. |

## Decision Gate G0 (first 2 hours)

Task 0 measures these before any building:
1. Gemini key tier (AI Studio → API keys → Billing Tier).
2. A Live soak test: 4 minutes at 0.5–1 fps video, with compression at 12K/6K, logging `prompt_token_count` per turn and any 429 `quotaId`.
3. A Flash-Lite vs 3.8 Flash `box_2d` bake-off on 3 real shelf-row frames, timing each call.
4. iPhone `getUserMedia` probe: achieved resolution, frame rate, still-grab time, wake lock, and whether the session survives 4 minutes. WebXR is not tested (iOS has none).

It then picks:
- **Path A, WebXR:** ARCore metric pose and depth give room geometry and spine scale. Choose it if the phone supports it, the camera image is at least 1280 px, frames can be read at 2 fps or more, and the session survives 4 minutes.
- **Path B, getUserMedia:** 1920×1080 video, printed ArUco markers per shelf unit, and DA V2 metric depth rescaled by the marker for the room.

Both paths use ArUco when it's visible and fall back to `book_format_prior` (flagged) when it isn't.

## Review Focus

1. **No marker and no WebXR** (the evaluators' unseen shelf). Expected: dims via `book_format_prior` with confidence ≤ 0.4 and a review entry; never a crash. Owner: Task 9.
2. **The same book in 3 overlapping frames.** Expected: counted once, `frame_ref` set to the sharpest frame. Owner: Task 8.
3. **Live session past 2 minutes, a GoAway, or a 429 mid-sweep.** Expected: a transparent resume with stills still flowing. Owner: Task 4.
4. **A retailer page for a different ISBN, a search-redirect page, or an out-of-stock offer.** Expected: `not_found` or `equivalent_edition` + review; never silently priced. Owner: Task 12.
5. **A vision quota 429 storm after the sweep.** Expected: failover to the other bucket. At T+4 min the packet is finalised with what's done and the rest goes to review. Owner: Task 15.

---

## Phase overview and timebox

| Phase | Tasks | Window (IST) | Owner |
|---|---|---|---|
| 0. Spikes + accounts + physical prep | 0 | Mon 26 Oct, morning | You + agent |
| 1. Ground truth (parallel) | GT | Mon 26 Oct, afternoon | You |
| 2. Core | 1–3 | Mon 26 Oct | Agent |
| 3. Live agent + web app | 4–6 | Mon 26 to Tue 27 Oct, morning | Agent |
| 4. Vision + room + items | 7–11 | Tue 27 Oct | Agent |
| 5. Pricing | 12–14 | Tue 27 Oct | Agent |
| 6. Orchestration + eval | 15–16 | Tue 27 Oct, evening | Agent |
| 7. Real sweeps, measured fix, docs, video | 17–18 | Wed 28 Oct, by 19:00 | You + agent |
| Submit | 19 | Wed 28 Oct, by 20:30 | You |

**Quota discipline:** develop against fixtures. Keep Wednesday's full Gemini daily quota for rehearsal and the demo, and another day's for the review call.

### Things only you can do

- [ ] **Keys (all free):**
  - Put your Gemini key in `.env` and check its tier in AI Studio.
  - In the same Google Cloud project, enable the Books API and create a Books key.
  - Optional: a free Hugging Face account and token, and a free Roboflow account (no card).
- [ ] **Space:** one enclosed room with at least 60 books on at least 2 shelving units and at least 8 non-book items. No people and no personal documents in frame.
- [ ] **Markers:** print the sheet from `tools/make_markers.py` at 100%, measure its 10 cm check line, and set `ARUCO_MARKER_SIZE_CM`.
- [x] **Phone:** iPhone (Safari). Path B is decided.
- [ ] **Ground truth:** Task GT, before any tuning.
- [ ] **Demo:** one unedited take of 6 minutes or less, from the iPhone screen recorder (mic on), with a second phone over the shoulder as backup. Upload to YouTube unlisted with a Drive mirror.

### Code you should write yourself

The review call asks you to explain the code line by line, so these are left for you (5–15 lines each):
1. `accept_identification()` in `libclaim/vision/resolve.py`.
2. `appraisal_reasons()` in `libclaim/pricing/appraisal.py`.
3. `derive_used()` in `libclaim/pricing/books.py`.

---

## File structure

```
cozmo-library-agent/
  README.md  .env.example  pyproject.toml
  config/
    locales/IN.yaml GB.yaml US.yaml   currency, sources order, thresholds, user_agent_contact env name
    pricing.yaml             condition_factors.v1, used_value_policy, bookswagon.enabled=false
    item_sources/IN.yaml     category -> listing URLs + must/exclude terms (generic, not room-specific)
    taxonomy.yaml            item categories, book format priors (cm)
  libclaim/
    settings.py models.py sweep.py llm.py
    capture/frame_store.py quality.py coverage.py
    live/bridge.py tools.py monitor.py prompt.md
    server.py
    vision/keyframes.py spines.py rfdetr.py masks.py tracks.py scale.py reading.py ocr.py resolve.py items.py
    room/geometry.py depth.py
    pricing/http.py appraisal.py books.py items.py
    pricing/sources/bookchor.py pustakkosh.py bookswagon.py penguin_uk.py listings.py fx.py
    packet/schema.py totals.py review.py report.py templates/report.html.j2
    review/checker.py        (stretch)
    pipeline.py
  web/                       Vite TS: capture.ts audio.ts ws.ts xr.ts ui.ts main.ts
  tools/ make_markers.py eval.py calibrate_used.py spike/
  ground_truth/              books.csv spines20.csv prices15.csv items.csv room.json
  runs/<sweep_id>/           frames/ stages/*.json cache/ transcript.jsonl claim_packet.json report.html
  docs/ brief.txt PLAN.md spike-results.md architecture.md failure-log.md results.md
  tests/                     mirrors libclaim/, fixtures in tests/fixtures/
```

Two dependency sets:
- **Base** (`pip install -e .`): fastapi, uvicorn, google-genai, opencv, numpy, httpx, pydantic, rapidfuzz, jinja2, rapidocr, onnxruntime-directml, aiolimiter, pytest.
- **ML extra** (`pip install -e .[ml]` in `.venv-ml`): torch 2.14.1+cu126, transformers, for SAM 2.1 and DA V2.

The pipeline runs without the ML extra: masks fall back to box edges, and room depth falls back to WebXR or flagged priors. That keeps "running in under 15 minutes" true.

Every stage is `run(sweep: SweepDir) -> StageOutput`. It reads earlier `stages/*.json` and writes `stages/<name>.json`. `python -m libclaim.pipeline runs/<id> --from <stage>` replays from any stage against the cache.

---

### Task 0: Spikes (Gate G0)

**Files:** `tools/spike/camera-probe.html`, `tools/spike/live_soak.py`, `tools/spike/box_bakeoff.py`, `docs/spike-results.md`.
- [ ] Run the reference app (`_reference/awesome-llm-apps/voice_ai_agents/insurance_claim_live_agent_team`; PowerShell steps in `research/01` §3) with the camera on for 3+ minutes. Record whether it dies at 2 minutes. This feeds the README's kept/changed/threw-away section.
- [ ] Write `live_soak.py`:
  - connect to `gemini-3.8-live` with compression `trigger_tokens=12000`, `target_tokens=6000` and resumption;
  - send 0.5 fps JPEG plus PCM for 4 minutes;
  - log `prompt_token_count` per turn, GoAway and 429 `quotaId`;
  - test `will_continue` streamed tool responses with SILENT / WHEN_IDLE / INTERRUPT;
  - test whether `send_client_content(turn_complete=False)` interrupts.
- [ ] Write `box_bakeoff.py`: 3 shelf-row photos (the user's own if available, otherwise public bookshelf photos) through `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, `gemini-3.8-flash` and local RF-DETR. Record the counts against a hand count, plus latency.
- [ ] Write `camera-probe.html`: on the iPhone through cloudflared, log the `getUserMedia` `getSettings()`, the still-grab time, wake-lock support and an AudioWorklet round trip.
- [ ] `docs/spike-results.md`: chosen path, injection method, Live tokens per minute, box quality per model, call latency.
- [ ] Commit `chore: capture, live-session and detector spikes`.

### Task GT: Ground truth (you, before tuning)

- [ ] `books.csv`: every book with unit, row, position, title, author, legible y/n, orientation.
- [ ] `spines20.csv`: 20 spines measured with a ruler, mixing formats.
- [ ] `room.json`: every wall, the ceiling height, the openings and the shelving fronts, tape-measured.
- [ ] `items.csv`: name, category, w/h/d and brand for at least 8 items.
- [ ] `prices15.csv`: 15 books checked by hand on the retailer the packet cites, same day, with screenshots.

### Task 1: Repo, settings, models, sweep dir

**Files:** Create `pyproject.toml` (base + `[ml]` extra), `.env.example`, `config/**`, `libclaim/{settings,models,sweep}.py`, `tests/test_settings.py`, `tests/test_sweep.py`.

**Produces:**
- `load_locale(cc) -> Locale`, with fields `country`, `currency`, `new_sources: list[str]`, `used_sources: list[str]`, `appraisal_threshold: Decimal`, `antiquarian_year: int`.
- `SweepDir(root)`, with methods `frame_path(seq, t_ms)`, `write_stage(name, model)` and `read_stage(name, cls)`, and `ledger.add(stage, seconds, tokens, usd_at_paid_rates)`.
- `Still(seq, t_ms, path, w, h, blur, glare, motion_dps, pose?, intrinsics?, depth_path?)`.

- [ ] Test: `load_locale("IN").currency == "INR"`, `load_locale("GB").new_sources[0] == "penguin_uk"`, and an unknown code raises `KeyError`. Stage IO round-trips, and the ledger is written to `stages/_ledger.json`.
- [ ] Commit `feat: settings, models, sweep directory`.

### Task 2: Packet schema, totals, review queue, report

**Files:** Create `libclaim/packet/*`, `tests/packet/test_totals.py`, `test_schema.py`.

**Produces:**
- `ClaimPacket`: the contract plus added fields `room.wall_area_net_m2`, `openings`, `shape`, `polygon_m`, `frame_refs`, `books[].scale_method`, `used_value.method`, `used_value.source_type`, `evidence`, `items[].dimensions_method`, `second_country`, `costs`, `latency_s`.
- `compute_totals(books, items) -> Totals`, with added sub-totals `books_used_value_observed` and `books_used_value_derived`.
- `build_review_queue(packet)` and `render_report(packet, sweep)`.

- [ ] Totals tests:
  - `needs_appraisal` books and books with no amount are excluded and counted in `excluded_from_totals`;
  - identified + unidentified + needs_appraisal equals book_count;
  - `shelf_run_m` is the sum of thicknesses;
  - observed + derived used equals `books_used_value`;
  - range items add their low and high.
- [ ] Schema test: every key path in the brief's contract exists in `ClaimPacket.model_json_schema()`.
- [ ] Report: each row links its frame thumbnail and price URL. The report includes the review queue, totals, costs and latency.
- [ ] Commit `feat: claim packet schema, totals, report`.

### Task 3: Frame store and capture quality

**Files:** `libclaim/capture/{frame_store,quality}.py`, `tests/capture/test_quality.py`.

**Produces:** `FrameStore.add(header, jpeg) -> Still`, `score(img) -> Quality(blur, glare)` and `gate(q, motion_dps, cfg) -> list[AlertCode]`.
- [ ] Test: a synthetic sharp checkerboard passes. Blurred with σ=5 it gives `blur`, more than 8% clipped area gives `glare`, and more than 60°/s gives `too_fast`.
- [ ] Commit `feat: frame store and capture quality gates`.

### Task 4: Live bridge

**Files:** `libclaim/live/{bridge.py,prompt.md}`, `tests/live/test_bridge.py`, using a fake session that replays recorded events.

**Produces:** `LiveBridge(sweep, tools, on_audio, on_transcript, on_interrupted)`, with methods `send_audio`, `send_frame`, `inject(event, scheduling)` and `close`. It uses compression at 12K/6K, resumption handles and GoAway reconnects, and writes `usage_metadata` to the ledger. Video goes at 0.5–1 fps at LOW resolution and stops after `end_sweep`.
- [ ] Test: GoAway leads to a reconnect with the last handle, and audio after the reconnect is delivered. A 429 leads to backoff and a reconnect. `interrupted` is forwarded once.
- [ ] Prompt rules:
  - Greet, confirm the country and currency, and explain the sweep in at most 2 sentences.
  - Direct the sweep unit by unit, row by row, holding each row for 2 s, then pan the wall-ceiling line.
  - Never state titles, counts, prices or measurements unless a tool returned them.
  - Ask whether art is an original or a print.
- [ ] Commit `feat: gemini live bridge with resumption`.

### Task 5: Live tools and capture monitor

**Files:** `libclaim/live/{tools,monitor}.py`, `libclaim/capture/coverage.py`, tests.

**Produces:**
- Tools: `record_user_statement(subject, kind, value)`, `exclude_region(description, shelf_id?)`, `set_locale(country, currency)`, `get_sweep_status()`, `end_sweep()`, and the `start_capture_monitor()` generator.
- `Monitor.push(event)`: at most 1 spoken item per 8 s, de-duplicated by region, never while the user is speaking, and stale alerts dropped.

- [ ] Test: two blur alerts for the same region within 8 s produce one push. "first edition" attaches to the stills in [t−4 s, t] and marks the matching tracks `needs_appraisal`. `exclude_region` moves books to excluded.
- [ ] Commit `feat: live tools, coverage map, capture monitor`.

### Task 6: Server and web app

**Files:** `libclaim/server.py`, `web/**`, `tests/test_server.py`.

**Produces:**
- The `/ws/sweep` protocol exactly as in `research/04` §6.2.
- Rear camera with the `ideal` constraint on phone, webcam on laptop.
- A 768 px live frame at 0.5–1 fps, and full-resolution stills at 2 fps over the tunnel (4 fps on localhost).
- AudioWorklet mic at 16 kHz in and playback at 24 kHz with flush.
- Wake lock, a live inventory list, alert toasts and an End sweep button.
- `xr.ts` (Path A only) attaches pose, intrinsics and depth to each still.

- [ ] Manual check on the laptop: talk, see stills arriving, and hear a blur alert when you shake the webcam. Then repeat on the phone through cloudflared.
- [ ] Commit `feat: sweep server and web capture app`.

### Task 7: Keyframes and LLM helper

**Files:** `libclaim/llm.py`, `libclaim/vision/keyframes.py`, tests.

**Produces:**
- `structured(models: list[str], parts, schema) -> T`:
  - per-model `aiolimiter` RPM limits;
  - an RPD counter in `runs/_quota.json`;
  - on 429, honour `RetryInfo.retryDelay`, then fail over to the next model;
  - log tokens to the ledger;
  - in tests, `LIBCLAIM_OFFLINE=1` raises if any live call is attempted.
- `select_keyframes(stills, events) -> Keyframes(rows, room)`: the sharpest still per 0.5 s window, dropping duplicates with pHash distance ≤ 6.

- [ ] Test: a fake client returning 429 on model 1 means the call succeeds on model 2 and the quota file counts both. 30 stills give 10 keyframes.
- [ ] Commit `feat: rate-limited structured LLM helper and keyframes`.

### Task 8: Spine detection, local counter, tracks

**Files:** `libclaim/vision/{spines,rfdetr,tracks}.py`, tests.

**Produces:**
- `detect_spines(keyframe) -> list[SpineDet(box_px, orientation, source)]`: one Gemini call per keyframe with both 2×1 tiles as two image parts and an `image_index`-keyed schema, followed by NMS.
- `rfdetr_boxes(img) -> list[Box]`: ONNX, DirectML with CPU fallback, threshold from config.
- `reconcile(gemini, rfdetr)`: IoU matching. A row whose counts differ by more than 10%, or whose gap exceeds 1.5× the median thickness, gets a review entry and a live re-capture alert. Up to 5 such rows per sweep get a `gemini-3.8-flash` re-query.
- `merge_tracks(...) -> list[BookTrack]`: ORB + RANSAC homography, or 3D centres ≤ 3 cm in Path A.

- [ ] Test: three synthetic overlapping frames give exactly N tracks with the sharpest `best_frame`. Fixture boxes where the two detectors disagree by 20% produce a review flag.
- [ ] Commit `feat: spine detection with independent local counter`.

### Task 9: Scale chain and spine dimensions

**Files:** `libclaim/vision/{scale,masks}.py`, `tools/make_markers.py`, tests.

**Produces:**
- `measure(track, stills, cfg) -> SpineDims(height_cm, thickness_cm, scale_method, confidence)`. The chain is `webxr_depth` → `aruco_homography` → `book_format_prior` (confidence ≤ 0.4, plus a review entry).
- Thickness comes from shared boundaries, using SAM 2.1 masks when the ML extra is present and box edges otherwise. The row sum is constrained to the measured run.
- Open Library `physical_dimensions` is a flagged cross-check only.

- [ ] Test: a rendered 16 cm marker and a 20×3 cm rectangle measure within 3%. With no marker, the method is `book_format_prior` and the review flag is set.
- [ ] Commit `feat: metric scale chain for spines`.

### Task 10: Spine reading, OCR, resolution

**Files:** `libclaim/vision/{reading,ocr,resolve}.py`, tests (Open Library fixtures recorded).

**Produces:**
- `read_spines(crops) -> list[Reading(track_id, title_verbatim, author_verbatim, publisher_verbatim, legible)]`: 24 rotated crops per call, keyed by `crop_id`, with the prompt "transcribe only characters you can see; empty if unsure".
- `ocr_text(crop) -> (text, score)` using RapidOCR.
- `resolve(reading, ocr, dims) -> Resolution(status, title, author, edition, isbn, id_confidence, match_score, margin, source_urls)`. It queries Open Library first and Google Books only on a miss or a disagreement.
- **You write** `accept_identification(reading, ocr, candidates) -> bool`. Starting gates:
  - OCR token overlap ≥ 0.6 with the VLM read;
  - score ≥ 90 with a margin ≥ 8;
  - at least 2 distinctive tokens, or the author matched;
  - the catalogues agree, or one scores ≥ 95.

  Calibrate the gates on ground truth to at most 1.5% wrong.

- [ ] Test: "SAPIENS / HARARI / VINTAGE" plus a matching OCR read resolves to Sapiens (Vintage). "EMMA" alone is `unidentified`. A VLM read that OCR doesn't support is `unidentified`, with both texts kept.
- [ ] Commit `feat: spine reading with OCR corroboration and catalogue resolution`.

### Task 11: Room geometry and non-book items

**Files:** `libclaim/room/{geometry,depth}.py`, `libclaim/vision/items.py`, tests.

**Produces:**
- `room_from_points(points, frames, shelves) -> RoomGeometry(length_m, width_m, height_m, floor_area_m2, wall_area_m2, wall_area_net_m2, shelved_wall_area_m2, shape, polygon_m, openings, scale_method, confidence, frame_refs)`:
  - Path A gets its points from WebXR.
  - Path B gets them from DA V2 metric depth, rescaled by the median ArUco edge.
  - The geometry is: RANSAC floor, walls from the band above the shelving, a Manhattan-snapped polygon, the shoelace area and a ray-plane ceiling.
- `detect_items(room_keyframes) -> list[ItemDet]`: 4 keyframes per call, taxonomy enum, deduplicated in 3D or by adjacent frames, sized from the planes or depth. Art is `needs_appraisal` unless the user said "print".

- [ ] Test: a synthetic 4×3×2.7 m cloud with a shelf block 0.35 m proud gives floor 12.0 ± 0.3 and gross wall area 37.8 ± 1.0. An L-shaped cloud gives `L` and the correct shoelace area.
- [ ] Commit `feat: room geometry and non-book items`.

### Task 12: Free price sources with evidence cache

**Files:** `libclaim/pricing/http.py`, `libclaim/pricing/sources/{bookchor,pustakkosh,bookswagon,penguin_uk,listings,fx}.py`, tests (all fixtures recorded from `research/05` §8; no network).

**Produces:**
- `PriceQuote(amount, currency, source, url, retrieved_at, kind: listing|derived_from_new|converted, condition, isbn_on_page, availability, cache_ref)`.
- `cached_get(url) -> CachedResponse`: sha1 key, raw body plus the parsed JSON-LD snippet saved, robots.txt checked once per host, 1 req/s per host, User-Agent contact from `.env`.
- Book sources expose `async quote(isbn, condition) -> PriceQuote | NotFound(evidence)`.
- `listings.range(category, locale) -> ItemRange | NotEnough(rows)`.
- `fx.rate(base, quote, on_or_before) -> FxRecord`.

- [ ] Test:
  - A Bookchor fixture yields `NewCondition` and `UsedCondition` offers with `isbn_on_page` equal to the query.
  - A search-redirect fixture yields `NotFound`, and so does a page with a different ISBN.
  - An `OutOfStock` offer is returned with `availability` set.
  - An IKEA listing fixture yields P25/P75 rows that each carry a URL.
  - A replay with no network is byte-identical.
- [ ] Commit `feat: free compliant price sources with evidence cache`.

### Task 13: Appraisal gate and book pricing

**Files:** `libclaim/pricing/{appraisal,books}.py`, `tools/calibrate_used.py`, tests.

**Produces:**
- **You write** `appraisal_reasons(resolution, reading, statements, quotes, locale) -> list[str]`. Reasons: a user statement; spine text with signed or first-edition wording; `year < antiquarian_year`; no ISBN; used ≥ threshold; used ≥ 1.5× new.
- `price_book(resolution, locale) -> BookPrices`:
  - follows the source order from the locale;
  - flags a disagreement between two retailers of more than 25% for review;
  - when there's no local price, converts at the ECB rate on or before `retrieved_at` (`converted=True`, keeping both records);
  - when there's no price at all, leaves the amounts empty and the book goes to `excluded_from_totals`.
- **You write** `derive_used(new_quote, condition, cfg) -> PriceQuote`.
- `calibrate_used.py` builds `condition_factors.v1` from observed used/new pairs, saving the snapshot to `config/calibration/`. The pairs are kept separate from `prices15.csv`.

- [ ] Test: no IN price but a GB price gives `converted=True` with the FX URL. No price at all gives an excluded line. A "Signed" spine gives `needs_appraisal` with no amounts.
- [ ] Commit `feat: appraisal gate and book pricing`.

### Task 14: Item pricing and second-country run

**Files:** `libclaim/pricing/items.py`, tests.

**Produces:**
- `price_item(item, locale) -> ItemPrice`: a range when there are at least 5 matches, otherwise `needs_appraisal` with the rows kept.
- `second_country(packet, cc="GB", n=10)` writes `packet.second_country`, with per-book `source_type` set to `local_listing` or `converted`.

- [ ] Test: 7 fixture rows give the expected nearest-rank P25/P75 with URLs. 2 rows give `needs_appraisal`.
- [ ] Commit `feat: item pricing and second-country repricing`.

### Task 15: Pipeline orchestration

**Files:** `libclaim/pipeline.py`, `tests/test_pipeline.py` (12-frame fixture sweep, all external calls faked).

**Produces:** `async run_pipeline(sweep, from_stage=None) -> ClaimPacket`.
- Detection, reading, resolution and pricing start during the sweep as rows close.
- After `end_sweep` the remaining stages run.
- A T+4 min deadline finalises the packet, sending anything unfinished to review.
- It emits `pipeline.progress`, records latency per stage, and pushes a code-computed summary for the readback.

- [ ] Test: the fixture sweep gives a valid `ClaimPacket`, every `frame_ref` exists, and the totals equal `compute_totals`. `--from price` makes zero network calls. A forced stall triggers the deadline path, which produces review entries.
- [ ] Commit `feat: pipeline orchestration with replay and deadline`.

### Task 16: Eval harness

**Files:** `tools/eval.py`, `tests/test_eval.py`.

**Produces:** `python tools/eval.py runs/<id> ground_truth/ > docs/results.md`, with one table per pass bar over all shelves. Books are matched to ground truth per row by sequence alignment.
- [ ] Test: a toy truth of 10 books with 1 miss and 1 wrong title gives count error 10%, accuracy 8 of 9 legible, and confidently-wrong 1 of 9 identified.
- [ ] Commit `feat: ground-truth evaluation`.

### Task 17: Real sweeps, measured fix, demo

- [ ] Do 2 or more rehearsal sweeps (watching the quota), running `eval.py` after each.
- [ ] Find the worst error's root cause, change one thing, replay with `--from <stage>`, and record the before and after numbers.
- [ ] Record the demo in one take. Copy that sweep's packet, report and frames to `submission/`.

### Task 18: Docs

- [ ] `README.md`:
  - a quickstart that works in under 15 minutes, with the base install only;
  - the optional ML extra in `.venv-ml`;
  - the cloudflared step;
  - what was kept, changed and thrown away from the reference;
  - models and AI tools used;
  - the free-tier data-use note;
  - licences (RF-DETR weights Apache, SAM 2.1 Apache, DA V2 small Apache, RapidOCR Apache);
  - excluded sources and why;
  - the assumptions table.
- [ ] `docs/architecture.md`: a mermaid pipeline diagram, which model does what, where scale comes from, where prices come from.
- [ ] `docs/failure-log.md`: the 3 worst errors, root causes and changes, the measured fix, cost per sweep ($0 billed, plus tokens and the paid-rate equivalent), and latency per stage.

### Task 18b (stretch, only if Tasks 0–18 are done by Wed 12:00): Checker agent

**Files:** `libclaim/review/checker.py`, tests.

**Produces:** `check_packet(packet, sweep) -> list[Disagreement]`. `gemma-4-31b-it` gets each line's frame crop and claim and answers through structured output whether it agrees, disagrees or can't tell. Disagreements go to `review_queue` as `checker_disagrees`; values are never edited.
- [ ] Test: a mocked disagreement adds a review entry and leaves the packet values unchanged.
- [ ] Commit `feat: independent checker agent`.

### Task 19: Submit

- [ ] Create a private GitHub repo (free), add the reviewers as collaborators, and push.
- [ ] Reply in thread `1a173b21681bc485` with the repo, the video link, the packet, the results sheet, the architecture note and the failure log.
