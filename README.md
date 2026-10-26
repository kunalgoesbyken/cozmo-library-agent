# Library Contents Claim Agent

A live voice and vision agent for a home-contents claim. The policyholder opens a web page on a phone, talks to the agent and walks the room once with the camera on. The agent directs the sweep (steady, slower, closer, a missed shelf). When the sweep ends, background stages count and read every book spine, measure what can be measured, price what can be priced from free sources, detect the non-book items and estimate the room. Within 5 minutes it reads a summary back and writes a traceable `claim_packet.json` and an HTML report.

The design rule: the live model only talks and coaches. Every number in the packet is computed by code from saved full-resolution stills and from cached HTTP responses. No language model writes a number, a price or a total.

## The result contract

Each sweep writes `runs/<sweep_id>/`:

| File | What it is |
|---|---|
| `claim_packet.json` | The contract from `docs/brief.txt` (fields only added, none dropped) plus `evidence`, `costs`, `latency_s`, `second_country` and per-line `box`, `crop_ref`, `scale_method`, `edition_unverified`, `equivalent_edition`, `corroboration` |
| `report.html` | Human-readable report built from the packet: per-line crop, click for the full frame with the box outlined, price links, review queue, totals, cost and latency |
| `frames/` | Every saved full-resolution still, plus `frames/crops/` thumbnails. Every `frame_ref` in the packet exists here |
| `stages/*.json` | One file per pipeline stage, so any stage can be replayed |
| `cache/` | Raw HTTP responses and parsed price evidence; `llm/` holds recorded Gemini replies |

An empty field means unknown. Totals come from `libclaim/packet/totals.py`. Notes on fields that are easy to misread: [docs/packet-notes.md](docs/packet-notes.md).

Documents: [architecture](docs/architecture.md), [failure log](docs/failure-log.md), [results](docs/results.md), [E2E findings](docs/e2e-findings.md), [spike results](docs/spike-results.md), [ML extra verification](docs/ml-verification.md), [plan](docs/PLAN.md).

## What has and has not been measured

Be aware of this before reading any number in this repo.

- Measured: counting on three public shelf photos (52 of 55, 22 of 22, 19 of 19 against my own hand count, not an official ground-truth sheet), the pipeline end to end on those photos against live Gemini and live catalogue and retailer pages, the live voice server against the real Live API for a 6-minute laptop session, and the optional ML models on this laptop.
- Not measured: any run with an iPhone (microphone, AudioWorklet playback, wake lock, camera permission, tunnel cookies, a real GoAway reconnect), any run with printed ArUco markers (so no marker-calibrated spine dimensions exist; every dimension in the test runs is a format prior and is labelled as an estimate), any ground-truth sheet (`ground_truth/` holds only templates, so `tools/eval.py` evaluates only time and cost), and context compression firing in a long session.
- Price coverage is limited by what free sources list. In the latest 3-photo run, 24 books were titled, 19 got an ISBN, and one book got a price (a used listing, INR 135, for an equivalent edition). See [failure log](docs/failure-log.md).

## Quickstart (base install, about 10 minutes plus downloads)

Needs Python 3.12.4 or later 3.12 (`requires-python` is `>=3.12.4,<3.13`), Node.js for the web build, and a Gemini API key. Windows is the tested platform.

Windows (PowerShell):

```
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
copy .env.example .env
cd web
npm install
npm run build
cd ..
New-Item -ItemType Directory -Force _scratch\models
curl.exe -L -o _scratch\models\rfdetr-nano-bookshelf.onnx https://huggingface.co/Geraldine/rf-detr-nano-bookshelf/resolve/main/onnx/rfdetr-nano-bookshelf.onnx
```

Linux or macOS:

```
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
cp .env.example .env
(cd web && npm install && npm run build)
mkdir -p _scratch/models
curl -L -o _scratch/models/rfdetr-nano-bookshelf.onnx https://huggingface.co/Geraldine/rf-detr-nano-bookshelf/resolve/main/onnx/rfdetr-nano-bookshelf.onnx
```

Linux and macOS are untested here. `pyproject.toml` pins `onnxruntime-directml`, which publishes Windows wheels (verify); on other systems replace that line with plain `onnxruntime` (same version) before installing. The code lists `DmlExecutionProvider` then `CPUExecutionProvider`, so it falls back to CPU.

Without the RF-DETR file the pipeline still runs: the independent local spine counter is skipped and counts rely on Gemini boxes alone (flagged in the review queue). The file is 115 MB; the model card declares Apache-2.0.

Edit `.env` (names only here; never commit it, `.env` is gitignored):

| Key | Purpose |
|---|---|
| `GEMINI_API_KEY` | Required. Your own Gemini API key |
| `GOOGLE_BOOKS_API_KEY` | Google Books key from the same Cloud project (primary catalogue search). Sent only in a request header, never in a URL or log |
| `LIVE_MODEL`, `LIVE_VOICE` | Live voice model and voice name |
| `VISION_MODELS` | Comma-separated models for spine boxes, spine reading and items; calls alternate between them |
| `TIEBREAK_MODEL`, `CHECKER_MODEL` | Tie-break model (at most a few calls per sweep) and the optional checker |
| `DEFAULT_COUNTRY`, `SECOND_COUNTRY` | Primary and second pricing locale (`config/locales/<CC>.yaml`: IN, GB, US) |
| `ARUCO_MARKER_SIZE_CM` | Measured side of your printed marker (see Markers). Empty means no metric scale |
| `HTTP_CONTACT_EMAIL` | Put your own contact address here; it goes only into the HTTP User-Agent of catalogue and retailer requests, from the environment, never from code |
| `FREE_TIER` | `true` if your Gemini key is on the free tier, `false` if billed. Empty reports `billed_usd` as unknown |
| `HF_TOKEN`, `ROBOFLOW_API_KEY` | Optional, not needed for the default path |
| `PORT` | Server port, default 8000 |
| `ACCESS_TOKEN` | Optional, at least 24 characters; otherwise a random token is generated per start |
| `PUBLIC_ORIGIN` | Your tunnel URL, for the exact Origin check (see Tunnel) |

Run the server:

```
.venv\Scripts\python -m libclaim.server        # Windows
.venv/bin/python -m libclaim.server            # Linux, macOS
```

It prints `Open http://localhost:8000/?t=<token>`. Open exactly that URL. The token is required for the page, the WebSocket and the report files. The server binds to 127.0.0.1 only.

Run the tests (offline, no network, no Gemini; about 3 to 4 minutes):

```
.venv\Scripts\python -m pytest -q
```

## Optional ML extra (`.venv-ml`)

Adds torch, transformers, SAM 2.1 (edge-accurate masks for metric measurement) and Depth Anything V2 (room depth). The pipeline runs without it: masks fall back to box edges, and without depth the room is left unmeasured (low confidence, review entry). About 3 GB, gitignored. The `--extra-index-url` is needed because plain PyPI gives a CPU-only torch on Windows.

```
py -3.12 -m venv .venv-ml
.venv-ml\Scripts\python -m pip install -e ".[ml]" --extra-index-url https://download.pytorch.org/whl/cu126
```

Run the server from `.venv-ml\Scripts\python -m libclaim.server`. Models download anonymously from Hugging Face on first use (447 MB). Verified on a GTX 1660 Ti: [docs/ml-verification.md](docs/ml-verification.md).

## Phone use: tunnel, iPhone and Safari

Safari needs HTTPS for the camera and microphone, so use a cloudflared quick tunnel (no account):

1. Install cloudflared (`winget install Cloudflare.cloudflared` on Windows, or the release binary from GitHub).
2. Start the server and note the token.
3. In a second terminal: `cloudflared tunnel --url http://localhost:8000`. It prints `https://<random>.trycloudflare.com`.
4. Put that URL in `.env` as `PUBLIC_ORIGIN` and restart the server. Without it any `*.trycloudflare.com` origin is accepted and the server logs a warning.
5. On the iPhone open `https://<random>.trycloudflare.com/?t=<token>`. Allow the camera and microphone. The tunnel URL changes on every run.

iOS Safari has no WebXR, so capture is plain `getUserMedia` with the rear camera: audio goes through an AudioWorklet, video goes at about 1 frame per second to the live model at low media resolution, and full-resolution stills are saved on the laptop for the pipeline. The page asks for a screen wake lock where Safari has one. Keep the phone screen on and the laptop awake. The iPhone path itself has not been run yet (see above).

## Markers: how metric scale is obtained

Scale comes from a printed ArUco marker (4x4_50 dictionary), never from a model's guess. Without a marker, dimensions are left empty and a format prior is shown as an estimate.

```
.venv\Scripts\python tools\make_markers.py --units 3 markers.pdf
```

(Linux and macOS: `.venv/bin/python tools/make_markers.py --units 3 markers.pdf`; if `libclaim` is not importable, run it with `PYTHONPATH=.`.)

- One page per shelf unit; unit n gets marker id n. The default marker side is 16 cm (`markers.print_size_cm` in `config/vision.yaml`).
- Print at 100% with no fit-to-page. Measure the 10 cm check line on the sheet with a ruler. If it is not 10 cm, fix the print setting.
- Measure the marker side and set `ARUCO_MARKER_SIZE_CM` to it (16 for the default sheet). `--size-cm` must agree with that variable.
- Tape each marker flat on the shelf front edge at book height, in plain view, one per unit. The scale stage assumes the spines sit 2.5 cm behind the shelf front edge (`scale.spine_recess_cm`) and corrects for that depth, so a marker on the front edge matters. A marker on the wall behind the books would be wrong. The 2.5 cm recess is an assumption and has not been calibrated against a ruler.
- Keep the marker visible in the sweep for each row (at least 100 px across: `scale.min_marker_px`). A marker seen in one frame scales neighbouring frames through the frame-to-frame homography at reduced confidence.

## Replay and evaluation

```
python -m libclaim.pipeline runs/<id> [--from <stage>] [--online]
```

Stages: `keyframes spines tracks scale reading resolution room items appraisal pricing packet`. Replay is offline by default and uses the recorded Gemini replies and HTTP cache; `--online` allows live requests. `--from scale` reuses recorded reads; `--from reading` re-reads.

```
python tools/eval.py runs/<id> ground_truth/ > docs/results.md
```

Fill `ground_truth/` first (see [ground_truth/README.md](ground_truth/README.md) for the file formats, matching rules and every threshold, which live in `config/eval.yaml`). Missing files make only their own bars "NOT EVALUATED".

Other tools: `tools/e2e_images.py <images> --hand-counts <json>` feeds still photos through the streaming pipeline with the real clients (spends Gemini quota, capped by `--cap-usd`); `tools/calibrate_used.py <isbn file> <cache dir>` recalibrates the used-condition factors from observed new and used pairs; `tools/record_catalog.py` re-records the catalogue test fixtures; `tools/spike/` holds the Gate G0 spikes.

## Kept, changed and thrown away from the reference app

Reference: `voice_ai_agents/insurance_claim_live_agent_team` in awesome-llm-apps (Apache-2.0). It describes what it sees; this agent has to measure, count and price. The reference sets no `context_window_compression` and no `session_resumption`, and never handles `GoAway`. Per the Live API docs, audio plus video sessions are capped at about 2 minutes without compression, so with the camera on it dies in about two minutes (read from the code and the docs; I did not watch it die on a camera).

| | Reference | This project |
|---|---|---|
| Kept | FastAPI server that proxies the browser WebSocket to Gemini Live, so the key stays server-side | Same pattern, one WebSocket `/ws/sweep` |
| Kept | `gemini-3.8-live`, barge-in via `interrupted`, NON_BLOCKING tools with `SILENT`, `WHEN_IDLE` and `INTERRUPT` scheduling | Same; the capture monitor is a NON_BLOCKING tool whose streamed responses carry coaching and inventory updates |
| Kept | Independent check of what the model claims to see, on the exact saved frame | Generalised: every read is corroborated by a second source (OCR, catalogue) before it counts |
| Changed | No compression, resumption or GoAway handling; 2-minute cap | Compression at 12K/6K tokens, resumption handles, GoAway and 429 reconnect with backoff, a text-turn fallback when injection arrives before the monitor starts |
| Changed | Last 512 px frame at 1 fps, nothing persisted | 1 fps low-resolution video for the model; every full-resolution still saved as evidence with quality gates (blur, glare, speed, occlusion) and perceptual-hash dedupe |
| Changed | ScriptProcessor mic capture, JSON-wrapped base64 audio | AudioWorklet capture and playback, binary frames |
| Changed | Localhost only, trivial owner cookie | Access token, Origin check (exact `PUBLIC_ORIGIN`), pre-auth size and connection caps, size limits, SSRF-guarded price client |
| Changed | Output is a Markdown claim and a ZIP | Typed `claim_packet.json` and HTML report, totals computed in code, review queue, replayable stages |
| Changed | Numbers do not need to be right | Counted, measured and priced numbers, each with a frame or a price source, checked by `tools/eval.py` |
| Thrown away | Google ADK sequential agent graph (2 LLM nodes, 5 rule nodes) | Plain Python stages over files; only vision, reading and item detection call a model |
| Thrown away | Mock policy directory, fraud-signal keyword rules, claim narrative, document checklist | Not part of this brief |
| Thrown away | Sketch generation, optional live avatar, in-memory session store with TTL | Sessions are directories under `runs/` |

## Models and AI tools used

| Tool | Where | What it does |
|---|---|---|
| Gemini Live (`LIVE_MODEL`, default `gemini-3.8-live`) | `libclaim/live/` | Voice conversation and coaching with low-resolution camera context. Calls tools. States no number unless a tool returned it |
| Gemini flash-lite family (`VISION_MODELS`: `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, alternated per `config/llm.yaml`) | `vision/boxes.py`, `reading.py`, `items.py` | Spine boxes, transcription of spine crops (title, author, publisher as printed), non-book item detection. Structured-output schemas only; no regex over model output |
| Gemini flash (`TIEBREAK_MODEL`) | `pipeline.py` | Re-queries a shelf row when Gemini boxes and the local detector disagree, at most `reconcile.max_requery_rows` (5) rows per sweep |
| Gemma (`CHECKER_MODEL`) | Configured only | Stretch checker; not part of the default path |
| RF-DETR nano, bookshelf ONNX (`Geraldine/rf-detr-nano-bookshelf`) via onnxruntime | `vision/rfdetr.py` | Independent local spine counter and live gap check. Compared with Gemini boxes; disagreements go to the review queue |
| RapidOCR 3.10 (PP-OCRv6 small, onnxruntime, CPU) | `vision/ocr.py` | Second, independent read of each spine crop. A title is accepted only when OCR supports the model's read or the catalogue strongly agrees |
| ArUco (OpenCV) | `vision/scale.py` | Metric scale from a printed marker, via homography |
| SAM 2.1 hiera-small (optional `[ml]`) | `vision/masks.py` | Edge-accurate spine masks for metric measurement, used only when a mask agrees with the detector box |
| Depth Anything V2 Metric-Indoor-Small (optional `[ml]`) | `room/depth.py` | Monocular metric depth for room geometry, rescaled by markers when seen. Without markers it is a prior and the room confidence is low |
| rapidfuzz | `vision/resolve.py`, `tools/eval.py` | Fuzzy title, author and publisher matching |

Development of this repository was assisted by an AI coding assistant (the brief asks for AI tools to be disclosed). No attribution lines are added to commits.

Free-tier data use: Google's terms for unpaid Gemini API use allow it to use submitted content, including images and audio, to improve its products, and human reviewers may read it (verify the current terms for your key). Do not put people, personal documents or anything private in frame, or use a billed key (`FREE_TIER=false`) if that matters to you.

Budget guard: every Gemini call goes through `libclaim/budget.py`. `check()` before and `record()` after write the token usage to `runs/spend.jsonl`, priced at paid-tier rates from `config/model_prices.yaml`. The guard refuses calls once the ledger reaches USD 0.80 (`DEFAULT_CAP_USD`), and the live server turns that into a `budget` error and a `closed` reason the phone shows. Per-model request limits (`config/llm.yaml`) and a daily counter in `runs/_quota.json` keep the free-tier RPM and RPD limits. `FREE_TIER` only changes how the packet describes billing (`billed_usd` is `0.0` for `true`, the paid-rate amount for `false`, `null` when unset); the ledger always reports `usd_at_paid_rates`.

## Licences

Verified from installed package metadata in this repository's environments, 2026-10-26:

| Component | Licence |
|---|---|
| fastapi, pydantic, pyyaml, rapidfuzz, aiolimiter, pytest, pytest-timeout, onnxruntime-directml | MIT |
| uvicorn, httpx, protego, python-dotenv | BSD-3-Clause |
| jinja2 | BSD |
| numpy | BSD-3-Clause with 0BSD, MIT, Zlib and CC0 parts |
| pillow | MIT-CMU |
| google-genai, opencv-python, rapidocr, pytest-asyncio | Apache-2.0 |
| torch (2.14.1 build) | BSD-style with Apache-2.0 and other component licences |
| torchvision | BSD |
| transformers, accelerate | Apache-2.0 |
| Node build tools (typescript, vite) | not checked here; verify |

Models and data:

| Model | Licence |
|---|---|
| SAM 2.1 hiera-small | Apache-2.0 (model card front matter in the local Hugging Face cache) |
| RF-DETR nano bookshelf ONNX | Declared Apache-2.0 on its card (from earlier research); the training dataset licence is not verified. Verify before redistributing |
| Depth Anything V2 Metric-Indoor-Small | The local model card carries no licence field. Upstream describes the small variant as Apache-2.0; verify |
| RapidOCR PP-OCRv6 models | Packaged with Apache-2.0 RapidOCR; verify the upstream PaddleOCR model licence |
| Gemini models | Google API terms; not redistributable |
| Test shelf photos (Wikimedia Commons, not in git) | CC BY 3.0 (Blytonite) and CC BY-SA 4.0 (Roman Eisele); attribution is in [docs/spike-results.md](docs/spike-results.md) |

Ultralytics (AGPL-3.0) and Piper (GPL-3.0) are not used.

## Excluded sources and why

| Source | Why not |
|---|---|
| SerpApi, fal, Replicate | Paid or need a card; the brief allows only the user's Gemini key |
| Google Cloud Vision | Paid, needs billing |
| ISBNdb | Paid |
| Anthropic API | Paid, not the provided key |
| Amazon, Flipkart | Terms forbid scraping |
| AbeBooks, Barnes & Noble | Terms forbid scraping |
| Gemini Search grounding used to find pages to scrape | Would route scraping through the model instead of honouring each site's robots.txt and terms |
| Bookswagon | Terms ban harvesting; the code exists but `bookswagon.enabled` is `false` in `config/pricing.yaml` |
| Open Library `/search` | Its robots.txt disallows `/search` (covering `/search.json`) for all agents. Google Books is the primary title and author search; Open Library is used only for `/isbn/{isbn}.json` and `/works/{work}/editions.json`, which robots.txt allows |
| Penguin UK, Crossword, Sapna, others | Only used where robots.txt allows the exact path; a robots block is recorded as `not_found` evidence and the line is excluded from totals |

Retail requests follow robots.txt per host, 1 request per second per host (2 for one host), use an identified User-Agent, and respect `Retry-After`. Prices come only from retailer pages for the matched ISBN: Bookchor JSON-LD, then Pustakkosh store API for India; Penguin UK RRP for GB; item ranges from robots-allowed listing pages (IKEA India, Urban Ladder, Vijay Sales, Nilkamal); FX from Frankfurter (ECB).

## Assumptions

| Ambiguity | Decision |
|---|---|
| `wall_area_m2` gross or net | Gross. `wall_area_net_m2` and `openings[]` are added fields |
| Shelf run | Sum of measured spine thicknesses, per row. With no scale it is `null` and `shelf_run_estimated_m` holds a format-prior estimate |
| Dimensions without a marker | Left empty. The format prior goes to `dims_estimated_from_prior` and is shown as "est."; `dims.fill_from_prior` (`config/pipeline.yaml`) is the owner's switch |
| Spine recess | Spines sit 2.5 cm behind the shelf front edge; the marker is on the front edge. Uncalibrated |
| Confidently wrong | Wrong among `identified` divided by all `identified` |
| Title accuracy | Correct identifications divided by legible ground-truth spines |
| Bars with an ambiguous wording | Read strictly: dimensions need at least 80% within 15% and a mean error of at most 15%; prices need at least 80% within 25%, mean at most 25%, and at least 10 of 15 same-source rows. If the evaluators read these more leniently, the reported pass rate is conservative |
| Used value with no listing | `derived_from_new`: sourced new price times a versioned condition factor (`config/calibration/condition_factors.v1.json`), labelled and sub-totalled separately. Switch: `used_value_policy` |
| Default condition | `good`, overridable by voice |
| Equivalent edition | When the printed edition cannot be matched, the most recent paperback ISBN-13 whose title matches is priced, labelled `equivalent_edition` and sent to review. An old work with an unverified edition is never auto-priced |
| Appraisal | Art, signed copies, first editions, antiquarian (before 1950) and above a per-locale used threshold (INR 5,000, GBP 60) become `needs_appraisal` and are not priced |
| Second country | GB. Only some books have a free GB print price; the rest are converted at the ECB rate and labelled |
| Deadline | 240 s after the sweep ends; vision work is cut at 150 s and finished locally, the rest goes to review |
| Single user | One live session at a time (`max_sessions: 1`) |

## Functions meant for the owner to rewrite

Three small, pure, config-driven functions hold the judgement calls. They are reference implementations; rewrite them to change policy. Thresholds live in `config/`.

1. `accept_identification(reading, ocr, candidates)` in `libclaim/vision/resolve.py`: decides whether a read becomes an identification and how it was corroborated (`ocr`, `catalogue_only`, or `none`). Thresholds: `gates` in `config/resolve.yaml`.
2. `appraisal_reasons(resolution, reading, statements, quotes, locale)` in `libclaim/pricing/appraisal.py`: the reasons a book goes to a human appraiser instead of being priced.
3. `derive_used(new_quote, condition, cfg)` in `libclaim/pricing/books.py`: the used value derived from a sourced new price when no used listing exists.

## Known limits and needs-human list

- iPhone: everything on the phone is unverified (microphone, AudioWorklet playback, flush on interrupt, wake lock, camera permission, cookies over https, a real GoAway). Needs a human with the phone.
- Marker: print the sheet, check the 10 cm line, set `ARUCO_MARKER_SIZE_CM`, tape it on the shelf front edge. Until then all spine dimensions are empty or estimates, and `box_height_trim` and the 2.5 cm recess are uncalibrated.
- Ground truth: `books.csv`, `spines20.csv`, `prices15.csv`, `items.csv` and `room.json` do not exist yet, so only time and cost are evaluated.
- Task 17 tuning: gates, vision cutoff, row detection and window sizes are tuned on three public photos only, not on real sweeps.
- Long runs: compression at 12K/6K never fired in the soak tests, so behaviour past about 10 minutes of live context is unverified.
- Identification and price coverage are low on the test photos (see [failure log](docs/failure-log.md)); free sources do not list most editions.
- Room area: the test runs had no metric depth in the base install and no room photos, so room figures are untested against a tape.
- Linux and macOS install is untested (see the onnxruntime note).
