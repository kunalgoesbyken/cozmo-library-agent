# 06 - Free stack: Gemini key plus zero-card services

Constraint (2026-10-26): the only credential is a Gemini API key. Everything else must be free and must not need a card. Pricing sources (SerpApi, eBay and so on) are out of scope here; another note covers them.

Researched 2026-10-26. Labels used below:
- **[docs]**: official page, fetched today.
- **[live]**: tested today from this laptop without any key.
- **[2nd]**: secondary source, usually a copy of the AI Studio rate-limit table, which needs a login to see.
- **(unverified)**: could not be confirmed.

Scratch work is in `_scratch/` (gitignored), including scripts, downloaded pages and logs.

---

## 0. TL;DR

1. **The key is almost certainly free tier.** Paid needs a linked billing account and prepayment. Check AI Studio → API keys → "Billing Tier" (§1.1).
2. **Gemini Live is free.** `gemini-3.8-live` input and output are "Free of charge" **[docs]**. The free-tier limit is **65K input TPM [2nd]**, with RPM "Unlimited". Concurrent sessions are unpublished (one is enough). The 3.8 Live free tier also lists Search grounding as "Supported".
3. **`gemini-3.8-flash` free is 5 RPM / 20 RPD [2nd]**, so it can't be the vision workhorse. **Flash-Lite 3.5 and 3.1 give 10 RPM / 500 RPD each [2nd]**, as separate buckets. With batching (2 tiles per detection call, 24 crops per reading call, 4 keyframes per item call) a sweep needs **~35–40 calls**. About 2/3 run during the sweep, and the post-sweep remainder takes **~1 min**. **One sweep fits on the free tier**, with ~25 sweeps per day of Flash-Lite headroom but only ~4 per day if 3.8 Flash tie-breaks are used.
4. **Cut from the plan as not card-free:** SerpApi, fal, Replicate, Cloud Vision, ISBNdb, Anthropic.
   - Replace with a **local RF-DETR bookshelf ONNX** model (Apache, 42–73 ms/frame on DirectML **[live]**) as the independent spine counter.
   - **Local SAM 2.1-small** for masks.
   - **Roboflow free (no card) hosted SAM 3**, about 20k images/month, as an optional third opinion.
   - **RapidOCR** (46–75 ms/crop CPU **[live]**) instead of Cloud Vision.
   - **DA V2 metric depth + ArUco + plane fit** instead of hosted VGGT or MapAnything.
5. **Google Books keyless is dead today**: HTTP 429 on the shared anonymous quota **[live]**. Create a free Books key in the Gemini key's GCP project (no billing). **Open Library works keyless** and returns `physical_dimensions` (cm) **[live]**.
6. **cloudflared quick tunnel works with no account [live]**: WS RTT 36–68 ms, 2.8 MB/s. localhost.run works but has a 1.2–1.6 s RTT. ngrok free has a 1 GB/month cap.
7. **D: has ~1.7 GB free.** Put any torch venv (cu126, about 3 GB) on C:.

---

## 1. Gemini key: free or paid, and what the free tier allows

### 1.1 Is the key free-tier or paid?

I could not tell without using the key, and the task said not to use keys. Here is how to settle it in 30 seconds:
- **AI Studio → API keys.** Each key's project shows a *Billing Tier* column. A free project shows "Free tier" next to a "Set up billing" button. **[docs]** https://ai.google.dev/gemini-api/docs/billing, plus forum threads.
- **AI Studio → Rate limit** (https://aistudio.google.com/rate-limit) shows the project's active per-model limits.
- **Programmatically:** the first `429 RESOURCE_EXHAUSTED` names the tier in its `QuotaFailure` details. On the free tier, the `quotaId` values end in `-FreeTier`, for example `GenerateRequestsPerMinutePerProjectPerModel-FreeTier` with metric `generativelanguage.googleapis.com/generate_content_free_tier_requests`. Log `error.details` in `llm.py`. **[2nd]** https://github.com/google-gemini/gemini-cli/issues/8437
- **Paid needs a linked billing account**, and the billing docs say it also needs a prepayment of at least $10 **[2nd, quoting the billing docs]**. If the user never added a card, the key is **free tier**. Plan for that.

`docs/PLAN.md` says "make sure billing is on, because free-tier rate limits are too low". The no-card constraint rules that out. The rest of this note designs for the free tier.

### 1.2 Free-tier availability per model (pricing page, last updated 2026-10-22) **[docs]**

Source: https://ai.google.dev/gemini-api/docs/pricing. Cells are quoted from the page HTML, saved at `_scratch/dl/pages/pricing.html`.

| Model | Free tier tokens | Free-tier Google Search grounding | Free data "used to improve our products" |
|---|---|---|---|
| `gemini-3.8-live` (and `-extended-thinking`, `3.1-flash-live-preview`) | **Free of charge** (input and output) | **"Supported"** (the free-tier cell) | Yes |
| `gemini-3.8-flash` | Free of charge | **Not available** | Yes |
| `gemini-3.6-flash` | Free of charge | Not available | Yes |
| `gemini-3.5-flash-lite` | Free of charge | Not available ("can be tested in AI Studio") | Yes |
| `gemini-3.1-flash-lite` | Free of charge (Batch is free as well) | Not available | Yes |
| `gemini-2.5-flash-lite` | Free of charge | Free, up to 500 RPD (shared with Flash). But 2.5 access is "limited to developers who previously used them" [2nd], so assume you can't use it | Yes |
| `gemini-embedding-2` | Free of charge (text, image, audio, video) | n/a | Yes |
| `gemma-4-31b-it`, `gemma-4-26b-a4b-it` | Free of charge (**paid tier: "Not available"**, so free only) | Not available | Yes |
| `gemini-3.1-pro-preview`, `gemini-4-argon` | **Not available** on the free tier | n/a | n/a |
| `gemini-3.5-transcribe-live` (streaming STT) | Free of charge | Not supported | Yes |
| `gemini-3.8-flash-tts`, `-flash-lite-tts` | Free of charge | n/a | Yes |

**So Live is on the free tier.** A real-time voice fallback is only a contingency (§1.6), not a requirement.

### 1.3 Free-tier rate limits

The official rate-limits page (https://ai.google.dev/gemini-api/docs/rate-limits, updated 2026-10-15) **no longer prints per-model numbers**. It says to "view your active rate limits in AI Studio". The numbers below come from a copy of AI Studio's "Free Tier comparison" table published on 2026-10-19 and 2026-10-22 ([scriptbyai free tier](https://www.scriptbyai.com/gemini-api-free-tier-limits/), [scriptbyai all tiers](https://www.scriptbyai.com/gemini-api-rate-limits/)). They are **[2nd]**: confirm them on the user's AI Studio Rate limit page on day 1.

| Model | RPM | Input TPM | RPD | Notes |
|---|---|---|---|---|
| `gemini-3.8-flash` | **5** | 250K | **20** | Tier 1 is 1K / 2M / 10K |
| `gemini-3.6-flash` | 5 | 250K | 20 | separate per-model bucket |
| `gemini-3.5-flash-lite` | **10** | 250K | **500** | Tier 1 is 4K / 4M / 150K |
| `gemini-3.1-flash-lite` | 10 | 250K | 500 | separate per-model bucket |
| `gemma-4-31b-it` / `26b-a4b-it` | 15 | unlimited | 1,500 | from a single Korean note, **(unverified)** ([tilnote](https://tilnote.io/pages/6a51b3398d9a59d14c1c174c)) |
| `gemini-embedding-2` | 100 | 30K | not listed | |
| `gemini-3.8-live` | "Unlimited" | **65K** | not listed | Tier 1 is 150K TPM. **Concurrent sessions are not published.** A forum claim says "3 in AI Studio" and a Google reply says limits are TPM-based ([forum](https://discuss.ai.google.dev/t/official-concurrent-session-rps-limits-for-gemini-live-api-where-are-they-documented/174664)), so **(unverified)**. We need 1 session. |
| `gemini-3.5-transcribe-live` | Unlimited | 20K | not listed | |
| `gemini-3.8-flash-tts` | 3 | 10K | not listed | too low for conversational TTS |
| Google Search grounding | only Live is "Supported" on free; Flash and Flash-Lite are "Not available" | | 5,000/month figure is the paid-tier allowance | We don't need grounding: prices must come from cited, cached HTTP responses |

General rules **[docs]**:
- Limits are **per project, not per key**. Extra keys in the same project add nothing. Don't create extra projects to multiply quota, because that is quota circumvention.
- RPD resets at midnight Pacific, which is 12:30 or 13:30 IST.
- Spend-based limits don't apply to the free tier.

### 1.4 Can one sweep run on the free tier? Yes, if we redesign the vision calls

The current plan expects 60–200 vision calls on `gemini-3.8-flash` within 5 minutes. On the free tier that model allows **5 RPM and 20 calls per day**, so a single sweep would use up a whole day's quota. **That plan does not fit.** The redesign:

1. **Primary vision model: Flash-Lite, on two separate buckets.** `gemini-3.5-flash-lite` and `gemini-3.1-flash-lite` each give 10 RPM and 500 RPD. A per-model token-bucket limiter alternates between them, giving **20 RPM and 1,000 RPD combined**.
   - `gemma-4-26b-a4b-it` is an optional third bucket (15 RPM, unverified) for text-only or reading work. Gemma's `box_2d` grounding is not documented, so **don't use it for detection** until it is tested.
   - `gemini-3.8-flash` (20 RPD) is kept for at most about 5 tie-break calls per sweep, on disputed rows only.
2. **Batch inside each call.**
   - **Detection:** one call per row keyframe. Send both 2×1 tiles as two image parts and ask for a list keyed by `image_index`. Boxes stay normalised per image.
   - **Reading:** 24 rotated spine crops per call as separate image parts at `media_resolution` LOW/MEDIUM (280–560 tokens each), keyed by `crop_id`. The plan's batch of 8 becomes 24.
   - **Items:** 4 room keyframes per call.
3. **Run most of it during the sweep.** Rows close every ~10–15 s, and the sweep lasts 3 min at 20 RPM, which gives about **60 calls of capacity before the sweep even ends**.

**Call budget for one sweep [est]** (2 units, ~12 rows, ~150 books, ~12 room keyframes):

| Stage | Calls | Model | When |
|---|---|---|---|
| Spine detection | ~20 (1 per row keyframe, 2 tiles each) | Flash-Lite ×2 | during the sweep |
| Re-query of gap or disputed rows | ~5 | `gemini-3.8-flash` | post-sweep |
| Spine reading | ~7 (150 crops / 24) | Flash-Lite ×2 | rows during the sweep, the rest post-sweep |
| Items | ~3–4 | Flash-Lite | post-sweep |
| **Total** | **~35–40** | | **Post-sweep part: ~15 calls ≈ 1 min at 20 RPM** |

TPM check: the largest call is 24 crops × 560 + 1K prompt ≈ 15K tokens, so even 10 such calls in a minute is about 150K, below 250K.

Daily headroom: about 1,000 Flash-Lite RPD / ~35 per sweep is roughly 25 sweeps a day, but the 3.8 Flash cap of 20 RPD allows only about 4 sweeps a day that use tie-breaks. So:
- develop against **recorded fixtures and the response cache**, not live calls;
- save a full day's quota for the demo day, and another for the review call (Wednesday 28 Oct, plus the call day).

If Flash-Lite boxes turn out too loose on dense rows (measure at G0 against ground truth):
- spend the 3.8 Flash calls on the densest rows first;
- add the local or Spaces SAM cross-check (§2) as the independent second detector. It needs no Gemini quota.

**Implementation requirements for `llm.py`:**
- a per-model async limiter (`aiolimiter`, or a 6 s / 3 s spacing per model);
- on 429, honour `RetryInfo.retryDelay`, then fall over to the other Flash-Lite bucket;
- log `quotaId` to the ledger;
- a daily-usage counter in `runs/_quota.json`, so the eval harness refuses to burn quota during replay.

### 1.5 Live session on the free tier

One ~9-minute session is greeting + 3 min sweep + ≤5 min processing + readback.
- Live is **free** **[docs]**, and RPM is "Unlimited" **[2nd]**.
- The binding limit is **65K input TPM [2nd]**.
- Raw stream at 1 fps LOW video (70 tok/frame) plus 25 tok/s audio is about **5.7K tok/min**.
- The risk is per-turn context re-processing. Billing re-charges the whole context window each turn (04 §1.10). If TPM counts the same way, 4–5 turns a minute on a 12–16K context comes to 50–80K/min, **which would exceed 65K**. (unverified)

Mitigations, all cheap:
- **Tighten compression** to `trigger_tokens ≈ 12,000` and sliding window `target_tokens ≈ 6,000`. The plan has no number, and the docs example of 25K/8K is too loose for the free tier.
- **Send video at 0.5 fps** except when the monitor wants the model to look.
- **Throttle monitor pushes.** Keep the 1 per 8 s rule. SILENT pushes still add context tokens.
- **Stop sending video** after `end_sweep`, during processing.
- **Test it in G0:** run the Task 0 soak with `usage_metadata.prompt_token_count` logged per turn. If you get a 429 `...PerMinute...-FreeTier` from Live, halve the video rate.
- Connection lifetime is ~10 min anyway, so session resumption (already in Task 4) covers a 9-minute session.

### 1.6 If Live fails on the free tier (contingency, all free)

| Option | How | Limits | Fit |
|---|---|---|---|
| **Chrome Web Speech API** | `SpeechRecognition` (continuous, interimResults) + `speechSynthesis` in the phone page | In Chrome, recognition is **server-based** ("audio is sent to a web service… won't work offline") **[docs]** https://developer.mozilla.org/en-US/docs/Web/API/SpeechRecognition. `processLocally` / `SpeechRecognition.install()` exist for on-device packs (support on Android Chrome unverified). Android Chrome stops continuous recognition after silence, so restart on `onend`. Barge-in is manual: cancel `speechSynthesis` on `onspeechstart`. | **Best fallback.** Zero quota, low latency, and audio never leaves Google/Chrome. The agent brain then becomes text. |
| **Gemini standard API, streaming text** | `generate_content_stream` with transcript + last 2 frames → text → `speechSynthesis` | One request per turn. Use **`gemma-4-26b-a4b-it`** (separate 15 RPM bucket, unverified) or `gemini-3.6-flash` sparingly (20 RPD). Don't spend the Flash-Lite vision buckets on chat. | Fine for ≤10 turns/min. Tools become structured output. |
| **`gemini-3.5-transcribe-live`** | free streaming STT over WebSocket | 20K TPM [2nd] | If Web Speech is flaky on the phone |
| **Kokoro-82M** (Apache-2.0) | `pip install kokoro-onnx` (pure wheel, 0.6.1) + `onnxruntime`. `kokoro` 0.9.4 (torch) also installs | CPU real-time factor well under 1 on a laptop **(unverified, not run)** | Server-side TTS if `speechSynthesis` voices are poor. Streams 24 kHz PCM over the existing WS. |
| **Piper** (`piper-tts` 1.8.0, `cp39-abi3-win_amd64` wheel exists **[live pip download]**) | `pip install piper-tts` | **GPL-3.0** (OHF-Voice/piper1-gpl). Fine for a private take-home, but note it in the README | Fastest CPU TTS |
| `gemini-3.8-flash-tts` | | 3 RPM free [2nd] | Too low |

### 1.7 Free-tier data use

Gemini API terms, effective 2026-03-23 **[docs]** https://ai.google.dev/gemini-api/terms:
- For Unpaid Services, "human reviewers may read, annotate, and process your API input and output"
- and "Do not submit sensitive, confidential, or personal information to the Unpaid Services".

That is acceptable here:
- the brief already forbids people and personal documents in frame;
- the capture monitor should still drop stills where Flash-Lite item detection returns `person` or `document_with_text`;
- **say so in the README.**

The terms also say apps offered to users in the **EEA, Switzerland or UK may use only Paid Services**. The user and reviewers are in India, so this is fine, but note it as a deployment caveat. The GB locale (pricing country) is unaffected.

---

## 2. Free replacements for fal SAM 3.1, fal VGGT and Replicate MapAnything

The detailed checks were done by a helper agent. Its scratch files are `_scratch/scripts/{spaces,view_api,sam3_test,sam3_real}.py` and `_scratch/venv`.

### 2.1 Machine facts **[live]**

- **GPU:** GTX 1660 Ti, 6 GB, compute capability 7.5. About **0.85 GB** is already held by the display.
- **Driver:** 572.60, which tops out at CUDA 12.8. **cu130 wheels need a newer driver (R580+).**
- **Python:** 3.12.4.
- **Drive D: is nearly full.** About 1.7 GB is free after cleanup, against 68 GB on C:. A torch CUDA venv is about 3 GB, so **put optional ML venvs and `PIP_CACHE_DIR` on C:**, or free up D: first.

### 2.2 Hosted, free

| Option | Free allowance | Card? | Tested | Verdict |
|---|---|---|---|---|
| **HF Spaces via `gradio_client` 2.7.2** | ZeroGPU daily GPU time: **2 min anonymous, 5 min with a free account**, 40 min with PRO. Resets 24 h after first use; 60 s default per call **[docs]** https://huggingface.co/docs/hub/spaces-zerogpu | No (email signup) | **[live]** `prithivMLmods/SAM3-Demo` `/run_image_segmentation`, prompt "book spine", on a real 960×344 Commons shelf photo: connect 4.4 s, **predict 69 s** (queue + cold GPU), **52 masks**, scores 0.76–0.93, against about 52–55 true. A synthetic PIL shelf gave 0 masks (unrealistic input). | Accurate but **too slow and too quota-limited for the per-sweep path**. Use it for dev and eval spot checks only. Frames go to the Space owner, so prefer the official `facebook/*` and `depth-anything/*` Spaces. |
| Spaces up today **[live, runtime API]** | `facebook/sam3`-based: `prithivMLmods/SAM3-Demo`, `merve/SAM3-video-segmentation` (`akhaliq/sam3` errors). SAM 2: `fffiloni/SAM2-Image-Predictor`. Depth: `depth-anything/Depth-Anything-V2` (`/on_submit(image)`), `depth-anything/depth-anything-3` (multi-step UI). Recon: `facebook/vggt` (running, but `gradio_client` couldn't read its API, cause unverified), `facebook/map-anything` (fragile multi-step UI). MoGe: `Ruicheng/MoGe-2` `/run(...)`. UniDepth: no maintained Space | | | |
| **HF Inference Providers** | **Free accounts now get no monthly credits** (PRO gets $2). HF's own inference "focuses mostly on CPU" **[docs]** https://huggingface.co/docs/inference-providers/pricing | n/a | | **Not an option.** |
| **Roboflow Serverless API: SAM 3** | Free plan has **10 credits/month and "does not require a credit card"**; projects stay private **[docs]** https://roboflow.com/pricing. SAM 3 costs **0.5 credits per 1,000 images**, so about **20,000 images/month free**. RF-DETR-Nano costs 0.125, Depth Anything V2/V3 Small 0.3125, PP-OCR det+rec 0.375 **[docs]** https://docs.roboflow.com/deployment/roboflow-cloud/serverless-api/model-pricing. Text prompt via `POST /sam3/concept_segment` https://docs.roboflow.com/deploy/supported-models/sam3 | **No** ("on-demand credits require a card"; included credits don't) | Not tested (needs a free API key) | **Best hosted replacement for fal SAM 3.1.** Latency and rate limit are unpublished ("Standard"), **(unverified)**. Credits reset monthly. |

### 2.3 Local on the GTX 1660 Ti (Windows, no Docker or WSL)

Half precision is unreliable on GTX 16xx cards: there are widespread NaN and black-output reports (https://github.com/pytorch/pytorch/issues/58123), and Turing has no bf16. **Run fp32 and treat any fp16 result as untrusted.**

VRAM and latency figures are **estimates**: torch and the weights were not downloaded because D: is full.

| Model | Params / licence / access | Install route | 6 GB fit (fp32) | Latency [est] | Verdict |
|---|---|---|---|---|---|
| **SAM 2.1 hiera-small** (tiny is 39M) | 46M, **Apache-2.0**, open | transformers 5.20.2 `Sam2Model`. Avoid PyPI `sam2` 1.1.0, a third-party fork. The official repo needs `SAM2_BUILD_CUDA=0` on Windows | Easy | 0.3–0.6 s encode at ~1024 px; box-prompt decode takes ms. 20–40 keyframes in **10–30 s** | **Use for spine masks** prompted by detector boxes. |
| **SAM 3** | 860M, SAM License (custom, permissive, has a trade-controls clause), **gated, manual approval** (3.1 also gated) | transformers `Sam3Model`. PyPI `sam3` 0.1.4 pins `huggingface-hub<2`, which conflicts with transformers 5.20 | About 3.4 GB of weights at 1008 px, borderline | 2–5 s/image (unverified) | Request access now. Optional local counter. |
| **Depth Anything V2 Metric-Indoor-Small** | 25M, Apache-2.0 for the small variant | transformers | Easy | 50–100 ms/frame | **Use for room fallback depth.** |
| DA3METRIC-LARGE | 0.33B, Apache-2.0 (DA3-Large/Giant are CC BY-NC) | `depth-anything-3` 0.1.1 is heavy (xformers, numpy<2, open3d, pycolmap) | About 1.4 GB | 0.5–1 s/frame | Optional upgrade, in a separate venv. |
| MapAnything | 1.23B. Use **`facebook/map-anything-apache`** (Apache-2.0); `facebook/map-anything` is CC-BY-NC | `pip install --no-deps git+https://github.com/facebookresearch/map-anything uniception==0.1.7` (its pins have Windows wheels) | fp16 weights about 2.5 GB, so fp16 with NaN risk | 30–90 s for 20 frames | **High risk. Last resort only.** |
| VGGT-1B | 1.26B, **CC-BY-NC-4.0** (VGGT-1B-Commercial is gated) | git, `--no-deps` (it pins torch 2.3.1) | fp32 about 5 GB, which leaves no room, so fp16 with NaN risk | 8–15 frames in 20–60 s | **Avoid on this GPU.** |
| MoGe-2 | 35M–331M, MIT | git dependencies including Triton-based FlexGEMM; risky on Windows (unverified) | ViT-S/B fit | n/a | Skip. |
| UniDepth | CC BY-NC 4.0, not on PyPI, README targets CUDA 11.8 | | | | Skip. |

### 2.4 Exact Windows installs (Python 3.12), verified by `pip download` / dry-run **[live]**

```powershell
# optional ML venv on C: (D: is full)
py -3.12 -m venv C:\venvs\libclaim-ml
C:\venvs\libclaim-ml\Scripts\python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
C:\venvs\libclaim-ml\Scripts\python -m pip install transformers==5.20.2 accelerate==1.15.0 timm==1.0.30 huggingface_hub==2.3.0 gradio_client==2.7.2
```

- The **cu126 index is the right one for driver 572.60.** It has torch 2.14.1 and torchvision 0.29.1 for cp312 win_amd64 (pip dry-run: "Would install torch-2.14.1+cu126").
- cu128 stops at 2.11, and cu130 needs a newer driver. Plain PyPI torch on Windows is CPU-only.
- PyTorch says **2.14 is the last release with CUDA 12.x wheels**; CUDA 13 builds still cover sm_75 https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432. So **pin 2.14.1**. That 2.14+cu126 includes sm_75 is **(unverified)**; check with `torch.cuda.get_arch_list()`.
- Other cp312 win_amd64 wheels that download fine:
  - onnxruntime 1.31.0, onnxruntime-gpu 1.31.0, onnxruntime-directml 1.24.4
  - xformers 0.0.35 (Turing kernels unverified, and not needed)
  - open3d 0.20.0, pycolmap 4.2.1, rerun-sdk 0.24.1
  - `triton-windows` 3.8.0.post29 (official triton has no Windows wheel)

### 2.5 Recommendation

1. **Spine masks (cross-check):** local SAM 2.1-small in fp32, prompted by detector boxes.
2. **Independent count:** the local RF-DETR bookshelf model (§3), with Roboflow-hosted SAM 3 "book spine" as an optional third opinion.
3. **Room fallback (no WebXR):**
   - ArUco scale + per-frame DA V2 metric depth + floor, wall and ceiling plane fit (local, under 1 min for 40 frames).
   - Optional dev check: one `facebook/map-anything` Space run with a free HF token.
   - **Drop fal VGGT and Replicate MapAnything.**

---

## 3. Free spine detection options

The detailed checks were done by a helper agent. Its scratch files are `_scratch/scripts/rfdetr_test.py` and `_scratch/rfdetr_out/`.

### 3.1 Ready-made permissive detector, run locally **[live]**

**`Geraldine/rf-detr-nano-bookshelf`** (https://huggingface.co/Geraldine/rf-detr-nano-bookshelf, updated 2026-09-25):
- RF-DETR-nano, one class `book`, 384 px input.
- **Weights are declared Apache-2.0.** The training dataset licence was not verified.
- Ships `onnx/rfdetr-nano-bookshelf.onnx` (115 MB), which runs on numpy + onnxruntime with no torch.
- Reported 0.974 mAP@50 on Roboflow `rateshelf/bookshelf-recognition-2` (4,624 images). The current weights are adapted to library shelves with VLM-made labels: 0.676 mAP@50 against those labels.

| Test photo (Wikimedia Commons) | Count by eye | Detected | CPU i5-9300H | DirectML on 1660 Ti |
|---|---|---|---|---|
| German hardbacks 1920×888 | 20 + 1 partial + 1 flat | 21 tight boxes, **missed the flat book** | 285 ms | **42–73 ms** (about 200 ms for 5 tiles) |
| Enid Blyton, dense thin paperbacks + flat stacks, 1920×688 | about 58–62 | 51 at threshold 0.5, 55 at 0.35 | about 0.5 s | 42–66 ms |

**Verdict:**
- A good **free, local, independent counter and box cross-check**, and fast enough to also run live during the sweep for "missed shelf" or "gap" alerts.
- Not reliable alone on dense thin paperbacks: it double-boxed some and missed some.
- It gives boxes, not masks; use SAM 2.1 for edges.

### 3.2 Roboflow

- **Free plan:** no card and 10 credits a month (§2.2). Self-hosted `inference` is free on every plan.
- **Dependency clash:** `inference` 1.8.2 pins `numpy<2.4`, `opencv<4.13` and `onnxruntime<1.22`, which conflicts with the plan's pins (numpy 2.5.4, opencv 5.0). Use the hosted HTTP API, or a separate venv.
- **Hosted training:** 1 credit per 30 GPU-minutes.

### 3.3 Public spine datasets

Licences were read from the page HTML.

| Dataset | Images | Task | Licence |
|---|---|---|---|
| [Book spine detection (P2)](https://universe.roboflow.com/book-detection/book-spine-detection-p2) | 7,063 | instance seg | CC BY 4.0 |
| [book_spine_instance_segmentation](https://universe.roboflow.com/minhaz/book_spine_instance_segmentation-uqybb) | 898 | instance seg | CC BY 4.0 |
| [LibVision Book Spines](https://universe.roboflow.com/libvision/book-spines-vh9b8) | 546 | detection | CC BY 4.0 |
| [Open Shelves](https://github.com/capjamesg/open-shelves) | - | - | CC BY-NC-SA 4.0, archived. **Avoid** |

Other weights:
- `izi0/mmrotate-bookspine` (MIT) needs mmrotate, which is painful on Windows.
- `ranathungaWK/...yolov8` and `quist99/...yolov8` are bare `best.pt` files with no licence, and AGPL if trained with Ultralytics. **Avoid.**
- Whether the Roboflow datasets have a trained hosted model is **(unverified)**: the Universe pages render in JS.

### 3.4 Fine-tuning on the 1660 Ti in under 1 hour

Ultralytics 8.4.189 is a pure wheel and installs on Windows. YOLO26 is the current family.
- **`check_amp` disables AMP on GTX 16xx**, so training runs in fp32 (ultralytics `utils/checks.py`).
- Estimates (not measured, because D: is full):
  - yolo11n-seg at 640 px on 7k images: about 3–6 min per epoch, so **10–15 epochs per hour**;
  - the s model: about 2.5× slower.
- That is feasible on a subset, but tight. Free, card-free alternatives are Roboflow hosted training and Colab's free T4.

**Licence:** Ultralytics models are **AGPL-3.0** by default, and Ultralytics claims even internal use needs its Enterprise licence unless fully open-sourced (https://www.ultralytics.com/license). A private repo shared with reviewers gives them the source, so AGPL compliance is plausible, but **avoid Ultralytics**.

Permissive alternatives:
- `rfdetr` 1.11.2: Nano/S/M/L detection and every segmentation size are Apache-2.0; the other sizes are PML-1.0. Native Windows is unverified, but the ONNX route works.
- RT-DETR or D-FINE through transformers (Apache-2.0).

The brief also says "if you fine-tune, justify why a pretrained model was not enough". The pretrained RF-DETR bookshelf model plus Gemini means **no fine-tuning is needed**.

---

## 4. Free OCR backup

The detailed checks were done by a helper agent. Its scratch files are `_scratch/scripts/{ocr_test,ocr_speed}.py`, `_scratch/venv-ocr` (CPU) and `_scratch/venv-dml` (DirectML).

| Option | Install (Windows, Python 3.12) | Tested **[live]** | Verdict |
|---|---|---|---|
| **RapidOCR 3.10.0** (Apache-2.0, released 2026-10-08) | `pip install rapidocr onnxruntime`. For GPU use `onnxruntime-directml` with `EngineConfig.onnxruntime.use_dml: True` (no CUDA install). Models (PP-OCRv6 det+rec, zh+en) auto-download. The old `rapidocr_onnxruntime` package is legacy (1.4.4, Jan 2025) | Synthetic spines with text rotated 90°: **3/3 exact** (scores 0.98–1.0), both whole-shelf and ±90° crops. Recognition only: **46–75 ms/crop on CPU**. Whole shelf: 610 ms CPU vs **97 ms DirectML**. **The default det config upscales thin crops (2–7 s/crop)**; set `params={"Det.limit_type":"max","Det.limit_side_len":960}` to get 37–270 ms | **Pick this.** Real stylised spines are untested. |
| PaddleOCR 3.7.0 + paddlepaddle 3.3.1 (CPU on PyPI) | GPU builds come from Paddle's own index (`https://www.paddlepaddle.org.cn/packages/stable/cu126/`, cu118/cu126 3.3.1, cu129/cu130 3.4.0) | Not run | The install docs say "GPU CUDA capability over 7.5", and the 1660 Ti is exactly 7.5, so support is **(unverified)**. RapidOCR runs the same models with less hassle, so skip Paddle. |
| Google Cloud Vision | Setup says "Verify that billing is enabled for your Google Cloud project" **[docs]** https://docs.cloud.google.com/vision/docs/setup. The 1,000 free units/month https://cloud.google.com/vision/pricing still need a billing account | | **Excluded** (needs a card). |

For spines, run both ±90° rotations through recognition and keep the higher-scoring read. Use the OCR tokens only as **corroboration**: a reading is accepted when its catalogue match shares at least one distinctive token with the OCR output. OCR never adds text the VLM didn't see.

---

## 5. Free tunnels for phone HTTPS

| Option | Account / card | Tested | Limits | Verdict |
|---|---|---|---|---|
| **cloudflared quick tunnel** | **None.** Portable `cloudflared.exe` 2026.10.3 from GitHub releases; `winget install Cloudflare.cloudflared` also works | **[live]** URL in 6 s. GET RTT 0.12–0.22 s (first request 1.0 s). 400 KB upload in 0.23–0.50 s (0.8–1.7 MB/s, bound by laptop uplink) | No uptime guarantee (CLI banner). 200 in-flight requests. No SSE. Random URL per run **[docs]** https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/ | **Keep as primary.** **WebSocket [live]** (`websockets` 17.3 echo server, `_scratch/scripts/ws_tunnel_test.py`): small-message RTT 36–68 ms; 20 × 250 KB binary frames in 1.78 s (**2.8 MB/s, 11 stills/s**); the connection survived 130 s idle with default pings. Still to re-test from the phone on mobile data in G0. |
| **localhost.run** | **None** (`ssh -R 80:localhost:8000 nokey@localhost.run`, using the built-in Windows OpenSSH) | **[live]** URL in 5 s. GET RTT **1.19–1.63 s** | Free domains rotate; "create an account… for a longer lasting domain" (banner) | Backup only. About 6× slower RTT than cloudflared from India. |
| **ngrok free** | Account required. No card needed at sign-up **(unverified on the page)** | not tested | **1 GB/month out**, 20K HTTP requests/month, interstitial on HTML **[docs]** https://ngrok.com/docs/pricing-limits/free-plan-limits | 1 GB/month is about 3 sweeps of 2 fps stills plus audio both ways. Avoid. |
| **Tailscale Funnel** | Account (Personal plan is free; card not required, **unverified**) + MagicDNS + HTTPS certs + `funnel` node attribute | not tested | Ports 443/8443/10000 only. "Non-configurable bandwidth limits", undisclosed **[docs]** https://tailscale.com/kb/1223/funnel | Stable `*.ts.net` URL, but undisclosed bandwidth and more setup. Second backup. |

Uplink sizing [est]: 2 fps × ~250 KB stills + 1 fps × ~60 KB live frames + 32 KB/s audio is about **0.6 MB/s**. That is well inside the 2.8 MB/s measured laptop-to-edge-to-laptop. The phone's own uplink is the real cap. **Default to 2 fps stills** and go to 3–4 fps only if `bufferedAmount` stays low.

---

## 6. Free catalogue APIs

| API | Key / card | Tested **[live]** | Limits **[docs]** | Use |
|---|---|---|---|---|
| **Open Library** `search.json`, `/works/<id>/editions.json`, `/isbn/<isbn>.json`, covers | none | `search.json?title=sapiens&author=harari` gave 200 in 1.2 s, `/works/OL17075811W` (86 editions). `/isbn/9780099590088.json` gave 200 in 2.1 s, **`physical_dimensions: "19.7 x 12.7 x 3.8 centimeters"`**, publisher Vintage. Cover by ISBN gave 200 in 1.1 s, 15 KB JPEG | **1 req/s anonymous, 3 req/s with `User-Agent: AppName (email)`**. "Do not bulk download". Covers by ISBN are limited to **100 requests per IP per 5 min** (403 after), while covers by CoverID/OLID are not limited. https://openlibrary.org/developers/api, https://openlibrary.org/dev/docs/api/covers | **Primary resolver.** Use `fields=` to keep responses small. `physical_dimensions` gives a free cross-check of measured spine height and thickness, and an edition-height prior (Task 9/10). |
| **Google Books** `volumes` | **A key is needed in practice.** Keyless calls gave **HTTP 429 "Queries per day" on the shared anonymous project** (`project_number:624717413613`) all 3 times today. Docs: public data "must be accompanied by an identifier… API key" https://developers.google.com/books/docs/v1/using | 429 ×3 without a key | Default about 1,000 req/day per project **(unverified, forum)**. Billing not mentioned in the docs | Enable "Books API" in the **same GCP project as the Gemini key** and create a **separate** unrestricted-to-Books key. AI Studio keys are restricted to the Generative Language API. No billing is needed to enable the API (verify in console, ~2 min). Secondary resolver. |
| **Wikidata** SPARQL / `wbsearchentities` | none | SPARQL by ISBN-13 `P212` gave Q122931444 (2015 paperback) in 1.8 s. `wbsearchentities` gave work Q12411055 plus 2 edition items in 0.5 s | WDQS: 60 s query timeout and per-IP throttling; send a User-Agent **(limits from memory, unverified)** | Optional third signal for work/edition and "first edition" year checks (appraisal gate). Not needed for MVP. |
| **ISBNdb** | **Paid** ($14.95–$14.99/mo and up; 7-day trial, card requirement unverified) | not tested (403 to bots) | | **Excluded** by the constraint. |

---

## 7. Free demo recording and hosting

- **Recording:** Android's built-in screen recorder (Android 11+) with "Device audio and microphone". The 04 §5 caveat about call-mode audio routing still applies, so do a 20 s test and keep the in-app MediaRecorder backup track. Free, and needs no account.
- **Hosting:**
  - **YouTube unlisted:** default upload limit is 15 minutes; longer needs phone verification, and no card is mentioned. https://support.google.com/youtube/answer/71673. A 6-minute video is fine.
  - **Google Drive link** (15 GB free; "anyone with the link" viewer) as a mirror.
  - Put both links in the email.
- **Repo:** GitHub Free gives "unlimited private repositories" with **unlimited collaborators** **[docs]** https://docs.github.com/en/get-started/learning-about-github/githubs-plans. Branch protection and code owners are not available on Free private repos, which doesn't matter here. Invite reviewers as collaborators (read access is enough).

---

## 8. Final table: one row per pipeline stage of `docs/PLAN.md`

Latency labels:
- **M** means measured today without a key.
- **E** means estimate.
- Gemini call latency could not be measured, because that needs the key. Flash-Lite is assumed at 3–10 s per batched call (E), so **time it in G0**.

| # | Stage (PLAN) | Free choice | Limits | Latency | Risk | Fallback |
|---|---|---|---|---|---|---|
| 0 | Phone access (Task 0, 6) | cloudflared quick tunnel, no account | No uptime SLA; 200 in-flight; URL changes per run | M: WS RTT 36–68 ms, 2.8 MB/s; GET 0.12–0.22 s | Tunnel drops mid-demo | localhost.run (M: RTT 1.2–1.6 s), then Tailscale Funnel (account, no card) |
| 1 | Capture + WebXR or `getUserMedia` (Task 0, 6) | Browser APIs, ArUco printed on paper | none | n/a | WebXR unsupported on the phone | Path B: ArUco + `getUserMedia` |
| 2 | Live agent (Task 4, 5) | `gemini-3.8-live`, free tier | **65K input TPM [2nd]**; RPM unlimited; concurrency unpublished; ~10 min connection; 2 min A+V without compression | E: same as paid | **Context re-processing exceeds 65K TPM** at 4+ turns/min with a 16K context (unverified) | Compression 12K/6K, 0.5 fps video, no video after `end_sweep`. Last resort: Web Speech STT + `speechSynthesis` + Gemma-4 text brain |
| 3 | Capture quality gates (Task 3) | OpenCV local | none | E: <10 ms/frame | none | none |
| 3b | Live "missed shelf / gap" check (new, optional) | RF-DETR-nano bookshelf ONNX on DirectML | none | **M: 42–73 ms/frame** | Misses flat books and thin spines | Gemini detection on the closed row |
| 4 | Keyframes (Task 7) | OpenCV + pHash local | none | E: <2 s per sweep | none | Time-based sampling |
| 5 | Spine detection (Task 8) | `gemini-3.5-flash-lite` + `gemini-3.1-flash-lite` `box_2d`, alternating, 1 call per row keyframe (2 tiles), **reconciled with RF-DETR boxes** | 10 RPM + 10 RPM, 500 + 500 RPD, 250K TPM each [2nd] | E: 3–10 s/call, ~20 calls, mostly during the sweep | **Flash-Lite boxes looser than 3.8 Flash on dense rows (unverified)**; quota burn during dev | `gemini-3.8-flash` re-query of disputed rows (≤5/sweep, 20 RPD). Roboflow SAM 3 "book spine" (free credits). Review entry + live re-capture |
| 6 | Cross-frame tracks (Task 8) | ORB + RANSAC homography local | none | E: <5 s | Double counts on overlap | Per-row single frame |
| 7 | Scale + spine dims (Task 9) | WebXR depth or ArUco homography; **SAM 2.1-small fp32 masks** for edges; Open Library `physical_dimensions` as cross-check | Optional ML venv about 3 GB (on C:) | E: 10–30 s for 20–40 keyframes (SAM) | fp16 NaN on the 1660 Ti (run fp32); SAM install on Windows | Box edges + shared-boundary method; `book_format_prior` flagged |
| 8 | Spine reading (Task 10) | Flash-Lite, **24 rotated crops per call** at LOW/MEDIUM + **RapidOCR corroboration** | as row 5; ~7 calls | E: 5–15 s/call. M: RapidOCR 46–75 ms/crop CPU (~10 s per 150 crops) | VLM fills in plausible titles | `unidentified` + review; 3.8 Flash on top-value candidates only |
| 9 | Resolution (Task 10) | **Open Library** (keyless, UA) first; **Google Books** with a free key only on miss or disagreement; Wikidata optional | OL 3 req/s identified; covers by ISBN 100 per 5 min; GB about 1,000/day (unverified); **GB keyless gives 429** | M: OL 0.9–2.1 s per call. E: 150 books ≈ 60–90 s at 3 req/s, overlapped with the sweep | GB daily cap (about 6 sweeps/day if GB is always queried) | Cache every response; resolve rows as they close |
| 10 | Room geometry (Task 11) | WebXR points. **Path B:** ArUco + DA V2 Metric-Indoor-Small (local fp32) + plane fit | none (local) | E: 40 frames < 10 s depth, < 10 s numpy | Monocular depth scale drift; shelving hides walls | Rescale depth by ArUco; ceiling-line pan; review entry. MapAnything Space dev-only |
| 11 | Non-book items (Task 11) | Flash-Lite, 4 room keyframes per call, fixed taxonomy | as row 5; ~3–4 calls | E: 5–15 s/call | Small items missed | Roboflow SAM 3 text prompts per category (free credits) |
| 12 | Appraisal gate (Task 13) | Code; optional Wikidata publication year | none | E: ms | none | Default to `needs_appraisal` |
| 13 | Pricing (Task 12–14) | **Out of scope here.** SerpApi is not card-free (see the pricing note) | | | | |
| 14 | Packet, totals, report (Task 2, 15) | Code + Jinja2 | none | E: <2 s | none | none |
| 15 | Orchestration (Task 15) | asyncio with **per-model limiters** (10/10/5 RPM), 429 `retryDelay`, failover between Flash-Lite buckets, `runs/_quota.json` | daily RPD | **E: post-sweep total ≈ 1.5–3 min** (≈15 Gemini calls + resolution tail + SAM/depth) | One 429 storm stalls the packet | Queue with deadline: at T+4 min, finalise with what's done and route the rest to review |
| 16 | Checker agent (Task 18b, stretch) | `gemma-4-31b-it` (free, different model family) instead of `claude-opus-5-5` (paid) | 15 RPM / 1,500 RPD (unverified) | E: 5–20 s/batch | Less independent than another vendor | Skip (stretch) |
| 17 | Demo + submission (Task 17–19) | Android screen recorder; YouTube unlisted (≤15 min without verification) + Drive mirror; GitHub Free private repo with collaborators | none relevant | n/a | Agent voice missing from the recording | In-app MediaRecorder track; second phone |

**Cost per sweep becomes $0.** The brief still asks for it, so the ledger should report $0 billed, together with the token counts and a "would cost at paid rates" figure.

---

## 9. Plan tasks that change

- **Header "Models and services":**
  - Vision becomes `gemini-3.5-flash-lite` + `gemini-3.1-flash-lite` (with `gemini-3.8-flash` for tie-breaks).
  - The spine cross-check becomes local RF-DETR-nano bookshelf + SAM 2.1-small (optionally Roboflow SAM 3).
  - The room fallback becomes DA V2 metric depth + ArUco.
  - The checker becomes Gemma 4.
  - Drop the fal and Anthropic rows.
- **"Things only you can do":**
  - Remove "make sure billing is on" and the Anthropic and fal keys.
  - Add: confirm the tier in AI Studio; enable the Books API in the same project and create a Books key.
  - Optional: free HF account (token), free Roboflow account (no card), and request access to `facebook/sam3`.
  - Free space on D:, or use C: for venvs.
  - SerpApi Starter ($25) is now outside the constraint; see the pricing note.
- **Task 0 (G0):**
  - Add a free-tier Live soak that logs `prompt_token_count` per turn and any 429 `quotaId`.
  - Add a **Flash-Lite vs 3.8 Flash box_2d bake-off** on 3 real row frames (costs 3 of the 20 daily 3.8 Flash calls).
  - Time Flash-Lite calls.
  - The cloudflared WS check is already done on the laptop; repeat it from the phone.
- **Task 1:** `.env.example`:
  - drop `ANTHROPIC_API_KEY` and `FAL_KEY`;
  - `VISION_MODEL` becomes `VISION_MODELS=gemini-3.5-flash-lite,gemini-3.1-flash-lite`, plus `TIEBREAK_MODEL=gemini-3.8-flash`;
  - add optional `HF_TOKEN` and `ROBOFLOW_API_KEY`;
  - `CHECKER_MODEL=gemma-4-31b-it`.
- **Task 4:**
  - compression `trigger_tokens≈12000` and `target_tokens≈6000`;
  - live video 0.5–1 fps at LOW;
  - stop video after `end_sweep`.
- **Task 6:** stills default to 2 fps over the tunnel.
- **Task 7 (`llm.py`):**
  - per-model RPM limiters, RPD counter, `RetryInfo` handling and bucket failover;
  - multi-image requests with index-keyed schemas;
  - a quota guard so tests and replay never call the live API.
- **Task 8:**
  - one detection call per keyframe (both tiles);
  - **new `vision/rfdetr.py`** (ONNX, DirectML or CPU) whose boxes are reconciled with Gemini's;
  - optional SAM 2.1 box-prompted masks.
- **Task 9:** add Open Library `physical_dimensions` as an edition-height cross-check, flagged and never a primary scale.
- **Task 10:**
  - reading batch goes from 8 to 24;
  - **RapidOCR corroboration** replaces Cloud Vision;
  - Open Library first, Google Books only on miss or disagreement (1,000/day cap).
- **Task 11:**
  - Path B drops MapAnything/VGGT hosted and uses DA V2 Metric-Indoor-Small + ArUco + planes;
  - items are batched 4 keyframes per call.
- **Task 15:**
  - "Gemini 16" concurrency becomes per-model RPM limits;
  - detection and reading run during the sweep;
  - a T+4 min deadline finalises the packet.
- **Task 18 (README):**
  - free-tier data-use disclosure ("human reviewers may read…");
  - licences (RF-DETR weights Apache, SAM 2.1 Apache, DA V2 small Apache; no Ultralytics; Piper GPL only if used);
  - the optional ML venv install (§2.4).
- **Task 18b:** the checker uses Gemma 4 instead of Claude.

---

## Sources

**Gemini:**
- https://ai.google.dev/gemini-api/docs/pricing (updated 2026-10-22)
- https://ai.google.dev/gemini-api/docs/rate-limits (updated 2026-10-15)
- https://aistudio.google.com/rate-limit (login)
- https://ai.google.dev/gemini-api/docs/billing
- https://ai.google.dev/gemini-api/terms (effective 2026-03-23)
- https://ai.google.dev/gemini-api/docs/image-understanding (updated 2026-09-23)
- https://ai.google.dev/gemini-api/docs/media-resolution (updated 2026-10-13)
- https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api

**Secondary, for limits:**
- https://www.scriptbyai.com/gemini-api-free-tier-limits/ (2026-10-19)
- https://www.scriptbyai.com/gemini-api-rate-limits/ (2026-10-22)
- https://tilnote.io/pages/6a51b3398d9a59d14c1c174c
- https://discuss.ai.google.dev/t/official-concurrent-session-rps-limits-for-gemini-live-api-where-are-they-documented/174664
- https://github.com/google-gemini/gemini-cli/issues/8437

**Voice:**
- https://developer.mozilla.org/en-US/docs/Web/API/SpeechRecognition
- https://github.com/OHF-Voice/piper1-gpl
- https://huggingface.co/hexgrad/Kokoro-82M (licence from memory, unverified)

**Models and hosting:**
- https://huggingface.co/docs/hub/spaces-zerogpu
- https://huggingface.co/docs/inference-providers/pricing
- https://roboflow.com/pricing
- https://docs.roboflow.com/deployment/roboflow-cloud/serverless-api/model-pricing
- https://docs.roboflow.com/deploy/supported-models/sam3
- https://huggingface.co/Geraldine/rf-detr-nano-bookshelf
- https://dev-discuss.pytorch.org/t/notice-cuda-12-6-wheels-will-no-longer-be-published-from-pytorch-2-15-drops-maxwell-pascal-volta/3432
- https://github.com/pytorch/pytorch/issues/58123
- https://www.ultralytics.com/license

**OCR:**
- https://github.com/RapidAI/RapidOCR
- https://docs.cloud.google.com/vision/docs/setup
- https://cloud.google.com/vision/pricing

**Tunnels:**
- https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/
- https://localhost.run/docs/
- https://ngrok.com/docs/pricing-limits/free-plan-limits
- https://tailscale.com/kb/1223/funnel

**Catalogue:**
- https://openlibrary.org/developers/api
- https://openlibrary.org/dev/docs/api/covers
- https://developers.google.com/books/docs/v1/using
- https://query.wikidata.org/

**Demo:**
- https://support.google.com/youtube/answer/71673
- https://docs.github.com/en/get-started/learning-about-github/githubs-plans

**Note on test hygiene:** the live Open Library, Google Books and Wikidata test (`_scratch/scripts/catalog_test.py`) sent a User-Agent that included the user's email. So did a helper's single request to the Wikimedia Commons API. Open Library asks for a contact in the UA, and the plan has `OPENLIBRARY_CONTACT_EMAIL` for that purpose. The email was not asked for in these tests, though. Use a generic UA in tests from now on.
