# Failure log

What went wrong, found by running real photos and a real live session through the pipeline, not by unit tests. Test data: three public shelf photos (Wikimedia Commons: Blyton, German, Latin; 93 books by my hand count of 96, no markers in any of them), fed as one still each. Details and the other findings are in [e2e-findings.md](e2e-findings.md).

## The three worst errors

### 1. OCR was so slow that the deadline cut identification to nothing

- Symptom: with the default 240 s deadline, only 19 of 93 reads finished. 75 books carried "reading did not finish before the packet was finalised"; none was identified. Batch run time was 307 to 393 s after the sweep when the deadline was lifted, against a 300 s bar.
- Root cause: RapidOCR's default detector setting (`limit_type: min`, `limit_side_len: 736`) scales the short side of every crop up to 736 px, so a 100 x 660 spine became about 736 x 4858 before detection. `crop_spine` also always ran both text rotations, and every segment competed for the same CPU. About 2.9 s per call, 5.8 s per spine.
- Measurement: profile on the 93 real crops (173 calls) in [e2e-findings.md](e2e-findings.md), Fix wave F1.
- Fix: never upscale (`det_limit_type: max`, 960 px), classifier off, 2 threads per call, one pool of 3 workers shared by all segments, the second rotation only when the first has low yield (156 calls for 93 crops). Reading results stream into per-book resolution, and books resolve concurrently under the catalogue rate limiter.

| | Before | After |
|---|---|---|
| OCR per call | 2.91 s | 0.15 s (0.13 s per crop wall time with the pool) |
| Legible reads OCR corroborates (of 78) | 31 | 49 |
| Replay `--from reading` | 240 s, cut at the deadline, 0 resolved | 23.0 s |
| Live latest run, after the sweep | 307 s (recorded E2E-A) | 106 s |

### 2. Dimensions looked measured but were class constants, and SAM made them worse

- Symptom: all 93 books had a spine height and thickness that took one of four constants (17.5, 21.0, 23.5, 30.0 cm), with `shelf_run_m` built from them. With the ML extra, 28 of the 52 Blyton books became 30 cm `large_format`, though they are 18 to 20 cm paperbacks (3 without SAM).
- Root cause (two faults): (a) with no marker the format prior was written into the contract fields as if measured; the brief says an empty field is correct when the system does not know, and a filled field with no evidence is a defect. (b) SAM masks match detector height (median ratio 0.99) but cover one spine of a detector box that straddles several, so a narrower mask gave a bigger aspect ratio and the prior picked a larger format.
- Fix: the prior uses the detector box only; a metric measurement takes a SAM mask only when its height is within 10% and its width ratio within 0.5 to 1.15 of the detector box. Without a metric scale the contract fields are empty, the prior goes to `dims_estimated_from_prior` and the report prints "est.". `shelf_run_m` is measured-only (`null` otherwise).

| | Before | After |
|---|---|---|
| Blyton books at 30 cm, with SAM / without | 28 / 3 | 3 / 3 |
| `spine_height_cm` filled with no scale | 93 | 0 |
| `shelf_run_m` | 2.615 (base), 2.835 (ml) | `null`; `shelf_run_estimated_m` 2.615 |
| Review queue entries | 282 | 192 (one entry per shelf row instead of one per book) |

Still true afterwards: no real measurement exists. The marker path is covered by synthetic tests only; the prior classes are a poor guess for these photos.

### 3. The identification gate rejected books that were read correctly

- Symptom: 4 of 93 identified, 12 `needs_appraisal`, 77 unidentified; 48 of those had the reason `ocr_disagrees`. The Gemini read was right in the cases I checked by eye ("The Secret of Skytop Hill and other stories", "The Wishing Chair Collection"), but the gate wanted the OCR text to overlap the read, and OCR returned an empty string for almost every spine of the low-resolution Blyton photo.
- Root cause: one corroboration rule with no way to say "OCR has nothing". Also, an edition needed a printed publisher before any ISBN was chosen, so even identified books could not be priced (0 of 93 priced).
- Fix: `accept_identification` returns how a read was corroborated. `ocr` is the old rule. `catalogue_only` accepts an empty or weak OCR when it does not contradict the read, the author was read and matches the best catalogue record, and score, title similarity and margin to the next work pass strict thresholds; such lines get half the confidence and a review note. OCR text that shares no word with the read still rejects. A read without an author and without OCR support is rejected. Safety guard: the same title read in overlapping boxes is kept only once. When the printed edition is unknown, the most recent paperback ISBN-13 with a matching title is used, labelled `equivalent_edition` and sent to review. Hold-out tests: an invented title, the right title by a different author, IT and EMMA with no author, OCR that reads other words.
- Measured: in the first F1 hybrid replay identified rose from 4 to 9 and `needs_appraisal` fell from 12 to 9. After the later catalogue change (Open Library `/search` is disallowed by its robots.txt, so Google Books is the only title search) a live run of the same photos gave 14 identified and 10 `needs_appraisal` (24 titled, 19 with an ISBN) of 93. That last number is not like-for-like: the reads, the OCR and the catalogue all changed. No identified title was wrong in the 16 I checked by eye in the first run; I did not check the 14 of the latest run by eye.

An earlier bug of the same family, found in review before these photos: a segment split recounted overlapping books (21 became 40). Fixed with carry-over dedupe across segments; the Blyton count is now 52 against my hand count of 55 (a miss, not an over-count).

## Cost per sweep and latency per stage

From `runs/f3-live/claim_packet.json` (`costs`, `latency_s`), the latest live run: three photos, 93 books, pipeline stages only (no live voice session in this run). Not in git, `runs/` is ignored.

| Stage | Wall seconds | Gemini USD at paid rates |
|---|---|---|
| keyframes | 0.1 | 0 |
| spines (3 calls) | 11.0 | 0.0028 |
| tracks | 1.1 | 0 |
| scale | 1.1 | 0 |
| reading (5 calls) | 27.5 | 0.0121 |
| resolution | 78.9 | 0 |
| room | 0.5 | 0 |
| items (1 call) | 18.8 | 0.0004 |
| appraisal | 0.0 | 0 |
| pricing | 0.2 | 0 |
| packet | 0.0 | 0 |
| After the sweep (wall) | 106.0 | total 0.0152 |

- 9 Gemini calls, 121,046 tokens, USD 0.0152 at paid rates. `billed_usd` is `null` ("unknown billing") because `FREE_TIER` is unset; with a free-tier key the expected bill is USD 0.
- Resolution is the longest post-sweep stage: it is bound by one request per second to the catalogue host. Stages overlap, so the rows do not add up to 106 s. The `_busy` entries in the packet are summed task time (for resolution 2,457 s), not wall time.
- Room took 0.5 s only because the base install has no depth model; it returned no measurement.
- The live voice side is not in this run. The recorded 6-minute live session on the laptop (E2E-B, 150 stills, 158 books, no local detector) cost about USD 0.137 at paid rates and took about 240 s after the sweep, before the F1 fixes; that was not re-measured after them. The Gate G0 soak cost about USD 0.014 per minute of live audio and video.
- 106 s is for three stills over four seconds. A real two-shelf sweep sends many more stills and will take longer; it was not measured.

## What is still wrong

- Identification: 69 of 93 books are unidentified in the latest run (41 `no_catalogue_match`, 15 illegible, 7 `not_corroborated`, 6 `not_confident`). Nothing in the repo shows the true title accuracy, because there is no ground-truth sheet. The latest 14 identified titles are not checked by hand.
- Catalogue: title search depends on one source (Google Books) after Open Library `/search` was excluded for robots.txt. The test fixtures for it are hand-built from earlier recordings, and the identification rate against the previous two-source setup is unmeasured. A series prefix on a title ("Famous Five: ...") or a sibling title can leave a margin of 14 against the gate of 15, so "Five Go To Smuggler's Top" and "Naughty Amelia Jane" stay unidentified. Roman-numeral volumes are not handled.
- Prices: one book of 93 got a price (a used listing, INR 135, for The Wishing Chair Collection, an equivalent edition from Bookchor). `books_replacement_cost` is 0.0. Bookchor and Pustakkosh do not list most of these editions, and no other free source qualifies under their terms or robots.txt. Totals are therefore close to empty, and "excluded from totals" is the honest figure.
- Equivalent editions: the "most recent matching ISBN-13" rule can pick a print-on-demand reprint. It is labelled and reviewed, not verified.
- Dimensions: all 93 books are `book_format_prior`. No marker was used, so there are no marker-calibrated dimensions, the 2.5 cm recess is an assumption, and `box_height_trim` is uncalibrated. Pass bar not evaluated.
- Room: no area result from these runs (base install, no room photos); depth model accuracy against a tape is unmeasured. Pass bars not evaluated.
- Detector: overlapping boxes on tilted stacks and one box spanning two spines remain (NMS thresholds left alone on purpose); the Blyton count was 3 short on stills. On a simulated pan in a real browser the tracker first counted 123 books for 96 true; merging frame-edge slivers and dropping boxes seen in at most 40% of the frames that view them (`config/vision.yaml`, `tracks`) gave 93, but that pan was synthetic and no marker was at book scale.
- Noise: the review queue has 216 entries for 93 books (10 high, 137 medium, 69 low); unidentified books still get two entries each.
- Live and phone: no iPhone run, no real GoAway, compression at 12K/6K never observed firing, the abrupt-disconnect path takes minutes when 150 stills arrive in a burst and Gemini rate limits bind.
- Tuning: gates and cutoffs were tuned on three public photos, not on a real library sweep.
- Ground truth: none of the five files exists, so [results.md](results.md) can evaluate only time and cost.
- Google Books as a price source: switched off (`enabled: false` in `config/pricing.yaml`). Live queries on 2026-10-26 showed that `isbn:` returns nothing with the project key, a bare ISBN query returns unrelated volumes, and every `FOR_SALE` volume was an ebook, which is not a replacement cost. Google Books stays the title-search catalogue.
