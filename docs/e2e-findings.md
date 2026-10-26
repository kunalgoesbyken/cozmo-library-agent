
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

## Real shelf photos through the streaming pipeline (E2E A)

Run on 2026-10-26 with `python tools/e2e_images.py _scratch/e2e_images --hand-counts tools/spike/box_photos.json` (real `Llm` client, real Open Library, Google Books and retailer HTTP, local RF-DETR via DirectML). Photos: the three spike photos (blyton, german, library), each fed as one still, 2 s apart, zero motion. Runs: base venv with live Gemini (`e2e-a-base`, `e2e-a-base2`), then `e2e-a-base3` (base venv) and `e2e-a-ml2` (`.venv-ml`, Depth Anything V2 and SAM 2.1 active), both replaying the recorded Gemini replies from the `llm` cache (no new Gemini spend). Gemini spend for the task is about USD 0.021 at paid rates (the packet ledgers of the two live runs: 0.0057 and 0.0151). The shared `runs/spend.jsonl` moved more because other agents were writing to it at the same time. Output under `runs/` is not in git.

Counts (hand count in brackets): blyton 52 (55), german 22 (22), library 19 (19); 93 books. After the fixes below: 4 identified, 77 unidentified, 12 needs_appraisal; 0 books priced. Every field of the claim_packet contract was validated with a throwaway script (types, empty when unknown, frame_refs exist, totals add up, source, url and retrieved_at present whenever an amount is set): 0 contract problems in base3 and ml2. report.html parses with balanced tags and all 373 frame images exist.

Fixed in this task (each with an offline unit test):
- RF-DETR `session.run` was called from several threads at once on one DirectML session (the lock covered only the load). The first run died with a native segmentation fault, and with `-X faulthandler` the DML fused-node errors appeared in every spine detection. `rfdetr_boxes` now holds the lock around the run.
- Open Library and Google Books return nothing for a title with a trailing volume number or shelfmark ("Gödel, Escher, Bach 30017", "Erzählungen 1", "Transhimalaja 2"). Resolution now retries once without trailing digit-only tokens when the first search found nothing. Gödel, Escher, Bach and Transhimalaja moved from `no_catalogue_match` to identified. Roman-numeral volumes ("In Nacht und Eis I") are not handled.
- In `.venv-ml` the room depth model failed with `ImportError` ("metric depth model could not be loaded") because the depth model and SAM both do a lazy `from transformers import ...` in different threads at the same time (reproduced standalone: `cannot import name 'Sam2Model' from 'transformers'`). Loading is now serialised with one lock. After the fix room and items ran with depth (room 0.46 s).

Open, not fixed:

1. OCR is the throughput bottleneck. Severity: high. `ocr_text` on one 100x660 crop takes about 2.4 s on this CPU (RapidOCR, `use_dml: false`), and `crop_spine` runs it once per rotation, so about 5 s per spine and about 8 minutes for 93 spines (reading_busy 610 to 740 s). With the default `deadline_s: 240` the first run finished only 19 of 93 reads: 75 books got "reading did not finish before the packet was finalised" and none were identified. Concurrent segments also compete for the same CPU. In a real sweep the OCR overlaps the pan, but a long sweep over 100 books will still be cut. Proposed: OCR one rotation first (the aspect ratio tells which way text runs) and try the other only when the yield is low; downscale crops; cap onnxruntime threads per segment; raise `deadline_s` for batch use (the driver has `--deadline-s`).
2. Identification rate is low: 4 identified plus 12 needs_appraisal of 93 (17 percent titled, 4 percent with an ISBN). Where books are lost: `ocr_disagrees` 48 (the gate needs the OCR text to overlap the VLM text; on the low-resolution Blyton photo OCR returns an empty string for almost every spine although the VLM read them correctly, for example "The Secret of Skytop Hill and other stories" and "The Wishing Chair Collection"), `no_catalogue_match` 12, `illegible` 15 (the 21 Famous Five spines and the lying stacks), `not_confident` 4. No identified title was wrong against the photos (all 16 checked by eye), so the gate is safe, but it is too strict when OCR returns nothing. Proposed: when OCR is empty and the VLM read is legible, accept only when a second independent VLM read agrees, with a lower `id_confidence`, instead of rejecting.
3. No book got a price. Severity: high for the demo, expected for these photos. 12 titled books have no ISBN, because `pick_edition` needs a publisher printed on the spine. The 3 that had an ISBN in the first run (Little Brownie House 9781841354675, Three Strange Travellers 9781841350219, Eco 9783446243828) are not listed by Bookchor or Pustakkosh (`not_found`; Bookswagon `disabled`; Penguin UK `blocked` by robots.txt for non-Penguin titles). The price path itself works on a mainstream ISBN: Pride and Prejudice 9780141439518 gave INR 286 from Bookchor (checked against the live page: price 286, MRP 350, ISBN on page) and GBP 9.99 from Penguin UK, with used value derived from new (143 and 5.00, factor v1 "good"). `second_country` is present with 0 books because no book was priced. Proposed: pick the edition without a printed publisher (for example the dominant ISBN of the work, flagged `edition_unverified`), or search retailers by title and author; both need care to avoid a wrong-edition price.
4. Dimensions are format priors, not measurements, on photos without a marker: all 93 books have `scale_method: book_format_prior`, and heights take one of four constants (17.5, 21.0, 23.5, 30.0 cm). With the ML extra (SAM masks) 26 of the 52 Blyton books flipped to `large_format` 30.0 cm, though they are about 18 to 20 cm paperbacks (base venv: 1 of 52), so SAM refinement made the prior worse on this low-resolution photo. `shelf_run_m` (2.6 m base, 2.8 m ml) is built from the prior thicknesses, so it is not a measurement either. The review queue gets one medium entry per book ("book_format_prior ...: ARUCO_MARKER_SIZE_CM is not set"), 93 of 282 entries, which buries real items and leaks an env var name. Proposed: one aggregate entry per segment, and leave `spine_height_cm` empty when only a class prior exists (or label it in the report).
5. `excluded_from_totals` counts every unpriced book (93 of 93 here), not only lines the user excluded or that need an appraiser, so it reads as "everything was excluded". The brief does not define it. Severity: low.
6. Report: every row thumbnail is the whole saved frame scaled to 48 px height (373 images for 93 books), so a 1920x688 shelf photo becomes an unreadable strip, and the packet has no per-book box to find a book in the frame (the box exists only in `stages/tracks.json`). Severity: low to medium. Proposed: crop each book box into a small file in the sweep dir and reference that, and add an optional `box` to the packet line.
7. Possible duplicate or merged boxes: `b01-016` and `b01-017` both read "The Secret of Killimooin" (lying stack, box IoU 0.45), and `b01-041` and `b01-043` overlap at IoU 0.47. Overall the blyton count is 3 below the hand count, so the error is a miss, not an over-count. Severity: low.
8. Latency: `after_sweep` was 240 s in the first run (deadline cut), and 307 to 393 s once the deadline was lifted. Resolution takes 160 to 207 s wall for 93 books because books are resolved one at a time inside a segment (about 3 catalogue calls each at 1 request per second) and only after that segment's reading ends. Proposed: resolve books of a segment concurrently under the catalogue rate limiter.
9. `FREE_TIER` is unset in `.env`, so `costs.billed_usd` is `null` ("unknown billing") on every run. Not a bug; set it before recording the demo.
10. Driver note: stills need epoch-based `t_ms`. Relative values (0, 2000, 4000) produced `captured_at: 1970-01-01`; the driver now uses the wall clock as the base. `sweep.device` is empty for batch runs because there is no hello.

## Fix wave F2

Replayed offline on copies of `runs/e2e-a-base3` (base venv) and `runs/e2e-a-ml2` (`.venv-ml`, SAM): the scale stage was recomputed from the recorded tracks, then the packet stage re-ran with all other stages as recorded (`runs/f2-base`, `runs/f2-ml`; no network, no Gemini). Output is under `runs/`, not in git.

Findings 4, 5, 6 and 7:

- **4, SAM made heights worse.** Root cause, measured on the Blyton frame with the real boxes: SAM masks match the detector height (median ratio 0.99) but the mask width covers one spine of a detector box that straddles several (for example `b01-021`: box 53 px, mask 36 px; stacked books give slivers, `b01-002` mask height 0.41 of the box). The format prior chose its class from the aspect ratio of the mask box, so a narrower box gave a larger aspect and a bigger format: 28 of the 52 Blyton books became 30 cm `large_format` (3 without SAM). Fix: priors use the detector box only (and SAM is no longer run when there is no scale); a metric measurement takes a mask only when its height is within `masks.height_tolerance` (0.1) and its width ratio within `masks.min_width_ratio`/`max_width_ratio` (0.5 to 1.15) of the detector box, else the trimmed detector box. Tests: synthetic bleeding and sliver masks, a mask that would flip the class, and an opt-in `LIBCLAIM_ML=1` test on the recorded real shelf (`LIBCLAIM_REAL_RUN`).
- **4, priors presented as measurements.** Spine fields are empty unless a metric scale was achieved. The prior goes to `dims_estimated_from_prior {height_cm, thickness_cm, book_format}`; `dims.fill_from_prior: false` in `config/pipeline.yaml` is the owner's switch. `shelf_run_m` is measured only (`null` when nothing was measured) and `shelf_run_estimated_m` holds the prior run. The report prints "est." in italics. The review queue has one entry per shelf row (`6 books on S1.1 have no measured size (no scale marker was visible); ...`), no env var names (also removed from the room depth review text).
- **5, `excluded_from_totals`.** Now counts identified books with no sourced replacement price (this includes books the user excluded) and items that are not `needs_appraisal` and have no price range. `needs_appraisal` and unidentified lines have their own counters. See `docs/packet-notes.md`.
- **6, thumbnails.** Each book and item gets `crop_ref` (`frames/crops/<id>.jpg`, longest side at most `crops.max_side_px` = 256, about 4x the 64 px display height), pixel `box` and `frame_size`, written during packet assembly. The report shows the crop and opens the full frame with the box outlined. All 562 frame and crop refs of the replay exist.
- **7, duplicate boxes.** Not a bug in track merging or NMS: the run is one still and one segment, so no track was merged. `b01-016`/`b01-017` are overlapping axis-aligned boxes of a tilted lying stack; `b01-041`/`b01-043` is a single detector box spanning two spines. Both pairs sit under the NMS thresholds (0.6 IoU, 0.8 containment; the second pair has containment 0.74) and loosening them would merge real neighbours while the count is already 3 below the hand count. Documented in `docs/packet-notes.md`, no code change.

Before and after (93 books, 3 photos, no marker in any of them):

| | base before | base after | ml before | ml after |
|---|---|---|---|---|
| Blyton books at 30 cm class | 3 | 3 | 28 | 3 |
| All books by prior class 17.5 / 21 / 23.5 / 30 cm | 23 / 11 / 44 / 15 | 23 / 11 / 44 / 15 | 9 / 15 / 27 / 42 | 23 / 11 / 44 / 15 |
| `spine_height_cm` filled | 93 | 0 | 93 | 0 |
| `shelf_run_m` / `shelf_run_estimated_m` | 2.615 / - | null / 2.615 | 2.835 / - | null / 2.615 |
| Review queue entries | 282 | 192 | 286 | 195 |
| No-marker entries | 93 | 4 | 93 | 4 |
| `excluded_from_totals` | 93 | 4 | 93 | 4 |

The prior classes are still a poor guess (the Blyton paperbacks are about 18 to 20 cm and the aspect picks 17.5 to 23.5 cm), which is why they are only estimates now. Report check: `runs/f2-ml/report.html` rendered to PNG with headless Edge; the books table shows the crop thumbnails, "est. 23.5 / 4.5" cells, and the opened full frame with the box outlined (`runs/f2-ml/report_books.png`, `report_top.png`).

Open: unidentified books still produce two per-book entries ("spine could not be identified" plus "identification: <reason>"); those are individually distinct enough to keep for now. The resolver is still given the prior height and thickness by `pipeline.resolve_book` (F3 owns `pipeline.py`), so that call should pass nothing when `scale_method` is `book_format_prior`.

## Fix wave F1

Findings 1, 2, 3 and 8. Measured offline on copies of `runs/e2e-a-base3` (`runs/f1-before`, `runs/f1-after`, `runs/f1-hybrid`, not in git). No Gemini call was made. Live HTTP: 24 requests (2 Open Library robots.txt, 1 googleapis robots.txt, 7 Google Books searches, 3 Open Library editions lists, 11 retailer pages including redirects), 4 over the planned 20 because Bookchor and Penguin UK redirect unknown ISBNs.

**1, OCR throughput.** Root cause: RapidOCR's default detector setting (`limit_type: min`, `limit_side_len: 736`) scales the short side of every crop up to 736 px, so a 100 x 660 spine became about 736 x 4858 before detection, and `crop_spine` always ran both rotations. Profile on the 93 real crops (one engine, one call at a time, both rotations, 173 calls):

| detector input | s per call | legible reads OCR corroborates (of 78) |
|---|---|---|
| default (min side 736) | 2.91 | 30 |
| max side 960 (never upscale) | 0.15 | 42 |
| max side 640 | 0.13 | 42 |
| min side 128 / 160 / 256 | 0.21 / 0.27 / 0.48 | 44 / 47 / 45 |
| max side 960, DirectML | 0.18 | - |
| max side 960, text-direction classifier off | 0.11 | 42 |

Upscaling hurts both speed and quality. DirectML does not help these small inputs. Chosen (all in `config/resolve.yaml` `ocr`): max side 960, classifier off (an upside-down line then reads as low-yield garbage, which is what tells the two rotations apart), 2 intra-op threads, one shared engine in a pool of 3 workers shared by all segments, first rotation from `reading.rotations`, the second only when the yield (score x characters) is below 32. That is 156 OCR calls for 93 crops. Wall time per crop over all 93 crops: 0.50 s (1 worker, 1 thread), 0.22 s (1 worker, 2 threads), 0.13 s (3 workers, 2 threads, the default), 0.12 s (4 x 2). Replay `--from reading` on the old code: reading busy 546 s and cut at the 240 s deadline, 0 books resolved. New code: the reading stage of all three segments takes 22.9 s, of which OCR is 12.0 s. Measured on the real reads, OCR now supports 49 of 78 legible titles (31 before) and is empty for 11 (41 before).

The new rotation choice changes the image sent to Gemini for 41 of 93 crops, so the recorded reading batches no longer match their cache keys and a pure `--from reading` replay ends with `reading_failed` for all books. The identification numbers below therefore use a hybrid replay: the recorded Gemini reads, the new OCR text for the same crops, then `--from resolution` with the new code. A live reading run (5 Gemini batch calls) is needed for an end-to-end number.

**2, identification gate.** `accept_identification` now returns how a read was corroborated: `ocr` (the old rule: OCR text covers at least 0.6 of the title words) or `catalogue_only` (OCR empty or weak and not contradicting the read, the author read and matched in the best catalogue record, score at least 95, title similarity at least 95, margin at least 15 to the next work, and the old lone-candidate and margin gates). `catalogue_only` lines get `id_confidence` = score x 0.5 and a review note. OCR text that shares no word with the read still rejects (`ocr_disagrees`); a read without an author and without OCR support is rejected before any catalogue call (`not_corroborated`). OCR words run together ("FAMOUSFIVE", "GemeinsameWerke1") now count as support for words of 4 letters or more. Hold-out tests: an invented title, the right title by a different author, IT and EMMA without an author, OCR that reads other words, a title whose runner-up is one word longer. A new safety rule: when the same title is identified in boxes of one frame that overlap (IoU at least 0.25), only the first keeps it and the others become `repeated_read`. On this shelf "Well Done, Secret Seven" was read from three overlapping boxes (`b01-040`, `b01-041`, `b01-043`, see F2 finding 7), and only `b01-040` keeps the title.

Open Library's robots.txt has `Disallow: /search` for every agent, which covers `/search.json`. No live Open Library search was made. Of the 48 former `ocr_disagrees` books, 42 now end as `catalogue_unavailable` in the replay because their searches are not cached. Google Books (googleapis.com has no robots.txt) was queried for 7 of them: "Well Done, Secret Seven" is identified from the catalogue only. "Five Go To Smuggler's Top" and "Naughty Amelia Jane" stay unidentified at a margin of 14 (the runner-ups are "Famous Five: Five Go To Smuggler's Top" and "Amelia Jane is Naughty Again"). Google Books returned nothing for the other 4.

**3, edition choice and prices.** When the printed edition is unknown (no publisher on the spine, a publisher that matches no edition with an ISBN, or several ISBNs from that publisher), the resolver takes the most recent edition with an ISBN-13 whose title matches the read (title similarity at least 95, so no translation), preferring `editions.preferred_formats` (paperback). The line keeps `edition: ""` unless the printed publisher matched, the resolution has `edition_unverified` and `equivalent_edition`, and prices from that ISBN carry the source "Bookchor (equivalent edition, ISBN ...)" plus an `equivalent_edition_priced` review note. `no_isbn` now only means that no edition with an ISBN was found. A new appraisal reason `old_work_edition_unverified` keeps an equivalent edition from being priced when the work is older than `antiquarian_year` and the spine shows no matching publisher. Second-country repricing reuses the stored resolution, so the label survives there too. Of the 12 old `no_isbn` books, 4 now have an equivalent ISBN (The Secret of the Lost Necklace, The Wishing Wand, Der Orientalist, The Cathedral and the Bazaar). Vagantenbeichte and Lateinische Gedichte deutscher Humanisten have no ISBN-13 edition with a matching title, and the rest are works from before 1950 or have no fetched editions. Live check of three equivalent ISBNs (9780596001315, 9783940731050, 9780753727041): none is listed at Bookchor or Pustakkosh (`not_found` with the page as evidence), and Penguin UK redirects to a path that robots.txt disallows. They are excluded from the totals. No other free source qualifies (research/05: Bookswagon terms, Crossword and Sapna robots.txt, Amazon and Flipkart terms), so none was added.

**8, resolution latency.** Books are resolved concurrently under the per-host catalogue limiter (identical in-flight requests are shared), and a book's resolution starts as soon as its reading batch returns instead of after the whole segment. Price prefetch and final pricing also run concurrently. Test: 30 books with a 50 ms catalogue resolve in parallel, and resolution starts while the second reading batch is still pending.

Before and after (93 books):

| | before (`e2e-a-base3`) | after (hybrid replay) |
|---|---|---|
| identified | 4 | 9 |
| needs_appraisal | 12 | 9 |
| unidentified | 77 | 75 (42 `catalogue_unavailable`, 15 illegible, 7 not_confident, 7 no_catalogue_match, 2 not_corroborated, 2 repeated_read) |
| books with an ISBN | 4 | 9 (5 equivalent editions) |
| priced | 0 | 0 |
| OCR per crop | about 5.8 s (2.9 s x 2 rotations) | 0.22 s on one worker, 0.13 s with the pool |
| replay `--from reading`, after_sweep | 240 s, cut at the deadline, 0 resolved | 23.0 s (Gemini cache misses, see above) |
| replay `--from resolution`, after_sweep | not run (the recorded live run spent 160.7 s in resolution, 307 s after the sweep) | 0.43 s (all requests cached) |

Projected for a live run of these photos, not measured: OCR 12 s, Gemini reading about 25 s for the longest segment, catalogue requests about 95 s at 1 request per second on the Open Library host, retailers about 25 s, so about 150 s after the sweep against 307 to 393 s before.

## Fix wave F3

- Catalogue: title and author search is Google Books only; Open Library is used for `/isbn/{isbn}.json`, `/works/{work}/editions.json` (robots.txt disallows `/search`). Search fixtures under `tests/fixtures/catalog` for Google Books and the isbn lookups were hand-built from earlier recorded data (no live HTTP in this wave); `tools/record_catalog.py` re-records them.
- Live: exclusion `shelf_id` is validated against shelves the pipeline has reported (unknown id returns a recoverable tool error listing known shelves); `connecting` is sent once; inventory upserts are sent only when an item changed; coverage regions move from `sweep` to the shelf once the pipeline places a frame; injections before `start_capture_monitor` go out as a text turn (silent ones are dropped); a budget stop sends an `error` (code `budget`) and `closed` with reason `budget_exceeded`.
- Abrupt disconnect: the code path is the same `finish()` as `end_sweep` with the normal deadline. The roughly 3 minutes seen is 20 stills sent in one burst with no local detector, bound by the Gemini per-model request limits and `vision_cutoff_s`; no code change.
- Replay: `--from scale` reuses recorded (non-failed) reads, so changed crop images no longer discard them; `--from reading` still re-reads.
- Packet book lines carry `edition_unverified`, `equivalent_edition`, `corroboration`.
- Security: SSRF guard rejects 64:ff9b::/96, 64:ff9b:1::/48, ::ffff:0:0/96 and ::/96 outright and checks the embedded IPv4 of 6to4 and Teredo; the price client has `trust_env=False`; `requires-python` is `>=3.12.4`. Pre-auth identity uses only a valid `CF-Connecting-IP`.

## F3 live re-run (`runs/f3-live`)

Run on 2026-10-26 after wave F3, with `tools/e2e_images.py` on the same three photos, base venv (no `[ml]` extra), live Gemini (9 calls: 3 spine, 5 reading, 1 items), live Google Books and retailer pages. Not in git (`runs/` is ignored). The live voice session was not part of this run.

| | E2E-A before fixes (`e2e-a-base3`) | After F1 (hybrid replay) | F3 live re-run |
|---|---|---|---|
| Counted (hand count) | 52 (55), 22 (22), 19 (19) | same | 52 (55), 22 (22), 19 (19) |
| identified | 4 | 9 | 14 |
| needs_appraisal | 12 | 9 | 10 |
| unidentified | 77 | 75 | 69 (41 no_catalogue_match, 15 illegible, 7 not_corroborated, 6 not_confident) |
| Books with an ISBN | 4 | 9 | 19 |
| Books with a price | 0 | 0 | 1 (used, INR 135, The Wishing Chair Collection, equivalent edition via Bookchor) |
| `spine_height_cm` filled | 93 | 0 | 0 (all `book_format_prior`) |
| Review queue entries | 282 | 192 | 216 |
| Time after the sweep | 307 s | 23 s (reading replay, partly cached) | 106 s |
| Gemini USD at paid rates | 0.0151 | none (replay) | 0.0152 |

The columns are not a controlled comparison: the reads, the OCR settings and the catalogue source changed between them, and the F1 column is a replay. The 14 identified titles of the F3 run were not checked against the photos by hand. Totals: replacement cost 0.0, used value 135.0 (observed), 14 lines excluded from totals.

What the run showed:

- The streaming pipeline now finishes inside the 5-minute bar on these photos (106 s). Resolution (78.9 s, one catalogue request per second) is the longest post-sweep stage, then reading (27.5 s).
- Identification doubled again once OCR no longer starves the gate, but still misses most books: 41 end as `no_catalogue_match` with Google Books as the only title search, since Open Library's `/search` is disallowed by its robots.txt.
- Price coverage did not improve with identification: Bookchor and Pustakkosh list almost none of the editions found. This is a limit of free sources, not a code defect, and the packet says so per line (`not_found` with the page as evidence).
- Cost and latency per stage are in [failure-log.md](failure-log.md).

Still unverified: iPhone behaviour, marker-calibrated dimensions, a real GoAway reconnect, compression firing, and every pass bar that needs a ground-truth sheet.
