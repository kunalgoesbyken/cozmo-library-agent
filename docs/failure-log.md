# Failure log: the demo run

Run `e5bee844d3f74a4b836be8b350e71d93`, the sweep in the demo video. No physical shelf was available, so a fake camera panned over two public shelf photos (41 Pelican paperbacks and 19 Latin and German classics, 60 books). At 30 s the camera whips too fast on purpose to test recovery. Full results: [results.md](results.md). The development-phase log is in [failure-log-dev.md](failure-log-dev.md).

## The three worst errors

### 1. Going back over a blurred stretch counted the same books twice (86 counted for 60)

- Symptom: in the first take with the deliberate whip, the agent correctly said "The image is too blurry. Please redo this row", and the user went back. The packet then held 86 books. One blurred still had become a shelf of its own (S2) with 23 unreadable "books".
- Root cause: the capture gate flagged the stills below `blur_floor`, and the agent acted on that flag. But keyframe selection still passed the best still of each 1.5 s window to spine detection, even when every still in the window was blurred. So the rejected pass and the redo were both counted.
- Fix (`libclaim/vision/keyframes.py`): a window whose best still is below `blur_floor` gives no keyframe. The still is kept as evidence and is never counted.
- Measured on the same recording, replayed from the keyframes stage:

| | Before | After |
|---|---|---|
| Books counted (truth 60) | 86 | 63 |
| Count error | 43% | 5.0% |
| Final take with the fix live | | 62 (3.3%) |

### 2. Title identification is 55.6%, below the 70% bar

First fix, measured across two recordings of the same shelves: Google Books is now retried with a plain `title author` query when the strict `intitle:/inauthor:` query returns nothing. In an earlier take that query returned nothing for 40 correctly read spines. Identified books went from 5 (take `26799d7a`) to 13 (take `2cfa238a`), with none wrong.

The next fixes were measured on take `2cfa238a`, replayed from the resolution stage against the same ground truth after each change:

| Change | Correct of 54 legible | Confidently wrong |
|---|---|---|
| Starting point (with the retry above) | 13 | 0 of 13 |
| A missing ISBN no longer sends a book to `needs_appraisal`. The book stays `identified` with a blank price, which is what the brief asks for unknowns | 26 | 5 of 31 |
| A catalogue title that contains the author ("Chartres, By George Henderson") is scored and shown without the author, when the spine agrees. Same-title records with differently spelled authors (Peter Merin and Peter Thoene) are no longer counted as rivals | 30 | 1 of 31 |

- Root cause of the remaining misses: the spine reads are right (OCR and Gemini agree on "War Gaming / Andrew Wilson"), but Google Books has no record of most 1960s–70s Pelican editions. Open Library has them, but its `robots.txt` disallows the search endpoint, so the pipeline does not use it.
- The one confidently wrong title is "Die Mosella" against the hand-read "Mosella". The spine prints "AUSONIUS · MOSELLA", so it is arguable. It is still counted as wrong (1 of 31, 3.2% against a 3% bar).
- What I would change: add a catalogue that permits title search for old editions (a library union catalogue, or a local Open Library data dump). I would also accept two independent Gemini reads plus OCR at a lower `id_confidence`, and measure that change against the confidently-wrong bar.

### 3. Two of 62 books priced

- Symptom: replacement cost INR 2,786.62 comes from only two books (Victorian Cities and Palladio, Penguin UK new prices in GBP converted at 126.78). Both match the live Penguin UK pages (GBP 14.99 and 6.99, checked 2026-10-28). The other 29 identified books have a blank price and are listed in `excluded_from_totals`.
- Root cause: the books are out of print. Bookchor and Pustakkosh list none of their ISBNs, and the ISBN the catalogue returns is often a later reprint, priced only as an `equivalent_edition`. The used value is a derived condition factor, not an observed listing.
- What I would change: add used-book marketplaces that publish condition and price for out-of-print titles (AbeBooks, eBay India), and allow a title-and-author retailer search when no exact ISBN matches. Those lines would be marked `edition_unverified`.

Also missed: 0 of 2 non-book items (a spiral-bound notebook and a bookend at the edges of the pan). The items detector runs on keyframes chosen for sharp spines.

## Cost per sweep

| | USD |
|---|---|
| Live voice (`gemini-3.8-live`) | 0.0745 |
| Spine boxes | 0.0230 |
| Spine reading | 0.0079 |
| Items | 0.0002 |
| Total at paid-tier rates | 0.1057 |
| Billed (free tier key) | 0.00 |

237,553 tokens.

## Latency per stage (seconds)

Streaming stages start during the sweep, so their spans overlap it. Busy time sums concurrent work.

| Stage | Span, first start to last end | Busy, summed work time |
|---|---|---|
| keyframes | 104.7 | 1.2 |
| spines | 108.6 | 186.0 |
| tracks | 76.3 | 13.5 |
| scale | 73.0 | 0.0 |
| reading | 87.1 | 47.9 |
| resolution | 96.9 | 1905.0 (concurrent per-book lookups) |
| items | 11.6 | 11.6 |
| room | 5.1 | 5.1 |
| appraisal, pricing, packet | under 0.2 | under 0.2 |
| **Packet ready after the sweep ended** | **58.0** | bar: under 300 |
