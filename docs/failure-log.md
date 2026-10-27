# Failure log: the demo run

Run `2cfa238aa7524fbeaebaf3f84598bf3b`, the sweep in the demo video. No physical shelf was available, so a fake camera panned over two public shelf photos (41 Pelican paperbacks and 19 Latin and German classics, 60 books). Full results: [results.md](results.md). The development-phase log is in [failure-log-dev.md](failure-log-dev.md).

## The three worst errors

### 1. Only 13 of 54 legible spines were identified (24%, bar 70%)

- Symptom: 35 books stayed `unidentified` and 13 went to `needs_appraisal`, although the spine reads were right. Gemini read "MODERN GERMAN ART PETER THOENE", RapidOCR read "PETER PELICAN MODFRN GFRMAN ART", and the book still stayed unidentified.
- Root cause (two faults): (a) Google Books was queried only as `intitle:X inauthor:Y`. For 40 correctly read spines in an earlier take this returned zero results ("Modern German Art" is catalogued under the author's real name). (b) The gate then needs a catalogue score of at least 90 and a margin of at least 8 between different works. Old Pelican titles have several reissues with differently written author names, so the top two groups are near ties and 22 books end `not_confident`.
- Fixed during this work: when the strict query is empty, a loose `title author` query runs (`libclaim/catalog/googlebooks.py`). That moved `no_catalogue_match` from 40 books to 1 and identified books from 5 to 13. Confidently wrong stayed at 0 of 13.
- What I would change next: merge catalogue groups by normalised author surname before computing the margin, and accept a double agreement (two independent Gemini reads plus OCR) at a lower `id_confidence` instead of rejecting. I would measure the change against the confidently-wrong bar before shipping it.

### 2. Two of 61 books priced

- Symptom: replacement cost INR 2,786.62 comes from only two books (Victorian Cities and Palladio, Penguin UK new prices in GBP converted at 126.78). Both match the live Penguin UK pages (GBP 14.99 and 6.99, checked 2026-10-28). Eleven identified books have no price.
- Root cause: the books are 1960s–70s Pelicans. Bookchor and Pustakkosh list none of the ISBNs the catalogue returns, and the ISBN found is often a later reprint, so it is priced only as an `equivalent_edition`. The used value is a derived condition factor, not an observed listing.
- What I would change: add used-book marketplaces that publish condition and price (AbeBooks, eBay India) as sources for out-of-print titles, and let a title-and-author search on a retailer stand in when no exact ISBN matches. Mark it `edition_unverified`.

### 3. Non-book items, spine size and room were not measured

- Symptom: 0 of 2 non-book items were found (a spiral-bound notebook and a white bookend at the shelf edges). Every spine size is a format prior labelled "est.", and the room is unknown.
- Root cause: the items detector runs on keyframes picked for sharp spines, and the two objects sit at the edges of the pan. The photos have no ArUco marker, so there is no metric scale. A flat photo has no room geometry for depth to measure.
- What I would change: run item detection on a wide frame at the start and end of every segment, not only on spine keyframes. With a physical room, the printed marker and the depth stage would apply, as described in [architecture.md](architecture.md).

## Live session issues seen in the video

- The agent coached once at the start, then stayed quiet during the pan. Capture alerts ("Hold steady, that is blurry") still showed on screen.
- A "missed a spot" alert fired after the sweep had ended.
- The agent asked the user to "pan along the line where the wall meets the ceiling", a room step that makes no sense for a single shelf.

## Cost per sweep

| | USD |
|---|---|
| Live voice (`gemini-3.8-live`) | 0.0468 |
| Spine boxes | 0.0276 |
| Spine reading | 0.0079 |
| Items | 0.0002 |
| Total at paid-tier rates | 0.0826 |
| Billed (free tier key) | 0.00 |

204,572 tokens.

## Latency per stage (seconds)

Streaming stages start during the sweep, so their spans overlap it. Busy time sums concurrent work.

| Stage | Span, first start to last end | Busy, summed work time |
|---|---|---|
| keyframes | 94.1 | 1.1 |
| spines | 108.4 | 174.7 |
| tracks | 66.8 | 10.9 |
| scale | 62.0 | 0.0 |
| reading | 75.3 | 45.4 |
| resolution | 91.9 | 2125.2 (concurrent per-book lookups) |
| items | 11.9 | 11.9 |
| room | 4.5 | 4.5 |
| appraisal, pricing, packet | under 0.3 | under 0.3 |
| **Packet ready after the sweep ended** | **78.7** | bar: under 300 |
