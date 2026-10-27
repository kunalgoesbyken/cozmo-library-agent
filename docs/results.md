# Results against ground truth

> Ground truth for this run comes from the two public shelf photos the demo camera panned over, because no physical shelf was available: books counted and titles read by hand from the full-resolution photos ([ground_truth/SOURCE.md](../ground_truth/SOURCE.md)). Spine rulers, room tape measurements and eight non-book items do not exist for photos, so those bars are NOT EVALUATED rather than filled with guesses. Two prices were hand-checked on the cited retailer page (Penguin UK, GBP); the packet reports INR, so they are listed but not comparable.
>
> The one confidently wrong book is "Die Mosella" against the hand-read "Mosella": the spine prints "AUSONIUS · MOSELLA" and the catalogue edition is titled "Die Mosella", so it is arguably right, but it is counted as wrong here rather than editing the ground truth after the fact.


Run: `e5bee844d3f74a4b836be8b350e71d93`

How the bars are read in this project (the brief gives tolerances, not how to aggregate them):

- Book count: error within 5.0% of the ground-truth row count.
- Titles: correct identifications over legible ground-truth spines at least 70.0%, and confidently wrong over all identified books at most 3.0%.
- Spine dimensions: the share of all 20 hand-measured spines within 15.0% must be at least 80.0% and the mean error at most 15.0%, for height and thickness. Unmatched spines and missing predictions count as failures; fewer than the required spines means not evaluated.
- Prices: over rows with the same source type, share within 25.0% at least 80.0% and mean error at most 25.0%; at least 10 comparable rows are required, otherwise not evaluated.
- Items: at least 80.0% found with the correct category, from at least 8 items.
- Floor within 10.0%, wall within 15.0%.
- Time: seconds after the sweep ended under 300.

## Pass bars

| Pass bar | Result | Note |
|---|---|---|
| Book count | PASS |  |
| Title identification | FAIL |  |
| Spine dimensions | NOT EVALUATED | spines20.csv not found in the ground-truth directory |
| Book prices | NOT EVALUATED | comparable 0 of 2; the bar needs at least 10 same-source rows |
| Non-book items | NOT EVALUATED | items.csv has 2 items; the bar needs at least 8 |
| Floor area | NOT EVALUATED | room.json not found in the ground-truth directory |
| Wall area | NOT EVALUATED | room.json not found in the ground-truth directory |
| Time to packet | PASS |  |

Could not be evaluated: Spine dimensions, Book prices, Non-book items, Floor area, Wall area.

## Book count

| True count | Packet count | Error | Aligned misses | Aligned insertions | Bar (within 5.0%) |
|---|---|---|---|---|---|
| 60 | 62 | 3.3% | 0 | 2 | PASS |

## Title identification

| Measure | Value | Bar |
|---|---|---|
| Legible ground-truth spines | 54 |  |
| Correctly identified | 30 of 54 |  |
| Accuracy | 55.6% | >= 70.0% |
| Packet books with status identified | 31 |  |
| Confidently wrong | 1 of 31 |  |
| Confidently wrong rate | 3.2% | <= 3.0% |
| Result | FAIL |  |

## Confidently wrong books

| Book | Slot | Ground truth | Packet title | Match score | Frame |
|---|---|---|---|---|---|
| b04-005 | S4 #5 | Mosella | Die Mosella | 78 | frames/00208_1793132912946.jpg |

## Unidentified and review queue

| Measure | Count |
|---|---|
| Unidentified books | 31 |
| Books needing appraisal | 0 |
| Review queue entries | 150 |
| Review queue, high | 0 |
| Review queue, medium | 112 |
| Review queue, low | 38 |

## Spine dimensions

Not evaluated: spines20.csv not found in the ground-truth directory.

## Book prices

| Hand-checked books | Comparable (same source type) | Mismatched source | Not priced | Mean error | Median error | Within 25.0%, of comparable | Result |
|---|---|---|---|---|---|---|---|
| 2 | 0 of 2 | 2 | 0 | n/a | n/a | n/a | NOT EVALUATED |

Mismatched source rows, excluded from the bar:

| ISBN | Title | Source | Hand-checked | Packet | Why |
|---|---|---|---|---|---|
| 9780140135824 | Victorian Cities | Penguin UK | 14.99 | 1900.43 | source Penguin UK (equivalent edition, ISBN 9780140135824) differs from Penguin UK; currency INR differs from GBP |
| 9780141936383 | Palladio | Penguin UK | 6.99 | 886.19 | currency INR differs from GBP |

## Non-book items

| Ground-truth items | Found | Found with correct category | Found with wrong category | Rate | Bar (>= 80.0%) |
|---|---|---|---|---|---|
| 2 | 0 | 0 | 0 | 0.0% | NOT EVALUATED |

| Item | True category | Packet item | Packet category | Correct |
|---|---|---|---|---|
| spiral-bound notebook | other | not found |  | no |
| white bookend | storage | not found |  | no |

## Floor area

Not evaluated: room.json not found in the ground-truth directory.

## Wall area

Not evaluated: room.json not found in the ground-truth directory.

## Time to packet

| Seconds after the sweep ended | Source | Bar | Result |
|---|---|---|---|
| 58.0 | after_sweep | < 300 | PASS |

## Cost and latency per stage

| Stage | Seconds | USD at paid rates |
|---|---|---|
| spines | 108.6 | 0.0230 |
| live | 112.1 | 0.0745 |
| reading | 87.1 | 0.0079 |
| items | 11.6 | 0.0002 |
| keyframes | 104.7 | 0.0000 |
| tracks | 76.3 | 0.0000 |
| scale | 73.0 | 0.0000 |
| resolution | 96.9 | 0.0000 |
| room | 5.1 | 0.0000 |
| appraisal | 0.0 | 0.0000 |
| pricing | 0.2 | 0.0000 |
| packet | 0.0 | 0.0000 |
| keyframes_busy | 1.2 | 0.0000 |
| spines_busy | 186.0 | 0.0000 |
| tracks_busy | 13.5 | 0.0000 |
| scale_busy | 0.0 | 0.0000 |
| resolution_busy | 1905.0 | 0.0000 |
| reading_busy | 47.9 | 0.0000 |
| items_busy | 11.6 | 0.0000 |
| room_busy | 5.1 | 0.0000 |
| appraisal_busy | 0.0 | 0.0000 |
| pricing_busy | 0.2 | 0.0000 |
| after_sweep | 58.0 | 0.0000 |
| total (stage sum) | 2846.2 | 0.1057 |
