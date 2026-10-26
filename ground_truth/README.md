# Ground truth

Fill these files in by hand, then run:

    python tools/eval.py runs/<id> ground_truth/ > docs/results.md

Copy each `*.template.csv` to the same name without `.template` (for example `books.template.csv` to `books.csv`) and add rows. Nothing here is generated or guessed; every value is something you counted, measured with a ruler or tape, or looked up yourself. Any file you leave out is skipped, and `docs/results.md` lists the pass bars that could not be evaluated. A file that is present but half filled in (blank numbers, a missing column, an empty file) makes only its own bars not evaluated, with the reason shown. All files are UTF-8.

Thresholds, tolerances and the shelf label format live in `config/eval.yaml`.

## books.csv

One row per physical book on the shelves, legible or not.

| Column | Meaning |
|---|---|
| unit | Shelf unit (bookcase) name or number, as you label it |
| row | Row within the unit, counted from the top |
| position | 1-based position along the row, left to right |
| title | Title as printed. Leave empty if you cannot read it |
| author | Author as printed, optional |
| legible | `y` if a person can read the title from the spine, otherwise `n` |
| orientation | `upright` or `flat`, optional |

The packet's `books[].shelf` must equal the shelf label built from `unit` and `row` with `shelf_label` in `config/eval.yaml` (default `{unit}.{row}`, so unit 2 row 3 is `2.3`). The pipeline labels a shelf with its segment label, followed by `.N` when the segment has several rows, so name your units to match those labels or change `shelf_label`. Packet books on shelves that match no ground-truth row are counted as insertions, and the results say when no packet shelf label matches at all.

How books are matched: per row, ground-truth books and packet books (ordered by position) are aligned with an edit-distance alignment. A miss and an insertion each cost `gap_cost`; pairing two books costs one minus their normalised title similarity (rapidfuzz, `title.scorer`), or `unknown_cost` when either title is empty. A paired book is correct when the packet status is `identified` and the similarity is at least `title.threshold`.

Title threshold behaviour: `token_sort_ratio` is word-order independent but not subset-tolerant. A subtitle added or dropped (`Dune` against `Dune: Messiah`, about 50) is not a match, which is deliberate: it is a different title. Titles that differ only by a volume or edition number (`Collected Works Volume 1` against `Volume 2`, about 96) do score above the default 85 and would count as correct, so check multi-volume sets by hand and raise `title.threshold` if you have many.

Definitions:

- Book count error: `abs(packet books - rows in books.csv) / rows in books.csv`, bar within 5%.
- Title accuracy: correct identifications divided by ground-truth rows with `legible = y`, bar at least 70%. A missed legible spine counts as not correct.
- Confidently wrong: packet books with status `identified` that sit on a legible ground-truth spine with a different title, or that match no ground-truth book at all, divided by all packet books with status `identified`. Bar at most 3%. An identified book on a spine you marked `n` is not counted as wrong because it cannot be checked.

## spines20.csv

Twenty spines measured with a ruler, mixing formats.

| Column | Meaning |
|---|---|
| unit, row, position | The same slot as in `books.csv` |
| height_cm | Spine height |
| thickness_cm | Spine thickness |

Each spine is looked up through the book alignment, and only counts when the aligned packet title matches the ground-truth title (`title.threshold`), so a mis-aligned neighbour is never measured. Only when `books.csv` is absent (or unusable) is the packet looked up directly by shelf label and position. A slot key that occurs twice (in `spines20.csv`, in `books.csv` or in the packet) is reported and left unmatched, never overwritten.

Reported per dimension: mean, median and p90 of `abs(packet - ruler) / ruler` over compared spines, and the share of all measured spines within 15%. Spines with no usable packet book, and spines whose dimension the packet did not produce, count as failures against the full sample. This project's reading of the bar: the share within 15% must be at least `dims.min_share_within` (80%) and the mean error at most 15%, for both height and thickness. With fewer than `dims.min_sample` (20) rows the bar is reported as not evaluated.

## prices15.csv

Fifteen books checked by hand on the retailer the packet cites, on the same day.

| Column | Meaning |
|---|---|
| isbn | ISBN of the book; matched to the packet by `books[].isbn` (hyphens ignored) |
| title | Title, for readability |
| source | Retailer name, for example `bookchor`; compared with the packet's source |
| url | Product page |
| price | Price you saw, a number |
| currency | ISO code, for example `INR`; must match the packet's `sweep.currency` |
| checked_on | Date you checked, `YYYY-MM-DD` |
| condition | `new` compares with the packet's `replacement_cost`; anything else (`good`, `used`) compares with `used_value` |
| screenshot | Path or file name of your screenshot |

A row has the same source type when the packet source matches `source` (rapidfuzz, `price.source_threshold`), the currency matches, and a used price is `observed` rather than `derived`. Rows that differ are listed separately and excluded from the bar; rows with no packet price are listed as not compared. An empty packet currency also counts as a mismatch. This project's reading of the bar: over the comparable (same-source) rows, the share within 25% must be at least `price.min_share_within` (80%) and the mean error at most 25%. At least `price.min_comparable` (10) rows must be comparable, otherwise the bar is reported as not evaluated; the summary shows `comparable X of 15`.

## items.csv

At least 8 non-book items.

| Column | Meaning |
|---|---|
| name | What it is, for example `reading chair` |
| category | One of `item_categories` in `config/taxonomy.yaml` |
| w_cm, h_cm, d_cm | Dimensions, optional |
| brand | Brand or model, optional |

An item is found when a packet item's `brand_model` and `description` fuzzily match `brand` and `name` (`items.threshold`, one packet item per ground-truth item). Matching prefers candidates whose category is equal, then falls back to name similarity alone; an item matched with the wrong category is counted as found but wrong, and shown separately. It counts only when the category matches. The bar is at least 80% found with the correct category, and needs at least `items.min_truth` (8) items, otherwise it is not evaluated.

## room.json

Tape-measured. All lengths in metres.

    {
      "ceiling_height_m": 2.7,
      "walls": [
        {"id": "A", "length_m": 4.0, "interior_angle_deg": 90},
        {"id": "B", "length_m": 3.0}
      ],
      "openings": [{"kind": "door", "wall": "A", "width_m": 0.9, "height_m": 2.1}],
      "shelving_fronts": [{"unit": "1", "wall": "B", "width_m": 1.8, "height_m": 2.0}]
    }

- `walls`: every wall in order around the room, walking counter-clockwise. `interior_angle_deg` is the interior angle at the end of that wall and defaults to 90; use 270 for an inside corner of an L-shaped room.
- `openings`: doors and windows, each with width and height.
- `shelving_fronts`: the front face of each shelf unit.

Formulas, computed in code and not typed in:

- Floor area: the walls are walked as a polygon (each wall turns left by `180 - interior_angle_deg`) and the shoelace formula gives the area. If the end of the last wall is more than `closure_tolerance_m` from the start, the floor bar is reported as not evaluated.
- Gross wall area: `sum(wall lengths) * ceiling_height_m`, compared with the packet's `room.wall_area_m2` (gross).
- Net wall area: gross minus the sum of `width_m * height_m` over openings. Shelved wall area: the sum of `width_m * height_m` over `shelving_fronts`. Both are shown without a bar.

Bars: floor within 10%, wall within 15%. Both are reported as not evaluated when the walls do not close, when numbers are left blank or `null` (as in `room.template.json`), or when `walls` has fewer than 3 entries.

## Time and cost

No ground truth is needed. Time to packet is the packet's `latency_s.after_sweep` (`time.elapsed_key`), or the sum of the post-sweep stages when that key is absent; stages in `time.streaming_stages` run during the sweep and are excluded from that sum. At least one post-sweep stage timing must exist (from `latency_s` or `stages/_ledger.json`), otherwise the bar is not evaluated. Expected stages with no timing are listed. The bar is under 300 seconds. Cost and latency per stage come from the same sources.
