# Claim packet notes

Notes on the packet fields that are easy to misread. The README should link here.

## Spine dimensions

- `spine_height_cm` and `spine_thickness_cm` are filled only when a metric scale was achieved (`scale_method: aruco_homography`). The brief says an empty field is correct when the system does not know.
- When no scale marker was visible the system only has a format prior (the nearest of the book formats in `config/taxonomy.yaml`, picked from the box aspect ratio). Those numbers are class constants, not measurements, so they go to `dims_estimated_from_prior` (`height_cm`, `thickness_cm`, `book_format`) and the contract fields stay `null`. `scale_method` stays `book_format_prior`.
- The report prints these as `est. 21.0 / 2.5` in italics.
- `dims.fill_from_prior` in `config/pipeline.yaml` (default `false`) lets the owner copy the estimate into the contract fields. `dims_estimated_from_prior` is still written, so the line stays marked as an estimate.
- Open Library dimension cross-checks run only on measured dimensions.
- SAM masks refine a metric measurement only when the mask agrees with the detector box (`masks.height_tolerance`, `masks.min_width_ratio`, `masks.max_width_ratio` in `config/vision.yaml`). Otherwise the trimmed detector box is used. Format priors never use masks.

## Totals

- `shelf_run_m` is the sum of measured thicknesses, in metres, and `null` when no book was measured.
- `shelf_run_estimated_m` is the sum of the prior thicknesses of the books that were not measured, and `null` when there are none. It is an estimate, never a measurement.
- `excluded_from_totals` counts lines that would be priced but have no price: identified books with no sourced replacement price (including books the user excluded), and items that are not `needs_appraisal` with no price range. This follows the brief: "If no price exists at all, say so and exclude it from the total."
- `needs_appraisal` books and items are also left out of every money total, but they are counted by `books_needs_appraisal` and the item `status`, not by `excluded_from_totals`. Unidentified books are counted by `books_unidentified`. So `books_identified + books_unidentified + books_needs_appraisal = book_count`, and `excluded_from_totals` is a subset of the identified books plus the non-appraisal items.
- All totals are computed in code from the lines (`libclaim/packet/totals.py`).

## Review queue

- Books with no measured size produce one entry per shelf row (`ref_id` is the shelf label, for example `S1.2`, `line_ref` is `books`) with the count of books and the reason, for example "46 books on S1.2 have no measured size (no scale marker was visible); sizes are estimates from the book format".
- Per-book entries remain for reasons that are specific to a book (identification, pricing, appraisal, a measurement note, a disagreement with Open Library).
- Review text never names environment variables.

## Locating a line in its frame

- `frame_ref` is the saved full frame. `box` is `[x1, y1, x2, y2]` in pixels of that frame and `frame_size` is `[width, height]` of the frame.
- `crop_ref` is a small JPEG of the box (under `frames/crops/<id>.jpg`, longest side capped by `crops.max_side_px` in `config/pipeline.yaml`), written while the packet is assembled. Review entries repeat the `crop_ref` of their line.
- The report shows the crop; clicking it opens the full frame with the box outlined.

## Known detector limits

- Two boxes can overlap on tilted or lying stacks (`b01-016`/`b01-017`, IoU 0.45) and on dense thin spines (`b01-041`/`b01-043`, IoU 0.47: one detector box spans two spines). They come from the detector, not from track merging (one still, one segment), and they sit below the NMS thresholds (`nms_iou: 0.6`, `nms_containment: 0.8`) on purpose: lowering them merges real neighbours and the count is already below the hand count.
