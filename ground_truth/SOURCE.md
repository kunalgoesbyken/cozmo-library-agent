# Where this ground truth comes from

No physical shelf was available for the demo, so the demo camera was a fake device (Chromium `--use-file-for-fake-video-capture`) panning over two public photos at their native resolution:

| Shelf in the packet | Photo | Licence | Books |
|---|---|---|---|
| S1 | [Forty-odd Pelican books](https://commons.wikimedia.org/wiki/File:Forty-odd_Pelican_books.jpg), middle row | CC BY-SA 4.0, by Hoary | 41 |
| S3 | [Part of a bookshelf containing books by Latin poets and historians (1)](https://commons.wikimedia.org/wiki/File:Part_of_a_bookshelf_containing_books_by_Latin_poets_and_historians_(1).jpg) | CC BY-SA 4.0, by Roman Eisele | 19 |

- `books.csv`: every book counted and every legible spine read by hand from the full-resolution photos, left to right. Spines with no readable title are `legible = n`.
- `items.csv`: the two non-book objects in the frame (a spiral-bound notebook at the left edge of the Pelican row and a white bookend at the right edge). The bar needs 8, so it is not evaluated.
- `prices15.csv`: the two books the packet priced, checked by hand on the cited Penguin UK pages on 2026-10-28. The packet reports INR, so the GBP rows are not comparable.
- `spines20.csv` and `room.json` are absent on purpose. A photo cannot be measured with a ruler or a tape, and nothing here is guessed.
