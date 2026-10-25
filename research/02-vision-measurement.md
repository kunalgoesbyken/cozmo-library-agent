# 02 - Vision and measurement: options and recommendations

Scope: everything in the brief that has to be *measured* from one phone sweep: frame selection, spine detection and counting, spine reading and resolution, metric scale, room dimensions and areas, and non-book items. Pricing and the live voice loop are covered in other notes.

Researched 2026-10-25. Model IDs and prices change often, so check them again before you build. Claims I could not check against a primary source are marked **(unverified)**.

---

## 0. TL;DR decisions

| Question | Decision |
|---|---|
| Capture stack | **Android Chrome WebXR `immersive-ar`** with `camera-access` + `depth-sensing` + `hit-test`. This gives a metric 6-DoF pose per frame, metric depth and a floor plane at no cost. **Fallback:** plain `getUserMedia` 4K video + printed ArUco markers + hosted MapAnything/VGGT. Spend the first hour proving WebXR works on the actual phone. |
| Scale source (books) | A **printed ArUco marker on each shelf unit** (primary, homography on the shelf-front plane). WebXR depth/pose is the second source and cross-check. **Standard book formats** are only a sanity check and a last-resort fallback, and every use is flagged. |
| Scale source (room) | **ARCore metric tracking via WebXR.** Fallback: MapAnything (metric) or VGGT reconstruction, rescaled with ArUco corners. |
| Double counting | **Shelf-row keyframes.** The agent tells the user to hold landscape on each row for about 2 s, giving one primary frame per row. Rows that need several frames are merged by homography (or by 3D position with WebXR), with fuzzy title matching as a tie-break. |
| Spine detection | **Gemini Flash `box_2d`** on tiled row keyframes (primary), cross-checked by **SAM 3.1 on fal** with the prompt "book spine". When the two disagree, the row goes to the review queue and the agent asks for a re-capture while the user is still there. |
| Spine reading | **Gemini on rotated per-spine crops** (verbatim transcription), corroborated by **Google Cloud Vision OCR**. Resolved via **Open Library + Google Books** with RapidFuzz scoring and strict acceptance gates. |
| Room geometry | Metric point cloud from WebXR (hit-tests + depth). Find the floor plane, fit Manhattan wall planes using points *above the shelving* (the wall-ceiling junction), then build a rectilinear polygon. Ceiling height comes from ray-casting the wall-ceiling edge onto the wall plane. |
| Non-book items | Gemini detection with a fixed taxonomy on room keyframes. Items are deduplicated by 3D position and sized via depth or a plane homography. |
| GPU vs hosted | **Hosted for all models.** The GTX 1660 Ti runs only OpenCV, plus optional PaddleOCR or DA3-Metric experiments. Reasons are in section 8. |

---

## 1. Frame selection from a continuous sweep

### 1.1 What Gemini Live sees is not enough
The Gemini Live API takes video as discrete frames at **1 FPS, at about 768x768** ([Firebase Live API limits](https://firebase.google.com/docs/ai-logic/live-api/limits-and-specs)). That is enough for conversation ("slow down", "is that an original?") and nowhere near enough to read spines. So **run two streams from the phone**:

1. A low-res stream at 1 FPS to the Live session, for talk and coaching.
2. A high-res evidence stream at 2-4 FPS as JPEG (quality 90), posted to the backend. Each frame carries a timestamp, plus pose, intrinsics and depth when WebXR is active. Every saved frame becomes a `frame_ref` (`frames/<sweep>/<seq>_<ts_ms>.jpg` with a JSON sidecar).

Also: **audio+video Live sessions are capped at 2 minutes unless `contextWindowCompression` is enabled**, and connections reset periodically, which `sessionResumption` handles ([Gemini Live session management](https://ai.google.dev/gemini-api/docs/live-session)). A 3-minute sweep hits the cap, so enable both.

### 1.2 Quality gates (on-device, cheap, real-time)
Run these in JS on a 320 px-wide downscale of every evidence frame, and on the server again at full resolution:

| Gate | Method | Use |
|---|---|---|
| Blur | Variance of Laplacian on grayscale ([pyimagesearch](https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/)). Use a **relative** threshold: keep a frame if it scores above about 0.6x the rolling median of the last 2 s, plus an absolute floor tuned on your own phone. | Drop blurred frames; tell the agent "hold still". |
| Motion | `DeviceMotionEvent.rotationRate` above about 60 deg/s, or WebXR pose angular velocity. | Predicts blur before it shows up in the image. It is the most reliable trigger for "slow down". |
| Glare | Fraction of pixels with V above 250 in HSV within the spine region; over 3% means glare. | "Tilt the phone a little" (glossy jackets). |
| Too far | Median detected spine width in px is under about 25 px at 1080p. | "Step closer". |
| Near-duplicate | dHash/pHash Hamming distance below 6 compared with the last kept frame. With WebXR, pose delta below 0.15 m and below 8 deg. | Avoids storing and processing redundant frames. |

Keyframes are the sharpest frame in each window of about 0.5 s that passes all the gates.

### 1.3 Avoiding double counting (the hard part)

| Approach | How | Pros | Cons |
|---|---|---|---|
| **A. Shelf-row keyframes (recommended)** | The agent tells the user to hold landscape on each shelf row for about 2 s. A row 80-120 cm wide fits in one landscape frame from about 1 m (roughly 65 deg HFOV gives about 1.27 m of coverage). Shelf boards separate rows physically, so rows rarely double count. | Simplest. One frame per row is the evidence frame. Count = sum over rows. | Wide rows need 2 frames. Depends on the agent's directing working. |
| B. Homography chaining within a row | ORB/SIFT matches + RANSAC homography between consecutive row frames. The shelf front is close to planar, so a homography fits well. Project boxes from frame i into frame i+1, merge at IoU > 0.3, and union-find into tracks. | Robust, about 50 lines of OpenCV. Handles a row split over 2-3 frames. | Recessed or tilted books add parallax error, which the IoU threshold absorbs. |
| C. 3D clustering (WebXR only) | Back-project each spine box centre to world coordinates with pose + depth (or a hit-test), then DBSCAN with eps about 3 cm. | Handles revisits and re-captures anywhere in the sweep. | Depends on WebXR depth quality, which is fine on textured shelves. |
| D. Per-shelf panorama | OpenCV `Stitcher` in `SCANS` (affine) mode, then detect on the mosaic ([OpenCV Stitcher](https://docs.opencv.org/4.x/d2/d8d/classcv_1_1Stitcher.html)). | One image per unit, which is nice for the report. | Ghosting and seams can split or duplicate spines. Fails with parallax when the user rotates in place. Use only as a visual and a count cross-check. |
| E. Text sequence alignment | Align the left-to-right title sequence across frames, Needleman-Wunsch on fuzzy title similarity. | Good tie-break for B and C. | Useless on unreadable spines. |

**Recommendation:** A as the capture protocol, B for multi-frame rows (C when WebXR is up), and E as the tie-break. Discard any box flagged `truncated` (cut by the image edge) when an overlapping frame has the same spine complete. Each book's `frame_ref` is the sharpest, most frontal frame of its track.

---

## 2. Spine detection and counting

### 2.1 Options compared

| Option | Output | Dense vertical rows | Flat / stacked | Notes |
|---|---|---|---|---|
| **Gemini Flash/Pro `box_2d`** | `[ymin,xmin,ymax,xmax]` normalised 0-1000, with JSON schema and an optional polygon mask ([image understanding docs](https://ai.google.dev/gemini-api/docs/image-understanding), [Vertex bbox](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/bounding-box-detection)) | Good semantic separation (book vs box file vs magazine). Can **merge or miss thin spines** in rows of more than about 40 books. Boxes are loose, by a few %. | Good when prompted explicitly ("books lying flat, count each one in a stack"). | No confidence scores and no NMS. On RF100-VL, Gemini 2.5 Pro scored 11.6 zero-shot mAP vs 15.7 for Grounding DINO, partly because there are no scores ([RF100-VL](https://arxiv.org/abs/2505.20612)). Books are an in-distribution concept, so this gap matters less here. The same call can also return the visible text. Current IDs: `gemini-3.8-flash` and `gemini-3.6-flash` (stable), `gemini-3.1-pro-preview` ([models](https://ai.google.dev/gemini-api/docs/models)). |
| **SAM 3 / 3.1** (concept prompt "book spine") | Instance masks + IDs for *all* matches ([SAM 3 paper](https://arxiv.org/abs/2511.16719)) | Built for exhaustive instances. CountBench 93.8% acc / 0.12 MAE vs 92.4% / 0.24 for Gemini 2.5 Pro. **Masks give the tightest geometry** for measuring height and thickness. | Prompt "book lying flat" separately. Thin pages-side-out stacks can merge. | Hosted: `fal-ai/sam-3-1/image` and `fal-ai/sam-3-1/image-rle` on fal. 848M params, which is heavy for 6 GB (unverified). |
| Qwen3-VL (2B to 235B-A22B) / Qwen3.5 | Relative 0-1000 boxes and points ([Qwen3-VL repo](https://github.com/QwenLM/Qwen3-VL), [tech report](https://arxiv.org/abs/2511.21631)) | Similar class to Gemini. Qwen2.5-VL-72B scored 5.6 zero-shot mAP on RF100-VL. | Similar to Gemini. | Good *second-opinion* VLM over an API. Local 4B/8B at 4-bit fits 6 GB but is slow on Turing. Qwen3.5 (Feb 2026, open weights, native VL) is reported but its grounding is not documented ([news](https://alternativeto.net/news/2026/2/alibaba-launches-qwen3-5-with-open-weight-397b-model-and-broader-language-support)). |
| Grounding DINO / OWLv2 / YOLO-World | Boxes **with confidence** | Tend to box a whole row or merge neighbours. Need careful NMS and threshold tuning per scene. | Weak. | Highest zero-shot mAP on RF100-VL (15.7 / 13.6), but that measures generic domains, not spines. Not recommended as primary. |
| Fine-tuned YOLO-seg on public spine data | Masks + confidence, fast | Good in-domain. | Depends on the data. Flat books are likely under-represented. | [Book spine detection (P2)](https://universe.roboflow.com/book-detection/book-spine-detection-p2): 7.06k images, instance segmentation, CC BY 4.0. Also [book_spine_instance_segmentation](https://universe.roboflow.com/minhaz/book_spine_instance_segmentation-uqybb) (Nov 2025, CC BY 4.0), [LibVision Book Spines](https://universe.roboflow.com/libvision/book-spines-vh9b8), and [Open Shelves](https://github.com/capjamesg/open-shelves) (archived, NC licence). An Oriented R-CNN variant reports 90.22 mAP on spines ([PMC11679322](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11679322/)). Fine-tuning is allowed with justification, but it is a time sink. Keep it in reserve. |
| Classic CV | Sobel-x projection profile / Hough lines give spine boundaries | Cheap third opinion on vertical rows. | No. | Used only for gap and merge checks. |

### 2.2 Recommended detection recipe
1. For each row keyframe, cut **2-3 overlapping horizontal tiles** (15% overlap) and send them at high `media_resolution`, with thinking set low or minimal as the docs advise for segmentation. Ask for, per spine: `box_2d`, `orientation` (vertical | flat | leaning), `in_stack_index`, `truncated`, `legible` and `visible_text` (verbatim), using a JSON schema.
2. Run SAM 3.1 with "book spine" (and "book lying flat") on the same full frame.
3. **Reconcile:** do Hungarian matching on IoU. A spine is counted if both see it, or if one sees it *and* its width is consistent with its neighbours. Merged boxes are split when SAM shows 2+ masks inside one Gemini box; the reverse case is handled the same way.
4. **Row-tiling sanity check:** vertical spines should tile the occupied row with no gaps. An unexplained gap wider than about 1.5x the median spine thickness triggers a re-query of that crop at higher zoom. If the gap persists, a review entry is created and the agent asks *live* for "a closer look at the left end of the second shelf".
5. Any per-row count disagreement above 1 goes to the review queue with the row frame.

Expected result (my estimate, unverified): within 2-4% count error on typical home shelves with this two-model reconciliation. Gemini alone is likely 5-10% on dense rows of thin paperbacks. Measure this against your ground truth early.

---

## 3. Spine reading and resolution

### 3.1 VLM vs OCR

| | Strength on spines | Weakness | Cost |
|---|---|---|---|
| **Gemini (crop-level)** | Reads vertical, stylised and low-contrast text. Separates title, author and publisher from layout. Copes with partial words. | **Fills in plausible text** (hallucination) on blurry spines. Its self-reported "legible" flag is unreliable. | About $0.75/M input tokens for 3.8 Flash ([pricing](https://ai.google.dev/gemini-api/docs/pricing)) |
| Google Cloud Vision `TEXT_DETECTION` | Independent, literal tokens with boxes. Never invents a famous title. | Trouble with rotated or stylised text unless you pre-rotate. Fragmented tokens. | About $1.50 per 1000 images, first 1000/month free ([pricing](https://cloud.google.com/vision/pricing), [OCR docs](https://cloud.google.com/vision/docs/ocr)) |
| PaddleOCR 3 / PP-OCRv5 | Local, has a text-line orientation classifier, upgraded for vertical text ([PaddleOCR 3 report](https://arxiv.org/abs/2507.05595), [PP-OCRv5 docs](https://www.paddleocr.ai/v3.3.1/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5.html)) | Windows + paddle GPU install friction. CPU works but is slow. | Free |
| docTR | Local, decent | Same rotation issues | Free ([repo](https://github.com/mindee/doctr)) |

Prior work agrees that text-only spine pipelines plateau around 72% recall, and that combining text with a second signal raises it to about 91% ([Tsai et al., ACM MM 2011](https://web.stanford.edu/~bgirod/pdfs/Tsai_ACM_Multimedia_11.pdf)). The modern equivalent is VLM reading + OCR corroboration + catalogue agreement.

### 3.2 Recipe
1. Crop each spine from its best frame (mask bounding rect, padded 10%) and produce **two rotations** (+90 and -90 deg) so the text is horizontal. Send batches of about 20 crops per Gemini call. Prompt: *transcribe only characters you can see; use `?` for unreadable characters; return `{title_text, author_text, publisher_text, other_text, readable_fraction}`; never complete or correct titles.*
2. Run Cloud Vision on the same rotated crops (both rotations; keep the one with more dictionary words).
3. **Corroboration score** `c_ocr` = the share of VLM title tokens (3+ chars) found in the OCR tokens, using a fuzzy token match with RapidFuzz `ratio` of 85 or more ([RapidFuzz](https://github.com/rapidfuzz/RapidFuzz)).
4. Look up candidates:
   - Open Library `search.json?title=..&author=..&fields=key,title,author_name,first_publish_year,editions,editions.key,editions.title,editions.language` ([search API](https://openlibrary.org/dev/docs/api/search)). Send a `User-Agent` with app name and contact. Unidentified clients are limited to about 1 req/s, identified ones get 3x ([developers](https://openlibrary.org/developers/api)).
   - Google Books `volumes?q=intitle:..+inauthor:..` (key required, `maxResults` up to 40). It returns `industryIdentifiers`, `dimensions` (height/thickness) and `saleInfo` ([Books API](https://developers.google.com/books/docs/v1/using)).
5. **Catalogue score** per candidate: `0.6*title + 0.3*author + 0.1*publisher`, each `token_set_ratio`. Author or publisher is skipped (weights renormalised) only when it is not visible.

### 3.3 Acceptance gates (to stay at 3% or below confidently wrong)
A book is `identified` only if **all** of these hold:
- `c_ocr >= 0.6`, so the VLM read is backed by literal OCR.
- Best catalogue score >= 90 **and** a margin of at least 8 over the next distinct work.
- Either 2+ distinctive title tokens, or the author also matched (stops "Emma" or "It" resolving to an arbitrary work).
- Both catalogues agree on the work, or only one returns it and the score is >= 95.

**Edition** is filled only when the spine publisher (or imprint logo text) matches an edition's publisher. As a secondary check, the measured spine height must be within 10% of Google Books `dimensions.height`. Otherwise `edition` and `isbn` stay empty and status is identified at work level. Anything failing the gates is `unidentified`, with dimensions, the raw transcript kept as evidence, and a review-queue reason. **Calibrate the thresholds on your own ground truth** so that the observed wrong rate among accepted IDs is at most 1.5%. That leaves headroom for the reviewers' unseen shelf. Confidence = `min(c_ocr, catalogue_score/100) * margin_factor`. It is computed in code and never taken from the model.

---

## 4. Metric scale

### 4.1 Options

| Source | Accuracy (expected) | Cost to build | Notes |
|---|---|---|---|
| **Printed ArUco / AprilTag** (OpenCV `cv2.aruco.ArucoDetector`, OpenCV 4.7+ API; [tutorial](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)) | About 1-3% on the marker plane when the marker is at least 100 px wide (my estimate). Error grows with the depth offset between marker and object plane: 3 cm of book recess at 70 cm adds about 4%. | Very low | One marker gives 4 corners, which is enough for a pose. **Print-scale risk:** "fit to page" scaling silently changes the size, so add a 10 cm check line and measure it once with a ruler. Use 15-16 cm markers on A4 (DICT_4X4_50, distinct IDs), one per shelf unit and one on the floor. |
| Credit card (8.56 x 5.398 cm) / A4 (21.0 x 29.7 cm) | Card: poor at 1 m (too few pixels). A4: OK, but detecting the sheet's corners is less robust than ArUco. | Medium | Use A4 only to *carry* the ArUco. The sheet edge doubles as a fallback reference. |
| **WebXR on Android Chrome** (ARCore) | Tracking scale is typically within a few % at room scale. Studies: about 1 cm error dimensioning cuboids ([Brno thesis](https://excel.fit.vutbr.cz/submissions/2019/049/49.pdf)), up to 7.5% in a navigation study, and degradation below 30-40 lux and beyond 2 m ([NURE 2025](https://rt.nure.ua/article/view/350221)). Depth is best at 0.5-5 m and **imprecise on white featureless walls** ([ARCore Depth](https://developers.google.com/ar/develop/depth)). | Medium | Hit-test, anchors and Depth API are *shipped* in Chrome; plane detection is a developer trial only ([ARCore vs WebXR](https://developers.google.com/ar/develop/webxr/arcore-comparison)). Depth via `getDepthInformation(view)`, `getDepthInMeters(x,y)` in `cpu-optimized` mode, and `rawValueToMeters` ([Depth Sensing spec](https://immersive-web.github.io/depth-sensing/)). Camera pixels via `camera-access`: `view.camera` and `XRWebGLBinding.getCameraImage()` return a **WebGL texture only**, readable only inside the rAF callback ([Raw Camera Access](https://immersive-web.github.io/raw-camera-access/)). Supported on Chrome Android ([caniuse](https://caniuse.com/wf-webxr-camera)). Reports of WebGL context loss on Pixel 6 Pro and Xiaomi 13T ([Babylon forum](https://forum.babylonjs.com/t/using-camera-access-feature-in-webxr-causes-context-loss-and-flickering-screen-issues/51610)). Hit-tests closer than 30-40 cm are noisy ([forum](https://forum.babylonjs.com/t/webxr-anchors-and-hit-test/60387)). Camera image resolution is device-dependent and not in the spec, so **measure it on your phone** (unverified). |
| Known shelf dimensions | Good if the model is known, e.g. a flat-pack unit. | Low | The user can state it by voice. Record it as `scale_method: known_shelf(user_stated)`. Use it as a cross-check, not the primary source. |
| Standard book formats | ±5-10% per book; it cannot tell formats apart reliably | Low | UK A-format 11.1x17.8 cm, B-format 12.9x19.8, Demy 13.8x21.6, Royal 15.6x23.4 ([publishing.co.uk](https://publishing.co.uk/guides/standard-book-sizes-uk/)). US mass-market 4.25x6.87 in (10.8x17.5 cm), trade 5.5x8.5 / 6x9 in ([thebookcoverdesigner](https://thebookcoverdesigner.com/standard-book-sizes/)). Indian publishers mostly use B-format and Demy-like trims (**unverified**). Use: if the row's modal height cluster sits within 3% of a standard height, it confirms the marker scale. Otherwise it is a fallback with a flag. Prefer per-book Google Books `dimensions` once a book is identified, as an independent check. |
| Metric monocular depth: Depth Pro, Depth Anything V2/V3-Metric, Metric3D v2, UniDepth V2, MoGe-2 | Typically 5-15% absolute relative error indoors (my estimate from published benchmarks; varies by scene). MoGe-2 reports lower metric error than UniDepth, Depth Pro and Metric3D v2 ([MoGe-2](https://arxiv.org/abs/2507.02546)). | Low (hosted) | Not acceptable as the scale *source* ("the model estimated it"). Acceptable as a **shape** source rescaled by a marker. DA3METRIC-LARGE: 0.35B, Apache-2.0, `metric = focal * out / 300`. DA3NESTED (multi-view metric) is CC BY-NC ([DA3 repo](https://github.com/ByteDance-Seed/depth-anything-3); hosted on [Replicate](https://replicate.com/vufinder/depth-anything-v3-metric)). Repos: [Depth Pro](https://github.com/apple/ml-depth-pro), [UniDepth](https://github.com/lpiccinelli-eth/UniDepth), [Metric3D](https://github.com/YvanYin/Metric3D). |
| Feed-forward multi-view reconstruction: VGGT, MapAnything, MASt3R/DUSt3R | VGGT output is **up to scale**, so fix it with marker corners. MapAnything predicts a metric scale factor directly ([paper](https://arxiv.org/abs/2509.13414)), but still verify against the marker. | Medium (hosted) | VGGT-1B: under 1 s per scene. The original checkpoint is non-commercial; use the `VGGT-1B-Commercial` variant. The original OOMs around 300 frames on 80 GB until the May 2026 memory fix, which allows 2-3x more frames ([repo](https://github.com/facebookresearch/vggt), [FastVGGT](https://arxiv.org/abs/2509.02560)). Hosted at **`fal-ai/vggt-1b`** and the [HF Space](https://huggingface.co/spaces/facebook/vggt). MapAnything: Apache weights `facebook/map-anything-apache`, accepts optional intrinsics, `memory_efficient_inference` ([repo](https://github.com/facebookresearch/map-anything), [Replicate](https://replicate.com/vufinder/map-anything), [HF Space](https://huggingface.co/spaces/facebook/map-anything)). MASt3R-SLAM: 15 FPS on an RTX 4090 ([paper](https://arxiv.org/abs/2412.12392)), which is too heavy for a 1660 Ti on Windows. |

### 4.2 Recommended scale chain for spines (record which one was used per book)
1. **Marker in the same frame:** compute a homography from the marker corners to the metric plane, rectify the spine mask, then read height and thickness in cm. Correct for book recess when WebXR depth is available.
2. **Marker in an overlapping frame:** chain homographies (section 1.3 B) to propagate scale along the row.
3. **WebXR:** project the spine mask's endpoints with `depth x pixel / focal` (intrinsics from `projectionMatrix`), or hit-test them.
4. **Book format prior:** last resort, `scale_method = "book_format_prior"`, with an automatic review entry.

**Thickness is the precision risk.** A 1.5 cm paperback at 1080p and 1 m is about 22 px wide, so ±3 px of box looseness is ±14%. Mitigations:
- Capture the evidence stream at 4K when using `getUserMedia`.
- Have the agent request "step closer" until the median spine is at least 40 px.
- Measure thickness from **shared boundaries between adjacent masks** (one boundary error per book rather than two).
- Constrain the sum of thicknesses in a row to equal the measured occupied run between uprights.

Height is easier (about 200+ px), with expected error of 2-5%.

**Shelf run (m)** = sum of vertical spine thicknesses + horizontal extent of each flat stack, per row, summed. Write the definition into the packet.

---

## 5. Room dimensions and areas

### 5.1 Options and expected accuracy

| Method | Floor area error (expected) | Wall area error | Effort | Verdict |
|---|---|---|---|---|
| **WebXR metric point cloud** (floor plane + depth + hit-tests, Manhattan wall fit) | 3-8% | 5-12% | Medium | **Primary.** Simplest method that plausibly passes both bars. |
| VGGT / MapAnything on about 40-80 keyframes, scale from ArUco | 5-15% (unverified, depends on textureless walls) | 8-20% | Medium (hosted) | **Fallback** when WebXR is unavailable. |
| Per-frame metric depth (MoGe-2 / DA3) without registration | Hard to fuse into a room | - | High | No. |
| HorizonNet / LGT-Net (Manhattan layout) | Needs **equirectangular panoramas**; 3D IoU about 72-82% on benchmarks; metric from an assumed camera height ([HorizonNet](https://arxiv.org/abs/1901.03861), [LGT-Net](https://openaccess.thecvf.com/content/CVPR2022/html/Jiang_LGT-Net_Indoor_Panoramic_Room_Layout_Estimation_With_Geometry-Aware_Transformer_Network_CVPR_2022_paper.html), [ECCV'24 supp](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06150-supp.pdf)) | ~15-25% | - | High | No. Stitching a 360° pano from a walking sweep is fragile. |
| SpatialLM (point cloud to walls, doors, windows as text; [paper](https://arxiv.org/abs/2506.07491)) | Depends on point-cloud quality | - | High | Stretch only. A nice floor-plan bonus, not needed. |
| iPhone Pro: ARKit + **RoomPlan** (LiDAR) | Anecdotally 2-10 cm per wall, average 8.5 cm in one test, worse where cabinets hide the wall base ([Nomtek](https://www.nomtek.com/blog/getting-a-room-plan-with-arkit), [RoomPlan](https://developer.apple.com/augmented-reality/roomplan/)) | Similar | **Native Swift app; needs a Mac** | Best accuracy, but not feasible from Windows in 70 h. iOS Safari has no WebXR AR ([device support, Jun 2026](https://abratabia.com/webxr-games/webxr-device-support.php)). |

### 5.2 Recommended algorithm (WebXR)
**During the sweep (in the browser):**
- Every 3rd XR frame, run hit-tests from a 5x5 screen grid and sample the CPU depth map (about 1000 points per frame, `depth > 0`). Transform them with the view pose into world coordinates and stream them with the frame ID.
- Store each evidence frame's pose and intrinsics.

**Directing the sweep:** the agent asks for one slow pan along the **wall-ceiling line** of each wall, which is part of the same continuous pass. Shelving hides the floor-wall junction. Shelves stand 25-40 cm proud of the wall, so fitting walls to shelf fronts would understate a 4 m room by up to 15%.

**After the sweep (server, numpy/Open3D):**
1. **Floor:** RANSAC on the lowest large horizontal cluster (normal within 10° of gravity; WebXR local-floor gives the y-axis). Camera height comes for free.
2. **Wall points:** points 0.3 m or more above the tallest shelf top, or within the top 40 cm below the ceiling estimate, plus any floor-junction points not occluded by furniture. Project them onto the floor plane.
3. **Manhattan frame:** histogram of 2D normals (from local PCA), giving a dominant axis and its 90° partner.
4. **Polygon:** project onto a 2 cm occupancy grid, take the outer contour, simplify with Douglas-Peucker, and snap edges to the Manhattan axes. This produces a rectangle, an L shape or a U shape. Edges more than 15° off-axis are kept as-is (angled walls, bay windows) and noted in `shape_notes`.
   - **Length / width** = the oriented bounding box of the polygon.
   - **Floor area** = shoelace area of the polygon. Record the shape type and vertex list.
5. **Ceiling height:** where ceiling depth exists (often poor on white ceilings), take the median distance of the ceiling plane to the floor. Otherwise use **ray-to-wall-plane intersection**:
   - Locate the wall-ceiling edge pixel in 3+ frames per wall, via a Gemini "point to the line where wall meets ceiling" query or LSD line detection plus vanishing direction.
   - Cast that ray, intersect it with the fitted vertical wall plane, and take height = y_hit - y_floor.
   - Take the median across walls. Sloped ceilings get a per-wall height and a note.
6. **Wall area (gross)** = sum over polygon edges of edge length x that wall's height. **Doors and windows:**
   - Detect them on wall frames (Gemini `box_2d`) and project the box corners onto the wall plane, giving metric width and height.
   - Sanity check: interior doors are typically 0.7-0.9 x 2.0-2.1 m.
   - Report `wall_area_m2` (choose and document gross or net; I suggest **gross**, with `wall_area_net_m2` as an added field) plus an `openings` list.
7. **Shelved wall area** = sum over shelving units of front-face width x height. Get it from the unit's corner points on its front plane (hit-tests or depth), or from a marker homography in the fallback path. Report it separately.
8. **Confidence:**
   - Fraction of polygon perimeter backed by actual wall points.
   - Residual of the plane fits.
   - Agreement between the ARCore scale and the ArUco marker size measured in 3D: triangulate the marker corners using pose + depth, compare with the known 16 cm, and **apply that as a correction factor if it is off by more than 2%**.

   Every room number lists the frame IDs that contributed points.

### 5.3 Fallback algorithm (no WebXR)
1. Pick 40-80 keyframes (sharp, diverse) and run `fal-ai/vggt-1b` or MapAnything to get point maps and poses.
2. Detect ArUco corners in each frame, look up their 3D points in the point map, and compute the scale = 0.16 m / median reconstructed marker edge.
3. Run the same steps 1-8 as above.

Expect worse wall accuracy. Flag `scale_method: "multiview_recon+aruco"` and lower confidence.

---

## 6. Non-book items

- **Detection:** Gemini on room keyframes (about 1 per 1.5 s of sweep, plus the row frames) with a fixed taxonomy as an enum in the JSON schema: `shelving, seating, table, desk, lamp, coffee_machine, framed_art, portrait, mirror, rug, electronics_tv, electronics_audio, electronics_computer, clock, plant, decor_object, storage, curtain, other`. Per item: `box_2d`, `material`, `brand_model_text` (verbatim only), `is_art`, `truncated`. Gemini's semantic labels are far stronger than open-vocabulary detectors for "coffee machine vs kettle" style distinctions. SAM 3.1 with the category name can refine masks for sizing.
- **Dedup:** with WebXR, cluster box-centre 3D points per category (eps about 30 cm). Without it, use category + temporal adjacency + homography/embedding similarity. Keep the sharpest, least-truncated frame as `frame_ref`.
- **Dimensions** (approximate, ±20-30% is acceptable; the brief says "approximate"):
  - On the floor plane (rugs, furniture footprints): floor homography or plane hit-tests, which is accurate.
  - On walls (art, mirrors, TVs): project onto the fitted wall plane.
  - Free-standing objects: WebXR depth at the mask median, so size = px x depth / focal. Depth (d) comes from a second view, or from a category prior marked `estimated_from_prior`.
  - Fallback: marker-rescaled DA3-Metric/MoGe-2 depth, flagged.
- **Art:** set `is_art` and push it to the review queue as `needs_appraisal`. The live agent asks "original or print?" while the item is in view.
- **Counting toward the 80% bar:** small decor is where recall drops. Have the agent do one wide "room overview" pan at the start, and run the detector on those frames with the prompt "list every distinct object, including small ones".

---

## 7. Recommended pipeline

| # | Stage | Model / tool | Input | Output | Fallback |
|---|---|---|---|---|---|
| 0 | Capture | WebXR `immersive-ar` (camera-access, depth-sensing, hit-test, local-floor); mic via `getUserMedia` (audio only) | Phone | Evidence frames 2-4 FPS + pose/intrinsics/depth sidecars + 3D point stream; 1 FPS low-res to Gemini Live | `getUserMedia` 4K video + ArUco; frames + gyro only |
| 1 | Live QA | JS Laplacian / gyro / glare / spine-px gates | Each evidence frame | "slow down / closer / tilt / missed shelf" events to the agent | - |
| 2 | Keyframes | Server: OpenCV, pHash, pose deltas, row segmentation from agent events | Evidence frames | Row keyframes + room keyframes | Time-based sampling |
| 3 | Spine detection | `gemini-3.8-flash` box_2d (tiled) + `fal-ai/sam-3-1/image` "book spine" | Row keyframes | Reconciled spine masks per row, flags | `gemini-3.1-pro-preview` on disputed rows; Roboflow spine model |
| 4 | Cross-frame merge | Homography chaining / 3D DBSCAN + title alignment | Spines per frame | Unique book tracks with best `frame_ref` | Per-row single frame only |
| 5 | Scale + spine dims | ArUco homography, WebXR depth, book-format prior | Tracks + frames | `spine_height_cm`, `spine_thickness_cm`, `scale_method`, shelf run | Prior + review flag |
| 6 | Spine reading | Gemini on rotated crops + Cloud Vision OCR | Spine crops | Verbatim fields + `c_ocr` | PaddleOCR PP-OCRv5 local |
| 7 | Resolution | Open Library + Google Books + RapidFuzz gates | Transcripts + measured height | `identified / unidentified`, work/edition/ISBN, `id_confidence` | Unidentified + review |
| 8 | Room | Numpy/Open3D RANSAC, Manhattan polygon, ray-plane ceiling | WebXR point cloud + poses + wall-edge points | L/W/H, floor, wall gross/net, shelved area, openings, shape notes | VGGT/MapAnything + ArUco scale |
| 9 | Items | Gemini taxonomy detection (+ SAM 3.1 masks) + 3D dedup + plane/depth sizing | Room keyframes | Items with category, dims, `frame_ref` | Category priors flagged |
| 10 | Evidence | Deterministic code | All of the above | Every number has `frame_ref` + method + inputs; totals computed in code | - |

**Latency budget** (under 5 min after the sweep): stages 3, 6 and 9 run *during* the sweep as frames arrive, so the post-sweep work is mostly stages 4, 5, 7 and 8. With about 20 parallel Gemini calls, expect 60-120 s. Room geometry is under 10 s of numpy.

**Cost per sweep** (rough, unverified):
- About 60 row frames x 3 tiles + about 150 crop batches + about 120 room frames comes to roughly 0.5-1.5M input tokens on 3.8 Flash ($0.75/M through 2026), plus output: **about $1-2**.
- Cloud Vision: about 150-300 images, under $0.50 after the free tier.
- SAM 3.1 on fal: check `get_pricing`.
- Live session: separate.

---

## 8. GPU vs hosted decision

**Hosted for all ML inference. Local GPU is optional and only for experiments.** Reasons:
- **GTX 1660 Ti (Turing, 6 GB):** no bf16, no FlashAttention-2 (Ampere and later), and the GTX 16xx series is known for fp16 NaN/black-output problems in some PyTorch models. VGGT-1B, SAM 3 (about 848M) and Qwen3-VL-8B are all borderline at 6 GB on Windows, with no Docker or WSL. Getting xformers, triton and CUDA builds working on native Windows would burn hours the deadline does not have.
- **Hosted options exist for each piece:**
  - Gemini 3.8 Flash and 3.1 Pro (detection, reading, items).
  - fal: `fal-ai/sam-3-1/image`, `fal-ai/vggt-1b`.
  - Replicate: DA3-metric, MapAnything.
  - HF Spaces: VGGT, MapAnything.
  - Google Cloud Vision.
- **Local stays CPU-only:** OpenCV (ArUco, homography, Laplacian, stitching), numpy/Open3D geometry, RapidFuzz. PaddleOCR CPU is a possible offline OCR fallback. DA3METRIC-LARGE (0.35B, Apache-2.0) fits 6 GB in fp32 if you want to test a local depth fallback.

---

## 9. Risks (ranked)

1. **WebXR does not work as hoped on the phone:** ARCore or Depth not supported, camera-access image too low-res, context loss, or `readPixels` too slow. *Mitigation:* a 1-hour spike before anything else. Log `XRCamera.width/height`, depth availability, and read FPS. If it fails, switch to the `getUserMedia` 4K + ArUco + VGGT/MapAnything path.
2. **Gemini Live 2-minute audio+video cap** kills the session mid-sweep. *Mitigation:* `contextWindowCompression` + `sessionResumption`. Frames are already persisted separately.
3. **Shelving hides the walls**, so room dimensions come out short. *Mitigation:* fit walls from the wall-ceiling band; the agent asks for a ceiling-line pan. Check polygon vs shelf-front distance.
4. **Spine thickness misses 15%** on thin paperbacks. *Mitigation:* 4K capture or "step closer", shared-boundary measurement, row-sum constraint. Report per-format error.
5. **VLM merges or misses thin spines, or invents titles.** *Mitigation:* SAM cross-check, tiling, gap check, OCR corroboration, strict catalogue gates calibrated on ground truth.
6. **Printed marker scaled by the printer.** *Mitigation:* 10 cm check line, measured once with a ruler, size stored in config (not hardcoded). Bring printed markers to the live review.
7. **Unseen room at review** (different lighting, white walls, a dense antiquarian shelf). *Mitigation:* do not tune thresholds to one room. Report results on all shelves, including bad ones. Low-light warnings from the agent (ARCore degrades below 30-40 lux).
8. **Rate limits:** Open Library about 1 req/s unless you send an identifying User-Agent. *Mitigation:* identify the app, cache responses, query Google Books in parallel.

---

## Sources
- Gemini image understanding / box_2d: https://ai.google.dev/gemini-api/docs/image-understanding
- Vertex bounding-box detection: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/bounding-box-detection
- Gemini models list: https://ai.google.dev/gemini-api/docs/models
- Gemini pricing: https://ai.google.dev/gemini-api/docs/pricing
- Gemini Live session management: https://ai.google.dev/gemini-api/docs/live-session
- Live API limits (Firebase mirror): https://firebase.google.com/docs/ai-logic/live-api/limits-and-specs
- Roboflow on Gemini 2.5 detection: https://roboflow.com/blog/gemini-2-5-object-detection-segmentation
- RF100-VL benchmark: https://arxiv.org/abs/2505.20612
- SAM 3: https://arxiv.org/abs/2511.16719
- Qwen3-VL repo: https://github.com/QwenLM/Qwen3-VL
- Qwen3-VL technical report: https://arxiv.org/abs/2511.21631
- Qwen3.5 release: https://alternativeto.net/news/2026/2/alibaba-launches-qwen3-5-with-open-weight-397b-model-and-broader-language-support
- Roboflow spine datasets: https://universe.roboflow.com/book-detection/book-spine-detection-p2 , https://universe.roboflow.com/minhaz/book_spine_instance_segmentation-uqybb , https://universe.roboflow.com/libvision/book-spines-vh9b8
- Open Shelves: https://github.com/capjamesg/open-shelves
- Oriented R-CNN spine detection: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11679322/
- Tsai et al. hybrid spine recognition: https://web.stanford.edu/~bgirod/pdfs/Tsai_ACM_Multimedia_11.pdf
- PaddleOCR 3.0 report: https://arxiv.org/abs/2507.05595
- PP-OCRv5 docs: https://www.paddleocr.ai/v3.3.1/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5.html
- docTR: https://github.com/mindee/doctr
- Cloud Vision OCR: https://cloud.google.com/vision/docs/ocr
- Cloud Vision pricing: https://cloud.google.com/vision/pricing
- Open Library search API: https://openlibrary.org/dev/docs/api/search
- Open Library API and rate limits: https://openlibrary.org/developers/api
- Google Books API: https://developers.google.com/books/docs/v1/using
- RapidFuzz: https://github.com/rapidfuzz/RapidFuzz
- OpenCV ArUco: https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
- OpenCV Stitcher: https://docs.opencv.org/4.x/d2/d8d/classcv_1_1Stitcher.html
- Laplacian blur detection: https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
- Book sizes: https://publishing.co.uk/guides/standard-book-sizes-uk/ , https://thebookcoverdesigner.com/standard-book-sizes/
- WebXR vs ARCore feature status: https://developers.google.com/ar/develop/webxr/arcore-comparison
- WebXR Depth Sensing spec: https://immersive-web.github.io/depth-sensing/
- WebXR Raw Camera Access spec: https://immersive-web.github.io/raw-camera-access/
- WebXR camera support (caniuse): https://caniuse.com/wf-webxr-camera
- camera-access context-loss report: https://forum.babylonjs.com/t/using-camera-access-feature-in-webxr-causes-context-loss-and-flickering-screen-issues/51610
- Hit-test noise at close range: https://forum.babylonjs.com/t/webxr-anchors-and-hit-test/60387
- ARCore Depth API: https://developers.google.com/ar/develop/depth
- ARCore accuracy studies: https://rt.nure.ua/article/view/350221 , https://excel.fit.vutbr.cz/submissions/2019/049/49.pdf
- VGGT: https://github.com/facebookresearch/vggt , https://huggingface.co/spaces/facebook/vggt
- FastVGGT: https://arxiv.org/abs/2509.02560
- MapAnything: https://github.com/facebookresearch/map-anything , https://arxiv.org/abs/2509.13414 , https://replicate.com/vufinder/map-anything
- Depth Anything 3: https://github.com/ByteDance-Seed/depth-anything-3 , https://replicate.com/vufinder/depth-anything-v3-metric
- MoGe-2: https://arxiv.org/abs/2507.02546
- Depth Pro: https://github.com/apple/ml-depth-pro
- UniDepth: https://github.com/lpiccinelli-eth/UniDepth
- Metric3D: https://github.com/YvanYin/Metric3D
- MASt3R-SLAM: https://arxiv.org/abs/2412.12392
- SpatialLM: https://arxiv.org/abs/2506.07491
- HorizonNet: https://arxiv.org/abs/1901.03861
- LGT-Net: https://openaccess.thecvf.com/content/CVPR2022/html/Jiang_LGT-Net_Indoor_Panoramic_Room_Layout_Estimation_With_Geometry-Aware_Transformer_Network_CVPR_2022_paper.html
- HorizonNet / LGT-Net 3D IoU (ECCV 2024 supplementary): https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06150-supp.pdf
- RoomPlan: https://developer.apple.com/augmented-reality/roomplan/
- RoomPlan accuracy test: https://www.nomtek.com/blog/getting-a-room-plan-with-arkit
- WebXR device support (iOS): https://abratabia.com/webxr-games/webxr-device-support.php
- fal endpoints seen in the fal catalogue: `fal-ai/sam-3-1/image`, `fal-ai/sam-3-1/image-rle`, `fal-ai/vggt-1b`, `fal-ai/image-preprocessors/depth-anything/v2` (relative depth only)
