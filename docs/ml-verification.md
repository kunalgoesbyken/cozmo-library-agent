# Optional ML extra: verification (2026-10-26)

Everything here was measured on this laptop (GTX 1660 Ti 6 GB, driver 572.60, i5-9300H, Windows 11) in a fresh `.venv-ml` (Python 3.12, gitignored). The lazily-imported code in `libclaim/room/depth.py` and `libclaim/vision/masks.py` had never run before this check.

## Install

```
py -3.12 -m venv .venv-ml
.venv-ml\Scripts\python -m pip install -e ".[ml]" --extra-index-url https://download.pytorch.org/whl/cu126
```

- Installed: torch 2.14.1+cu126, torchvision 0.29.1+cu126, transformers 5.20.2, accelerate 1.15.0, huggingface_hub 2.3.0, numpy 2.5.4.
- `torch.cuda.is_available()` is True and `get_arch_list()` includes `sm_75`, so the 1660 Ti is supported by the 2.14.1+cu126 wheel (the research note had this as unverified).
- The `[ml]` extra only had `torch==2.14.1`, unpinned `torchvision` and `transformers`. It now pins `torchvision==0.29.1`, `transformers==5.20.2`, `accelerate==1.15.0`. Without `--extra-index-url`, pip takes the CPU-only torch from PyPI on Windows.
- Installing torch took about 6 minutes (download bound). HF cache for both models: 447 MB under `D:\hf-cache` (set per session via `HF_HOME`).
- Both models downloaded with no token and no account (`depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf`, `facebook/sam2.1-hiera-small`; these are the ids in `config/room.yaml` and `config/vision.yaml`).

## Does the existing code work against the real APIs?

Yes, on CPU, unchanged: `Sam2Processor`/`Sam2Model` with `input_boxes`, `multimask_output=False`, `post_process_masks`, and `AutoModelForDepthEstimation` with `post_process_depth_estimation(target_sizes=...)` all behaved as written.

One real gap: the code never moved anything to the GPU, so the extra ran on CPU even with CUDA available. Fixed with `masks.device()` (cuda when available, else cpu), used to place both models and the inputs, and results go back with `.cpu()`. Offline tests monkeypatch torch; two opt-in tests (`LIBCLAIM_ML=1`, `tests/vision/test_masks_ml.py`) run the real models on synthetic images and pass in `.venv-ml`.

## Metric depth (Depth Anything V2 Metric-Indoor-Small, fp32)

Images (Wikimedia Commons, CC0 or CC BY-SA, kept in `_scratch/ml/rooms`, not in git): two empty apartment rooms, a CC0 lecture hall, and a CC0 close-up of a wall (flat-wall check only).

| Image | Output | depth p5 / median / p95 (m) | Latency CUDA warm | Latency CPU warm (4 threads) | VRAM peak |
|---|---|---|---|---|---|
| empty room, window + cord | 960x1280 f32 | 1.81 / 2.90 / 3.23 | 0.18 s | 1.4-2.1 s | 244 MiB |
| empty room, corner windows | 960x1280 f32 | 2.53 / 3.41 / 4.42 | 0.15 s | 1.7 s | 244 MiB |
| lecture hall | 964x1280 f32 | 2.46 / 5.06 / 13.84 | 0.13 s | not run | 244 MiB |
| wall with frames | 873x1280 f32 | 1.92 / 2.05 / 2.17 | 0.20 s | 1.9 s | 257 MiB |

First call per process is 7-12 s on CUDA (weights load + CUDA init), about 1.3-13 s on CPU.

Sanity, using the depth plus the pinhole intrinsics from `libclaim.room.depth.camera` (fallback 931 px focal) and a RANSAC plane fit with no marker scaling:

- Both apartment rooms: the floor fits as a plane (81 % and 98 % inliers) with the camera 1.71 m and 1.67 m above it, which is a plausible handheld-phone height. Far wall 3.1 m and corner 4.6 m away: plausible for a small bedroom.
- Lecture hall: floor plane 1.79 m below the camera, ceiling plane 3.18 m above it (25 % inliers, fans and beams), so a ceiling at about 4.97 m. Plausible for a hall with clerestory windows, not checked against a tape.
- Wall close-up: the fitted normal is along the optical axis (x, y, z = -0.06, 0.02, -1.0), flat at 2.05 m with a 0.25 m p5-p95 spread. Sane for a wall.
- None of the apartment photos shows a ceiling, so a 2.4-3 m ceiling height was not measured on a normal room. Floor-to-camera height and the far-wall distances are the evidence for metric plausibility.

Verdict: DA V2 Metric-Indoor-Small gives plausible absolute metres (no ArUco scaling needed to land in the right range), as a prior. It is a single-image estimate; absolute error of 10-20 % is normal for this class of model and was not measurable here without ground truth. `depth.max_m: 10.0` in `config/room.yaml` clips the far part of large halls (the hall reached 14.8 m).

## SAM 2.1 small, box prompted (fp32)

Boxes came from the repo's RF-DETR detector (`rfdetr_boxes`) on `_scratch/shelf/{blyton,german,library}.jpg`; 12 evenly spaced boxes per photo through `masks.spine_boxes`. Overlays are in `_scratch/shelf/*.sam_*.png` (red = detector box, green = box tightened from the mask).

| Photo | 12 boxes CUDA warm | CPU warm | IoU(detector box, mask box) min / median / max |
|---|---|---|---|
| blyton (688x1920, dense) | 0.34 s | 4.6 s | 0.22 / 0.63 / 0.94 |
| german (888x1920) | 0.33 s | 4.5 s | 0.89 / 0.94 / 0.99 |
| library (1280x1920) | 0.38 s | 6.1 s | 0.69 / 0.93 / 0.98 |

- VRAM peak 803 MiB for 12 boxes, 2.5 GiB for all 51 boxes of blyton in one call (0.55 s). A shelf with 150+ boxes in one call would approach the 6 GB limit; batch the boxes if that arises.
- Edges: on isolated spines the mask box hugs the spine (height ratio median 0.99, width 0.97 on german and library). On the dense blyton shelf, detector boxes that straddle several thin spines produce a mask of one spine (width ratio median 0.65) and stacked or tilted books give slivers. These are detector-box problems, not mask-edge problems, and tight_boxes then measures one spine, not the pair.
- First call: about 7-12 s (load).

## Surprising or worth knowing

- `CUDA_VISIBLE_DEVICES=` (empty) does not hide the GPU on Windows torch; use `-1`. With `-1`, the onnxruntime-directml RF-DETR step segfaulted at process start, so CPU SAM numbers were measured by forcing `masks.device` to cpu instead.
- transformers logs a benign warning when loading SAM 2.1: "model type `sam2_video` to instantiate `sam2`". Results are correct.
- huggingface_hub warns about missing symlink support on Windows (cache works, a little more disk).
- Anonymous HF downloads worked and were fast (13 s and 33 s).
- Depth latency on CUDA is 0.13-0.2 s, about twice the research estimate of 50-100 ms; on CPU it is 1.3-2.1 s per frame. Both fit the sweep budget (40 frames: about 6 s CUDA, about 70 s CPU).
