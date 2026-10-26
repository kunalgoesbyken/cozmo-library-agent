from functools import cache
from importlib.util import find_spec

import numpy as np

from libclaim.vision.boxes import Box, load_vision_config, read_image

ML_MODULES = ("torch", "transformers")


def available() -> bool:
    return all(find_spec(name) is not None for name in ML_MODULES)


@cache
def sam_model(model_id: str):
    from transformers import Sam2Model, Sam2Processor

    return Sam2Processor.from_pretrained(model_id), Sam2Model.from_pretrained(model_id).eval()


def extent(lines: np.ndarray) -> tuple[float, float] | None:
    spans = [np.flatnonzero(line) for line in lines]
    spans = [span for span in spans if span.size]
    if not spans:
        return None
    return float(np.median([s[0] for s in spans])), float(np.median([s[-1] + 1 for s in spans]))


def band(start: float, length: float, fraction: float) -> slice:
    centre, half = start + length / 2, length * fraction / 2
    return slice(max(int(centre - half), 0), int(centre + half) + 1)


def mask_box(mask: np.ndarray, box: Box, fraction: float) -> Box:
    across = extent(mask[band(box.y1, box.height, fraction)])
    along = extent(mask[:, band(box.x1, box.width, fraction)].T)
    x1, x2 = across or (box.x1, box.x2)
    y1, y2 = along or (box.y1, box.y2)
    return Box(x1=x1, y1=y1, x2=x2, y2=y2)


def spine_boxes(path: str, boxes: list[Box]) -> list[Box] | None:
    if not boxes or not available():
        return None
    cfg = load_vision_config().masks
    try:
        processor, model = sam_model(cfg.model_id)
    except (ImportError, OSError):
        return None
    import torch

    rgb = np.ascontiguousarray(read_image(path)[:, :, ::-1])
    inputs = processor(images=rgb, input_boxes=[[[b.x1, b.y1, b.x2, b.y2] for b in boxes]], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    found = processor.post_process_masks(outputs.pred_masks, inputs["original_sizes"])[0][:, 0].numpy()
    return [mask_box(mask, box, cfg.band) for mask, box in zip(found, boxes)]
