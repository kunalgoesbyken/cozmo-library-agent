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


def mask_edges(mask: np.ndarray, box: Box, band: float) -> tuple[float, float]:
    middle = (box.y1 + box.y2) / 2
    half = box.height * band / 2
    rows = mask[max(int(middle - half), 0) : int(middle + half) + 1]
    spans = [np.flatnonzero(row) for row in rows]
    spans = [span for span in spans if span.size]
    if not spans:
        return box.x1, box.x2
    return float(np.median([s[0] for s in spans])), float(np.median([s[-1] + 1 for s in spans]))


def spine_edges(path: str, boxes: list[Box]) -> list[tuple[float, float]] | None:
    if not boxes or not available():
        return None
    import torch

    cfg = load_vision_config().masks
    processor, model = sam_model(cfg.model_id)
    rgb = np.ascontiguousarray(read_image(path)[:, :, ::-1])
    inputs = processor(images=rgb, input_boxes=[[[b.x1, b.y1, b.x2, b.y2] for b in boxes]], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    found = processor.post_process_masks(outputs.pred_masks, inputs["original_sizes"])[0][:, 0].numpy()
    return [mask_edges(mask, box, cfg.band) for mask, box in zip(found, boxes)]
