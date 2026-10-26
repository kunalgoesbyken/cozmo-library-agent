from functools import cache

import numpy as np
from rapidocr import RapidOCR

from libclaim.resolve_config import load_resolve_config

WORD_SEPARATOR = " "


@cache
def ocr_engine() -> RapidOCR:
    cfg = load_resolve_config().ocr
    return RapidOCR(params={"Global.log_level": cfg.log_level, "EngineConfig.onnxruntime.use_dml": cfg.use_dml})


def ocr_text(crop: np.ndarray) -> tuple[str, float]:
    result = ocr_engine()(crop)
    if not result.txts:
        return "", 0.0
    return WORD_SEPARATOR.join(result.txts), float(np.mean(result.scores))
