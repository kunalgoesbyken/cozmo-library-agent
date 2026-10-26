from concurrent.futures import ThreadPoolExecutor
from functools import cache
from typing import Any

import numpy as np
from rapidocr import RapidOCR

from libclaim.resolve_config import load_resolve_config

WORD_SEPARATOR = " "
POOL_PREFIX = "ocr"


def ocr_params() -> dict[str, Any]:
    cfg = load_resolve_config().ocr
    return {
        "Global.log_level": cfg.log_level,
        "Global.use_cls": cfg.use_cls,
        "Det.limit_type": cfg.det_limit_type,
        "Det.limit_side_len": cfg.det_limit_side_len,
        "EngineConfig.onnxruntime.use_dml": cfg.use_dml,
        "EngineConfig.onnxruntime.intra_op_num_threads": cfg.threads,
        "EngineConfig.onnxruntime.inter_op_num_threads": cfg.inter_threads,
    }


@cache
def ocr_engine() -> RapidOCR:
    return RapidOCR(params=ocr_params())


@cache
def ocr_pool() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=load_resolve_config().ocr.workers, thread_name_prefix=POOL_PREFIX)


def ocr_text(crop: np.ndarray) -> tuple[str, float]:
    result = ocr_engine()(crop)
    if not result.txts:
        return "", 0.0
    return WORD_SEPARATOR.join(result.txts), float(np.mean(result.scores))
