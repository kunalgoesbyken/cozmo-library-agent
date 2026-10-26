import os
from pathlib import Path

import cv2
import pytest

from libclaim.models import Still
from libclaim.settings import ROOT
from libclaim.vision.rfdetr import rfdetr_boxes
from libclaim.vision.spines import detect_spines

SHELF = ROOT / "_scratch" / "shelf"

pytestmark = pytest.mark.skipif(os.environ.get("LIBCLAIM_LIVE") != "1", reason="LIBCLAIM_LIVE=1 required")


async def test_live_detection_on_shelf_photo(monkeypatch):
    monkeypatch.delenv("LIBCLAIM_OFFLINE", raising=False)
    path = next(Path(SHELF).glob("*.jpg"))
    img = cv2.imread(str(path))
    height, width = img.shape[:2]
    still = Still(seq=0, t_ms=0, path=str(path), w=width, h=height, blur=1.0, glare=0.0, motion_dps=0.0)
    dets = await detect_spines(still)
    assert len(dets) > 0
    local = rfdetr_boxes(img)
    assert local is None or isinstance(local, list)
