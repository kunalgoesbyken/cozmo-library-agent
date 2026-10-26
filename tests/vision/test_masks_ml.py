import os
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from libclaim.models import Still
from libclaim.room import depth
from libclaim.vision import masks
from libclaim.vision.boxes import Box, iou

ml_only = pytest.mark.skipif(os.environ.get("LIBCLAIM_ML") != "1", reason="needs the [ml] extra and LIBCLAIM_ML=1")


def fake_torch(cuda: bool):
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda))


@pytest.mark.parametrize(("cuda", "expected"), [(True, "cuda"), (False, "cpu")])
def test_device_follows_cuda_availability(monkeypatch, cuda, expected):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(cuda))
    assert masks.device() == expected


class FakeInputs(dict):
    def __init__(self):
        super().__init__(original_sizes="sizes")
        self.moved_to = None

    def to(self, target):
        self.moved_to = target
        return self


class FakeTensor:
    def __init__(self, array):
        self.array = array
        self.calls = []

    def __getitem__(self, key):
        return FakeTensor(self.array[key])

    def cpu(self):
        self.calls.append("cpu")
        return self

    def numpy(self):
        assert "cpu" in self.calls
        return self.array


def test_spine_boxes_moves_inputs_to_the_device_and_results_back(monkeypatch, tmp_path):
    inputs = FakeInputs()
    mask = np.zeros((1, 1, 40, 40), bool)
    mask[0, 0, 5:35, 15:25] = True

    class Processor:
        def __call__(self, **_):
            return inputs

        def post_process_masks(self, *_):
            return [FakeTensor(mask)]

    class Model:
        def __call__(self, **_):
            return SimpleNamespace(pred_masks="masks")

    path = tmp_path / "a.png"
    cv2.imwrite(str(path), np.zeros((40, 40, 3), np.uint8))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        no_grad=nullcontext,
    ))
    monkeypatch.setattr(masks, "available", lambda: True)
    monkeypatch.setattr(masks, "sam_model", lambda model_id: (Processor(), Model()))
    found = masks.spine_boxes(str(path), [Box(x1=10, y1=0, x2=30, y2=40)])
    assert inputs.moved_to == "cuda"
    assert found == [Box(x1=15, y1=5, x2=25, y2=35)]


def shelf_image():
    image = np.full((480, 640, 3), 90, np.uint8)
    spans = [(60, 130), (150, 230), (250, 310), (330, 420), (440, 500)]
    for index, (x1, x2) in enumerate(spans):
        image[60:430, x1:x2] = (40 + 40 * index, 200 - 30 * index, 60 + 35 * index)
    return image, [Box(x1=x1 - 8, y1=50, x2=x2 + 8, y2=440) for x1, x2 in spans]


@ml_only
def test_sam_masks_hug_box_prompted_spines(tmp_path):
    image, boxes = shelf_image()
    path = tmp_path / "shelf.png"
    cv2.imwrite(str(path), image)
    found = masks.spine_boxes(str(path), boxes)
    assert found is not None and len(found) == len(boxes)
    truth = [Box(x1=x1, y1=60, x2=x2, y2=430) for x1, x2 in [(60, 130), (150, 230), (250, 310), (330, 420), (440, 500)]]
    assert all(iou(a, b) > 0.9 for a, b in zip(found, truth))


@ml_only
def test_metric_depth_returns_a_finite_positive_map_at_image_size(tmp_path):
    image, _ = shelf_image()
    path = tmp_path / "room.png"
    cv2.imwrite(str(path), image)
    still = Still(seq=0, t_ms=0, path=str(path), w=640, h=480, blur=1.0, glare=0.0, motion_dps=0.0)
    result = depth.metric_depth(still)
    assert result.shape == (480, 640) and np.isfinite(result).all() and result.min() > 0
