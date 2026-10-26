import cv2
import numpy as np
import pytest

from libclaim.packet.crops import Cropper
from libclaim.stages import CropConfig
from libclaim.sweep import SweepDir
from libclaim.vision.boxes import Box

CFG = CropConfig(dir="crops", max_side_px=100, jpeg_quality=80, margin_frac=0.0)


@pytest.fixture
def sweep(tmp_path):
    run = SweepDir(tmp_path / "run")
    run.frames.mkdir(parents=True)
    image = np.zeros((300, 600, 3), np.uint8)
    image[50:250, 100:160] = (0, 0, 255)
    cv2.imwrite(str(run.frames / "a.jpg"), image)
    return run


def test_crop_is_written_under_the_sweep_with_box_and_frame_size(sweep):
    found = Cropper(sweep, CFG).locate("b01-001", str(sweep.frames / "a.jpg"), Box(x1=100, y1=50, x2=160, y2=250))
    assert found.crop_ref == "frames/crops/b01-001.jpg"
    assert found.box == [100, 50, 160, 250] and found.frame_size == [600, 300]
    crop = cv2.imread(str(sweep.root / found.crop_ref))
    assert max(crop.shape[:2]) == 100
    assert crop[..., 2].mean() > 200 and crop[..., 0].mean() < 60


def test_small_boxes_are_not_upscaled(sweep):
    found = Cropper(sweep, CFG).locate("i_001", "a.jpg", Box(x1=100, y1=50, x2=140, y2=90))
    assert cv2.imread(str(sweep.root / found.crop_ref)).shape[:2] == (40, 40)


def test_box_is_clamped_to_the_frame(sweep):
    found = Cropper(sweep, CFG).locate("b", "a.jpg", Box(x1=-20, y1=-5, x2=700, y2=400))
    assert found.box == [0, 0, 600, 300]


def test_margin_widens_the_crop_inside_the_frame(sweep):
    cfg = CFG.model_copy(update={"margin_frac": 0.5, "max_side_px": 1000})
    found = Cropper(sweep, cfg).locate("b", "a.jpg", Box(x1=100, y1=50, x2=200, y2=150))
    assert found.box == [100, 50, 200, 150]
    assert cv2.imread(str(sweep.root / found.crop_ref)).shape[:2] == (200, 200)


@pytest.mark.parametrize("path", ["", "missing.jpg"])
def test_missing_frame_gives_an_empty_locator(sweep, path):
    found = Cropper(sweep, CFG).locate("b", path, Box(x1=0, y1=0, x2=10, y2=10))
    assert found.crop_ref == "" and found.box == [] and found.frame_size == []
    assert not (sweep.frames / "crops").exists()


def test_degenerate_box_gives_an_empty_locator(sweep):
    assert Cropper(sweep, CFG).locate("b", "a.jpg", Box(x1=50, y1=50, x2=50, y2=90)).crop_ref == ""


def test_each_frame_is_decoded_once(sweep, monkeypatch):
    reads = []
    real = cv2.imread
    monkeypatch.setattr(cv2, "imread", lambda *args: reads.append(args) or real(*args))
    cropper = Cropper(sweep, CFG)
    for key in "abc":
        cropper.locate(key, "a.jpg", Box(x1=0, y1=0, x2=20, y2=20))
    assert len(reads) == 1
