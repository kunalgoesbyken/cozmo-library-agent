import importlib.util

import cv2
import numpy as np
from PIL import Image

from libclaim.settings import ROOT
from libclaim.vision.boxes import load_vision_config

CM_PER_INCH = 2.54


def tool():
    spec = importlib.util.spec_from_file_location("make_markers", ROOT / "tools" / "make_markers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def detect(gray):
    cfg = load_vision_config().scale
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.aruco_dictionary))
    corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters()).detectMarkers(gray)
    return corners, ids


def test_png_sheets_have_unit_ids_physical_size_and_check_line(tmp_path):
    size_cm = 15.0
    written = tool().make_markers(tmp_path / "markers.png", units=2, size_cm=size_cm)
    assert len(written) == 2
    dpi = load_vision_config().markers.dpi
    for marker_id, path in enumerate(written):
        image = Image.open(path)
        assert round(image.info["dpi"][0]) == dpi
        gray = np.array(image.convert("L"))
        corners, ids = detect(gray)
        assert ids.flatten().tolist() == [marker_id]
        side_px = np.linalg.norm(corners[0][0][0] - corners[0][0][1])
        assert abs(side_px / dpi * CM_PER_INCH - size_cm) < 0.05
        below = gray[int(corners[0][0][:, 1].max()) + 1 :] < 128
        line_px = below.sum(axis=1).max()
        check_cm = load_vision_config().markers.check_line_cm
        assert abs(line_px / dpi * CM_PER_INCH - check_cm) < 0.05


def test_pdf_has_one_page_per_unit(tmp_path):
    written = tool().make_markers(tmp_path / "markers.pdf", units=3)
    assert len(written) == 1
    data = written[0].read_bytes()
    assert data.startswith(b"%PDF")
    assert data.count(b"/Type /Page") - data.count(b"/Type /Pages") == 3
