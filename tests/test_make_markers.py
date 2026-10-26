import importlib.util
import zlib

import cv2
import numpy as np
import pytest
from PIL import Image, PdfParser

from libclaim.settings import ROOT, Settings
from libclaim.vision.boxes import load_vision_config

CM_PER_INCH = 2.54
POINTS_PER_INCH = 72


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


def check_page(gray, marker_id, size_cm, dpi):
    corners, ids = detect(gray)
    assert ids.flatten().tolist() == [marker_id]
    side_px = np.linalg.norm(corners[0][0][0] - corners[0][0][1])
    assert abs(side_px / dpi * CM_PER_INCH - size_cm) < 0.05
    below = gray[int(corners[0][0][:, 1].max()) + 1 :] < 128
    line_px = below.sum(axis=1).max()
    assert abs(line_px / dpi * CM_PER_INCH - load_vision_config().markers.check_line_cm) < 0.05


def pdf_pages(path):
    pdf = PdfParser.PdfParser(filename=str(path))
    pages = []
    for ref in pdf.pages:
        page = pdf.read_indirect(ref)
        image = pdf.read_indirect(page[b"Resources"][b"XObject"][b"Im0"])
        width, height = image.dictionary[b"Width"], image.dictionary[b"Height"]
        assert image.dictionary[b"BitsPerComponent"] == 1
        assert image.dictionary[b"Filter"] == b"FlateDecode"
        bits = np.frombuffer(zlib.decompress(image.buf), np.uint8).reshape(height, -1)
        gray = (np.unpackbits(bits, axis=1)[:, :width] * 255).astype(np.uint8)
        pages.append((page[b"MediaBox"], gray))
    pdf.close()
    return pages


def test_png_sheets_are_lossless_with_unit_ids_size_and_check_line(tmp_path):
    written = tool().make_markers(tmp_path / "markers.png", units=2, settings=Settings(aruco_marker_size_cm=15.0))
    assert len(written) == 2
    dpi = load_vision_config().markers.dpi
    for marker_id, path in enumerate(written):
        image = Image.open(path)
        assert image.mode == "1"
        assert round(image.info["dpi"][0]) == dpi
        check_page(np.array(image.convert("L")), marker_id, 15.0, dpi)


def test_pdf_pages_decode_to_detectable_markers_at_physical_size(tmp_path):
    written = tool().make_markers(tmp_path / "markers.pdf", units=3, settings=Settings())
    cfg = load_vision_config().markers
    pages = pdf_pages(written[0])
    assert len(pages) == 3
    for marker_id, (media_box, gray) in enumerate(pages):
        width_cm, height_cm = (value / POINTS_PER_INCH * CM_PER_INCH for value in media_box[2:])
        assert (round(width_cm, 2), round(height_cm, 2)) == tuple(cfg.page_cm)
        check_page(gray, marker_id, cfg.print_size_cm, cfg.dpi)


def test_size_differing_from_the_setting_fails_loudly(tmp_path):
    with pytest.raises(ValueError, match="ARUCO_MARKER_SIZE_CM"):
        tool().make_markers(tmp_path / "m.pdf", units=1, size_cm=16.0, settings=Settings(aruco_marker_size_cm=15.0))
