import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from libclaim.vision.boxes import VisionConfig, load_vision_config

CM_PER_INCH = 2.54
PDF_SUFFIX = ".pdf"
FONT = cv2.FONT_HERSHEY_SIMPLEX
INK = 0
PAPER = 255


def to_px(cm: float, dpi: int) -> int:
    return round(cm / CM_PER_INCH * dpi)


def write_label(sheet: np.ndarray, text: str, top: int, height: int, margin: int) -> int:
    weight = max(1, height // 10)
    scale = cv2.getFontScaleFromHeight(FONT, height, weight)
    (width, _), _ = cv2.getTextSize(text, FONT, scale, weight)
    room = sheet.shape[1] - 2 * margin
    if width > room:
        scale *= room / width
        (width, _), _ = cv2.getTextSize(text, FONT, scale, weight)
    cv2.putText(sheet, text, ((sheet.shape[1] - width) // 2, top + height), FONT, scale, INK, weight, cv2.LINE_AA)
    return top + 2 * height


def sheet_for(marker_id: int, size_cm: float, cfg: VisionConfig) -> np.ndarray:
    markers = cfg.markers
    dpi = markers.dpi
    width, height = (to_px(cm, dpi) for cm in markers.page_cm)
    sheet = np.full((height, width), PAPER, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg.scale.aruco_dictionary))
    side = to_px(size_cm, dpi)
    left, top = (width - side) // 2, to_px(markers.margin_cm, dpi)
    sheet[top : top + side, left : left + side] = cv2.aruco.generateImageMarker(dictionary, marker_id, side)
    line = to_px(markers.check_line_cm, dpi)
    stroke = max(1, to_px(markers.line_width_cm, dpi))
    tick = to_px(markers.label_height_cm, dpi)
    line_left, line_top = (width - line) // 2, top + side + to_px(markers.gap_cm, dpi)
    sheet[line_top : line_top + stroke, line_left : line_left + line] = INK
    sheet[line_top - tick : line_top + tick, line_left : line_left + stroke] = INK
    sheet[line_top - tick : line_top + tick, line_left + line - stroke : line_left + line] = INK
    margin = to_px(markers.margin_cm, dpi)
    cursor = line_top + 2 * tick
    for text in (
        f"{markers.check_line_cm:g} cm check line - print at 100%, no fit to page",
        f"shelf unit {marker_id} - marker id {marker_id} - {cfg.scale.aruco_dictionary}",
        f"marker side {size_cm:g} cm - measure it, set ARUCO_MARKER_SIZE_CM",
    ):
        cursor = write_label(sheet, text, cursor, tick, margin)
    return sheet


def make_markers(out: Path, units: int, size_cm: float | None = None) -> list[Path]:
    cfg = load_vision_config()
    side = size_cm or cfg.markers.print_size_cm
    dpi = cfg.markers.dpi
    pages = [Image.fromarray(sheet_for(marker_id, side, cfg)) for marker_id in range(units)]
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == PDF_SUFFIX:
        pages[0].save(out, save_all=True, append_images=pages[1:], resolution=dpi)
        return [out]
    written = []
    for marker_id, page in enumerate(pages):
        path = out.with_name(f"{out.stem}-{marker_id}{out.suffix}")
        page.save(path, dpi=(dpi, dpi))
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Printable ArUco marker sheets, one per shelf unit")
    parser.add_argument("out", type=Path, help="output .pdf (one page per unit) or image path (one file per unit)")
    parser.add_argument("--units", type=int, required=True, help="number of shelf units; unit n gets marker id n")
    parser.add_argument("--size-cm", type=float, help="printed marker side in cm")
    args = parser.parse_args()
    for path in make_markers(args.out, args.units, args.size_cm):
        print(path)


if __name__ == "__main__":
    main()
