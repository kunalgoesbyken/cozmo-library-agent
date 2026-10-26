import argparse
import zlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from libclaim.settings import Settings, load_settings
from libclaim.vision.boxes import VisionConfig, load_vision_config

CM_PER_INCH = 2.54
POINTS_PER_INCH = 72
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
    cv2.putText(sheet, text, ((sheet.shape[1] - width) // 2, top + height), FONT, scale, INK, weight, cv2.LINE_8)
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
        f"marker side {size_cm:g} cm - this must equal ARUCO_MARKER_SIZE_CM",
        "tape to the shelf front edge at book height",
    ):
        cursor = write_label(sheet, text, cursor, tick, margin)
    return sheet


def pdf_stream(header: str, data: bytes) -> bytes:
    return f"<< {header} /Length {len(data)} >>\nstream\n".encode() + data + b"\nendstream"


def pdf_bytes(pages: list[np.ndarray], dpi: int) -> bytes:
    kids = " ".join(f"{3 + 3 * i} 0 R" for i in range(len(pages)))
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode()]
    for index, page in enumerate(pages):
        height, width = page.shape
        w_pt, h_pt = width / dpi * POINTS_PER_INCH, height / dpi * POINTS_PER_INCH
        first = 3 + 3 * index
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w_pt:.4f} {h_pt:.4f}] /Contents {first + 1} 0 R "
            f"/Resources << /XObject << /Im0 {first + 2} 0 R >> >> >>".encode()
        )
        objects.append(pdf_stream("", f"q {w_pt:.4f} 0 0 {h_pt:.4f} 0 0 cm /Im0 Do Q".encode()))
        bits = zlib.compress(np.packbits(page > INK, axis=1).tobytes())
        header = (f"/Type /XObject /Subtype /Image /Width {width} /Height {height} /ColorSpace /DeviceGray "
                  "/BitsPerComponent 1 /Filter /FlateDecode")
        objects.append(pdf_stream(header, bits))
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def marker_side(size_cm: float | None, settings: Settings, cfg: VisionConfig) -> float:
    configured = settings.aruco_marker_size_cm
    if size_cm is not None and configured is not None and size_cm != configured:
        raise ValueError(f"--size-cm {size_cm:g} differs from ARUCO_MARKER_SIZE_CM {configured:g}, which the app uses")
    return size_cm or configured or cfg.markers.print_size_cm


def make_markers(out: Path, units: int, size_cm: float | None = None, settings: Settings | None = None) -> list[Path]:
    cfg = load_vision_config()
    side = marker_side(size_cm, settings or load_settings(), cfg)
    dpi = cfg.markers.dpi
    pages = [sheet_for(marker_id, side, cfg) for marker_id in range(units)]
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == PDF_SUFFIX:
        out.write_bytes(pdf_bytes(pages, dpi))
        return [out]
    written = []
    for marker_id, page in enumerate(pages):
        path = out.with_name(f"{out.stem}-{marker_id}{out.suffix}")
        Image.fromarray(page).convert("1", dither=Image.Dither.NONE).save(path, dpi=(dpi, dpi))
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Printable ArUco marker sheets, one per shelf unit")
    parser.add_argument("out", type=Path, help="output .pdf (one page per unit) or .png path (one file per unit)")
    parser.add_argument("--units", type=int, required=True, help="number of shelf units; unit n gets marker id n")
    parser.add_argument("--size-cm", type=float, help="printed marker side in cm; defaults to ARUCO_MARKER_SIZE_CM")
    args = parser.parse_args()
    try:
        written = make_markers(args.out, args.units, args.size_cm)
    except ValueError as error:
        parser.error(str(error))
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
