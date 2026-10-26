from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from libclaim.packet.models import ClaimPacket
from libclaim.sweep import SweepDir

TEMPLATE_DIR = Path(__file__).parent / "templates"
REPORT_NAME = "report.html"

_env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=select_autoescape(["html", "j2"]),
)


def render_report(packet: ClaimPacket, sweep: SweepDir) -> Path:
    html = _env.get_template("report.html.j2").render(packet=packet)
    path = sweep.root / REPORT_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
