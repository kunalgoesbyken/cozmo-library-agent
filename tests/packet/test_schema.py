import json

from libclaim.models import FxRecord
from libclaim.packet.models import BookReplacementCost, DimsEstimate, ClaimPacket, SecondCountry, SecondCountryBook
from libclaim.packet.report import render_report
from libclaim.packet.review import build_review_queue
from libclaim.packet.totals import compute_totals
from libclaim.sweep import SweepDir

CONTRACT = {
    "sweep": ["id", "captured_at", "device", "duration_s", "country", "currency"],
    "room": [
        "length_m", "width_m", "height_m", "floor_area_m2", "wall_area_m2",
        "shelved_wall_area_m2", "scale_method", "confidence",
    ],
    "totals": [
        "book_count", "books_identified", "books_unidentified", "shelf_run_m",
        "books_replacement_cost", "books_used_value", "items_replacement_cost_low",
        "items_replacement_cost_high", "excluded_from_totals",
    ],
    "review_queue[]": ["ref_id", "reason"],
    "books[]": [
        "id", "shelf", "position", "frame_ref", "status", "title", "author", "edition",
        "isbn", "spine_height_cm", "spine_thickness_cm", "id_confidence",
        "replacement_cost", "used_value",
    ],
    "books[].replacement_cost": ["amount", "source", "url", "retrieved_at", "converted"],
    "books[].used_value": ["amount", "source", "url", "retrieved_at", "condition_assumed"],
    "items[]": [
        "id", "category", "description", "brand_model", "frame_ref", "dimensions_cm",
        "status", "replacement_cost", "confidence",
    ],
    "items[].dimensions_cm": ["w", "h", "d"],
    "items[].replacement_cost": ["low", "high", "source", "url", "retrieved_at"],
}

ADDED = {
    "room": ["wall_area_net_m2", "openings", "shape", "polygon_m", "frame_refs"],
    "books[]": [
        "scale_method", "evidence", "crop_ref", "box", "frame_size", "dims_estimated_from_prior",
        "edition_unverified", "equivalent_edition", "corroboration",
    ],
    "books[].used_value": ["method", "source_type"],
    "items[]": ["dimensions_method", "crop_ref", "box", "frame_size"],
    "items[].replacement_cost": ["url_high"],
    "totals": ["books_used_value_observed", "books_used_value_derived", "shelf_run_estimated_m"],
    "": ["second_country", "costs", "latency_s"],
    "costs": ["usd_at_paid_rates", "billed_usd", "billing", "tokens", "by_stage_usd"],
}


def resolve(schema, defs, ref_or_node):
    node = ref_or_node
    if "anyOf" in node:
        node = next(n for n in node["anyOf"] if n.get("type") != "null")
    if "items" in node:
        node = node["items"]
    if "$ref" in node:
        node = defs[node["$ref"].split("/")[-1]]
    return node


def properties_at(schema, path):
    defs = schema["$defs"]
    node = schema
    for part in [p for p in path.replace("[]", "").split(".") if p]:
        node = resolve(schema, defs, node["properties"][part])
    return node["properties"]


def test_every_contract_key_path_exists_in_schema():
    schema = ClaimPacket.model_json_schema()
    assert {"sweep", "room", "books", "items", "totals", "review_queue"} <= set(schema["properties"])
    for path, keys in CONTRACT.items():
        assert set(keys) <= set(properties_at(schema, path)), path


def test_added_fields_exist_in_schema():
    schema = ClaimPacket.model_json_schema()
    for path, keys in ADDED.items():
        assert set(keys) <= set(properties_at(schema, path)), path


def packet():
    return ClaimPacket.model_validate(
        {
            "sweep": {"id": "s1", "captured_at": "2026-10-26T10:00:00Z", "currency": "INR"},
            "books": [
                {
                    "id": "b1", "status": "identified", "title": "Dune", "frame_ref": "frames/00001_250.jpg",
                    "replacement_cost": {"amount": "499.00", "url": "https://shop.example/dune", "source": "shop"},
                },
                {"id": "b2", "status": "unidentified", "frame_ref": "frames/00002_500.jpg"},
            ],
            "items": [
                {
                    "id": "i1", "status": "range", "frame_ref": "frames/00003_750.jpg",
                    "replacement_cost": {"low": "100", "high": "200", "url": "https://shop.example/lamp"},
                },
                {"id": "i2", "status": "needs_appraisal", "frame_ref": "frames/00004_900.jpg"},
            ],
            "costs": {"usd_at_paid_rates": 0.1234, "tokens": 900, "by_stage_usd": {"vision": 0.1234}},
            "latency_s": {"vision": 12.5},
        }
    )


def test_money_dumps_as_json_numbers_and_unknown_stays_null():
    data = json.loads(packet().model_dump_json())
    assert data["books"][0]["replacement_cost"]["amount"] == 499.0
    assert data["books"][1]["replacement_cost"]["amount"] is None


def test_review_queue_flags_unknowns_with_frame_refs():
    built = packet()
    queue = build_review_queue(built)
    by_id = {entry.ref_id: entry for entry in queue}
    assert set(by_id) == {"b2", "i2", "room"}
    assert by_id["b2"].frame_ref == "frames/00002_500.jpg"
    assert by_id["b2"].line_ref == "books[1]"
    assert by_id["i2"].severity == "high"


def test_report_links_frames_prices_queue_totals_costs_and_latency(tmp_path):
    built = packet()
    built.totals = compute_totals(built.books, built.items)
    built.review_queue = build_review_queue(built)
    path = render_report(built, SweepDir(tmp_path))
    html = path.read_text(encoding="utf-8")
    assert path == tmp_path / "report.html"
    for ref in ("frames/00001_250.jpg", "frames/00002_500.jpg", "frames/00003_750.jpg"):
        assert f'src="{ref}"' in html
        assert f'<a href="{ref}">' in html
    assert 'href="https://shop.example/dune"' in html
    assert 'href="https://shop.example/lamp"' in html
    assert "Review queue (3)" in html
    assert "spine could not be identified" in html
    assert "499.00" in html
    assert "0.1234" in html
    assert "12.50 s" in html
    assert "http://" not in html.replace("https://", "")
    assert "<script" not in html


def located_packet():
    built = packet()
    built.books[0].crop_ref = "frames/crops/b1.jpg"
    built.books[0].box = [100, 50, 300, 250]
    built.books[0].frame_size = [1000, 500]
    built.books[0].dims_estimated_from_prior = DimsEstimate(height_cm=21.0, thickness_cm=2.5, book_format="trade_paperback")
    built.books[0].scale_method = "book_format_prior"
    return built


def test_report_shows_the_crop_and_outlines_the_box_in_the_full_frame(tmp_path):
    html = render_report(located_packet(), SweepDir(tmp_path)).read_text(encoding="utf-8")
    assert 'src="frames/crops/b1.jpg"' in html
    assert 'left: 10.00%; top: 10.00%; width: 20.00%; height: 40.00%;' in html
    assert 'href="frames/00001_250.jpg"' in html and "box 100, 50, 300, 250" in html


def test_report_marks_prior_dimensions_as_estimates_and_unmeasured_run_as_unknown(tmp_path):
    built = located_packet()
    built.totals = compute_totals(built.books, built.items)
    html = render_report(built, SweepDir(tmp_path)).read_text(encoding="utf-8")
    assert 'class="est"' in html and "est. 21.0 / 2.5" in html
    assert "Shelf run, measured" in html and '<span class="unknown">unknown</span>' in html
    assert "est. 0.025 m" in html


def test_report_does_not_link_non_http_urls(tmp_path):
    built = packet()
    built.books[0].replacement_cost.url = "javascript:alert(1)"
    html = render_report(built, SweepDir(tmp_path)).read_text(encoding="utf-8")
    assert "javascript:" not in html


def test_report_renders_second_country_table_and_item_bound_links(tmp_path):
    built = packet()
    built.items[0].replacement_cost.url_high = "https://shop.example/lamp-high"
    built.second_country = SecondCountry(
        country="GB",
        currency="GBP",
        books=[
            SecondCountryBook(
                book_id="b1", title="Dune", isbn="9780000000001", status="identified", source_type="converted",
                replacement_cost=BookReplacementCost(amount="5.99", source="Penguin UK", url="https://uk.example/dune", converted=True),
                original_amount="499.00", original_currency="INR",
                fx=FxRecord(
                    base="INR", quote="GBP", rate="0.0095", rate_date="2026-10-23", provider="ECB",
                    url="https://fx.example/rate", retrieved_at="2026-10-23T09:00:00Z", cache_ref="fx",
                ),
            )
        ],
    )
    html = render_report(built, SweepDir(tmp_path)).read_text(encoding="utf-8")
    assert "Second country: GB (GBP)" in html and "SecondCountry(" not in html
    assert "converted" in html and "Dune" in html and "5.99" in html
    assert 'href="https://fx.example/rate"' in html and 'href="https://uk.example/dune"' in html
    assert 'href="https://shop.example/lamp"' in html and 'href="https://shop.example/lamp-high"' in html
