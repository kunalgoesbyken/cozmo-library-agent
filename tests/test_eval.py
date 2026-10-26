import importlib.util
import json
import sys
from decimal import Decimal

import pytest

from libclaim.packet.models import (
    Book,
    BookReplacementCost,
    ClaimPacket,
    Costs,
    Dimensions,
    Item,
    ReviewEntry,
    Room,
    SweepInfo,
    UsedValue,
)
from libclaim.settings import ROOT
from libclaim.sweep import Ledger

TITLES = [
    "The Selfish Gene",
    "Gödel Escher Bach",
    "Thinking Fast and Slow",
    "A Brief History of Time",
    "Sapiens",
    "The Pragmatic Programmer",
    "Clean Code",
    "Zen and the Art of Motorcycle Maintenance",
    "Dune",
    "The Left Hand of Darkness",
]


def tool():
    spec = importlib.util.spec_from_file_location("eval_tool", ROOT / "tools" / "eval.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_tool"] = module
    spec.loader.exec_module(module)
    return module


ev = tool()
CFG = ev.load_eval_config()


def gt_books(legible_except=(5,)):
    return [
        ev.GtBook(unit="1", row="1", position=i + 1, title=t, legible=(i + 1) not in legible_except)
        for i, t in enumerate(TITLES)
    ]


def book(position, title, status="identified", shelf="1-1", **kw):
    return Book(id=f"b{shelf}{position}", shelf=shelf, position=position, status=status, title=title,
                frame_ref=f"frames/{position}.jpg", **kw)


def toy_packet():
    books = []
    for i, title in enumerate(TITLES, start=1):
        if i == 5:
            continue
        books.append(book(i, "Some Other Novel" if i == 8 else title))
    return ClaimPacket(books=books)


def test_toy_truth_count_accuracy_and_confidently_wrong():
    packet = toy_packet()
    truth = gt_books()
    matches = ev.match_books(truth, packet, CFG)
    count = ev.count_section(truth, packet, CFG)
    main, listing = ev.title_sections(truth, matches, packet, CFG)
    assert count.metrics["error"] == pytest.approx(0.10)
    assert count.status == ev.FAIL
    assert (main.metrics["correct"], main.metrics["legible"]) == (8, 9)
    assert (main.metrics["confidently_wrong"], main.metrics["identified"]) == (1, 9)
    assert main.status == ev.FAIL
    assert listing.metrics["count"] == 1
    assert "frames/8.jpg" in "\n".join(listing.lines)
    assert "Some Other Novel" in "\n".join(listing.lines)


def test_alignment_handles_insertion_and_miss_by_title():
    packet = toy_packet()
    packet.books.insert(2, book(99, "Phantom Volume"))
    matches = ev.match_books(gt_books(), packet, CFG)
    inserted = [m for m in matches if m.gt is None]
    missed = [m for m in matches if m.book is None]
    assert [m.book.title for m in inserted] == ["Phantom Volume"]
    assert [m.gt.position for m in missed] == [5]


def test_unidentified_books_do_not_break_alignment_and_count_as_misses():
    books = [book(i, t) for i, t in enumerate(TITLES, start=1)]
    books[3] = book(4, "", status="unidentified")
    packet = ClaimPacket(books=books)
    truth = gt_books(legible_except=())
    matches = ev.match_books(truth, packet, CFG)
    main, _ = ev.title_sections(truth, matches, packet, CFG)
    assert (main.metrics["correct"], main.metrics["legible"]) == (9, 10)
    assert main.metrics["confidently_wrong"] == 0
    assert main.status == ev.PASS


def test_identified_book_on_illegible_spine_is_not_counted_wrong():
    packet = ClaimPacket(books=[book(i, t) for i, t in enumerate(TITLES, start=1)])
    truth = gt_books()
    matches = ev.match_books(truth, packet, CFG)
    main, _ = ev.title_sections(truth, matches, packet, CFG)
    assert main.metrics["confidently_wrong"] == 0
    assert main.status == ev.PASS


def test_dims_within_and_out_of_tolerance():
    packet = ClaimPacket(
        books=[
            book(1, "A", spine_height_cm=22.0, spine_thickness_cm=3.0),
            book(2, "B", spine_height_cm=30.0, spine_thickness_cm=3.0),
        ]
    )
    truth = [
        ev.GtSpine(unit="1", row="1", position=1, height_cm=20.0, thickness_cm=3.0),
        ev.GtSpine(unit="1", row="1", position=2, height_cm=20.0, thickness_cm=4.0),
        ev.GtSpine(unit="1", row="1", position=3, height_cm=20.0, thickness_cm=4.0),
    ]
    section = ev.spines_section(truth, [], packet, CFG)
    height, thickness = section.metrics["height"], section.metrics["thickness"]
    assert height["n"] == 2
    assert height["mean"] == pytest.approx(0.3)
    assert height["share_within"] == pytest.approx(0.5)
    assert thickness["mean"] == pytest.approx(0.125)
    assert thickness["share_within"] == pytest.approx(0.5)
    assert section.metrics["unmatched"] == 1
    assert section.status == ev.FAIL


def test_dims_pass_when_mean_error_within_tolerance():
    packet = ClaimPacket(books=[book(1, "A", spine_height_cm=21.0, spine_thickness_cm=3.3)])
    truth = [ev.GtSpine(unit="1", row="1", position=1, height_cm=20.0, thickness_cm=3.0)]
    assert ev.spines_section(truth, [], packet, CFG).status == ev.PASS


def priced_book(isbn, amount, source, condition="new", source_type="observed"):
    if condition == "new":
        return book(int(isbn), "T", isbn=isbn, replacement_cost=BookReplacementCost(amount=Decimal(amount), source=source))
    return book(int(isbn), "T", isbn=isbn,
                used_value=UsedValue(amount=Decimal(amount), source=source, source_type=source_type))


def gt_price(isbn, price, source, condition="new"):
    return ev.GtPrice(isbn=isbn, source=source, price=price, currency="INR", condition=condition)


def test_prices_report_source_mismatch_separately():
    packet = ClaimPacket(
        sweep=SweepInfo(currency="INR"),
        books=[
            priced_book("1", "110", "bookchor"),
            priced_book("2", "500", "amazon"),
            priced_book("3", "100", "bookchor", condition="good", source_type="derived"),
            book(4, "T", isbn="4"),
        ],
    )
    truth = [
        gt_price("1", 100.0, "Bookchor"),
        gt_price("2", 100.0, "bookchor"),
        gt_price("3", 100.0, "bookchor", condition="good"),
        gt_price("4", 100.0, "bookchor"),
        gt_price("5", 100.0, "bookchor"),
    ]
    section = ev.prices_section(truth, packet, CFG)
    assert section.metrics["n"] == 1
    assert section.metrics["mean"] == pytest.approx(0.10)
    assert section.metrics["mismatched"] == 2
    assert section.metrics["not_priced"] == 2
    assert section.status == ev.PASS
    assert "Mismatched source rows" in "\n".join(section.lines)


def test_prices_fail_when_error_exceeds_tolerance():
    packet = ClaimPacket(books=[priced_book("1", "200", "bookchor")])
    section = ev.prices_section([gt_price("1", 100.0, "bookchor")], packet, CFG)
    assert section.status == ev.FAIL


def test_items_found_with_correct_category():
    packet = ClaimPacket(
        items=[
            Item(id="i1", category="furniture", description="oak reading chair", status="range"),
            Item(id="i2", category="decor", brand_model="Philips lamp", description="table lamp", status="range"),
        ]
    )
    truth = [
        ev.GtItem(name="reading chair", category="furniture"),
        ev.GtItem(name="table lamp", category="lighting", brand="Philips"),
        ev.GtItem(name="globe", category="decor"),
    ]
    section = ev.items_section(truth, packet, CFG)
    assert (section.metrics["found"], section.metrics["correct"]) == (2, 1)
    assert section.status == ev.FAIL


ROOM = ev.RoomTruth(
    ceiling_height_m=2.5,
    walls=[ev.Wall(length_m=4.0), ev.Wall(length_m=3.0), ev.Wall(length_m=4.0), ev.Wall(length_m=3.0)],
    openings=[ev.RoomOpening(width_m=1.0, height_m=2.0)],
    shelving_fronts=[ev.ShelvingFront(width_m=2.0, height_m=2.0)],
)


def test_room_formulas():
    assert ev.floor_area_m2(ROOM, CFG) == pytest.approx(12.0)
    assert ev.gross_wall_area_m2(ROOM) == pytest.approx(35.0)
    assert ev.openings_area_m2(ROOM) == pytest.approx(2.0)
    assert ev.shelving_area_m2(ROOM) == pytest.approx(4.0)


def test_l_shaped_floor_area():
    walls = [(4.0, 90), (2.0, 90), (2.0, 270), (2.0, 90), (2.0, 90), (4.0, 90)]
    room = ev.RoomTruth(
        ceiling_height_m=2.5,
        walls=[ev.Wall(length_m=length, interior_angle_deg=angle) for length, angle in walls],
    )
    assert ev.floor_area_m2(room, CFG) == pytest.approx(12.0)


def test_unclosed_walls_are_not_evaluated():
    room = ev.RoomTruth(ceiling_height_m=2.5, walls=[ev.Wall(length_m=4.0), ev.Wall(length_m=3.0), ev.Wall(length_m=4.0)])
    floor, wall = ev.room_sections(room, ClaimPacket(), CFG)
    assert floor.status == ev.NOT_EVALUATED
    assert "do not close" in floor.reason
    assert wall.status == ev.FAIL


def test_area_errors():
    packet = ClaimPacket(room=Room(floor_area_m2=12.9, wall_area_m2=41.0))
    floor, wall = ev.room_sections(ROOM, packet, CFG)
    assert floor.status == ev.PASS
    assert wall.status == ev.FAIL
    off = ClaimPacket(room=Room(floor_area_m2=14.0, wall_area_m2=36.0))
    floor, wall = ev.room_sections(ROOM, off, CFG)
    assert (floor.status, wall.status) == (ev.FAIL, ev.PASS)


def test_time_to_packet_excludes_live_stages(tmp_path):
    packet = ClaimPacket(
        latency_s={"capture": 600.0, "vision": 100.0, "price": 120.0},
        costs=Costs(usd_at_paid_rates=0.5, by_stage_usd={"vision": 0.3, "price": 0.2}),
    )
    ledger = Ledger(tmp_path / "_ledger.json")
    time, stages = ev.time_sections(packet, ledger, CFG)
    assert time.metrics["seconds"] == pytest.approx(220.0)
    assert time.status == ev.PASS
    assert "0.3000" in "\n".join(stages.lines)
    slow = ClaimPacket(latency_s={"vision": 400.0})
    assert ev.time_sections(slow, ledger, CFG)[0].status == ev.FAIL


def test_time_falls_back_to_ledger(tmp_path):
    ledger = Ledger(tmp_path / "_ledger.json")
    ledger.add("vision", 50.0, 10, 0.1)
    time, _ = ev.time_sections(ClaimPacket(), ledger, CFG)
    assert time.metrics["seconds"] == pytest.approx(50.0)


def test_honesty_counts():
    packet = ClaimPacket(
        books=[book(1, "A"), book(2, "", status="unidentified"), book(3, "", status="needs_appraisal")],
        review_queue=[ReviewEntry(ref_id="b1", severity="high", reason="x", line_ref="books[0]")],
    )
    section = ev.honesty_section(packet)
    assert section.metrics == {"unidentified": 1, "review": 1}


def write_run(tmp_path, packet):
    run = tmp_path / "run"
    run.mkdir()
    (run / "claim_packet.json").write_text(packet.model_dump_json(), encoding="utf-8")
    return run


def test_partial_truth_reports_unevaluated_bars(tmp_path, capsys):
    run = write_run(tmp_path, toy_packet())
    truth = tmp_path / "truth"
    truth.mkdir()
    (truth / "books.csv").write_text(
        "unit,row,position,title,author,legible,orientation\n"
        + "".join(
            f"1,1,{i},{t},,{'n' if i == 5 else 'y'},upright\n" for i, t in enumerate(TITLES, start=1)
        ),
        encoding="utf-8",
    )
    assert ev.main(["eval.py", str(run), str(truth)]) == 0
    out = capsys.readouterr().out
    assert "Could not be evaluated:" in out
    for name in ("Spine dimensions", "Book prices", "Non-book items", "Floor area", "Wall area"):
        assert f"| {name} | NOT EVALUATED |" in out
    assert "| Book count | FAIL |" in out
    assert "| Title identification | FAIL |" in out
    assert "8 of 9" in out
    assert "1 of 9" in out


def test_all_truth_files_missing(tmp_path, capsys):
    run = write_run(tmp_path, toy_packet())
    truth = tmp_path / "empty"
    truth.mkdir()
    sections = ev.evaluate(run, truth)
    statuses = {s.title: s.status for s in sections}
    assert statuses["Book count"] == ev.NOT_EVALUATED
    assert statuses["Title identification"] == ev.NOT_EVALUATED
    assert statuses["Floor area"] == ev.NOT_EVALUATED


def test_room_json_round_trip(tmp_path):
    path = tmp_path / "room.json"
    path.write_text(json.dumps({"ceiling_height_m": 2.5, "walls": [{"length_m": 4.0}, {"length_m": 3.0},
                                                                  {"length_m": 4.0}, {"length_m": 3.0}]}),
                    encoding="utf-8")
    assert ev.floor_area_m2(ev.load_room(path), CFG) == pytest.approx(12.0)


def test_main_usage_error(capsys):
    assert ev.main(["eval.py"]) == 2
