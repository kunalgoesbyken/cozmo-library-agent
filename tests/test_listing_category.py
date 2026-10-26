from types import SimpleNamespace

from libclaim.pipeline import item_sources, listing_category


def det(description):
    return SimpleNamespace(description=description, brand_model="")


def test_the_item_itself_decides_the_category_not_the_furniture_it_stands_on():
    sources = item_sources("IN")
    assert listing_category(det("Table lamp with pleated shade on the desk/shelf unit"), sources) == "table lamp"
    assert listing_category(det("Tall bookshelf with a table lamp on top"), sources) == "bookshelf"


def test_unmatched_description_has_no_listing_category():
    assert listing_category(det("Large decorative vase"), item_sources("IN")) == ""
    assert listing_category(det("Large decorative vase"), None) == ""
