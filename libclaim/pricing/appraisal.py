from decimal import Decimal

from libclaim.live.tools import Statement
from libclaim.models import PriceQuote
from libclaim.pricing.config import load_pricing_config
from libclaim.settings import Locale
from libclaim.vision.reading import Reading
from libclaim.vision.resolve import Resolution, tokens

NEW = "new"


def appraisal_reasons(
    resolution: Resolution, reading: Reading, statements: list[Statement], quotes: list[PriceQuote], locale: Locale
) -> list[str]:
    cfg = load_pricing_config().books
    texts = [reading.title_verbatim, reading.author_verbatim, reading.publisher_verbatim, resolution.edition, resolution.vlm_text, resolution.ocr_text]
    spine = set(tokens(" ".join(texts)))
    new = max((q.amount for q in quotes if q.condition == NEW), default=Decimal(0))
    used = max((q.amount for q in quotes if q.condition != NEW), default=Decimal(0))
    checks = [
        ("user_statement", any(s.needs_appraisal for s in statements)),
        ("spine_wording", any(set(tokens(term)) <= spine for term in cfg.appraisal_triggers)),
        ("antiquarian_year", resolution.year is not None and resolution.year < locale.antiquarian_year),
        ("old_work_edition_unverified", resolution.equivalent_edition and not resolution.edition
         and resolution.work_year is not None and resolution.work_year < locale.antiquarian_year),
        ("no_isbn", not resolution.isbn),
        ("used_over_threshold", used >= locale.appraisal_threshold),
        ("used_over_new", new > 0 and used >= cfg.used_new_ratio * new),
    ]
    return [name for name, hit in checks if hit]
