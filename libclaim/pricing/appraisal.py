from decimal import Decimal

from libclaim.pricing.config import load_pricing_config
from libclaim.settings import Locale
from libclaim.vision.resolve import tokens

NEW = "new"


def appraisal_reasons(resolution, reading, statements, quotes, locale: Locale) -> list[str]:
    cfg = load_pricing_config().books
    spine = set(tokens(" ".join([reading.title_verbatim, reading.author_verbatim, reading.publisher_verbatim, resolution.ocr_text])))
    new = max((q.amount for q in quotes if q.condition == NEW), default=Decimal(0))
    used = max((q.amount for q in quotes if q.condition != NEW), default=Decimal(0))
    checks = [
        ("user_statement", any(s.needs_appraisal for s in statements)),
        ("spine_wording", any(set(tokens(term)) <= spine for term in cfg.appraisal_triggers)),
        ("antiquarian_year", resolution.year is not None and resolution.year < locale.antiquarian_year),
        ("no_isbn", not resolution.isbn),
        ("used_over_threshold", used >= locale.appraisal_threshold and used > 0),
        ("used_over_new", new > 0 and used >= cfg.used_new_ratio * new),
    ]
    return [name for name, hit in checks if hit]
