from datetime import date
from decimal import Decimal

import httpx

from libclaim.models import FxRecord
from libclaim.pricing.config import PricingConfig, load_pricing_config
from libclaim.pricing.http import PriceFetchError, PriceHttp
from libclaim.pricing.sources.common import OK


class Fx:
    def __init__(self, http: PriceHttp, config: PricingConfig | None = None):
        self.http = http
        self.config = config or load_pricing_config()

    async def rate(self, base: str, quote: str, on_or_before: date) -> FxRecord:
        cfg = self.config.sources.fx
        url = str(
            httpx.URL(
                cfg.url,
                params={"base": base, "quotes": quote, "providers": cfg.provider, "date": on_or_before.isoformat()},
            )
        )
        response = await self.http.cached_get(url)
        rows = response.json(parse_float=Decimal) if response.status == OK else []
        if not rows:
            raise PriceFetchError(f"no {cfg.provider} rate for {base}/{quote} on or before {on_or_before}")
        row = rows[0]
        return FxRecord(
            base=row["base"],
            quote=row["quote"],
            rate=row["rate"],
            rate_date=date.fromisoformat(row["date"]),
            provider=cfg.provider_label,
            url=url,
            retrieved_at=response.retrieved_at,
            cache_ref=response.cache_ref,
        )
