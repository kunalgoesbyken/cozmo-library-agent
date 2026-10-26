import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from libclaim.capture.quality import AlertCode
from libclaim.live.bridge import LiveConfig
from libclaim.live.session import Scheduling


@dataclass(frozen=True)
class Alert:
    code: AlertCode
    region: str
    raised_at: float


class Monitor:
    def __init__(
        self,
        inject: Callable[[dict[str, Any], Scheduling], Awaitable[None]],
        config: LiveConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
        user_speaking: Callable[[], bool] = lambda: False,
    ):
        self.inject = inject
        self.config = config
        self.clock = clock
        self.user_speaking = user_speaking
        self.last_spoken_at: float | None = None
        self.region_spoken_at: dict[str, float] = {}

    def should_speak(self, event: Alert, now: float) -> bool:
        age = now - event.raised_at
        if not 0 <= age <= self.config.alert_ttl_seconds or self.user_speaking():
            return False
        if self.last_spoken_at is not None and now - self.last_spoken_at < self.config.monitor_min_gap_seconds:
            return False
        last_for_region = self.region_spoken_at.get(event.region)
        return last_for_region is None or now - last_for_region >= self.config.monitor_region_cooldown_seconds

    async def push(self, event: Alert) -> bool:
        now = self.clock()
        if not self.should_speak(event, now):
            return False
        self.last_spoken_at = now
        self.region_spoken_at[event.region] = now
        await self.inject({"alert": event.code.value, "region": event.region}, Scheduling.WHEN_IDLE)
        return True
