import asyncio

from libclaim.capture.quality import AlertCode
from libclaim.live.bridge import load_live_config
from libclaim.live.monitor import Alert, Monitor
from libclaim.live.session import Scheduling

CONFIG = load_live_config()


class Harness:
    def __init__(self):
        self.now = 100.0
        self.speaking = False
        self.injected = []
        self.monitor = Monitor(self.inject, CONFIG, clock=lambda: self.now, user_speaking=lambda: self.speaking)

    async def inject(self, event, scheduling):
        self.injected.append((event, scheduling))

    def push(self, code, region, age=0.0):
        return asyncio.run(self.monitor.push(Alert(code, region, self.now - age)))


def test_two_blur_alerts_same_region_within_gap_push_once():
    h = Harness()
    assert h.push(AlertCode.blur, "A1") is True
    h.now += 3
    assert h.push(AlertCode.blur, "A1") is False
    assert h.injected == [({"alert": "blur", "region": "A1"}, Scheduling.WHEN_IDLE)]


def test_different_region_waits_for_min_gap():
    h = Harness()
    h.push(AlertCode.blur, "A1")
    h.now += CONFIG.monitor_min_gap_seconds - 0.1
    assert h.push(AlertCode.glare, "A2") is False
    h.now += 0.2
    assert h.push(AlertCode.glare, "A2") is True


def test_same_region_deduplicated_past_gap_until_cooldown():
    h = Harness()
    h.push(AlertCode.blur, "A1")
    h.now += CONFIG.monitor_min_gap_seconds + 1
    assert h.push(AlertCode.blur, "A1") is False
    h.now += CONFIG.monitor_region_cooldown_seconds
    assert h.push(AlertCode.blur, "A1") is True


def test_never_while_user_speaking():
    h = Harness()
    h.speaking = True
    assert h.push(AlertCode.blur, "A1") is False
    h.speaking = False
    assert h.push(AlertCode.blur, "A1") is True


def test_stale_alert_dropped():
    h = Harness()
    assert h.push(AlertCode.blur, "A1", age=CONFIG.alert_ttl_seconds + 0.1) is False
    assert h.injected == []
    assert h.push(AlertCode.blur, "A1", age=CONFIG.alert_ttl_seconds - 0.1) is True
