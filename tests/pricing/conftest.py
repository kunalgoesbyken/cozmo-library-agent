import json
from pathlib import Path

import httpx
import pytest

from libclaim.pricing.config import load_pricing_config
from libclaim.pricing.http import PriceHttp

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "pricing"
ROUTES = json.loads((FIXTURES / "routes.json").read_text(encoding="utf-8"))


class Site:
    def __init__(self, extra=None):
        self.routes = {**ROUTES, **(extra or {})}
        self.requests = []

    def handler(self, request):
        url = str(request.url)
        self.requests.append(url)
        route = self.routes.get(url)
        if route is None:
            return httpx.Response(404, content=b"")
        headers = {"content-type": route.get("type", "text/html")}
        if "location" in route:
            headers["location"] = route["location"]
        body = (FIXTURES / route["file"]).read_bytes() if "file" in route else route.get("body", "").encode()
        return httpx.Response(route["status"], content=body, headers=headers)


def fast_config(**http):
    config = load_pricing_config()
    fast = config.http.model_copy(
        update={"min_interval_seconds": 0.001, "host_min_interval_seconds": {}, "backoff_seconds": 0.0, **http}
    )
    return config.model_copy(update={"http": fast})


def make_http(tmp_path, site, config=None):
    client = httpx.AsyncClient(transport=httpx.MockTransport(site.handler), headers={"User-Agent": "test"}, follow_redirects=True)
    return PriceHttp(tmp_path / "cache", client=client, config=config or fast_config())


@pytest.fixture
def online(monkeypatch):
    monkeypatch.delenv("LIBCLAIM_OFFLINE")


@pytest.fixture
def site():
    return Site()


@pytest.fixture
def http(tmp_path, site, online):
    return make_http(tmp_path, site)
