import pytest


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("LIBCLAIM_OFFLINE", "1")
