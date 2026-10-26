import pytest

from libclaim.llm import load_llm_config


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    config = load_llm_config()
    monkeypatch.setenv(config.offline_env, config.offline_value)
