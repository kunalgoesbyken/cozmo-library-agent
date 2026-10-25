from decimal import Decimal

import pytest

from libclaim.settings import load_locale, load_settings


def test_locale_in():
    locale = load_locale("IN")
    assert locale.currency == "INR"
    assert locale.country == "IN"
    assert locale.appraisal_threshold == Decimal("5000")
    assert locale.antiquarian_year == 1950
    assert locale.used_sources == ["bookchor", "pustakkosh", "derived"]


def test_locale_gb():
    assert load_locale("GB").new_sources[0] == "penguin_uk"


def test_locale_unknown_raises_key_error():
    with pytest.raises(KeyError):
        load_locale("ZZ")


def test_settings_read_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.delenv("VISION_MODELS", raising=False)
    env = tmp_path / ".env"
    env.write_text("PORT=9100\nVISION_MODELS=a,b\n")
    settings = load_settings(env)
    assert settings.port == 9100
    assert settings.vision_models == ["a", "b"]


def test_process_env_overrides_file(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("PORT=9100\n")
    monkeypatch.setenv("PORT", "9200")
    assert load_settings(env).port == 9200


def test_user_agent_without_contact(tmp_path, monkeypatch):
    monkeypatch.delenv("HTTP_CONTACT_EMAIL", raising=False)
    env = tmp_path / ".env"
    env.write_text("")
    agent = load_settings(env).user_agent
    assert "@" not in agent
    assert agent.startswith("libclaim/")


def test_user_agent_contact_from_env_only(tmp_path, monkeypatch):
    monkeypatch.delenv("HTTP_CONTACT_EMAIL", raising=False)
    env = tmp_path / ".env"
    env.write_text("HTTP_CONTACT_EMAIL=who@example.test\n")
    assert "who@example.test" in load_settings(env).user_agent
