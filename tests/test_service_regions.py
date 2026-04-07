import pytest

from shuo.services.flux import _deepgram_environment, _deepgram_flux_options
from shuo.services.twilio_client import _twilio_client_options


def test_deepgram_region_defaults_to_us_endpoints():
    env = _deepgram_environment("us")

    assert env.base == "wss://api.deepgram.com"
    assert env.production == "wss://api.deepgram.com"
    assert env.agent == "wss://agent.deepgram.com"


def test_deepgram_eu_region_uses_eu_endpoints():
    env = _deepgram_environment("eu")

    assert env.base == "wss://api.eu.deepgram.com"
    assert env.production == "wss://api.eu.deepgram.com"
    assert env.agent == "wss://agent.eu.deepgram.com"


def test_deepgram_region_rejects_unknown_values():
    with pytest.raises(ValueError, match="DEEPGRAM_REGION"):
        _deepgram_environment("apac")


def test_flux_defaults_to_flux_english(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_MODEL", raising=False)
    monkeypatch.delenv("DEEPGRAM_LANGUAGE", raising=False)

    assert _deepgram_flux_options() == {
        "model": "flux-general-en",
        "encoding": "mulaw",
        "sample_rate": 8000,
    }


def test_flux_accepts_english_language_aliases():
    assert _deepgram_flux_options(language="en-US") == {
        "model": "flux-general-en",
        "encoding": "mulaw",
        "sample_rate": 8000,
    }


def test_flux_rejects_non_english_language():
    with pytest.raises(ValueError, match="DEEPGRAM_LANGUAGE"):
        _deepgram_flux_options(language="pt-BR")


def test_flux_rejects_non_flux_model():
    with pytest.raises(ValueError, match="DEEPGRAM_MODEL"):
        _deepgram_flux_options(model="nova-3")


def test_twilio_defaults_to_us_region_and_edge(monkeypatch):
    monkeypatch.delenv("TWILIO_REGION", raising=False)
    monkeypatch.delenv("TWILIO_EDGE", raising=False)

    assert _twilio_client_options() == {
        "region": "us1",
        "edge": "ashburn",
    }


def test_twilio_region_and_edge_can_be_overridden(monkeypatch):
    monkeypatch.setenv("TWILIO_REGION", "ie1")
    monkeypatch.setenv("TWILIO_EDGE", "dublin")

    assert _twilio_client_options() == {
        "region": "ie1",
        "edge": "dublin",
    }
