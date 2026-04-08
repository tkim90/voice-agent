import json

from shuo.livekit_ptbr import (
    bootstrap_env,
    build_dispatch_metadata,
    build_elevenlabs_voice_settings,
    build_sip_participant_kwargs,
    build_turn_handling,
    livekit_env,
)


def test_bootstrap_env_aliases():
    env = {
        "LIVEKIT_WS_URL": "wss://example.livekit.cloud",
        "ELEVENLABS_API_KEY": "secret",
    }

    bootstrap_env(env)

    assert env["LIVEKIT_URL"] == "wss://example.livekit.cloud"
    assert env["ELEVEN_API_KEY"] == "secret"


def test_build_dispatch_metadata_keeps_phone_number_and_caller_id():
    payload = json.loads(
        build_dispatch_metadata(
            "+14045634375",
            greeting="Cumprimente em portugues.",
            participant_name="Cliente",
            sip_number="+15551234567",
        )
    )

    assert payload == {
        "phone_number": "+14045634375",
        "greeting": "Cumprimente em portugues.",
        "participant_name": "Cliente",
        "sip_number": "+15551234567",
    }


def test_build_sip_participant_kwargs_reads_trunk_and_livekit_phone(monkeypatch):
    monkeypatch.setenv("SIP_OUTBOUND_TRUNK_ID", "trunk-123")
    monkeypatch.setenv("LIVEKIT_PHONE_NUMBER", "+15551234567")

    kwargs = build_sip_participant_kwargs(
        "room-1",
        {"phone_number": "+14045634375", "participant_name": "Cliente"},
    )

    assert kwargs["room_name"] == "room-1"
    assert kwargs["sip_trunk_id"] == "trunk-123"
    assert kwargs["sip_call_to"] == "+14045634375"
    assert kwargs["participant_identity"] == "+14045634375"
    assert kwargs["participant_name"] == "Cliente"
    assert kwargs["sip_number"] == "+15551234567"
    assert kwargs["wait_until_answered"] is True
    assert kwargs["krisp_enabled"] is True


def test_build_turn_handling_falls_back_when_turn_detector_files_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "livekit.plugins.turn_detector.multilingual":
            raise RuntimeError(
                'livekit-plugins-turn-detector initialization failed. Could not find file "languages.json".'
            )
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    turn_handling = build_turn_handling()

    assert turn_handling["interruption"]["mode"] == "vad"
    assert "turn_detection" not in turn_handling


def test_livekit_env_ignores_legacy_global_model_settings(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_MODEL", "flux-general-en")
    monkeypatch.delenv("LIVEKIT_DEEPGRAM_MODEL", raising=False)

    assert livekit_env("LIVEKIT_DEEPGRAM_MODEL", "nova-3") == "nova-3"


def test_build_elevenlabs_voice_settings_uses_livekit_defaults(monkeypatch):
    monkeypatch.delenv("LIVEKIT_ELEVENLABS_STABILITY", raising=False)
    monkeypatch.delenv("LIVEKIT_ELEVENLABS_SIMILARITY_BOOST", raising=False)
    monkeypatch.delenv("LIVEKIT_ELEVENLABS_SPEED", raising=False)
    monkeypatch.delenv("LIVEKIT_ELEVENLABS_USE_SPEAKER_BOOST", raising=False)

    settings = build_elevenlabs_voice_settings()

    assert settings.stability == 0.55
    assert settings.similarity_boost == 0.90
    assert settings.speed == 0.88
    assert settings.use_speaker_boost is False


def test_build_elevenlabs_voice_settings_reads_overrides(monkeypatch):
    monkeypatch.setenv("LIVEKIT_ELEVENLABS_STABILITY", "0.61")
    monkeypatch.setenv("LIVEKIT_ELEVENLABS_SIMILARITY_BOOST", "0.95")
    monkeypatch.setenv("LIVEKIT_ELEVENLABS_SPEED", "1.05")
    monkeypatch.setenv("LIVEKIT_ELEVENLABS_USE_SPEAKER_BOOST", "1")

    settings = build_elevenlabs_voice_settings()

    assert settings.stability == 0.61
    assert settings.similarity_boost == 0.95
    assert settings.speed == 1.05
    assert settings.use_speaker_boost is True
