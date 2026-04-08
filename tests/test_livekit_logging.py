import asyncio
import logging
from types import SimpleNamespace

from shuo.livekit_logging import LiveKitSessionLogger


class _FakeHistory:
    def __init__(self, count: int):
        self._messages = [object()] * count

    def messages(self):
        return self._messages


class _FakeSession:
    def __init__(self, history_count: int):
        self.history = _FakeHistory(history_count)
        self.output = SimpleNamespace(audio=None)
        self.callbacks = {}

    def on(self, event, callback):
        self.callbacks[event] = callback


def _messages(caplog) -> str:
    return "\n".join(record.getMessage() for record in caplog.records)


def test_livekit_session_logger_logs_transcript_llm_and_tts(caplog):
    caplog.set_level(logging.INFO)

    session = _FakeSession(history_count=2)
    logger = LiveKitSessionLogger()
    logger.attach_session(session)

    session.callbacks["user_input_transcribed"](
        SimpleNamespace(transcript="Ola, tudo bem?", is_final=True)
    )
    session.callbacks["speech_created"](SimpleNamespace(source="generate_reply"))

    asyncio.run(logger.text_output.capture_text("Oi"))
    asyncio.run(logger.text_output.capture_text(", tudo bem?"))
    logger.text_output.flush()
    logger._on_playback_started(SimpleNamespace())
    logger._on_playback_finished(SimpleNamespace(interrupted=False))

    logs = _messages(caplog)
    assert "LISTENING" in logs
    assert "RESPONDING" in logs
    assert "Start" in logs and "Agent" in logs
    assert 'User transcript ready for LLM (14 chars): "Ola, tudo bem?"' in logs
    assert 'Starting completion with 2 history messages; user="Ola, tudo bem?"' in logs
    assert "Turn started" in logs
    assert "LLM first token" in logs
    assert 'Completed response (2 chunks, 13 chars): "Oi, tudo bem?"' in logs
    assert 'Text chunk #1 (13 chars) [flush]: "Oi, tudo bem?"' in logs
    assert "Starting TTS stream (13 chars)" in logs
    assert "TTS first audio" in logs
    assert "TTS stream complete" in logs
    assert "Agent turn finished" in logs


def test_livekit_session_logger_logs_interruptions_like_reset(caplog):
    caplog.set_level(logging.INFO)

    session = _FakeSession(history_count=0)
    logger = LiveKitSessionLogger()
    logger.attach_session(session)

    session.callbacks["user_input_transcribed"](SimpleNamespace(transcript="Teste", is_final=True))
    session.callbacks["speech_created"](SimpleNamespace(source="generate_reply"))
    session.callbacks["user_state_changed"](SimpleNamespace(new_state="speaking"))

    logs = _messages(caplog)
    assert "RESPONDING" in logs
    assert "LISTENING" in logs
    assert "Reset" in logs and "Agent" in logs
    assert "Turn cancelled at +" in logs
    assert "LiveKit interruption" in logs


def test_livekit_session_logger_falls_back_to_assistant_item_text(caplog):
    caplog.set_level(logging.INFO)

    session = _FakeSession(history_count=4)
    logger = LiveKitSessionLogger()
    logger.attach_session(session)

    session.callbacks["user_input_transcribed"](
        SimpleNamespace(transcript="Voce me ouviu?", is_final=True)
    )
    session.callbacks["speech_created"](SimpleNamespace(source="generate_reply"))
    session.callbacks["conversation_item_added"](
        SimpleNamespace(
            item=SimpleNamespace(
                role="assistant",
                text_content="Sim, estou te ouvindo.",
                interrupted=False,
                metrics={"llm_node_ttft": 0.111},
            )
        )
    )

    logs = _messages(caplog)
    assert "LLM first token  +111ms" in logs
    assert 'Completed response (1 chunks, 22 chars): "Sim, estou te ouvindo."' in logs
    assert 'Text chunk #1 (22 chars) [flush]: "Sim, estou te ouvindo."' in logs
