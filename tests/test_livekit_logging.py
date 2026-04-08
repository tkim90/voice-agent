import asyncio
import logging
from types import SimpleNamespace

import shuo.livekit_logging as livekit_logging
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


def test_livekit_session_logger_writes_per_session_log(tmp_path):
    original_log_dir = livekit_logging.LOG_DIR
    livekit_logging.LOG_DIR = tmp_path

    try:
        session = _FakeSession(history_count=2)
        logger = LiveKitSessionLogger(room_name="room/alpha")
        logger.attach_session(session)

        session.callbacks["user_input_transcribed"](
            SimpleNamespace(transcript="Ola, tudo bem?", is_final=True)
        )
        session.callbacks["speech_created"](SimpleNamespace(source="generate_reply"))
        asyncio.run(logger.text_output.capture_text("Oi, tudo bem?"))
        logger.text_output.flush()
        logger._on_playback_started(SimpleNamespace())
        logger._on_playback_finished(SimpleNamespace(interrupted=False))

        files = list(tmp_path.glob("livekit-room-alpha-*.log"))
        assert len(files) == 1

        content = files[0].read_text()
        assert "Phase: LISTENING -> RESPONDING" in content
        assert 'Agent: User transcript ready for LLM (14 chars): "Ola, tudo bem?"' in content
        assert 'LLM: Completed response (1 chunks, 13 chars): "Oi, tudo bem?"' in content
        assert "Agent: TTS stream complete" in content
    finally:
        livekit_logging.LOG_DIR = original_log_dir
