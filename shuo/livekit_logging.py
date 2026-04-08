"""
Structured logging helpers for the LiveKit pt-BR agent.

This reuses the same shuo logger/action/service vocabulary as the Twilio path
so transcript, LLM, and spoken output logs stay consistent across both agents.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from livekit.agents import AgentSession
from livekit.agents.voice.io import TextOutput

from .log import Logger, ServiceLogger, preview_text
from .types import Phase, ResetAgentTurnAction, StartAgentTurnAction


@dataclass
class _TurnLogState:
    started_at: float
    transcript: str
    history_messages: int
    started_logged: bool = False
    interrupted: bool = False
    first_token_at: float | None = None
    first_token_logged: bool = False
    llm_output_logged: bool = False
    tts_chunks_logged: int = 0
    first_audio_logged: bool = False


class LiveKitLogTextOutput(TextOutput):
    """TextOutput wrapper that logs the final text flushed toward spoken output."""

    def __init__(
        self,
        owner: "LiveKitSessionLogger",
        *,
        next_in_chain: TextOutput | None = None,
    ) -> None:
        super().__init__(label="shuo_livekit_log", next_in_chain=next_in_chain)
        self._owner = owner
        self._buffer = ""
        self._delta_count = 0

    async def capture_text(self, text: str) -> None:
        if text:
            self._buffer += text
            self._delta_count += 1
            self._owner.on_llm_text_delta()

        if self.next_in_chain:
            await self.next_in_chain.capture_text(text)

    def flush(self) -> None:
        if self._buffer.strip():
            self._owner.on_llm_text_flush(self._buffer, self._delta_count)

        self._buffer = ""
        self._delta_count = 0

        if self.next_in_chain:
            self.next_in_chain.flush()


class LiveKitSessionLogger:
    """Bridges LiveKit session events into the repo's existing structured logs."""

    def __init__(self) -> None:
        self._conversation_log = Logger()
        self._agent_log = ServiceLogger("Agent")
        self._llm_log = ServiceLogger("LLM")
        self._tts_log = ServiceLogger("TTS")
        self._session: AgentSession[Any] | None = None
        self._phase = Phase.LISTENING
        self._active_turn: _TurnLogState | None = None
        self._text_output = LiveKitLogTextOutput(self)
        self._audio_attached = False

    @property
    def text_output(self) -> TextOutput:
        return self._text_output

    def attach_session(self, session: AgentSession[Any]) -> None:
        """Register session event listeners before the session starts."""
        self._session = session
        session.on("user_input_transcribed", self._on_user_input_transcribed)
        session.on("user_state_changed", self._on_user_state_changed)
        session.on("agent_state_changed", self._on_agent_state_changed)
        session.on("speech_created", self._on_speech_created)
        session.on("conversation_item_added", self._on_conversation_item_added)
        session.on("error", self._on_error)

    def attach_output_listeners(self) -> None:
        """Attach playback listeners once RoomIO has populated the session outputs."""
        if self._audio_attached or not self._session or not self._session.output.audio:
            return

        self._session.output.audio.on("playback_started", self._on_playback_started)
        self._session.output.audio.on("playback_finished", self._on_playback_finished)
        self._audio_attached = True

    def on_llm_text_delta(self) -> None:
        turn = self._active_turn
        if turn is None or turn.first_token_logged:
            return

        turn.first_token_logged = True
        turn.first_token_at = time.monotonic()
        self._agent_log.info(f"⏱  LLM first token  +{self._elapsed_ms(turn)}ms")

    def on_llm_text_flush(self, text: str, chunk_count: int) -> None:
        turn = self._active_turn
        if turn is None:
            return

        text = text.strip()
        if not text:
            return

        preview = preview_text(text, 120)
        summary = "Completed response" if not turn.interrupted else "Interrupted response"
        self._llm_log.info(
            f'{summary} ({chunk_count} chunks, {len(text)} chars): "{preview}"'
        )
        self._agent_log.info(
            f"LLM stream complete  +{self._elapsed_ms(turn)}ms  -> flushing TTS"
        )

        turn.tts_chunks_logged += 1
        self._tts_log.info(
            f'Text chunk #{turn.tts_chunks_logged} ({len(text)} chars) [flush]: '
            f'"{preview_text(text)}"'
        )
        self._tts_log.info(f"Starting TTS stream ({len(text)} chars)")
        turn.llm_output_logged = True

    def _start_turn(self, transcript: str) -> None:
        self._active_turn = _TurnLogState(
            started_at=time.monotonic(),
            transcript=transcript,
            history_messages=self._history_message_count(),
        )
        self._set_phase(Phase.RESPONDING)
        self._conversation_log.action(StartAgentTurnAction(transcript=transcript))
        self._agent_log.info(
            f'User transcript ready for LLM ({len(transcript)} chars): "{preview_text(transcript)}"'
        )
        self._llm_log.info(
            f"Starting completion with {self._active_turn.history_messages} history messages; "
            f'user="{preview_text(transcript, 80)}"'
        )

    def _cancel_turn(self, reason: str) -> None:
        turn = self._active_turn
        if turn is None or turn.interrupted:
            return

        turn.interrupted = True
        self._set_phase(Phase.LISTENING)
        self._conversation_log.action(ResetAgentTurnAction())
        self._agent_log.info(f"Turn cancelled at +{self._elapsed_ms(turn)}ms ({reason})")

    def _complete_turn(self) -> None:
        turn = self._active_turn
        if turn is None or turn.interrupted:
            self._active_turn = None
            return

        elapsed = self._elapsed_ms(turn)
        self._agent_log.info(f"TTS stream complete  +{elapsed}ms")
        self._agent_log.info(f"⏱  Agent turn finished  +{elapsed}ms total")
        self._set_phase(Phase.LISTENING)
        self._active_turn = None

    def _set_phase(self, phase: Phase) -> None:
        self._conversation_log.transition(self._phase, phase)
        self._phase = phase

    def _history_message_count(self) -> int:
        if not self._session:
            return 0
        return len(self._session.history.messages())

    @staticmethod
    def _item_text(item: Any) -> str:
        text = getattr(item, "text_content", None)
        return text or ""

    def _elapsed_ms(self, turn: _TurnLogState) -> int:
        return int((time.monotonic() - turn.started_at) * 1000)

    def _on_user_input_transcribed(self, event: Any) -> None:
        transcript = getattr(event, "transcript", "").strip()
        if not getattr(event, "is_final", False) or not transcript:
            return

        self._start_turn(transcript)

    def _on_user_state_changed(self, event: Any) -> None:
        if getattr(event, "new_state", None) == "speaking" and self._phase == Phase.RESPONDING:
            self._cancel_turn("LiveKit interruption")

    def _on_agent_state_changed(self, event: Any) -> None:
        new_state = getattr(event, "new_state", None)
        if new_state in {"idle", "listening"} and self._phase == Phase.RESPONDING:
            turn = self._active_turn
            if turn is not None and turn.interrupted:
                self._active_turn = None
                self._set_phase(Phase.LISTENING)

    def _on_speech_created(self, event: Any) -> None:
        if getattr(event, "source", None) != "generate_reply":
            return

        turn = self._active_turn
        if turn is None or turn.started_logged:
            return

        turn.started_logged = True
        self._agent_log.info("Turn started")

    def _on_conversation_item_added(self, event: Any) -> None:
        item = getattr(event, "item", None)
        if item is None or getattr(item, "role", None) != "assistant":
            return

        turn = self._active_turn
        if turn is None:
            return

        text = self._item_text(item).strip()
        if text and not turn.llm_output_logged:
            if not turn.first_token_logged:
                metrics = getattr(item, "metrics", {}) or {}
                llm_ttft = metrics.get("llm_node_ttft")
                if llm_ttft is not None:
                    turn.first_token_logged = True
                    turn.first_token_at = turn.started_at + float(llm_ttft)
                    self._agent_log.info(
                        f"⏱  LLM first token  +{int(float(llm_ttft) * 1000)}ms"
                    )

            self.on_llm_text_flush(text, 1)

        if getattr(item, "interrupted", False):
            turn.interrupted = True
            self._active_turn = None

    def _on_playback_started(self, _: Any) -> None:
        turn = self._active_turn
        if turn is None or turn.first_audio_logged:
            return

        turn.first_audio_logged = True
        elapsed = self._elapsed_ms(turn)
        if turn.first_token_at is not None:
            tts_latency = int((time.monotonic() - turn.first_token_at) * 1000)
            self._agent_log.info(
                f"⏱  TTS first audio  +{elapsed}ms  (TTS latency {tts_latency}ms)"
            )
        else:
            self._agent_log.info(f"⏱  TTS first audio  +{elapsed}ms")

    def _on_playback_finished(self, event: Any) -> None:
        if getattr(event, "interrupted", False):
            self._cancel_turn("playback interrupted")
            self._active_turn = None
            return

        self._complete_turn()

    def _on_error(self, event: Any) -> None:
        error = getattr(event, "error", None)
        if error is not None:
            self._agent_log.error("LiveKit session error", error)
