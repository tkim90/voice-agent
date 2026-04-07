"""
Deepgram Flux service -- always-on STT + turn detection.

A single persistent WebSocket to Deepgram using the v2 listen API.
Receives all Twilio audio continuously and emits turn events.

Replaces both local VAD (Silero) and separate STT (Deepgram v1).
"""

import os
import asyncio
from typing import Optional, Callable, Awaitable

from deepgram import AsyncDeepgramClient, DeepgramClientEnvironment

from ..log import ServiceLogger

log = ServiceLogger("Flux")


def _deepgram_environment(region: Optional[str] = None) -> DeepgramClientEnvironment:
    """Return a Deepgram endpoint configuration for the requested region."""
    resolved_region = (region or os.getenv("DEEPGRAM_REGION", "us")).strip().lower()

    if resolved_region in {"us", "default", "global"}:
        return DeepgramClientEnvironment(
            base="wss://api.deepgram.com",
            production="wss://api.deepgram.com",
            agent="wss://agent.deepgram.com",
        )

    if resolved_region == "eu":
        return DeepgramClientEnvironment(
            base="wss://api.eu.deepgram.com",
            production="wss://api.eu.deepgram.com",
            agent="wss://agent.eu.deepgram.com",
        )

    raise ValueError("DEEPGRAM_REGION must be one of: us, eu")


def _deepgram_flux_options(
    model: Optional[str] = None,
    language: Optional[str] = None,
) -> dict:
    """Return a Flux-compatible listen configuration."""
    resolved_model = (model or os.getenv("DEEPGRAM_MODEL", "flux-general-en")).strip()
    resolved_language = (
        language or os.getenv("DEEPGRAM_LANGUAGE", "en")
    ).strip().lower().replace("_", "-")

    if not resolved_language:
        resolved_language = "en"

    if resolved_model != "flux-general-en":
        raise ValueError("DEEPGRAM_MODEL must be flux-general-en for FluxService")

    if resolved_language not in {"en", "en-us", "en-gb", "english"}:
        raise ValueError("DEEPGRAM_LANGUAGE must be English for FluxService")

    return {
        "model": "flux-general-en",
        "encoding": "mulaw",
        "sample_rate": 8000,
    }


class FluxService:
    """
    Deepgram Flux streaming service.

    Audio format: mulaw 8kHz (direct from Twilio, no conversion needed).
    Turn events: StartOfTurn (barge-in), EndOfTurn (with transcript).
    """

    def __init__(
        self,
        on_end_of_turn: Callable[[str], Awaitable[None]],
        on_start_of_turn: Callable[[], Awaitable[None]],
        on_interim: Optional[Callable[[str], Awaitable[None]]] = None,
    ):
        self._on_end_of_turn = on_end_of_turn
        self._on_start_of_turn = on_start_of_turn
        self._on_interim = on_interim

        self._api_key = os.getenv("DEEPGRAM_API_KEY", "")
        self._log_interim = os.getenv("SHUO_LOG_INTERIM", "").lower() in {"1", "true", "yes", "on"}
        self._client: Optional[AsyncDeepgramClient] = None
        self._connection = None
        self._cm = None
        self._listener_task: Optional[asyncio.Task] = None
        self._running = False
        self._last_interim = ""

    @property
    def is_active(self) -> bool:
        return self._running and self._connection is not None

    async def start(self) -> None:
        """Connect to Deepgram Flux (always-on for the duration of the call)."""
        if self._running:
            return

        try:
            deepgram_env = _deepgram_environment()
            self._client = AsyncDeepgramClient(
                api_key=self._api_key,
                environment=deepgram_env,
            )

            self._cm = self._client.listen.v2.connect(
                **_deepgram_flux_options()
            )
            self._connection = await self._cm.__aenter__()

            self._connection.on("message", self._on_message)
            self._connection.on("Error", self._on_error)

            self._listener_task = asyncio.create_task(
                self._connection.start_listening()
            )

            self._running = True
            log.connected()

        except Exception as e:
            log.error("Connection failed", e)
            await self._cleanup()
            raise

    async def send(self, audio_bytes: bytes) -> None:
        """Send audio chunk to Deepgram Flux."""
        if not self._connection or not self._running:
            return

        try:
            await self._connection.send_media(audio_bytes)
        except Exception as e:
            log.error("Send failed", e)

    async def stop(self) -> None:
        """Disconnect from Deepgram Flux."""
        self._running = False
        await self._cleanup()
        log.disconnected()

    async def _cleanup(self) -> None:
        """Clean up resources."""
        self._running = False

        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass
            self._listener_task = None

        if self._cm:
            try:
                await self._cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._cm = None

        self._connection = None
        self._client = None

    async def _on_message(self, message, *args, **kwargs) -> None:
        """Handle Flux messages -- parse TurnInfo events."""
        try:
            msg_type = getattr(message, "type", None)

            if msg_type == "TurnInfo":
                event = getattr(message, "event", None)

                if event == "EndOfTurn":
                    transcript = getattr(message, "transcript", "") or ""
                    await self._on_end_of_turn(transcript.strip())

                elif event == "StartOfTurn":
                    await self._on_start_of_turn()

            elif msg_type == "Results":
                channel = getattr(message, "channel", None)
                if channel:
                    alternatives = getattr(channel, "alternatives", None)
                    if alternatives:
                        alt = (
                            alternatives[0]
                            if isinstance(alternatives, list)
                            else alternatives
                        )
                        transcript = getattr(alt, "transcript", "")
                        if transcript:
                            cleaned = transcript.strip()
                            if self._log_interim and cleaned and cleaned != self._last_interim:
                                self._last_interim = cleaned
                                log.info(f'Interim transcript: "{cleaned}"')
                            if self._on_interim:
                                await self._on_interim(cleaned)

        except Exception as e:
            log.error("Message handling failed", e)

    async def _on_error(self, error, *args, **kwargs) -> None:
        """Handle Deepgram errors."""
        log.error("Deepgram: " + str(error))
