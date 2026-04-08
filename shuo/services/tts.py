"""
ElevenLabs Text-to-Speech service.

This implementation uses ElevenLabs' HTTP streaming endpoint rather than the
WebSocket API. In this app, the full assistant turn is available by the time
TTS starts, and the HTTP stream is materially more reliable for that pattern.
"""

import os
import base64
import asyncio
import json
from typing import Optional, Callable, Awaitable

import httpx

from ..log import ServiceLogger, preview_text

log = ServiceLogger("TTS")

TWILIO_FRAME_BYTES = 160  # 20ms of ulaw_8000 mono audio

class TTSService:
    """
    ElevenLabs streaming TTS service.

    Buffers text during LLM generation, then streams ulaw_8000 audio over HTTP
    on flush(). Audio is reframed into Twilio-friendly 20ms chunks.
    """

    def __init__(
        self,
        on_audio: Callable[[str], Awaitable[None]],
        on_done: Callable[[], Awaitable[None]],
    ):
        self._on_audio = on_audio
        self._on_done = on_done

        self._client: Optional[httpx.AsyncClient] = None
        self._stream_task: Optional[asyncio.Task] = None
        self._running = False

        self._api_key = os.getenv("ELEVENLABS_API_KEY", "")
        self._voice_id = os.getenv("ELEVENLABS_VOICE_ID", "GDzHdQOi6jjf8zaXhCYD")
        self._model_id = os.getenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5")

        self._pending_text = ""
        self._sent_chunks = 0
        self._sent_chars = 0
        self._received_audio_chunks = 0
        self._received_byte_chunks = 0
        self._last_text_preview = ""

    @property
    def is_active(self) -> bool:
        return self._running and self._client is not None

    def bind(
        self,
        on_audio: Callable[[str], Awaitable[None]],
        on_done: Callable[[], Awaitable[None]],
    ) -> None:
        """Rebind callbacks (used by connection pool to assign per-turn handlers)."""
        self._on_audio = on_audio
        self._on_done = on_done
        self._pending_text = ""
        self._sent_chunks = 0
        self._sent_chars = 0
        self._received_audio_chunks = 0
        self._received_byte_chunks = 0
        self._last_text_preview = ""

    async def start(self) -> None:
        """Prepare an HTTP client for this turn."""
        if self._running:
            return

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=10.0),
            headers={
                "xi-api-key": self._api_key,
                "Content-Type": "application/json",
                "Accept": "application/octet-stream",
            },
        )
        self._running = True
        self._pending_text = ""
        self._sent_chunks = 0
        self._sent_chars = 0
        self._received_audio_chunks = 0
        self._received_byte_chunks = 0
        self._last_text_preview = ""

        log.info(f"HTTP streaming ready ({self._model_id}, voice {self._voice_id[:8]}...)")
        log.connected()

    async def send(self, text: str) -> None:
        """Buffer text for synthesis."""
        if not self._running:
            return

        self._pending_text += text

    async def flush(self) -> None:
        """Start the HTTP audio stream for the buffered text."""
        if not self._running or not self._client:
            return

        text = self._pending_text.strip()
        self._pending_text = ""

        if not text:
            log.warning("Flush requested before any text was sent to TTS")
            return

        self._sent_chunks = 1
        self._sent_chars = len(text)
        self._last_text_preview = preview_text(text)

        log.info(
            f'Text chunk #1 ({self._sent_chars} chars) [flush]: '
            f'"{self._last_text_preview}"'
        )
        log.info(f"Starting HTTP TTS stream ({self._sent_chars} chars)")

        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass

        self._stream_task = asyncio.create_task(self._stream_audio(text))

    async def stop(self) -> None:
        """Close any active stream gracefully."""
        if not self._running:
            return

        if self._stream_task and not self._stream_task.done():
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass

        await self._cleanup()
        log.disconnected()

    async def cancel(self) -> None:
        """Abort the current stream immediately."""
        self._running = False

        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass

        await self._cleanup()
        log.cancelled()

    async def _cleanup(self) -> None:
        """Clean up resources."""
        self._running = False

        if self._client:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None

        self._stream_task = None

    async def _stream_audio(self, text: str) -> None:
        """Fetch streaming audio from ElevenLabs and forward framed chunks."""
        if not self._client:
            return

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self._voice_id}/stream"
        params = {"output_format": "ulaw_8000"}
        payload = {
            "text": text,
            "model_id": self._model_id,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "use_speaker_boost": False,
                "speed": 1.0,
            },
        }

        remainder = b""

        try:
            async with self._client.stream("POST", url, params=params, json=payload) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", errors="replace")
                    guidance = None
                    try:
                        parsed = json.loads(body)
                        detail = parsed.get("detail", {})
                        if (
                            response.status_code == 402
                            and detail.get("code") == "paid_plan_required"
                        ):
                            guidance = (
                                "Configured ELEVENLABS_VOICE_ID requires a paid plan. "
                                "Free-tier API use is blocked for many Voice Library voices. "
                                "Switch ELEVENLABS_VOICE_ID to a voice you own or a shared "
                                "voice with free_users_allowed=true, or upgrade ElevenLabs."
                            )
                    except json.JSONDecodeError:
                        pass

                    log.error(
                        "HTTP TTS request failed "
                        f"(status={response.status_code}, body={preview_text(body, 160)})"
                    )
                    if guidance:
                        log.warning(guidance)
                    await self._on_done()
                    return

                async for chunk in response.aiter_bytes():
                    if not self._running:
                        break

                    if not chunk:
                        continue

                    self._received_byte_chunks += 1
                    remainder += chunk

                    while len(remainder) >= TWILIO_FRAME_BYTES:
                        frame = remainder[:TWILIO_FRAME_BYTES]
                        remainder = remainder[TWILIO_FRAME_BYTES:]
                        await self._emit_audio_frame(frame)

                if self._running and remainder:
                    await self._emit_audio_frame(remainder)

                if self._running:
                    if self._received_audio_chunks == 0:
                        log.warning(
                            "HTTP TTS stream ended with no audio "
                            f"(chars={self._sent_chars}, text=\"{self._last_text_preview}\")"
                        )
                    else:
                        log.info(
                            f"TTS finalized with {self._received_audio_chunks} audio chunks "
                            f"from {self._received_byte_chunks} byte chunks"
                        )
                    await self._on_done()

        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.error("HTTP TTS stream failed", e)
            if self._running:
                await self._on_done()

    async def _emit_audio_frame(self, frame: bytes) -> None:
        """Emit one Twilio media payload frame."""
        self._received_audio_chunks += 1
        if self._received_audio_chunks == 1:
            log.info(
                f"First audio chunk received after {self._sent_chunks} text chunks "
                f"({self._sent_chars} chars)"
            )

        payload = base64.b64encode(frame).decode("ascii")
        await self._on_audio(payload)
