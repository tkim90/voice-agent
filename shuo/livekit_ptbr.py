"""
LiveKit-based outbound telephony experiment for Brazilian Portuguese.

This module keeps the existing Twilio-first codepath untouched and adds a
separate LiveKit worker for inbound and outbound telephony experiments.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from collections.abc import MutableMapping
from typing import Any

from dotenv import load_dotenv
from livekit import api, rtc
from livekit.agents import Agent, AgentSession, JobContext, JobProcess, WorkerOptions, cli, room_io

from .livekit_logging import LiveKitSessionLogger
from .log import compact_livekit_cli_logs, setup_logging

load_dotenv()
compact_livekit_cli_logs()

logger = logging.getLogger("shuo.livekit")

AGENT_NAME = "shuo-ptbr-telephony"
DEFAULT_GREETING = (
    "Cumprimente a pessoa em portugues brasileiro, diga que a ligacao chegou bem, "
    "e pergunte como voce pode ajudar."
)
DEFAULT_INSTRUCTIONS = """
Voce e um agente de voz para telefonia que conversa em portugues do Brasil.
Fale sempre em portugues brasileiro, com frases curtas, naturais e faceis de ouvir por telefone.
Nao use markdown, listas, emojis nem respostas longas.
Se a pessoa interromper, pare e escute.
Faca uma pergunta por vez e confirme dados importantes em voz alta.
"""


def bootstrap_env(env: MutableMapping[str, str] | None = None) -> MutableMapping[str, str]:
    """Normalize local env names to the names expected by LiveKit plugins."""
    target = os.environ if env is None else env

    if target.get("LIVEKIT_WS_URL") and not target.get("LIVEKIT_URL"):
        target["LIVEKIT_URL"] = target["LIVEKIT_WS_URL"]

    if target.get("ELEVENLABS_API_KEY") and not target.get("ELEVEN_API_KEY"):
        target["ELEVEN_API_KEY"] = target["ELEVENLABS_API_KEY"]

    return target


def first_env(*names: str, env: MutableMapping[str, str] | None = None) -> str | None:
    values = bootstrap_env(env)
    for name in names:
        value = values.get(name)
        if value:
            return value
    return None


def require_env(*names: str, env: MutableMapping[str, str] | None = None) -> str:
    value = first_env(*names, env=env)
    if value:
        return value
    raise RuntimeError(f"Missing required environment variable. Expected one of: {', '.join(names)}")


def outbound_trunk_id(env: MutableMapping[str, str] | None = None) -> str:
    return require_env("SIP_OUTBOUND_TRUNK_ID", "LIVEKIT_SIP_OUTBOUND_TRUNK_ID", env=env)


def livekit_env(name: str, default: str) -> str:
    value = os.getenv(name)
    if value:
        return value
    return default


def build_elevenlabs_voice_settings() -> Any:
    from livekit.plugins import elevenlabs

    return elevenlabs.VoiceSettings(
        stability=float(livekit_env("LIVEKIT_ELEVENLABS_STABILITY", "0.55")),
        similarity_boost=float(livekit_env("LIVEKIT_ELEVENLABS_SIMILARITY_BOOST", "0.90")),
        speed=float(livekit_env("LIVEKIT_ELEVENLABS_SPEED", "0.88")),
        use_speaker_boost=livekit_env("LIVEKIT_ELEVENLABS_USE_SPEAKER_BOOST", "0") != "0",
    )


def build_dispatch_metadata(
    phone_number: str,
    *,
    greeting: str | None = None,
    participant_name: str | None = None,
    sip_number: str | None = None,
) -> str:
    payload: dict[str, Any] = {"phone_number": phone_number}
    if greeting:
        payload["greeting"] = greeting
    if participant_name:
        payload["participant_name"] = participant_name
    if sip_number:
        payload["sip_number"] = sip_number
    return json.dumps(payload, ensure_ascii=False)


class BrazilianPortugueseAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=os.getenv("LIVEKIT_AGENT_INSTRUCTIONS", DEFAULT_INSTRUCTIONS).strip())


def prewarm(proc: JobProcess) -> None:
    """Load VAD once per worker process to reduce first-call latency."""
    bootstrap_env()
    from livekit.plugins import silero

    proc.userdata["vad"] = silero.VAD.load(
        min_silence_duration=float(os.getenv("LIVEKIT_VAD_MIN_SILENCE", "0.35")),
        prefix_padding_duration=float(os.getenv("LIVEKIT_VAD_PREFIX_PADDING", "0.25")),
    )


def build_turn_handling() -> dict[str, Any]:
    endpointing = {
        "min_delay": float(os.getenv("LIVEKIT_MIN_ENDPOINTING_DELAY", "0.20")),
        "max_delay": float(os.getenv("LIVEKIT_MAX_ENDPOINTING_DELAY", "1.20")),
    }
    interruption: dict[str, Any] = {
        "mode": "adaptive",
        "min_duration": float(os.getenv("LIVEKIT_MIN_INTERRUPTION_DURATION", "0.15")),
        "resume_false_interruption": os.getenv("LIVEKIT_RESUME_FALSE_INTERRUPTION", "1") != "0",
        "false_interruption_timeout": float(os.getenv("LIVEKIT_FALSE_INTERRUPTION_TIMEOUT", "2.0")),
    }
    turn_handling: dict[str, Any] = {
        "endpointing": endpointing,
        "interruption": interruption,
    }

    try:
        from livekit.plugins.turn_detector.multilingual import MultilingualModel

        turn_handling["turn_detection"] = MultilingualModel()
    except RuntimeError as error:
        message = str(error)
        if "languages.json" not in message:
            raise

        logger.warning(
            "Multilingual turn detector files are missing; falling back to VAD-only turn handling. "
            "Run `./.venv/bin/python livekit_ptbr_agent.py download-files` to restore multilingual "
            "turn detection and adaptive interruption handling."
        )
        interruption["mode"] = "vad"

    return turn_handling


def build_session(vad: Any) -> AgentSession:
    """Build a low-latency pt-BR pipeline."""
    bootstrap_env()
    from livekit.plugins import deepgram, elevenlabs, groq

    # Keep the pipeline aggressively streaming:
    # - Deepgram no_delay reduces STT buffering.
    # - preemptive_generation lets the LLM start before turn finalization.
    # - ElevenLabs auto_mode + streaming_latency prioritizes early audio chunks.
    return AgentSession(
        vad=vad,
        stt=deepgram.STT(
            model=livekit_env("LIVEKIT_DEEPGRAM_MODEL", "nova-3"),
            language=livekit_env("LIVEKIT_DEEPGRAM_LANGUAGE", "pt-BR"),
            interim_results=livekit_env("LIVEKIT_DEEPGRAM_INTERIM_RESULTS", "1") != "0",
            smart_format=True,
            no_delay=True,
            endpointing_ms=int(livekit_env("LIVEKIT_DEEPGRAM_ENDPOINTING_MS", "25")),
            vad_events=livekit_env("LIVEKIT_DEEPGRAM_VAD_EVENTS", "1") != "0",
        ),
        llm=groq.LLM(
            model=livekit_env("LIVEKIT_LLM_MODEL", "llama-3.3-70b-versatile"),
            temperature=float(livekit_env("LIVEKIT_LLM_TEMPERATURE", "0.2")),
            max_completion_tokens=int(livekit_env("LIVEKIT_LLM_MAX_COMPLETION_TOKENS", "256")),
        ),
        tts=elevenlabs.TTS(
            api_key=require_env("ELEVENLABS_API_KEY", "ELEVEN_API_KEY"),
            voice_id=livekit_env("LIVEKIT_ELEVENLABS_VOICE_ID", "G7ILShrCNLfmS0A37SXS"),
            voice_settings=build_elevenlabs_voice_settings(),
            model=livekit_env("LIVEKIT_ELEVENLABS_MODEL_ID", "eleven_flash_v2_5"),
            language=livekit_env("LIVEKIT_ELEVENLABS_LANGUAGE", "pt"),
            auto_mode=True,
            streaming_latency=int(livekit_env("LIVEKIT_ELEVENLABS_STREAMING_LATENCY", "4")),
        ),
        preemptive_generation=True,
        turn_handling=build_turn_handling(),
    )


def build_room_options(*, next_text_output: Any | None = None) -> room_io.RoomOptions:
    from livekit.plugins import noise_cancellation

    return room_io.RoomOptions(
        audio_input=room_io.AudioInputOptions(
            noise_cancellation=lambda params: (
                noise_cancellation.BVCTelephony()
                if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                else noise_cancellation.BVC()
            ),
        ),
        text_output=room_io.TextOutputOptions(next_in_chain=next_text_output),
    )


def parse_job_metadata(raw_metadata: str) -> dict[str, Any]:
    if not raw_metadata:
        return {}
    data = json.loads(raw_metadata)
    if not isinstance(data, dict):
        raise ValueError("Job metadata must be a JSON object")
    return data


def build_sip_participant_kwargs(room_name: str, metadata: dict[str, Any]) -> dict[str, Any]:
    phone_number = metadata.get("phone_number")
    if not phone_number:
        raise RuntimeError("Dispatch metadata must include phone_number")

    kwargs: dict[str, Any] = {
        "room_name": room_name,
        "sip_trunk_id": outbound_trunk_id(),
        "sip_call_to": phone_number,
        "participant_identity": phone_number,
        "participant_name": metadata.get("participant_name") or phone_number,
        "krisp_enabled": True,
        "wait_until_answered": True,
    }

    caller_id = metadata.get("sip_number") or first_env("LIVEKIT_PHONE_NUMBER")
    if caller_id:
        kwargs["sip_number"] = caller_id

    return kwargs


async def entrypoint(ctx: JobContext) -> None:
    bootstrap_env()
    ctx.log_context_fields = {"room": ctx.room.name}

    await ctx.connect()

    metadata = parse_job_metadata(ctx.job.metadata)
    phone_number = metadata.get("phone_number")

    session = build_session(ctx.proc.userdata["vad"])
    session_logger = LiveKitSessionLogger(room_name=ctx.room.name)
    session_logger.attach_session(session)
    session_started = asyncio.create_task(
        session.start(
            agent=BrazilianPortugueseAssistant(),
            room=ctx.room,
            room_options=build_room_options(next_text_output=session_logger.text_output),
        )
    )

    try:
        if phone_number:
            await ctx.api.sip.create_sip_participant(
                api.CreateSIPParticipantRequest(**build_sip_participant_kwargs(ctx.room.name, metadata))
            )

        await session_started
        session_logger.attach_output_listeners()

        if phone_number:
            await ctx.wait_for_participant(identity=phone_number)
        else:
            await ctx.wait_for_participant()

        session.generate_reply(
            instructions=metadata.get("greeting") or DEFAULT_GREETING,
            input_modality="audio",
        )
    except api.TwirpError as error:
        logger.error(
            "Failed to create SIP participant: %s (sip_status_code=%s sip_status=%s)",
            error.message,
            error.metadata.get("sip_status_code"),
            error.metadata.get("sip_status"),
        )
        if not session_started.done():
            session_started.cancel()
            try:
                await session_started
            except asyncio.CancelledError:
                pass
        ctx.shutdown()


def build_worker_options() -> WorkerOptions:
    bootstrap_env()
    return WorkerOptions(
        entrypoint_fnc=entrypoint,
        prewarm_fnc=prewarm,
        agent_name=os.getenv("LIVEKIT_AGENT_NAME", AGENT_NAME),
        ws_url=first_env("LIVEKIT_URL", "LIVEKIT_WS_URL"),
        api_key=first_env("LIVEKIT_API_KEY"),
        api_secret=first_env("LIVEKIT_API_SECRET"),
    )


async def dispatch_outbound_call(
    phone_number: str,
    *,
    room_name: str | None = None,
    greeting: str | None = None,
    participant_name: str | None = None,
    agent_name: str | None = None,
) -> tuple[str, str]:
    bootstrap_env()
    outbound_trunk_id()

    livekit = api.LiveKitAPI(
        url=require_env("LIVEKIT_URL", "LIVEKIT_WS_URL"),
        api_key=require_env("LIVEKIT_API_KEY"),
        api_secret=require_env("LIVEKIT_API_SECRET"),
    )

    resolved_room = room_name or (
        f"{(agent_name or os.getenv('LIVEKIT_AGENT_NAME', AGENT_NAME))}-{uuid.uuid4().hex[:8]}"
    )
    try:
        dispatch = await livekit.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=agent_name or os.getenv("LIVEKIT_AGENT_NAME", AGENT_NAME),
                room=resolved_room,
                metadata=build_dispatch_metadata(
                    phone_number,
                    greeting=greeting,
                    participant_name=participant_name,
                    sip_number=first_env("LIVEKIT_PHONE_NUMBER"),
                ),
            )
        )
        return dispatch.room, dispatch.agent_name
    finally:
        await livekit.aclose()


def run_worker() -> None:
    bootstrap_env()
    setup_logging(install_handler=False)
    compact_livekit_cli_logs()
    cli.run_app(build_worker_options())
