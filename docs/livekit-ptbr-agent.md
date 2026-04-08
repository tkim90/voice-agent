# LiveKit pt-BR Telephony Experiment

This repository now includes a separate LiveKit-based telephony agent for Brazilian Portuguese.

Files:

- `livekit_ptbr_agent.py` starts the LiveKit worker.
- `scripts/livekit_dispatch_call.py` optionally dispatches the worker to call a PSTN number.
- `shuo/livekit_ptbr.py` contains the shared worker, session, and dispatch logic.

## Required environment variables

Add these to `.env`:

- `LIVEKIT_URL` or `LIVEKIT_WS_URL`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`
- `DEEPGRAM_API_KEY`
- `GROQ_API_KEY`
- `ELEVENLABS_API_KEY`
- `ELEVENLABS_VOICE_ID=G7ILShrCNLfmS0A37SXS`

For inbound calls to a LiveKit phone number, the above is enough on the application side.
You must still configure LiveKit telephony so that your phone number routes inbound calls to this agent through a dispatch rule.

Only for outbound PSTN calls, also add:

- `SIP_OUTBOUND_TRUNK_ID` or `LIVEKIT_SIP_OUTBOUND_TRUNK_ID`
- `LIVEKIT_PHONE_NUMBER` for outbound caller ID if your trunk supports it

Recommended model values:

- `LIVEKIT_DEEPGRAM_MODEL=nova-3`
- `LIVEKIT_DEEPGRAM_LANGUAGE=pt-BR`
- `LIVEKIT_LLM_MODEL=llama-3.3-70b-versatile`
- `LIVEKIT_ELEVENLABS_MODEL_ID=eleven_flash_v2_5`
- `LIVEKIT_ELEVENLABS_VOICE_ID=G7ILShrCNLfmS0A37SXS`
- `LIVEKIT_ELEVENLABS_SPEED=0.88`
- `LIVEKIT_ELEVENLABS_STABILITY=0.55`
- `LIVEKIT_ELEVENLABS_SIMILARITY_BOOST=0.90`

The LiveKit worker intentionally uses `LIVEKIT_*` model settings so it does not inherit the older Twilio-path settings such as `DEEPGRAM_MODEL=flux-general-en`.

## Why this is low latency

The worker is configured to start producing speech as early as possible:

- Deepgram runs with `no_delay=True` and low endpointing.
- `preemptive_generation=True` lets the LLM start generating before turn finalization.
- ElevenLabs runs with `auto_mode=True` and `streaming_latency=4` to prioritize early audio chunks.
- Adaptive interruption handling is enabled with the multilingual turn detector.

## Run locally for inbound calls

1. Download model files:

```bash
.venv/bin/python livekit_ptbr_agent.py download-files
```

2. Start the LiveKit worker:

```bash
.venv/bin/python livekit_ptbr_agent.py dev
```

3. Configure your LiveKit phone number to dispatch inbound calls to the agent name `shuo-ptbr-telephony` or your `LIVEKIT_AGENT_NAME` override.

4. Call the LiveKit number from your phone.

## Optional outbound testing

1. Download model files:

```bash
.venv/bin/python livekit_ptbr_agent.py download-files
```

2. Start the LiveKit worker:

```bash
.venv/bin/python livekit_ptbr_agent.py dev
```

3. In another terminal, dispatch an outbound call:

```bash
.venv/bin/python scripts/livekit_dispatch_call.py +14045634375
```

## Important note about outbound telephony

LiveKit outbound PSTN calls still require an outbound SIP trunk. A LiveKit phone number alone is not enough to place outbound calls. Use `SIP_OUTBOUND_TRUNK_ID` or `LIVEKIT_SIP_OUTBOUND_TRUNK_ID` so the worker can create the SIP participant successfully.
