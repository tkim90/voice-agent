#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shuo.livekit_ptbr import dispatch_outbound_call


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dispatch the LiveKit pt-BR telephony agent to place an outbound call."
    )
    parser.add_argument("phone_number", help="Destination phone number in E.164 format, for example +14045634375")
    parser.add_argument("--room", dest="room_name", help="Optional LiveKit room name")
    parser.add_argument("--greeting", help="Optional initial greeting instruction for the agent")
    parser.add_argument("--participant-name", help="Optional SIP participant display name")
    parser.add_argument("--agent-name", help="Override the dispatched agent name")
    return parser.parse_args()


async def _main() -> None:
    args = parse_args()
    room_name, agent_name = await dispatch_outbound_call(
        args.phone_number,
        room_name=args.room_name,
        greeting=args.greeting,
        participant_name=args.participant_name,
        agent_name=args.agent_name,
    )
    print(f"Dispatched {agent_name} to room {room_name} for call {args.phone_number}")


if __name__ == "__main__":
    asyncio.run(_main())
