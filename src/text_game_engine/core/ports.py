from __future__ import annotations

from typing import Protocol

from .types import LLMTurnOutput, TurnContext


class LLMPort(Protocol):
    async def complete_turn(self, context: TurnContext, *, progress=None) -> LLMTurnOutput:
        ...


class ActorResolverPort(Protocol):
    def resolve_discord_mention(self, mention: str) -> str | None:
        ...
