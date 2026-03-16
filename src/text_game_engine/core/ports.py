from __future__ import annotations

from typing import Any, Awaitable, Callable, Protocol

from .types import LLMTurnOutput, TurnContext

# A progress callback receives a *phase* label and optional metadata dict.
# It may be sync (returning None) or async (returning an awaitable).
# Callers must treat it as best-effort: exceptions are swallowed and sync
# return values are accepted.
#
# Defined phases (open-ended — implementations may add more):
#   "thinking"  — LLM inference started; metadata is None
#   "tool_call" — a tool call is being executed; metadata {"tool": "<name>"}
#   "writing"   — final narration pass started; metadata is None
#   "refining"  — auto-fix / retry pass started; metadata is None
ProgressCallback = Callable[[str, dict[str, Any] | None], Awaitable[None] | None]


class LLMPort(Protocol):
    async def complete_turn(
        self,
        context: TurnContext,
        *,
        progress: ProgressCallback | None = None,
    ) -> LLMTurnOutput:
        ...


class ActorResolverPort(Protocol):
    def resolve_discord_mention(self, mention: str) -> str | None:
        ...
