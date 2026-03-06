from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


ChatRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: ChatRole
    content: str


@dataclass(frozen=True)
class CompletionRequest:
    messages: list[ChatMessage]
    model: str | None = None
    temperature: float = 0.8
    max_tokens: int = 2048
    stop: list[str] | None = None
    json_mode: bool = False
    provider_options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompletionResult:
    text: str
    finish_reason: str | None = None
    usage: dict[str, int] | None = None
    raw_response: dict[str, Any] | None = None


class ModelBackend(Protocol):
    async def complete(self, request: CompletionRequest) -> CompletionResult:
        ...
