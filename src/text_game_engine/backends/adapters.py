from __future__ import annotations

from .base import ChatMessage, CompletionRequest, ModelBackend


class BackendTextCompletionPort:
    """Adapt a backend-level chat client onto the emulator completion port."""

    def __init__(self, backend: ModelBackend, *, model: str | None = None):
        self._backend = backend
        self._model = model

    async def complete(
        self,
        system_prompt: str,
        prompt: str,
        *,
        temperature: float = 0.8,
        max_tokens: int = 2048,
    ) -> str | None:
        messages = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=prompt))
        result = await self._backend.complete(
            CompletionRequest(
                messages=messages,
                model=self._model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
        return result.text
