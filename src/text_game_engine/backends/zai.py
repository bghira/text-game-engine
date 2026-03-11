from __future__ import annotations

import asyncio
import logging
from typing import Any

from .base import ChatMessage, CompletionRequest, CompletionResult

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.z.ai/api/coding/paas/v4"
_DEFAULT_MODEL = "glm-5"


class ZAIBackend:
    """ZAI backend with streaming and early tool-call detection.

    Uses the OpenAI SDK pointed at the ZAI endpoint.  Streams responses so
    that tool-call JSON objects (``{"tool_call": ...}``) can be returned as
    soon as the JSON closes, without waiting for trailing thinking tokens.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = _DEFAULT_BASE_URL,
        model: str = _DEFAULT_MODEL,
        thinking_enabled: bool = True,
    ):
        self._api_key = api_key
        self._base_url = base_url
        self._model = model
        self._thinking_enabled = thinking_enabled

    # ------------------------------------------------------------------
    # ModelBackend protocol
    # ------------------------------------------------------------------

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        model = request.model or self._model
        thinking = request.provider_options.get(
            "thinking_enabled", self._thinking_enabled
        )
        messages = self._prepare_messages(request.messages)
        stream = await asyncio.to_thread(
            self._open_stream,
            messages,
            model=model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            thinking_enabled=bool(thinking),
        )
        text = await asyncio.to_thread(self._consume_stream, stream)
        return CompletionResult(
            text=text or "",
            finish_reason="stop",
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_messages(
        messages: list[ChatMessage],
    ) -> list[dict[str, str]]:
        """Map ChatMessages to ZAI-compatible dicts.

        ZAI uses ``assistant`` for the system-like preamble (instead of
        ``system``).  Convert accordingly.
        """
        out: list[dict[str, str]] = []
        for msg in messages:
            role = msg.role
            if role == "system":
                role = "assistant"
            out.append({"role": role, "content": msg.content})
        return out

    def _open_stream(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        thinking_enabled: bool,
    ) -> Any:
        from openai import OpenAI

        client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
        )
        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if thinking_enabled:
            request_kwargs["extra_body"] = {
                "thinking": {"type": "enabled"},
            }
        return client.chat.completions.create(**request_kwargs)

    @staticmethod
    def _consume_stream(stream: Any) -> str | None:
        """Consume a ZAI streaming response.

        Tracks JSON brace depth so that tool-call responses
        (``{"tool_call": ...}``) are returned as soon as the top-level
        object closes — without waiting for trailing thinking tokens.
        Non-tool-call responses are consumed in full.
        """
        chunks: list[str] = []
        brace_depth = 0
        in_json = False
        found_tool_call = False

        try:
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue
                text = delta.content
                if not text:
                    continue
                chunks.append(text)

                for ch in text:
                    if ch == "{":
                        if not in_json:
                            in_json = True
                        brace_depth += 1
                    elif ch == "}":
                        brace_depth -= 1
                        if in_json and brace_depth == 0:
                            so_far = "".join(chunks)
                            if (
                                '"tool_call"' in so_far
                                and '"narration"' not in so_far
                            ):
                                found_tool_call = True

                if found_tool_call:
                    stream.close()
                    break
        except Exception as exc:
            logger.warning("Error during ZAI stream consumption: %s", exc)

        return "".join(chunks) or None
