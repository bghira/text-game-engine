from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from .base import ChatMessage, CompletionRequest, CompletionResult

logger = logging.getLogger(__name__)

_RE_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


class OllamaBackend:
    """Native Ollama chat backend using `/api/chat`."""

    _RESERVED_PROVIDER_FIELDS = {"model", "messages", "stream", "options"}

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str = "http://127.0.0.1:11434",
        request_timeout: float = 300.0,
        keep_alive: str | None = None,
        options: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        think: bool = False,
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._request_timeout = request_timeout
        self._keep_alive = keep_alive
        self._options = dict(options or {})
        self._headers = dict(headers or {})
        self._think = think

    # Hard wall-clock cap on a single request (connect + think + generate).
    _TOTAL_REQUEST_TIMEOUT: float = 240.0  # 4 minutes

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        model = request.model or self._model
        if not model:
            raise ValueError("OllamaBackend requires a model name")
        payload = self._build_payload(request, model=model)

        delay = 2.0
        max_delay = 120.0
        while True:
            try:
                data = await asyncio.wait_for(
                    asyncio.to_thread(self._post_streaming, "/api/chat", payload),
                    timeout=self._TOTAL_REQUEST_TIMEOUT,
                )
            except (TimeoutError, asyncio.TimeoutError, OSError) as exc:
                logger.warning("Ollama request failed (%s) — retrying in %.0fs", exc, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay)
                continue
            except RuntimeError as exc:
                exc_str = str(exc)
                # Retry on 429, 502, 503, 504
                if any(f"HTTP {code}" in exc_str for code in ("429", "502", "503", "504")):
                    logger.warning("Ollama %s — retrying in %.0fs", exc_str[:100], delay)
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, max_delay)
                    continue
                raise

            message = data.get("message")
            text = ""
            if isinstance(message, dict):
                text = str(message.get("content") or "").strip()
            if not text:
                text = str(data.get("response") or "").strip()
            # Strip <think>...</think> blocks from the visible output.
            if text:
                text = _RE_THINK_BLOCK.sub("", text).strip()
            if not text:
                logger.warning("Ollama returned empty response — retrying in %.0fs", delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay)
                continue

            usage = self._extract_usage(data)
            return CompletionResult(
                text=text,
                finish_reason=self._coerce_text(data.get("done_reason")),
                usage=usage,
                raw_response=data,
            )

    def _build_payload(self, request: CompletionRequest, *, model: str) -> dict[str, Any]:
        provider_options = dict(request.provider_options or {})
        payload: dict[str, Any] = {
            "model": model,
            "messages": [self._message_payload(message) for message in request.messages],
            "stream": True,
        }
        if self._think:
            payload["think"] = True
        payload_options = dict(self._options)
        payload_options.setdefault("temperature", request.temperature)
        payload_options.setdefault("num_predict", max(1, int(request.max_tokens)))
        if request.stop:
            payload_options["stop"] = list(request.stop)
        provider_specific_options = provider_options.pop("options", None)
        if isinstance(provider_specific_options, dict):
            payload_options.update(provider_specific_options)
        if payload_options:
            payload["options"] = payload_options

        format_value = provider_options.pop("format", None)
        if request.json_mode and format_value is None:
            format_value = "json"
        if format_value is not None:
            payload["format"] = format_value

        keep_alive = provider_options.pop("keep_alive", self._keep_alive)
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive

        reserved = self._RESERVED_PROVIDER_FIELDS.intersection(provider_options)
        if reserved:
            names = ", ".join(sorted(reserved))
            raise ValueError(
                f"Ollama provider_options cannot override reserved payload fields: {names}"
            )
        payload.update(provider_options)
        return payload

    @staticmethod
    def _message_payload(message: ChatMessage) -> dict[str, str]:
        return {
            "role": str(message.role),
            "content": str(message.content),
        }

    def _post_streaming(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a streaming request and accumulate the response.

        Each streamed chunk resets the socket read timeout, so the model can
        think for arbitrarily long as long as tokens keep arriving within the
        timeout window.  The final chunk (``done: true``) carries the full
        metadata (usage counts, done_reason, etc.).
        """
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/x-ndjson",
            **self._headers,
        }
        req = urllib_request.Request(
            f"{self._base_url}{path}",
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            resp = urllib_request.urlopen(req, timeout=self._request_timeout)
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama request failed with HTTP {exc.code}: {detail.strip()}"
            ) from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"Ollama request failed: {exc.reason}") from exc

        content_parts: list[str] = []
        final_chunk: dict[str, Any] = {}
        try:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(chunk, dict):
                    continue
                # Accumulate content from streamed message fragments.
                msg = chunk.get("message")
                if isinstance(msg, dict):
                    part = msg.get("content")
                    if part:
                        content_parts.append(str(part))
                # The final chunk has done=true and carries metadata.
                if chunk.get("done"):
                    final_chunk = chunk
        finally:
            resp.close()

        # Build a merged response that looks like a non-streaming response.
        assembled_text = "".join(content_parts)
        result: dict[str, Any] = dict(final_chunk)
        result["message"] = {"role": "assistant", "content": assembled_text}
        # Keep .response for legacy callers that check it.
        result["response"] = assembled_text
        return result

    @staticmethod
    def _coerce_text(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _extract_usage(payload: dict[str, Any]) -> dict[str, int] | None:
        usage: dict[str, int] = {}
        prompt_tokens = payload.get("prompt_eval_count")
        completion_tokens = payload.get("eval_count")
        if isinstance(prompt_tokens, int):
            usage["prompt_tokens"] = prompt_tokens
        if isinstance(completion_tokens, int):
            usage["completion_tokens"] = completion_tokens
        if usage:
            usage["total_tokens"] = usage.get("prompt_tokens", 0) + usage.get(
                "completion_tokens", 0
            )
        return usage or None
