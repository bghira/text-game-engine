from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from .base import ChatMessage, CompletionRequest, CompletionResult

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.z.ai/api/coding/paas/v4"
_DEFAULT_MODEL = "glm-5-turbo"


class _RateLimited(Exception):
    """Raised when the ZAI endpoint returns 429."""


class ZAIBackend:
    """ZAI backend using the OpenAI-compatible coding endpoint.

    Sends standard ``/chat/completions`` requests via ``requests`` and
    supports early tool-call detection so that ``{"tool_call": ...}``
    payloads are returned as soon as the JSON object closes, without
    waiting for trailing thinking tokens.
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
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._thinking_enabled = thinking_enabled

    # ------------------------------------------------------------------
    # ModelBackend protocol
    # ------------------------------------------------------------------

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        model = request.model or self._model
        messages = self._prepare_messages(request.messages)

        delay = 2.0
        max_delay = 120.0
        while True:
            try:
                stream = await asyncio.to_thread(
                    self._open_stream,
                    messages,
                    model=model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                )
            except _RateLimited:
                logger.warning("ZAI 429 rate-limited — retrying in %.0fs", delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay)
                continue
            except Exception as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status in (405, 503):
                    logger.warning("ZAI %s — retrying in %.0fs", status, delay)
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, max_delay)
                    continue
                raise

            text = await asyncio.to_thread(self._consume_stream, stream)
            if not text:
                logger.warning("ZAI returned empty answer — retrying in %.0fs", delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay)
                continue

            return CompletionResult(
                text=text,
                finish_reason="stop",
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_messages(
        messages: list[ChatMessage],
    ) -> list[dict[str, str]]:
        """Map ChatMessages to OpenAI-compatible dicts."""
        out: list[dict[str, str]] = []
        for msg in messages:
            out.append({"role": msg.role, "content": msg.content})
        return out

    def _open_stream(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> Any:
        try:
            import requests as _requests
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "ZAI backend requires the 'requests' package."
            ) from exc

        url = f"{self._base_url}/chat/completions"
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        logger.warning(
            "ZAI request: url=%s model=%s msgs=%d",
            url, model, len(messages),
        )

        resp = _requests.post(
            url,
            headers=headers,
            json=body,
            stream=True,
            timeout=300,
        )

        logger.warning(
            "ZAI response: status=%d",
            resp.status_code,
        )

        if resp.status_code == 429:
            resp.close()
            raise _RateLimited("ZAI 429")
        if not resp.ok:
            try:
                err_body = resp.text
            except Exception:
                err_body = "(unreadable)"
            logger.warning("ZAI error response body: %s", err_body[:2000])
            resp.raise_for_status()
        return resp

    @staticmethod
    def _consume_stream(resp: Any) -> str | None:
        """Consume an OpenAI-compatible SSE stream.

        Parses ``data: {...}`` lines with ``choices[0].delta.content``.
        Tracks JSON brace depth so that tool-call responses are returned
        as soon as the top-level JSON object closes.
        """
        chunks: list[str] = []
        raw_lines: list[str] = []
        brace_depth = 0
        in_json = False
        found_tool_call = False
        in_string = False
        escape_next = False

        try:
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                raw_lines.append(raw_line)
                line = raw_line
                if line.startswith("data: "):
                    line = line[6:]
                elif line.startswith("data:"):
                    line = line[5:]
                else:
                    continue

                line = line.strip()
                if not line or line == "[DONE]":
                    continue

                try:
                    event = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue

                # OpenAI-style {"choices": [{"delta": {"content": "..."}}]}
                if isinstance(event, dict) and "choices" in event:
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    delta_obj = choices[0].get("delta") or {}
                    text = delta_obj.get("content")
                    if not text:
                        continue
                else:
                    continue

                chunks.append(text)

                for ch in text:
                    if escape_next:
                        escape_next = False
                        continue
                    if ch == "\\" and in_string:
                        escape_next = True
                        continue
                    if ch == '"':
                        in_string = not in_string
                        continue
                    if in_string:
                        continue
                    if ch == "{":
                        if not in_json:
                            in_json = True
                        brace_depth += 1
                    elif ch == "}":
                        if not in_json or brace_depth <= 0:
                            continue
                        brace_depth -= 1
                        if in_json and brace_depth == 0:
                            so_far = "".join(chunks)
                            if (
                                '"tool_call"' in so_far
                                and '"narration"' not in so_far
                            ):
                                found_tool_call = True

                if found_tool_call:
                    resp.close()
                    break
        except Exception as exc:
            logger.warning("Error during ZAI stream consumption: %s", exc)
        finally:
            try:
                resp.close()
            except Exception:
                pass

        result = "".join(chunks) or None
        logger.warning(
            "ZAI stream consumed: raw_lines=%d answer_chunks=%d result_len=%d",
            len(raw_lines), len(chunks), len(result or ""),
        )
        if not result:
            logger.warning(
                "ZAI stream returned empty answer. Raw SSE lines (first 50):\n%s",
                "\n".join(raw_lines[:50]),
            )
        return result
