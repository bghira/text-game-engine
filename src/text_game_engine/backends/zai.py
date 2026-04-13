from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from .base import ChatMessage, CompletionRequest, CompletionResult

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://chat.z.ai"
_DEFAULT_MODEL = "glm-5"


class _RateLimited(Exception):
    """Raised when the ZAI WebUI endpoint returns 429."""


class ZAIBackend:
    """ZAI backend using the Open WebUI completions endpoint.

    Streams responses via SSE (``data: {...}`` lines) and supports early
    tool-call detection so that ``{"tool_call": ...}`` payloads are returned
    as soon as the JSON object closes, without waiting for trailing thinking
    tokens.
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
        thinking = request.provider_options.get(
            "thinking_enabled", self._thinking_enabled
        )
        messages = self._prepare_messages(request.messages)

        burst_size = 5
        gap = 0.3
        pause = 2.0
        max_pause = 120.0
        while True:
            # Fire a burst of concurrent attempts to race past the
            # global rate-limit window during competition.
            async def _attempt() -> Any:
                return await asyncio.to_thread(
                    self._open_stream,
                    messages,
                    model=model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    thinking_enabled=bool(thinking),
                )

            tasks = []
            for i in range(burst_size):
                tasks.append(asyncio.ensure_future(_attempt()))
                if i < burst_size - 1:
                    await asyncio.sleep(gap)

            stream = None
            last_exc: Exception | None = None
            for coro in asyncio.as_completed(tasks):
                try:
                    result = await coro
                    stream = result
                    break
                except _RateLimited as exc:
                    last_exc = exc
                except Exception as exc:
                    # Non-rate-limit error — cancel remaining and raise.
                    for t in tasks:
                        t.cancel()
                    raise

            if stream is not None:
                # Cancel any still-running burst tasks.
                for t in tasks:
                    if not t.done():
                        t.cancel()
                break

            # All burst attempts hit 429 — back off then retry.
            logger.warning(
                "ZAI 429 rate-limited — burst of %d failed, retrying in %.0fs",
                burst_size, pause,
            )
            await asyncio.sleep(pause)
            pause = min(pause * 2, max_pause)

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
        """Map ChatMessages to WebUI-compatible dicts.

        The WebUI endpoint accepts standard ``system`` / ``user`` /
        ``assistant`` roles.
        """
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
        thinking_enabled: bool,
    ) -> Any:
        try:
            import requests as _requests
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "ZAI backend requires the 'requests' package."
            ) from exc

        chat_id = str(uuid.uuid4())
        request_id = str(uuid.uuid4())
        message_id = str(uuid.uuid4())
        parent_id = str(uuid.uuid4())

        body: dict[str, Any] = {
            "stream": True,
            "model": model,
            "messages": messages,
            "params": {},
            "extra": {},
            "features": {
                "image_generation": False,
                "web_search": False,
                "auto_web_search": False,
                "preview_mode": True,
                "flags": [],
                "vlm_tools_enable": False,
                "vlm_web_search_enable": False,
                "vlm_website_mode": False,
                "enable_thinking": bool(thinking_enabled),
            },
            "chat_id": chat_id,
            "id": request_id,
            "current_user_message_id": message_id,
            "current_user_message_parent_id": parent_id,
            "background_tasks": {
                "title_generation": False,
                "tags_generation": False,
            },
        }

        url = f"{self._base_url}/api/v2/chat/completions"
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "*/*",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        resp = _requests.post(
            url,
            headers=headers,
            json=body,
            stream=True,
            timeout=300,
        )
        if resp.status_code == 429:
            resp.close()
            raise _RateLimited("ZAI 429")
        resp.raise_for_status()
        return resp

    @staticmethod
    def _consume_stream(resp: Any) -> str | None:
        """Consume a ZAI WebUI SSE stream.

        Parses ``data: {...}`` lines.  Only ``phase: "answer"`` deltas are
        collected.  Tracks JSON brace depth so that tool-call responses
        (``{"tool_call": ...}``) are returned as soon as the top-level
        object closes, without waiting for trailing thinking tokens.
        """
        chunks: list[str] = []
        brace_depth = 0
        in_json = False
        found_tool_call = False
        in_string = False
        escape_next = False

        try:
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
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

                # WebUI format: {"type": "chat:completion", "data": {"delta_content": "...", "phase": "..."}}
                if isinstance(event, dict) and "data" in event:
                    inner = event["data"]
                    if isinstance(inner, dict):
                        phase = inner.get("phase", "")
                        delta = inner.get("delta_content")
                        if phase != "answer" or not delta:
                            continue
                        text = str(delta)
                    else:
                        continue
                # Fallback: OpenAI-style {"choices": [{"delta": {"content": "..."}}]}
                elif isinstance(event, dict) and "choices" in event:
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

        return "".join(chunks) or None
