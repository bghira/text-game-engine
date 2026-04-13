from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import uuid
from typing import Any

from .base import ChatMessage, CompletionRequest, CompletionResult

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://chat.z.ai"
_DEFAULT_MODEL = "glm-5"
_SIGNING_SECRET = "key-@@@@)))()((9))-xxxx&&&%%%%%"


def _extract_user_id_from_jwt(token: str) -> str:
    """Decode the JWT payload (no verification) and return the user ID."""
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return ""
        payload_b64 = parts[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        for field in ("id", "user_id", "uid", "sub"):
            val = payload.get(field)
            if val:
                return str(val)
    except Exception:
        pass
    return ""


def _compute_signature(
    message_text: str,
    request_id: str,
    timestamp_ms: int,
    user_id: str,
    secret: str = _SIGNING_SECRET,
) -> str:
    """Dual-layer HMAC-SHA256 signature for Z.ai WebUI endpoint."""
    # Base64-encode the prompt text (matching the frontend's btoa(binaryStr))
    prompt_b64 = base64.b64encode(message_text.encode("utf-8")).decode("ascii")

    # Sorted payload: alphabetical key order → requestId, timestamp, user_id
    sorted_payload = f"requestId,{request_id},timestamp,{timestamp_ms},user_id,{user_id}"
    canonical = f"{sorted_payload}|{prompt_b64}|{timestamp_ms}"

    # Layer 1: time-windowed key derivation (5-minute windows)
    window_index = timestamp_ms // (5 * 60 * 1000)
    derived = hmac.new(
        secret.encode(), str(window_index).encode(), hashlib.sha256
    ).hexdigest()

    # Layer 2: sign canonical string with derived key
    return hmac.new(
        derived.encode(), canonical.encode(), hashlib.sha256
    ).hexdigest()


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

        import time as _time
        import urllib.parse as _urlparse

        chat_id = str(uuid.uuid4())
        request_id = str(uuid.uuid4())
        message_id = str(uuid.uuid4())
        parent_id = str(uuid.uuid4())

        # Extract the last user message for signature_prompt.
        last_user_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_content = msg.get("content", "")
                break

        body: dict[str, Any] = {
            "stream": True,
            "model": model,
            "messages": messages,
            "signature_prompt": last_user_content,
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
            "variables": {},
            "chat_id": chat_id,
            "id": request_id,
            "current_user_message_id": message_id,
            "current_user_message_parent_id": parent_id,
            "background_tasks": {
                "title_generation": False,
                "tags_generation": False,
            },
        }

        ts = int(_time.time() * 1000)
        user_id = _extract_user_id_from_jwt(self._api_key or "")
        signature = _compute_signature(
            last_user_content, request_id, ts, user_id,
        )
        query_params = _urlparse.urlencode({
            "timestamp": ts,
            "requestId": request_id,
            "user_id": user_id,
            "version": "0.0.1",
            "platform": "web",
            "token": self._api_key or "",
            "user_agent": "Mozilla/5.0 (X11; Linux x86_64; rv:149.0) Gecko/20100101 Firefox/149.0",
            "language": "en-US",
            "languages": "en-US,en",
            "timezone": "America/Belize",
            "cookie_enabled": "true",
            "screen_width": "3840",
            "screen_height": "2160",
            "screen_resolution": "3840x2160",
            "viewport_height": "1047",
            "viewport_width": "1920",
            "viewport_size": "1920x1047",
            "color_depth": "24",
            "pixel_ratio": "1",
            "current_url": f"https://chat.z.ai/c/{chat_id}",
            "pathname": f"/c/{chat_id}",
            "search": "",
            "hash": "",
            "host": "chat.z.ai",
            "hostname": "chat.z.ai",
            "protocol": "https:",
            "referrer": "",
            "title": "Z.ai - Free AI Chatbot & Agent powered by GLM-5.1 & GLM-5",
            "timezone_offset": "360",
            "is_mobile": "false",
            "is_touch": "false",
            "max_touch_points": "0",
            "browser_name": "Firefox",
            "os_name": "Linux",
            "signature_timestamp": ts,
        })
        url = f"{self._base_url}/api/v2/chat/completions?{query_params}"
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "*/*",
            "Accept-Language": "en-US",
            "Accept-Encoding": "gzip, deflate",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:149.0) Gecko/20100101 Firefox/149.0",
            "X-FE-Version": "prod-fe-1.1.7",
            "X-Signature": signature,
            "Origin": "https://chat.z.ai",
            "Connection": "keep-alive",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Priority": "u=0",
            "Pragma": "no-cache",
            "Cache-Control": "no-cache",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
            headers["Cookie"] = f"token={self._api_key}"

        # Log equivalent curl command for debugging.
        _curl_parts = [f"curl '{url}'"]
        for hk, hv in headers.items():
            _safe_v = hv
            if hk in ("Authorization", "Cookie"):
                _safe_v = hv[:20] + "..." if len(hv) > 20 else hv
            _curl_parts.append(f"  -H '{hk}: {_safe_v}'")
        _body_json = json.dumps(body, ensure_ascii=False)
        _curl_parts.append(f"  --data-raw '{_body_json}'")
        logger.warning("ZAI request curl:\n%s", " \\\n".join(_curl_parts))

        resp = _requests.post(
            url,
            headers=headers,
            json=body,
            stream=True,
            timeout=300,
        )

        logger.warning(
            "ZAI response: status=%d headers=%s",
            resp.status_code,
            dict(resp.headers),
        )

        if resp.status_code == 429:
            resp.close()
            raise _RateLimited("ZAI 429")
        if not resp.ok:
            # Log the body for non-2xx so we can see rejection reasons.
            try:
                err_body = resp.text
            except Exception:
                err_body = "(unreadable)"
            logger.warning("ZAI error response body: %s", err_body[:2000])
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
