from __future__ import annotations

import asyncio

from text_game_engine.backends import (
    BackendTextCompletionPort,
    ChatMessage,
    CompletionRequest,
    OllamaBackend,
    build_backend,
    build_text_completion_port,
)


class RecordingOllamaBackend(OllamaBackend):
    def __init__(self, response_payload: dict, **kwargs):
        super().__init__(**kwargs)
        self.response_payload = response_payload
        self.calls: list[tuple[str, dict]] = []

    def _post_streaming(self, path: str, payload: dict) -> dict:
        self.calls.append((path, payload))
        return dict(self.response_payload)


class FakeStreamingResponse:
    def __init__(self, lines: list[bytes]):
        self._lines = lines
        self.closed = False

    def __iter__(self):
        return iter(self._lines)

    def close(self):
        self.closed = True


def test_ollama_backend_builds_native_chat_payload_and_parses_response():
    async def run_test():
        backend = RecordingOllamaBackend(
            {
                "message": {"role": "assistant", "content": "Hello from Ollama"},
                "done_reason": "stop",
                "prompt_eval_count": 12,
                "eval_count": 34,
            },
            model="llama3.1",
            keep_alive="15m",
            options={"top_k": 20},
        )
        result = await backend.complete(
            CompletionRequest(
                messages=[
                    ChatMessage(role="system", content="You are concise."),
                    ChatMessage(role="user", content="Say hello."),
                ],
                temperature=0.25,
                max_tokens=256,
                stop=["STOP"],
            )
        )
        assert result.text == "Hello from Ollama"
        assert result.finish_reason == "stop"
        assert result.usage == {
            "prompt_tokens": 12,
            "completion_tokens": 34,
            "total_tokens": 46,
        }
        assert backend.calls
        path, payload = backend.calls[0]
        assert path == "/api/chat"
        assert payload["model"] == "llama3.1"
        assert payload["stream"] is True
        assert payload["think"] is False
        assert payload["keep_alive"] == "15m"
        assert payload["messages"] == [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Say hello."},
        ]
        assert payload["options"]["temperature"] == 0.25
        assert payload["options"]["num_predict"] == 256
        assert payload["options"]["stop"] == ["STOP"]
        assert payload["options"]["top_k"] == 20

    asyncio.run(run_test())


def test_ollama_backend_json_mode_and_provider_options_override_defaults():
    async def run_test():
        backend = RecordingOllamaBackend(
            {"response": "{\"ok\":true}"},
            model="qwen2.5",
            keep_alive="10m",
            options={"seed": 7},
        )
        result = await backend.complete(
            CompletionRequest(
                messages=[ChatMessage(role="user", content="Return JSON.")],
                max_tokens=128,
                json_mode=True,
                provider_options={
                    "keep_alive": "1h",
                    "options": {"temperature": 0.1, "num_ctx": 32768},
                },
            )
        )
        assert result.text == "{\"ok\":true}"
        _, payload = backend.calls[0]
        assert payload["format"] == "json"
        assert payload["think"] is False
        assert payload["keep_alive"] == "1h"
        assert payload["options"]["seed"] == 7
        assert payload["options"]["temperature"] == 0.1
        assert payload["options"]["num_ctx"] == 32768
        assert payload["options"]["num_predict"] == 128

    asyncio.run(run_test())


def test_ollama_backend_sends_think_true_and_expands_token_budget():
    async def run_test():
        backend = RecordingOllamaBackend(
            {"message": {"role": "assistant", "content": "with thinking"}},
            model="qwen3",
            think=True,
        )
        result = await backend.complete(
            CompletionRequest(
                messages=[ChatMessage(role="user", content="Think this through.")],
                max_tokens=128,
            )
        )
        assert result.text == "with thinking"
        _, payload = backend.calls[0]
        assert payload["think"] is True
        assert payload["options"]["num_predict"] == backend._THINKING_NUM_PREDICT_FLOOR

    asyncio.run(run_test())


def test_ollama_backend_allows_provider_options_think_override():
    async def run_test():
        backend = RecordingOllamaBackend(
            {"message": {"role": "assistant", "content": "low effort"}},
            model="gpt-oss",
        )
        await backend.complete(
            CompletionRequest(
                messages=[ChatMessage(role="user", content="Use low reasoning.")],
                max_tokens=128,
                provider_options={"think": "low"},
            )
        )
        _, payload = backend.calls[0]
        assert payload["think"] == "low"
        assert payload["options"]["num_predict"] == backend._THINKING_NUM_PREDICT_FLOOR

    asyncio.run(run_test())


def test_ollama_backend_normalizes_string_false_think_values():
    async def run_test():
        backend = RecordingOllamaBackend(
            {"message": {"role": "assistant", "content": "without thinking"}},
            model="qwen3",
            think="false",
        )
        await backend.complete(
            CompletionRequest(
                messages=[ChatMessage(role="user", content="No reasoning trace.")],
                max_tokens=128,
            )
        )
        _, payload = backend.calls[0]
        assert payload["think"] is False
        assert payload["options"]["num_predict"] == 128

    asyncio.run(run_test())


def test_ollama_streaming_ignores_thinking_chunks_when_think_is_false(monkeypatch):
    response = FakeStreamingResponse(
        [
            b'{"message":{"thinking":"hidden reasoning"},"done":false}\n',
            b'{"message":{"content":"visible answer"},"done":false}\n',
            b'{"done":true,"done_reason":"stop","eval_count":2}\n',
        ]
    )

    def fake_urlopen(req, timeout):
        return response

    monkeypatch.setattr(
        "text_game_engine.backends.ollama.urllib_request.urlopen",
        fake_urlopen,
    )

    backend = OllamaBackend(model="qwen3")
    result = backend._post_streaming(
        "/api/chat",
        {"model": "qwen3", "messages": [], "stream": True, "think": False},
    )

    assert result["message"] == {"role": "assistant", "content": "visible answer"}
    assert result["response"] == "visible answer"
    assert response.closed is True


def test_ollama_streaming_keeps_thinking_chunks_when_requested(monkeypatch):
    response = FakeStreamingResponse(
        [
            b'{"message":{"thinking":"reasoning "},"done":false}\n',
            b'{"message":{"thinking":"trace"},"done":false}\n',
            b'{"message":{"content":"visible answer"},"done":false}\n',
            b'{"done":true,"done_reason":"stop","eval_count":3}\n',
        ]
    )

    def fake_urlopen(req, timeout):
        return response

    monkeypatch.setattr(
        "text_game_engine.backends.ollama.urllib_request.urlopen",
        fake_urlopen,
    )

    backend = OllamaBackend(model="qwen3")
    result = backend._post_streaming(
        "/api/chat",
        {"model": "qwen3", "messages": [], "stream": True, "think": True},
    )

    assert result["message"] == {
        "role": "assistant",
        "content": "visible answer",
        "thinking": "reasoning trace",
    }


def test_backend_text_completion_port_adapts_backend_to_emulator_surface():
    async def run_test():
        backend = RecordingOllamaBackend(
            {"message": {"role": "assistant", "content": "adapted output"}},
            model="mistral",
        )
        port = BackendTextCompletionPort(backend)
        result = await port.complete(
            "System instruction",
            "User prompt",
            temperature=0.6,
            max_tokens=300,
        )
        assert result == "adapted output"
        _, payload = backend.calls[0]
        assert payload["messages"] == [
            {"role": "system", "content": "System instruction"},
            {"role": "user", "content": "User prompt"},
        ]
        assert payload["options"]["temperature"] == 0.6
        assert payload["options"]["num_predict"] == 300

    asyncio.run(run_test())


def test_build_backend_supports_ollama_provider():
    backend = build_backend("ollama", model="llama3.1")
    assert isinstance(backend, OllamaBackend)


def test_build_text_completion_port_wraps_backend():
    port = build_text_completion_port("ollama", model="llama3.1")
    assert isinstance(port, BackendTextCompletionPort)
