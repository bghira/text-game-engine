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

    def _post_json(self, path: str, payload: dict) -> dict:
        self.calls.append((path, payload))
        return dict(self.response_payload)


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
        assert payload["stream"] is False
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
        assert payload["keep_alive"] == "1h"
        assert payload["options"]["seed"] == 7
        assert payload["options"]["temperature"] == 0.1
        assert payload["options"]["num_ctx"] == 32768
        assert payload["options"]["num_predict"] == 128

    asyncio.run(run_test())


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
