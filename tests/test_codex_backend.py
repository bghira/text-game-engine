from __future__ import annotations

import asyncio

from text_game_engine.backends import (
    BackendTextCompletionPort,
    ChatMessage,
    CodexCLIBackend,
    CompletionRequest,
    build_backend,
)


class RecordingCodexCLIBackend(CodexCLIBackend):
    def __init__(self, *, stdout: str, stderr: str = "", returncode: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.calls: list[tuple[list[str], str]] = []

    async def _run_exec_command(self, command: list[str], prompt: str) -> tuple[int, str, str]:
        self.calls.append((list(command), prompt))
        return self.returncode, self.stdout, self.stderr


def test_codex_backend_builds_exec_command_and_parses_jsonl_output():
    async def run_test():
        backend = RecordingCodexCLIBackend(
            stdout=(
                'Reading prompt from stdin...\n'
                '{"type":"thread.started","thread_id":"abc"}\n'
                '{"type":"turn.started"}\n'
                '{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"Hello from Codex"}}\n'
                '{"type":"turn.completed","usage":{"input_tokens":12,"cached_input_tokens":5,"output_tokens":7}}\n'
            ),
            model="gpt-5-codex",
            cd="/tmp",
            sandbox="read-only",
            config={"model_reasoning_effort": "low"},
            add_dirs=["/tmp/shared"],
        )
        result = await backend.complete(
            CompletionRequest(
                messages=[
                    ChatMessage(role="system", content="You are terse."),
                    ChatMessage(role="user", content="Say hello."),
                ]
            )
        )
        assert result.text == "Hello from Codex"
        assert result.finish_reason == "stop"
        assert result.usage == {
            "input_tokens": 12,
            "cached_input_tokens": 5,
            "output_tokens": 7,
            "total_tokens": 19,
        }
        command, prompt = backend.calls[0]
        assert command[:4] == ["codex", "exec", "--json", "-s"]
        assert "read-only" in command
        assert "--ephemeral" in command
        assert "--skip-git-repo-check" in command
        assert ["-C", "/tmp"] == command[command.index("-C") : command.index("-C") + 2]
        assert ["-m", "gpt-5-codex"] == command[command.index("-m") : command.index("-m") + 2]
        assert any(
            item.startswith("user_instructions=") and "<system_instructions>" in item and "You are terse." in item
            for item in command
        )
        assert any(
            item.startswith("model_reasoning_effort=") and '"low"' in item
            for item in command
        )
        assert prompt == "<user_request>\nSay hello.\n</user_request>"

    asyncio.run(run_test())


def test_codex_backend_formats_multi_message_prompt_and_flattens_provider_config():
    async def run_test():
        backend = RecordingCodexCLIBackend(
            stdout='{"type":"item.completed","item":{"type":"agent_message","text":"done"}}\n',
            model="gpt-5-codex",
        )
        await backend.complete(
            CompletionRequest(
                messages=[
                    ChatMessage(role="system", content="System rules."),
                    ChatMessage(role="user", content="First."),
                    ChatMessage(role="assistant", content="Second."),
                    ChatMessage(role="user", content="Third."),
                ],
                provider_options={
                    "config": {
                        "model_reasoning_effort": "medium",
                        "nested": {"flag": True},
                    },
                    "extra_args": ["--color", "never"],
                },
            )
        )
        command, prompt = backend.calls[0]
        assert "<conversation>" in prompt
        assert "<user_message>" in prompt
        assert "<assistant_message>" in prompt
        assert "Third." in prompt
        assert any(
            item.startswith("nested.flag=") and item.endswith("true") for item in command
        )
        assert command[-2:] == ["--color", "never"]

    asyncio.run(run_test())


def test_codex_backend_raises_runtime_error_from_json_error_event():
    async def run_test():
        backend = RecordingCodexCLIBackend(
            stdout='{"type":"error","message":"boom"}\n',
            returncode=1,
        )
        try:
            await backend.complete(
                CompletionRequest(messages=[ChatMessage(role="user", content="Hello")])
            )
        except RuntimeError as exc:
            assert "boom" in str(exc)
        else:
            raise AssertionError("expected RuntimeError")

    asyncio.run(run_test())


def test_backend_text_completion_port_supports_codex_backend():
    async def run_test():
        backend = RecordingCodexCLIBackend(
            stdout='{"type":"item.completed","item":{"type":"agent_message","text":"adapted"}}\n'
        )
        port = BackendTextCompletionPort(backend)
        result = await port.complete(
            "You are zork emulator.",
            "Return adapted",
        )
        assert result == "adapted"
        command, prompt = backend.calls[0]
        assert any(
            item.startswith("user_instructions=")
            and "<system_instructions>" in item
            and "You are zork emulator." in item
            for item in command
        )
        assert prompt == "<user_request>\nReturn adapted\n</user_request>"

    asyncio.run(run_test())


def test_build_backend_supports_codex_provider():
    backend = build_backend("codex", cd="/tmp")
    assert isinstance(backend, CodexCLIBackend)
