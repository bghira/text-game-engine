from __future__ import annotations

import asyncio

from text_game_engine.backends import (
    BackendTextCompletionPort,
    ChatMessage,
    ClaudeCLIBackend,
    CompletionRequest,
    GeminiCLIBackend,
    OpenCodeBackend,
    build_backend,
)


class RecordingGeminiCLIBackend(GeminiCLIBackend):
    def __init__(self, *, stdout: str, stderr: str = "", returncode: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.calls: list[list[str]] = []

    async def _run_command(self, command: list[str]) -> tuple[int, str, str]:
        self.calls.append(list(command))
        return self.returncode, self.stdout, self.stderr


class RecordingClaudeCLIBackend(ClaudeCLIBackend):
    def __init__(self, *, stdout: str, stderr: str = "", returncode: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.calls: list[list[str]] = []

    async def _run_command(self, command: list[str]) -> tuple[int, str, str]:
        self.calls.append(list(command))
        return self.returncode, self.stdout, self.stderr


class RecordingOpenCodeBackend(OpenCodeBackend):
    def __init__(self, *, stdout: str, stderr: str = "", returncode: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.calls: list[list[str]] = []

    async def _run_command(self, command: list[str]) -> tuple[int, str, str]:
        self.calls.append(list(command))
        return self.returncode, self.stdout, self.stderr


def test_gemini_backend_parses_json_and_wraps_system_prompt():
    async def run_test():
        backend = RecordingGeminiCLIBackend(
            stdout='noise\n{"response":"GEMINI-OK","stats":{"models":{"gemini":{"tokens":{"prompt":10,"candidates":2,"total":12,"thoughts":1,"tool":0}}}}}',
            model="gemini-2.5-flash",
        )
        result = await backend.complete(
            CompletionRequest(
                messages=[
                    ChatMessage(role="system", content="Be terse."),
                    ChatMessage(role="user", content="Say ok."),
                ]
            )
        )
        assert result.text == "GEMINI-OK"
        assert result.usage == {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "thought_tokens": 1,
            "tool_tokens": 0,
        }
        command = backend.calls[0]
        assert command[:2] == ["gemini", "-o"]
        assert "json" in command
        assert "-m" in command
        assert "SYSTEM:" in command[-1]

    asyncio.run(run_test())


def test_claude_backend_uses_system_prompt_flag_and_json_result():
    async def run_test():
        backend = RecordingClaudeCLIBackend(
            stdout='{"type":"result","subtype":"success","is_error":false,"result":"CLAUDE-OK","usage":{"input_tokens":20,"output_tokens":3,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}',
            model="sonnet",
        )
        result = await backend.complete(
            CompletionRequest(
                messages=[
                    ChatMessage(role="system", content="Be terse."),
                    ChatMessage(role="user", content="Say ok."),
                ]
            )
        )
        assert result.text == "CLAUDE-OK"
        assert result.usage == {
            "input_tokens": 20,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": 3,
            "total_tokens": 23,
        }
        command = backend.calls[0]
        assert "--system-prompt" in command
        assert "--tools" in command

    asyncio.run(run_test())


def test_opencode_backend_parses_jsonl_events():
    async def run_test():
        backend = RecordingOpenCodeBackend(
            stdout=(
                '{"type":"step_start"}\n'
                '{"type":"text","part":{"text":"OPEN"}}\n'
                '{"type":"text","part":{"text":"CODE-OK"}}\n'
                '{"type":"step_finish","part":{"reason":"stop","tokens":{"total":99,"input":80,"output":19,"reasoning":7}}}\n'
            ),
            model="opencode/gpt-5-nano",
        )
        result = await backend.complete(
            CompletionRequest(
                messages=[
                    ChatMessage(role="system", content="Be terse."),
                    ChatMessage(role="user", content="Say ok."),
                ]
            )
        )
        assert result.text == "OPEN\nCODE-OK"
        assert result.finish_reason == "stop"
        assert result.usage == {
            "input_tokens": 80,
            "output_tokens": 19,
            "reasoning_tokens": 7,
            "total_tokens": 99,
        }
        command = backend.calls[0]
        assert command[:3] == ["opencode", "run", "--format"]
        assert "SYSTEM:" in command[-1]

    asyncio.run(run_test())


def test_backend_text_completion_port_supports_claude_backend():
    async def run_test():
        backend = RecordingClaudeCLIBackend(
            stdout='{"type":"result","subtype":"success","is_error":false,"result":"adapted","usage":{"input_tokens":1,"output_tokens":1}}'
        )
        port = BackendTextCompletionPort(backend)
        result = await port.complete("System instruction", "User prompt")
        assert result == "adapted"

    asyncio.run(run_test())


def test_build_backend_supports_gemini_claude_and_opencode():
    assert isinstance(build_backend("gemini"), GeminiCLIBackend)
    assert isinstance(build_backend("claude"), ClaudeCLIBackend)
    assert isinstance(build_backend("opencode"), OpenCodeBackend)
