from __future__ import annotations

import asyncio
import json
from typing import Any

from .base import ChatMessage, CompletionRequest, CompletionResult


class ClaudeCLIBackend:
    """Claude Code CLI backend using `--print` JSON output."""

    def __init__(
        self,
        *,
        model: str | None = None,
        claude_bin: str = "claude",
        cd: str | None = None,
        request_timeout: float = 180.0,
        permission_mode: str = "dontAsk",
        tools: str | None = "",
        extra_args: list[str] | None = None,
    ):
        self._model = model
        self._claude_bin = claude_bin
        self._cd = cd
        self._request_timeout = float(request_timeout)
        self._permission_mode = permission_mode
        self._tools = tools
        self._extra_args = [str(item) for item in (extra_args or []) if str(item).strip()]

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        command = self._build_command(request)
        returncode, stdout, stderr = await self._run_command(command)
        if returncode != 0:
            detail = stderr.strip() or stdout.strip() or f"exit code {returncode}"
            raise RuntimeError(f"Claude CLI request failed: {detail}")
        payload = self._extract_last_json_object(stdout)
        if not isinstance(payload, dict):
            raise RuntimeError("Claude CLI returned no JSON result payload")
        if payload.get("is_error") is True:
            raise RuntimeError(f"Claude CLI request failed: {payload.get('result') or payload}")
        return CompletionResult(
            text=str(payload.get("result") or "").strip(),
            finish_reason="stop",
            usage=self._extract_usage(payload),
            raw_response=payload,
        )

    def _build_command(self, request: CompletionRequest) -> list[str]:
        system_prompt = "\n\n".join(
            message.content.strip()
            for message in request.messages
            if message.role == "system" and message.content.strip()
        )
        prompt = self._build_prompt(request.messages)
        model = request.model or self._model
        command = [
            self._claude_bin,
            "-p",
            prompt,
            "--output-format",
            "json",
            "--permission-mode",
            str(request.provider_options.get("permission_mode") or self._permission_mode),
        ]
        if model:
            command.extend(["--model", model])
        tools = request.provider_options.get("tools", self._tools)
        if tools is not None:
            command.extend(["--tools", str(tools)])
        if system_prompt:
            command.extend(["--system-prompt", system_prompt])
        extra_args = request.provider_options.get("extra_args")
        if isinstance(extra_args, list):
            command.extend(str(item) for item in extra_args if str(item).strip())
        command.extend(self._extra_args)
        return command

    async def _run_command(self, command: list[str]) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=self._cd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=self._request_timeout,
        )
        return proc.returncode, stdout.decode("utf-8", errors="replace"), stderr.decode(
            "utf-8", errors="replace"
        )

    @staticmethod
    def _build_prompt(messages: list[ChatMessage]) -> str:
        non_system = [message for message in messages if message.role != "system"]
        if len(non_system) == 1 and non_system[0].role == "user":
            return non_system[0].content
        parts: list[str] = []
        for message in non_system:
            parts.append(f"{str(message.role).upper()}:\n{message.content}")
        return "\n\n".join(parts).strip()

    @staticmethod
    def _extract_last_json_object(stdout: str) -> dict[str, Any] | None:
        lines = stdout.splitlines()
        for idx in range(len(lines) - 1, -1, -1):
            candidate = "\n".join(lines[idx:]).strip()
            if not candidate.startswith("{"):
                continue
            try:
                value = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        return None

    @staticmethod
    def _extract_usage(payload: dict[str, Any]) -> dict[str, int] | None:
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return None
        out: dict[str, int] = {}
        mapping = {
            "input_tokens": "input_tokens",
            "cache_creation_input_tokens": "cache_creation_input_tokens",
            "cache_read_input_tokens": "cache_read_input_tokens",
            "output_tokens": "output_tokens",
        }
        for src, dest in mapping.items():
            raw = usage.get(src)
            if isinstance(raw, int):
                out[dest] = raw
        total = out.get("input_tokens", 0) + out.get("output_tokens", 0)
        if total:
            out["total_tokens"] = total
        return out or None
