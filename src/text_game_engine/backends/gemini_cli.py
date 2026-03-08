from __future__ import annotations

import asyncio
import json
from typing import Any

from .base import ChatMessage, CompletionRequest, CompletionResult


class GeminiCLIBackend:
    """Gemini CLI backend using non-interactive JSON output."""

    _TEXT_COMPLETION_INSTRUCTIONS = (
        "You are being used as a text-completion backend, not as a coding agent. "
        "Do not inspect the workspace, read files, run shell commands, or infer hidden tasks "
        "from nearby repositories unless the prompt explicitly asks for that. Respond directly "
        "to the prompt content only."
    )

    def __init__(
        self,
        *,
        model: str | None = None,
        gemini_bin: str = "gemini",
        cd: str | None = None,
        request_timeout: float = 180.0,
        sandbox: bool | None = None,
        yolo: bool | None = None,
        approval_mode: str | None = None,
        extra_args: list[str] | None = None,
    ):
        self._model = model
        self._gemini_bin = gemini_bin
        self._cd = cd
        self._request_timeout = float(request_timeout)
        self._sandbox = sandbox
        self._yolo = yolo
        self._approval_mode = approval_mode
        self._extra_args = [str(item) for item in (extra_args or []) if str(item).strip()]

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        command = self._build_command(request)
        returncode, stdout, stderr = await self._run_command(command)
        if returncode != 0:
            detail = stderr.strip() or stdout.strip() or f"exit code {returncode}"
            raise RuntimeError(f"Gemini CLI request failed: {detail}")
        payload = self._extract_last_json_object(stdout)
        if not isinstance(payload, dict):
            raise RuntimeError("Gemini CLI returned no JSON response payload")
        text = str(payload.get("response") or "").strip()
        return CompletionResult(
            text=text,
            finish_reason="stop" if text else None,
            usage=self._extract_usage(payload),
            raw_response=payload,
        )

    def _build_command(self, request: CompletionRequest) -> list[str]:
        prompt = self._build_prompt(request.messages)
        model = request.model or self._model
        command = [
            self._gemini_bin,
            "-o",
            "json",
        ]
        if model:
            command.extend(["-m", model])
        sandbox = request.provider_options.get("sandbox", self._sandbox)
        if sandbox is True:
            command.append("--sandbox")
        yolo = request.provider_options.get("yolo", self._yolo)
        if yolo is True:
            command.append("--yolo")
        approval_mode = request.provider_options.get("approval_mode", self._approval_mode)
        if approval_mode:
            command.extend(["--approval-mode", str(approval_mode)])
        extra_args = request.provider_options.get("extra_args")
        if isinstance(extra_args, list):
            command.extend(str(item) for item in extra_args if str(item).strip())
        command.extend(self._extra_args)
        command.append(prompt)
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

    @classmethod
    def _build_prompt(cls, messages: list[ChatMessage]) -> str:
        system_prompt = "\n\n".join(
            message.content.strip()
            for message in messages
            if message.role == "system" and message.content.strip()
        )
        non_system = [message for message in messages if message.role != "system"]
        if len(non_system) == 1 and non_system[0].role == "user":
            user_text = non_system[0].content.strip()
            if system_prompt:
                return (
                    f"{cls._TEXT_COMPLETION_INSTRUCTIONS}\n\n"
                    f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_text}"
                ).strip()
            return f"{cls._TEXT_COMPLETION_INSTRUCTIONS}\n\nUSER:\n{user_text}".strip()
        parts = [cls._TEXT_COMPLETION_INSTRUCTIONS]
        if system_prompt:
            parts.extend(["SYSTEM:", system_prompt])
        for message in non_system:
            parts.extend([f"{str(message.role).upper()}:", message.content])
        return "\n\n".join(part.strip() for part in parts if part.strip())

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
        stats = payload.get("stats")
        if not isinstance(stats, dict):
            return None
        models = stats.get("models")
        if not isinstance(models, dict):
            return None
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "thought_tokens": 0,
            "tool_tokens": 0,
        }
        seen = False
        for data in models.values():
            if not isinstance(data, dict):
                continue
            tokens = data.get("tokens")
            if not isinstance(tokens, dict):
                continue
            seen = True
            usage["prompt_tokens"] += int(tokens.get("prompt") or 0)
            usage["completion_tokens"] += int(tokens.get("candidates") or 0)
            usage["total_tokens"] += int(tokens.get("total") or 0)
            usage["thought_tokens"] += int(tokens.get("thoughts") or 0)
            usage["tool_tokens"] += int(tokens.get("tool") or 0)
        if not seen:
            return None
        return usage
