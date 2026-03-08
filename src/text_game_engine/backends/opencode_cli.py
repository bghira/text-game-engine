from __future__ import annotations

import asyncio
import json
from typing import Any

from .base import ChatMessage, CompletionRequest, CompletionResult
from .prompt_formatting import (
    build_structured_system_instructions,
    build_structured_user_prompt,
)


class OpenCodeCLIBackend:
    """OpenCode CLI backend using `opencode run --format json`."""

    _TEXT_COMPLETION_INSTRUCTIONS = (
        "You are being used as a text-completion backend, not as a coding agent. "
        "Do not inspect files, run commands, or infer hidden tasks from the workspace unless "
        "the prompt explicitly asks for that. Respond directly to the prompt content only."
    )

    def __init__(
        self,
        *,
        model: str | None = None,
        opencode_bin: str = "opencode",
        cd: str | None = None,
        request_timeout: float = 180.0,
        agent: str | None = None,
        variant: str | None = None,
        extra_args: list[str] | None = None,
    ):
        self._model = model
        self._opencode_bin = opencode_bin
        self._cd = cd
        self._request_timeout = float(request_timeout)
        self._agent = agent
        self._variant = variant
        self._extra_args = [str(item) for item in (extra_args or []) if str(item).strip()]

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        command = self._build_command(request)
        returncode, stdout, stderr = await self._run_command(command)
        parsed = self._parse_output(stdout)
        if returncode != 0:
            detail = stderr.strip() or stdout.strip() or f"exit code {returncode}"
            raise RuntimeError(f"OpenCode request failed: {detail}")
        if parsed["error_message"]:
            raise RuntimeError(f"OpenCode request failed: {parsed['error_message']}")
        return CompletionResult(
            text=str(parsed["text"] or "").strip(),
            finish_reason=parsed["finish_reason"],
            usage=parsed["usage"],
            raw_response={"events": parsed["events"], "stderr": stderr},
        )

    def _build_command(self, request: CompletionRequest) -> list[str]:
        prompt = self._build_prompt(request.messages)
        model = request.model or self._model
        command = [
            self._opencode_bin,
            "run",
            "--format",
            "json",
        ]
        if model:
            command.extend(["-m", model])
        agent = request.provider_options.get("agent", self._agent)
        if agent:
            command.extend(["--agent", str(agent)])
        variant = request.provider_options.get("variant", self._variant)
        if variant:
            command.extend(["--variant", str(variant)])
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
        system_block = build_structured_system_instructions(
            base_instructions=cls._TEXT_COMPLETION_INSTRUCTIONS,
            system_prompt=system_prompt,
        )
        user_block = build_structured_user_prompt(messages)
        return "\n\n".join(part for part in [system_block, user_block] if part.strip()).strip()

    @staticmethod
    def _parse_output(stdout: str) -> dict[str, Any]:
        events: list[dict[str, Any]] = []
        texts: list[str] = []
        usage: dict[str, int] | None = None
        finish_reason: str | None = None
        error_message: str | None = None
        for raw_line in stdout.splitlines():
            line = raw_line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            events.append(event)
            if str(event.get("type") or "") == "text":
                part = event.get("part")
                if isinstance(part, dict):
                    text = str(part.get("text") or "").strip()
                    if text:
                        texts.append(text)
            elif str(event.get("type") or "") == "step_finish":
                part = event.get("part")
                if isinstance(part, dict):
                    finish_reason = str(part.get("reason") or "").strip() or None
                    tokens = part.get("tokens")
                    if isinstance(tokens, dict):
                        usage = {
                            "input_tokens": int(tokens.get("input") or 0),
                            "output_tokens": int(tokens.get("output") or 0),
                            "reasoning_tokens": int(tokens.get("reasoning") or 0),
                            "total_tokens": int(tokens.get("total") or 0),
                        }
            elif str(event.get("type") or "") == "error":
                error_message = str(event.get("message") or "").strip() or None
        return {
            "events": events,
            "text": "\n".join(texts).strip(),
            "usage": usage,
            "finish_reason": finish_reason,
            "error_message": error_message,
        }


# Backwards-compatible alias for earlier exports.
OpenCodeBackend = OpenCodeCLIBackend
