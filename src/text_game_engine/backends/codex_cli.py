from __future__ import annotations

import asyncio
import json
from typing import Any

from .base import ChatMessage, CompletionRequest, CompletionResult
from .prompt_formatting import (
    build_codex_structured_system_instructions,
    build_codex_structured_user_prompt,
)


class CodexCLIBackend:
    """Codex CLI backend using `codex exec` in non-interactive JSON mode."""

    _TEXT_COMPLETION_INSTRUCTIONS = (
        "You are being used as a text-completion backend, not as a coding agent. "
        "Do not inspect the workspace, read files, run shell commands, edit files, "
        "or infer hidden tasks from the repository unless the prompt explicitly asks for that. "
        "Respond directly to the prompt content only."
    )

    def __init__(
        self,
        *,
        model: str | None = None,
        codex_bin: str = "codex",
        cd: str | None = None,
        sandbox: str = "read-only",
        profile: str | None = None,
        request_timeout: float = 300.0,
        skip_git_repo_check: bool = True,
        ephemeral: bool = True,
        config: dict[str, Any] | None = None,
        add_dirs: list[str] | None = None,
        extra_args: list[str] | None = None,
    ):
        self._model = model
        self._codex_bin = codex_bin
        self._cd = cd
        self._sandbox = sandbox
        self._profile = profile
        self._request_timeout = float(request_timeout)
        self._skip_git_repo_check = bool(skip_git_repo_check)
        self._ephemeral = bool(ephemeral)
        self._config = dict(config or {})
        self._add_dirs = [str(item) for item in (add_dirs or []) if str(item).strip()]
        self._extra_args = [str(item) for item in (extra_args or []) if str(item).strip()]

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        command, prompt = self._build_exec_command(request)
        returncode, stdout, stderr = await self._run_exec_command(command, prompt)
        parsed = self._parse_exec_output(stdout)
        error_message = parsed["error_message"]
        if returncode != 0:
            detail = error_message or stderr.strip() or stdout.strip() or f"exit code {returncode}"
            raise RuntimeError(f"Codex CLI request failed: {detail}")
        if error_message:
            raise RuntimeError(f"Codex CLI request failed: {error_message}")
        return CompletionResult(
            text=str(parsed["text"] or "").strip(),
            finish_reason=parsed["finish_reason"],
            usage=parsed["usage"],
            raw_response={"events": parsed["events"], "stderr": stderr},
        )

    def _build_exec_command(self, request: CompletionRequest) -> tuple[list[str], str]:
        model = request.model or self._model
        system_prompt = "\n\n".join(
            message.content.strip()
            for message in request.messages
            if message.role == "system" and message.content.strip()
        )
        command = [
            self._codex_bin,
            "exec",
            "--json",
            "-s",
            self._sandbox,
        ]
        if self._ephemeral:
            command.append("--ephemeral")
        if self._skip_git_repo_check:
            command.append("--skip-git-repo-check")
        if self._cd:
            command.extend(["-C", self._cd])
        if self._profile:
            command.extend(["-p", self._profile])
        if model:
            command.extend(["-m", model])
        for directory in self._add_dirs:
            command.extend(["--add-dir", directory])
        config_pairs = dict(self._config)
        user_instructions = build_codex_structured_system_instructions(
            base_instructions=self._TEXT_COMPLETION_INSTRUCTIONS,
            system_prompt=system_prompt,
        )
        # `user_instructions` is the closest separate instructions channel that
        # `codex exec` exposes for the emulator's system/developer prompt.
        config_pairs["user_instructions"] = user_instructions
        provider_config = request.provider_options.get("config")
        if isinstance(provider_config, dict):
            self._flatten_config("", provider_config, config_pairs)
        for key, value in config_pairs.items():
            if value is None:
                continue
            command.extend(["-c", f"{key}={self._toml_literal(value)}"])
        extra_args = request.provider_options.get("extra_args")
        if isinstance(extra_args, list):
            command.extend(str(item) for item in extra_args if str(item).strip())
        command.extend(self._extra_args)
        prompt = self._build_prompt(request.messages)
        return command, prompt

    async def _run_exec_command(self, command: list[str], prompt: str) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(prompt.encode("utf-8")),
            timeout=self._request_timeout,
        )
        return proc.returncode, stdout.decode("utf-8", errors="replace"), stderr.decode(
            "utf-8", errors="replace"
        )

    @classmethod
    def _build_prompt(cls, messages: list[ChatMessage]) -> str:
        return build_codex_structured_user_prompt(messages)

    @classmethod
    def _parse_exec_output(cls, stdout: str) -> dict[str, Any]:
        events: list[dict[str, Any]] = []
        messages: list[str] = []
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
            event_type = str(event.get("type") or "").strip()
            if event_type == "item.completed":
                item = event.get("item")
                if isinstance(item, dict) and str(item.get("type") or "") == "agent_message":
                    text = str(item.get("text") or "").strip()
                    if text:
                        messages.append(text)
            elif event_type == "turn.completed":
                usage = cls._coerce_usage(event.get("usage"))
                finish_reason = "stop"
            elif event_type in {"error", "turn.failed"}:
                extracted = cls._extract_error_message(event)
                if extracted:
                    error_message = extracted
        return {
            "events": events,
            "text": messages[-1] if messages else "",
            "usage": usage,
            "finish_reason": finish_reason,
            "error_message": error_message,
        }

    @classmethod
    def _extract_error_message(cls, event: dict[str, Any]) -> str | None:
        direct = cls._coerce_text(event.get("message"))
        if direct:
            return direct
        payload = event.get("error")
        if isinstance(payload, dict):
            nested = cls._coerce_text(payload.get("message"))
            if nested:
                return nested
        return None

    @staticmethod
    def _coerce_usage(value: object) -> dict[str, int] | None:
        if not isinstance(value, dict):
            return None
        usage: dict[str, int] = {}
        for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
            raw = value.get(key)
            if isinstance(raw, int):
                usage[key] = raw
        total = 0
        if "input_tokens" in usage:
            total += usage["input_tokens"]
        if "output_tokens" in usage:
            total += usage["output_tokens"]
        if total:
            usage["total_tokens"] = total
        return usage or None

    @staticmethod
    def _coerce_text(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @classmethod
    def _flatten_config(cls, prefix: str, value: Any, out: dict[str, Any]) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                key_text = str(key).strip()
                if not key_text:
                    continue
                dotted = f"{prefix}.{key_text}" if prefix else key_text
                cls._flatten_config(dotted, nested, out)
            return
        if prefix:
            out[prefix] = value

    @classmethod
    def _toml_literal(cls, value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            return repr(value)
        if isinstance(value, str):
            return json.dumps(value)
        if isinstance(value, (list, tuple)):
            return "[" + ", ".join(cls._toml_literal(item) for item in value) + "]"
        raise TypeError(f"Unsupported Codex config value type: {type(value).__name__}")
