from __future__ import annotations

from typing import Any

from .adapters import BackendTextCompletionPort
from .base import ModelBackend
from .claude_cli import ClaudeCLIBackend
from .codex_cli import CodexCLIBackend
from .gemini_cli import GeminiCLIBackend
from .opencode_cli import OpenCodeCLIBackend
from .ollama import OllamaBackend
from .zai import ZAIBackend


def build_backend(provider: str, **config: Any) -> ModelBackend:
    normalized = str(provider or "").strip().lower()
    if normalized == "ollama":
        return OllamaBackend(**config)
    if normalized == "gemini":
        return GeminiCLIBackend(**config)
    if normalized == "claude":
        return ClaudeCLIBackend(**config)
    if normalized == "opencode":
        return OpenCodeCLIBackend(**config)
    if normalized in {"codex", "codex-cli", "codex_cli"}:
        return CodexCLIBackend(**config)
    if normalized == "zai":
        return ZAIBackend(**config)
    raise ValueError(f"Unsupported backend provider: {provider}")


def build_text_completion_port(provider: str, **config: Any) -> BackendTextCompletionPort:
    return BackendTextCompletionPort(build_backend(provider, **config))
