from __future__ import annotations

from typing import Any

from .adapters import BackendTextCompletionPort
from .base import ModelBackend
from .ollama import OllamaBackend


def build_backend(provider: str, **config: Any) -> ModelBackend:
    normalized = str(provider or "").strip().lower()
    if normalized == "ollama":
        return OllamaBackend(**config)
    raise ValueError(f"Unsupported backend provider: {provider}")


def build_text_completion_port(provider: str, **config: Any) -> BackendTextCompletionPort:
    return BackendTextCompletionPort(build_backend(provider, **config))
