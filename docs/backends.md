# Backends

`text-game-engine` now includes a backend layer for host-facing text generation integrations.

The first built-in provider is native Ollama over `/api/chat`, not the OpenAI compatibility layer.

## Public API

```python
from text_game_engine import (
    BackendTextCompletionPort,
    OllamaBackend,
    build_backend,
    build_text_completion_port,
)
```

## Native Ollama

Use `OllamaBackend` when you want local model inference through a running Ollama daemon.

```python
from text_game_engine import BackendTextCompletionPort, OllamaBackend

backend = OllamaBackend(
    model="llama3.1",
    base_url="http://127.0.0.1:11434",
    keep_alive="30m",
    options={
        "num_ctx": 32768,
    },
)

completion_port = BackendTextCompletionPort(backend)
```

The adapter exposes the existing `TextCompletionPort` shape used by `ZorkEmulator`, attachment summarization, setup classification, map generation, and related helper calls.

## Factory

If you want provider selection from config:

```python
from text_game_engine import BackendTextCompletionPort, build_backend

backend = build_backend(
    "ollama",
    model="qwen2.5:14b",
    base_url="http://127.0.0.1:11434",
)
completion_port = BackendTextCompletionPort(backend)
```

Or directly:

```python
from text_game_engine import build_text_completion_port

completion_port = build_text_completion_port(
    "ollama",
    model="qwen2.5:14b",
    base_url="http://127.0.0.1:11434",
)
```

## Current Scope

This backend layer currently targets the emulator-style `TextCompletionPort`.

`GameEngine` turn resolution still uses the separate `LLMPort` interface:

```python
GameEngine(uow_factory=..., llm=your_turn_llm_port)
```

If your host wants fully local turn resolution as well, implement `LLMPort` using the same backend and your own turn-prompt / JSON-decoding policy. That keeps the transport reusable without hardcoding one structured-output strategy into the engine package.
