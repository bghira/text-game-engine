# Backends

`text-game-engine` now includes a backend layer for host-facing text generation integrations.

The first built-in provider is native Ollama over `/api/chat`, not the OpenAI compatibility layer.

## Public API

```python
from text_game_engine import (
    BackendTextCompletionPort,
    ClaudeCLIBackend,
    CodexCLIBackend,
    GeminiCLIBackend,
    OpenCodeBackend,
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

## Codex CLI

Use `CodexCLIBackend` when you want emulator completions to run through a locally installed `codex` CLI.

```python
from text_game_engine import BackendTextCompletionPort, CodexCLIBackend

backend = CodexCLIBackend(
    cd="/home/kash/src/text-game-engine",
    sandbox="read-only",
    skip_git_repo_check=False,
)
completion_port = BackendTextCompletionPort(backend)
```

Notes:
- This backend uses `codex exec --json` under the hood.
- The emulator/system prompt is mapped onto Codex CLI's `user_instructions` config override.
- The task prompt is passed on stdin.
- Codex CLI does not expose direct equivalents for every chat completion knob, so `temperature`, `stop`, and `max_tokens` are only indirect hints unless you pass explicit CLI config through `provider_options["config"]`.
- Use a small or empty `cd` when possible; Codex is less likely to waste tokens behaving like a repo agent in a noisy workspace.

## Gemini CLI

```python
from text_game_engine import BackendTextCompletionPort, GeminiCLIBackend

backend = GeminiCLIBackend(
    cd="/tmp/tge-gemini",
)
completion_port = BackendTextCompletionPort(backend)
```

Gemini CLI does not expose a separate system-prompt flag, so system messages are wrapped into the final task prompt.

## Claude CLI

```python
from text_game_engine import BackendTextCompletionPort, ClaudeCLIBackend

backend = ClaudeCLIBackend(
    cd="/tmp/tge-claude",
)
completion_port = BackendTextCompletionPort(backend)
```

Claude Code exposes a real `--system-prompt` flag, so emulator system instructions are passed separately.

## OpenCode CLI

```python
from text_game_engine import BackendTextCompletionPort, OpenCodeBackend

backend = OpenCodeBackend(
    model="opencode/gpt-5-nano",
    cd="/tmp/tge-opencode",
)
completion_port = BackendTextCompletionPort(backend)
```

OpenCode does not expose a separate system-prompt flag in `run`, so system messages are wrapped into the task prompt. As with Codex, using a small or empty working directory reduces wasted context.

## Factory

If you want provider selection from config:

```python
from text_game_engine import BackendTextCompletionPort, build_backend

backend = build_backend(
    "codex",
    cd="/home/kash/src/text-game-engine",
    sandbox="read-only",
)
completion_port = BackendTextCompletionPort(backend)
```

Or directly:

```python
from text_game_engine import build_text_completion_port

completion_port = build_text_completion_port(
    "codex",
    cd="/home/kash/src/text-game-engine",
    sandbox="read-only",
)
```

## Current Scope

This backend layer currently targets the emulator-style `TextCompletionPort`.

`GameEngine` turn resolution still uses the separate `LLMPort` interface:

```python
GameEngine(uow_factory=..., llm=your_turn_llm_port)
```

If your host wants fully local turn resolution as well, implement `LLMPort` using the same backend and your own turn-prompt / JSON-decoding policy. That keeps the transport reusable without hardcoding one structured-output strategy into the engine package.
