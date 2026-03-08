# text-game-engine

Standalone Python package for running the Zork runtime extracted from
`discord-tron-master`, with a built-in SQL persistence layer.

## Features

- Full `ZorkEmulator` runtime surface for standalone hosts (not Discord-only).
- Core turn engine with optimistic CAS (`row_version`) and durable inflight leases.
- SQLAlchemy persistence layer (models, repos, unit-of-work, schema bootstrap).
- Timer lifecycle persistence (`scheduled_unbound -> scheduled_bound -> expired/cancelled -> consumed`).
- Rewind + snapshot model with memory visibility watermark support.
- Calendar events stored as absolute `fire_day` values (not countdown-only fields).
- Attachment text ingestion and chunked summarization utilities.
- Optional GLM-5 token counting utility (`glm_token_count`).

## Install

```bash
git clone https://github.com/bghira/text-game-engine
cd text-game-engine
pip install -e .
```

Optional GLM tokenizer support:

```bash
pip install -e ".[glm]"
```

## Documentation

- Backends: [`docs/backends.md`](docs/backends.md)
- Source material authoring: [`docs/source-material.md`](docs/source-material.md)
- SDK: [`docs/sdk.md`](docs/sdk.md)
- Persistence: [`docs/persistence.md`](docs/persistence.md)
- Examples index: [`docs/examples.md`](docs/examples.md)
- Examples folder: [`examples/README.md`](examples/README.md)
- Schema invariants: [`SCHEMA.md`](SCHEMA.md)
- Migration checklist: [`MIGRATION_CHECKLIST.md`](MIGRATION_CHECKLIST.md)

## Real Examples

- Minimal engine turn resolution: [`examples/minimal_engine_turn.py`](examples/minimal_engine_turn.py)
- Standalone Zork runtime flow: [`examples/zork_emulator_session.py`](examples/zork_emulator_session.py)
- Attachment chunking/summarization flow: [`examples/attachment_processing.py`](examples/attachment_processing.py)

## Ollama

Native Ollama support is available through the backend layer:

```python
from text_game_engine import BackendTextCompletionPort, OllamaBackend

backend = OllamaBackend(model="llama3.1")
completion_port = BackendTextCompletionPort(backend)
```

Pass `completion_port` into `ZorkEmulator(...)` for setup, summarization, map generation, and other emulator-side completions.

## Codex CLI

If you already have the `codex` CLI installed and authenticated, you can use it as a backend too:

```python
from text_game_engine import BackendTextCompletionPort, CodexCLIBackend

backend = CodexCLIBackend(
    cd="/path/to/repo-or-workdir",
    sandbox="read-only",
)
completion_port = BackendTextCompletionPort(backend)
```

The emulator system prompt is mapped onto Codex CLI's separate `user_instructions` config channel, and the task prompt is sent over stdin to `codex exec`.
Point `cd` at a small or empty working directory when possible; Codex performs better as a completion backend when it is not sitting in a large repo tree.

## Other local CLIs

`text-game-engine` now also includes local CLI backends for:
- `gemini`
- `claude`
- `opencode`

These follow the same `BackendTextCompletionPort(...)` adapter pattern as Ollama and Codex.
