# text-game-engine

Standalone Python package for running the Zork runtime extracted from
[bghira/discord-tron-master](https://github.com/bghira/discord-tron-master), with a built-in SQL persistence layer.

If you're looking for the webUI, that's in [bghira/text-game-webui](https://github.com/bghira/text-game-webui), which relies on this engine.

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

Optional extras:

```bash
pip install -e ".[glm]"
```

Accelerator-targeted installs:

```bash
# NVIDIA / CUDA
pip install -e ".[cuda]"

# AMD / ROCm
pip install -e ".[rocm]" --extra-index-url https://download.pytorch.org/whl/rocm7.1

# Apple silicon / MPS
pip install -e ".[apple]"
```

Notes:
- `cuda` is the convenience extra for hosts that want Torch plus NVIDIA monitoring support.
- `rocm` expects PyTorch ROCm wheels from the PyTorch index, so use the `--extra-index-url` above.
- `apple` is for Apple silicon hosts using the PyTorch MPS backend.

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

## Backend Performance Notes

Best-to-worst so far for Zork-style narrative quality and instruction adherence:

1. `glm-5`
   Regularly follows the full contract best. This is currently the strongest backend for narrative quality and overall reliability.
2. `gpt-5.4`
   Performs well after the newer context-wrapper changes.[1] It also handles adult content, including action and romance, well.
3. `glm-4.7`
   Strong follow-through and good adult-content handling.
4. `glm-4.6`
   Similar to 4.7, but weaker overall.
5. `claude` Sonnet 4.5-4.6
   Improved after the XML/example adapter layer.[2] Still performs poorly for adult situations, especially action and romance.
6. `qwen3.5:27b`
   Testing was limited, but it performed surprisingly well and landed roughly around Sonnet 4.5 quality.
7. `claude` Opus
   Currently the weakest in this stack: high contrivance, surprisingly poor instruction adherence, and would need more backend-specific prompt optimization.

Notes:
- The ordering above is also the current best-to-worst ranking for narrative quality.
- Adult-content support here refers to how well the model sustains consensual erotic, romantic, or violent/adult narrative situations inside the emulator contract, not a general policy statement.
- Backend adapters matter a lot. Claude improved materially once XML wrappers and example wrapping were added; GPT-5.4 improved materially once stronger context wrappers were added.

[1] GPT-5.4 note: newer context wrappers, stricter output-contract transport, and backend-specific prompt shaping improved reliability noticeably.

[2] Claude note: XML section wrapping plus `<example>`-style tool/output examples improved structure following, but did not fully solve weak adult-scene handling.
