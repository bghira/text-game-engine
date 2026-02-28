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
pip install text-game-engine
```

Optional GLM tokenizer support:

```bash
pip install text-game-engine[glm]
```

## Documentation

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
