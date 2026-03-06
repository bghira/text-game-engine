# AGENTS.md - text-game-engine

Operating notes for contributors and coding agents working in this repository.

## Mission

- Provide the standalone engine/runtime extracted from `discord-tron-master`.
- Keep runtime behavior, persistence, and tool-facing docs aligned.
- Treat `docs/` as canonical human-maintained documentation.

## Important Docs

- `README.md`: install, examples, and top-level entry points
- `docs/sdk.md`: host-facing integration surface
- `docs/persistence.md`: storage model and invariants
- `docs/examples.md`: runnable examples map
- `docs/source-material.md`: canonical authoring rules for `!zork source-material`

## Source-Material Rule

If you change source-material ingestion, format detection, `source_browse`,
`memory_search` source behavior, or attachment authoring expectations, update
`docs/source-material.md` in the same change.

## Repo Focus

- `src/text_game_engine/zork_emulator.py`: main runtime surface
- `src/text_game_engine/core/`: attachment handling, memory, engine primitives
- `src/text_game_engine/persistence/`: SQLAlchemy persistence layer
- `examples/`: concrete integration examples
- `tests/`: compatibility and parity coverage
