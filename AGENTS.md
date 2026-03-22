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

## Change Discipline

- When you change a method signature, helper contract, or data shape, update all
  call sites in the repo, not just the ones in the code path you are actively
  editing.
- Prefer repo-wide searches before finishing a refactor: find every caller,
  serializer, normalizer, prompt builder, and test that depends on the old
  contract.
- If a helper is used both from model-facing flows and internal maintenance
  flows, verify both paths still receive the new required inputs.
- When truncating or normalizing structured text, preserve critical suffixes or
  metadata fields instead of assuming the important part is always at the front.

## Repo Focus

- `src/text_game_engine/zork_emulator.py`: main runtime surface
- `src/text_game_engine/core/`: attachment handling, memory, engine primitives
- `src/text_game_engine/persistence/`: SQLAlchemy persistence layer
- `examples/`: concrete integration examples
- `tests/`: compatibility and parity coverage
