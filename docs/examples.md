# Examples

All examples are in `examples/`.

Run from repository root:

```bash
PYTHONPATH=src python examples/<file>.py
```

## Available Example Files

- `examples/minimal_engine_turn.py`
  - Seeds actor/campaign rows, executes one `GameEngine.resolve_turn`, and prints persisted state.
- `examples/zork_emulator_session.py`
  - Creates a standalone `ZorkEmulator` session and runs multiple `play_action` calls.
- `examples/attachment_processing.py`
  - Demonstrates `.txt` extraction plus chunked/condensed summarization with progress callbacks.

## Notes

- Examples use in-memory SQLite for zero external setup.
- Attachment example uses a stub completion port and a deterministic token counter for offline execution.
