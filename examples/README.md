# Examples

Run from repository root:

```bash
PYTHONPATH=src python examples/minimal_engine_turn.py
PYTHONPATH=src python examples/zork_emulator_session.py
PYTHONPATH=src python examples/attachment_processing.py
```

Files:

- `minimal_engine_turn.py`: direct `GameEngine` turn resolution with persisted state inspection.
- `zork_emulator_session.py`: standalone `ZorkEmulator` flow (`play_action` + inventory-decorated narration).
- `attachment_processing.py`: attachment text extraction and chunked summarization.
