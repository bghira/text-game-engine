# SDK Reference

This document lists the primary public APIs for `text-game-engine`.

## Top-Level Exports

```python
from text_game_engine import (
    GameEngine,
    ZorkEmulator,
    AttachmentProcessingConfig,
    AttachmentTextProcessor,
    extract_attachment_text,
    glm_token_count,
    TextCompletionPort,
    MemorySearchPort,
    TimerEffectsPort,
    IMDBLookupPort,
    MediaGenerationPort,
)
```

## Core Engine

Source: `src/text_game_engine/core/engine.py`

```python
GameEngine(
    uow_factory,
    llm,
    actor_resolver=None,
    clock=None,
    lease_ttl_seconds=90,
    max_conflict_retries=1,
)
```

Methods:

- `await resolve_turn(ResolveTurnInput, before_phase_c=None) -> ResolveTurnResult`
- `rewind_to_turn(campaign_id: str, target_turn_id: int) -> RewindResult`
- `filter_memory_hits_by_visibility(campaign_id: str, hits: list[dict]) -> list[dict]`

Important input/output dataclasses:

- `ResolveTurnInput`
- `ResolveTurnResult`
- `LLMTurnOutput`
- `TimerInstruction`
- `GiveItemInstruction`
- `TurnContext`
- `RewindResult`

Source: `src/text_game_engine/core/types.py`

## Zork Emulator Facade

Source: `src/text_game_engine/zork_emulator.py`

```python
ZorkEmulator(
    game_engine,
    session_factory,
    completion_port=None,
    map_completion_port=None,
    timer_effects_port=None,
    memory_port=None,
    imdb_port=None,
    media_port=None,
)
```

Common entry points:

- `get_or_create_actor(actor_id, display_name=None) -> Actor`
- `get_or_create_campaign(namespace, name, created_by_actor_id, campaign_id=None) -> Campaign`
- `get_or_create_player(campaign_id, actor_id) -> Player`
- `await play_action(campaign_or_ctx=None, actor_id=None, action=None, session_id=None, manage_claim=True, *, command_prefix="!", campaign_id=None) -> str | None`
- `await begin_turn(campaign_id_or_ctx, actor_id=None, *, command_prefix="!") -> tuple[str | None, str | None]`
- `end_turn(campaign_id, actor_id) -> None`
- `execute_rewind(campaign_id, target_discord_message_id, channel_id=None) -> tuple[int, int] | None`
- `register_timer_message(campaign_id, message_id, channel_id=None, thread_id=None) -> bool`
- `await start_campaign_setup(...) -> str`
- `await handle_setup_message(...) -> str`
- `build_prompt(campaign, player, action, recent_turns, campaign_state=None, *, party_snapshot=None) -> str`
- `await generate_map(campaign_or_ctx, actor_id=None, command_prefix="!") -> str`

Compatibility notes:

- Supports both direct-ID calls and Discord-like context objects for several methods.
- Keeps return-shape compatibility with upstream `discord-tron-master` behavior.

## Attachment Utilities

Source: `src/text_game_engine/core/attachments.py`

- `await extract_attachment_text(attachments, *, config=None, logger=None) -> str | None`
- `AttachmentTextProcessor(completion, token_count=glm_token_count, config=None, logger=None)`
- `await AttachmentTextProcessor.summarise_long_text(text, *, progress=None) -> str`
- `AttachmentProcessingConfig(...)`

Tokenizer utility:

- `glm_token_count(text: str) -> int` in `src/text_game_engine/core/tokens.py`
- Uses `zai-org/GLM-5` when `transformers` is installed, otherwise fallback estimate.

## Port Interfaces

Source: `src/text_game_engine/core/emulator_ports.py`

- `TextCompletionPort`
- `TimerEffectsPort`
- `MemorySearchPort`
- `IMDBLookupPort`
- `MediaGenerationPort`

Source: `src/text_game_engine/core/ports.py`

- `LLMPort` for `GameEngine`
- `ActorResolverPort` for give-item mention resolution
