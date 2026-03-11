# Migration Checklist

See `PARITY_IMPLEMENTATION_PLAN.md` for phased execution and acceptance criteria.

## Database

1. Apply `migrations/0001_initial.sql`.
2. Verify partial unique timer index exists.
3. Verify campaign row-version CAS updates work.
4. Verify inflight lease uniqueness on `(campaign_id, actor_id)`.
5. Verify outbox idempotency uniqueness on campaign + session scope + event + key.
6. Backfill existing campaigns/players/turns/snapshots.
7. Set `memory_visible_max_turn_id` to latest turn per campaign.
8. Enable outbox worker for `scene_image_requested`, `timer_scheduled`, `memory_prune_requested`.

## Code Migration

1. `ZorkEmulator` compatibility facade implemented in `src/text_game_engine/zork_emulator.py`.
2. Rewind compatibility implemented:
   - message-id target resolution
   - user-message fallback resolution
   - optional channel-scoped turn deletion behavior
3. Attachment utility extracted as reusable feature:
   - `extract_attachment_text(...)` in `src/text_game_engine/core/attachments.py`
   - `AttachmentTextProcessor.summarise_long_text(...)` in `src/text_game_engine/core/attachments.py`
   - `glm_token_count(...)` in `src/text_game_engine/core/tokens.py`
4. Package exports added for attachment utilities in:
   - `src/text_game_engine/core/__init__.py`
   - `src/text_game_engine/__init__.py`
5. Optional tokenizer dependency declared:
   - install with `pip install text-game-engine[glm]`

## Remaining for Downstream (`discord-tron-master`)

1. Replace local `_extract_attachment_text` / `_summarise_long_text` calls with `text_game_engine` utilities.
2. Add a thin adapter that maps current `GPT.turbo_completion(...)` usage to `AttachmentTextProcessor` completion port.
3. Keep existing UX text/progress-message wording unchanged while switching backend utility calls.
4. Run integration tests in Discord flow:
   - setup with `.txt` attachment
   - large-file rejection path
   - multi-chunk summary + condensation path
   - guard-token retry behavior

## Standalone Parity Status (vs original `discord_tron_master/classes/zork_emulator.py`)

1. Class-surface parity is effectively complete:
   - original class methods: 133
   - standalone class methods: 147
   - no missing original method names in standalone; additional standalone methods are adapter/compat helpers.
2. Attachment extraction + GLM-aware chunk/summary pipeline is extracted and wired into setup flow:
   - `_extract_attachment_text(...)` and `_summarise_long_text(...)` exist on `ZorkEmulator` and use shared `core/attachments.py`.
3. Prompt/runtime helpers are ported and prompt constants are restored to original-equivalent text:
   - JSON cleanup/parsing, story-context/model-state builders, inventory sanitization, character budget-fit helpers.
   - `build_prompt(...)` includes `CURRENT_GAME_TIME`, `SPEED_MULTIPLIER`, and `CALENDAR` user-prompt fields and appends calendar/roster tool prompt sections.
   - missing `game_time` is seeded to upstream default (`Day 1, Morning`) during prompt build.
   - latest upstream prompt deltas are mirrored:
     - `SYSTEM_PROMPT` includes nested `WORLD_STATE` structure requirement guidance.
     - `MEMORY_TOOL_PROMPT` includes aggressive/most-turns memory-search guidance.
   - latest upstream prompt budgets are mirrored:
     - `MAX_SUMMARY_CHARS=10000`, `MAX_STATE_CHARS=10000`, `MAX_NARRATION_CHARS=23500`, `MAX_CHARACTERS_CHARS=8000`.
   - setup variant-generation parity is mirrored:
     - retry example text (`retry: make it darker`), retry fallback prompt wording, setup-variant diagnostics (`_zork_log`), and IMDB `synopsis`-first fallback shaping.
   - calendar lifecycle/removal parity is mirrored:
     - `CALENDAR_TOOL_PROMPT` includes explicit `calendar_update.remove` rules (remove only when resolved by player action; overdue events remain).
     - `_apply_calendar_update(...)` no longer auto-prunes overdue events and keeps latest re-add per event name.
4. Timer/runtime compatibility is ported:
   - schedule/cancel/expire/emit flow with timer-effect ports and DB timer binding helpers.
   - includes recent-player-action race guard before firing timed events.
   - timed events resolve as narrator-only turns (no synthetic player turn).
5. `play_action(...)` output/turn parity:
   - narration is decorated with authoritative inventory line (`Inventory: ...` or `Inventory: empty`).
   - model-authored debug footers (`--- ... XP ...`) are stripped from narration before persistence/decorating.
   - timer countdown line is appended when a timer is scheduled.
   - timer-line epoch timestamps are generated from UNIX time (`time.time()`) and speed-multiplier scaling is applied before scheduling/decorating.
   - context-mode command shortcuts include `calendar`/`cal`/`events` and `roster`/`characters`/`npcs`.
   - OOC actions (`[OOC ...]`) do not create player turns.
   - timer interruption path records the interruption narrator note.
   - `give_item` transfer semantics match upstream post-update behavior, including explicit transfer support and fallback inference/refusal heuristics.
   - new character portrait parity is implemented (`_compose_character_portrait_prompt`, `_enqueue_character_portrait`, `record_character_portrait_url`) with auto-enqueue for newly introduced NPCs that include `appearance`.
   - attachment helper compatibility methods are present (`_summarise_chunk`, `_condense`) with upstream-equivalent guard-token retry/fallback behavior.
   - context-mode onboarding parity includes deterministic non-thread party-choice flow (`main party` / `new path`).
   - context-mode `look` and `inventory` fast paths mirror upstream short-circuit behavior.
6. Rewind cleanup parity:
   - full rewind and channel-scoped rewind delete embeddings for pruned turns in the same operation.
7. Runtime diagnostics parity:
   - `_zork_log(...)` appends timestamped sections to `zork.log` in the process working directory.
8. Avatar/scene storage and lifecycle helpers are ported:
   - room-scene URL persistence, pending-avatar accept/decline, scene/avatar enqueue adapters via `MediaGenerationPort`.
9. Legacy invocation shapes are accepted for key call sites:
   - `begin_turn(ctx, ...)`, `play_action(ctx, action, campaign_id=...)`
   - `start_campaign_setup(campaign, raw_name, attachment_summary=...)`
   - `handle_setup_message(ctx, content, campaign, ...)`
10. IMDB behavior parity:
   - standalone supports port-driven lookups plus built-in progressive IMDB fallback/enrichment when no port is bound.

## Residual Behavioral Differences

1. Media generation and IMDB network behavior are now port-driven (`MediaGenerationPort`, `IMDBLookupPort`) instead of direct Discord/HTTP calls.
2. Processing reaction helpers are host-object best-effort utilities; exact Discord permission/race semantics remain downstream-specific.
