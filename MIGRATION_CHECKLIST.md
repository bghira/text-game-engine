# Migration Checklist

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
