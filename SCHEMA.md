# Schema Invariants

## Core invariants

1. `tge_campaigns.row_version` is the optimistic CAS fence for turn commits.
2. `tge_inflight_turns(campaign_id, actor_id)` is unique (one in-flight turn per actor per campaign).
3. Active timers are unique per campaign (`status IN ('scheduled_unbound','scheduled_bound')`).
4. Outbox idempotency is scoped by campaign + session scope + event type + idempotency key.
5. `memory_visible_max_turn_id` is the hard visibility watermark for memory reads after rewind.

## Turn transaction model

1. Phase A: short tx to acquire lease + load context.
2. Phase B: no tx (LLM/tool work).
3. Phase C: short tx with lease token validation and campaign CAS update.
4. Any CAS mismatch rolls back all phase C writes.

## Timer state machine

- `scheduled_unbound` -> `scheduled_bound` via attach message.
- `scheduled_unbound|scheduled_bound` -> `cancelled`.
- `scheduled_unbound|scheduled_bound` -> `expired`.
- `expired` -> `consumed`.

All transitions are idempotent conditional updates.

## Rewind and memory

On rewind:

1. restore snapshot state
2. delete turns/snapshots after target turn
3. set `memory_visible_max_turn_id = target_turn_id`
4. enqueue `memory_prune_requested`

Memory queries must always filter by `turn_id <= memory_visible_max_turn_id`.
