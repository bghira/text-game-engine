# Persistence

`text-game-engine` includes a first-class SQL persistence layer.

## What Is Included

Source directory: `src/text_game_engine/persistence/sqlalchemy`

- Engine/session builders:
  - `build_engine(url)`
  - `build_session_factory(engine)`
  - `create_schema(engine)`
- Unit of work:
  - `SQLAlchemyUnitOfWork(session_factory)`
- ORM models:
  - `Campaign`, `Session`, `Actor`, `ActorExternalRef`, `Player`
  - `Turn`, `Snapshot`, `Timer`, `InflightTurn`
  - `MediaRef`, `Embedding`, `OutboxEvent`
- Repository implementations for campaigns, players, turns, snapshots, timers, inflight claims, and outbox.

## Schema and Migrations

- SQL migration file: `migrations/0001_initial.sql`
- Invariant spec: `SCHEMA.md`

Two bootstrap options:

1. Apply SQL migrations (recommended for production).
2. Call `create_schema(engine)` for local/dev bootstrap.

## Minimal Setup

```python
from text_game_engine.persistence.sqlalchemy import (
    SQLAlchemyUnitOfWork,
    build_engine,
    build_session_factory,
    create_schema,
)

engine = build_engine("sqlite+pysqlite:///game.db")
create_schema(engine)
session_factory = build_session_factory(engine)

def uow_factory():
    return SQLAlchemyUnitOfWork(session_factory)
```

## Concurrency and Consistency Model

- Campaign writes are CAS-protected by `tge_campaigns.row_version`.
- Inflight turn lock is unique on `(campaign_id, actor_id)`.
- Active timer is unique per campaign.
- Outbox idempotency is unique per campaign + session scope + event + key.
- Rewind sets `memory_visible_max_turn_id`; memory queries must filter by it.

Details are documented in `SCHEMA.md`.

## Custom DB Backends

If you need a non-SQLAlchemy backend, implement the protocols in:

- `src/text_game_engine/persistence/interfaces.py`

`GameEngine` only depends on the unit-of-work and repository protocol surface.
