# text-game-engine

Standalone text game engine with:

- host-agnostic core turn resolution
- SQL persistence layer with optimistic CAS fencing
- durable inflight turn leases
- timer lifecycle persistence
- outbox event dispatch model

See `SCHEMA.md` for persistence invariants.
