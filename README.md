# text-game-engine

Standalone text game engine with:

- host-agnostic core turn resolution
- SQL persistence layer with optimistic CAS fencing
- durable inflight turn leases
- timer lifecycle persistence
- outbox event dispatch model
- attachment text utility with GLM-5 token-aware chunking/summarization

See `SCHEMA.md` for persistence invariants.

## Attachment Utility

Use `AttachmentTextProcessor` for token-aware chunking/summarization and
`extract_attachment_text` for `.txt` attachment decode/size handling.

`glm_token_count` is exposed as a utility and lazy-loads the GLM-5 tokenizer.
Install optional tokenizer dependency with:

`pip install text-game-engine[glm]`
