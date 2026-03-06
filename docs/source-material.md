# Source Material Authoring

This document defines how to format `.txt` attachments for `!zork source-material`
and any standalone host built on `text-game-engine`.

## Why Format Matters

Source-material ingestion now uses staged format detection.

The engine samples the first chunk of the attachment and classifies the document as
one dominant format:

- `story`
- `rulebook`
- `generic`

That classification controls how the file is stored and how the model is expected to
use it later.

Practical rule: one attachment should contain one dominant format. Do not mix
story/setup prose and rulebook facts in the same file if you want reliable results.

## First-Chunk Rule

Format detection is driven by the first chunk of the file, not by a later appendix.

That means:

- Start the file with content that clearly matches the format you want.
- Do not put a long prose preamble in front of a rulebook.
- Do not put pages of rulebook keys in front of a story/setup prompt.
- If you have both kinds of material, split them into separate attachments.

## Supported Formats

## `story`

Use this for narrative or setup-prompt style material:

- prose scenes
- scripted dialogue
- sectioned campaign briefs
- story-generator prompts
- encounter/style guides written as paragraphs or bullet lists

Typical shape:

```text
HAPPY TAVERN 2.0 - STORY GENERATOR PROMPT

Genre: sex-comedy
Format: endless sitcom

PLAYER CHARACTER:
...

SETTING:
...
```

Guidelines:

- Write in natural prose.
- Keep sections grouped by topic.
- This format is good for scenario framing, tone, opening-state setup, and story
  generation guidance.
- If the content is mostly freeform prompt text, it may be treated more like setup
  guidance than a browsable canon index.

## `rulebook`

Use this for open-set canon that the model should browse and search as compact facts.

This is the preferred format for:

- character bibles
- location facts
- setting rules
- tone rules
- consent rules
- reusable canon details

Required shape:

- one fact per line
- each line starts with a stable key
- use `KEY: value`

Recommended key style:

- uppercase
- hyphen-separated
- descriptive but short

Example:

```text
TONE: The Happy Tavern is an adult comedy. The default mood is warm, chaotic, and funny.
TONE-RULES: No espionage, no political thrillers, no criminal plots, no trauma arcs, no crisis-driven storylines.
CHAR-MONET: Monet, mid-thirties, bar and room manager. Hoarse voice. Ruthlessly competent.
SETTING-ROOM2: Room Two has a king bed, a ceiling mirror, and a bluetooth speaker Carlo added.
RED-ROOM-AND-TAVERN: The Committee exists in the Happy Tavern ecosystem the way a benign tumor exists in a body.
```

Guidelines:

- Keep each line self-contained.
- Do not wrap one fact across multiple lines unless absolutely necessary.
- Prefer one key per fact, not one key per paragraph block.
- If one topic has multiple facts, split them into multiple keyed lines when practical.
- Use stable keys so later wildcard browsing is predictable.

Good:

```text
CHAR-SOL: Solomon "Sol" Kade, bouncer, 6'5", gentle until he isn't.
CHAR-SOL-DIALOGUE: Sol speaks softly and simply.
CHAR-SOL-SHIRTS: Sol names every Hawaiian shirt.
```

Less good:

```text
SOL-STUFF: Sol is tall and nice and here are seven unrelated paragraphs about him...
```

## `generic`

This is the fallback for mixed notes, dumps, or files that do not clearly behave like
story/setup prose or a `KEY: value` rulebook.

Typical examples:

- brainstorming notes
- copied fragments from multiple formats
- half-structured lore dumps
- raw memory exports

Generic files are still useful, but they are not the best format when you want clean
canon lookup behavior.

If something should be browsed as canon later, rewrite it as a rulebook file.

## Recommended Authoring Pattern

For campaigns like Happy Tavern, use separate attachments:

1. A story/setup prompt file for the generator-facing framing.
2. A rulebook file for canonical searchable facts.

That separation gives the best results:

- setup/story prompt shapes the campaign premise and tone
- rulebook material becomes concise source memory for `source_browse` and
  `memory_search`

## Happy Tavern Example Split

Use a story/setup attachment for material like:

- `HAPPY TAVERN 2.0 - STORY GENERATOR PROMPT`
- `PLAYER CHARACTER`
- `TONIGHT'S STATE`
- `EPISODE STRUCTURE`
- `OPENING SCENE`

Use a rulebook attachment for material like:

- `TONE: ...`
- `TONE-RULES: ...`
- `CHAR-MONET: ...`
- `SETTING-ROOM2: ...`
- `BLUE-ROOM-RULES: ...`
- `RED-ROOM-AND-TAVERN: ...`

## Authoring Checklist

- One attachment, one dominant format.
- Make the first chunk representative of the whole file.
- Use separate files for story/setup and rulebook canon.
- Use `KEY: value` lines for anything you want treated as concise searchable canon.
- Keep rulebook keys stable and predictable.
- Avoid mixing long prose paragraphs into the top of a rulebook file.
