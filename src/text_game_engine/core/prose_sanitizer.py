"""Strip recurrent LLM prose tics from generated narration.

The LLM repeatedly produces these patterns even with explicit prompt-level
bans, so they are stripped post-generation as a hard guarantee. Only
unquoted spans are touched — anything inside ``"..."`` is preserved verbatim
since that is character-voice dialogue and should never be mutated.

Patterns stripped:
1. ``A beat.`` / ``A beat of <word>.`` — lazy temporal markers.
2. ``Not <word>. <Word>.`` paired-fragment redefinitions ("Not pity. Recognition.").
3. Standalone single-word emotional-tag sentences from a blocklist
   ("Filing." / "Processing." / "Staying." / etc.).
4. Stray closing TTS tags such as ``</quiet>`` or ``[/emotive]``.
"""

from __future__ import annotations

import re

_BANNED_TAG_WORDS: frozenset[str] = frozenset(
    {
        # State/posture labels
        "absent",
        "certain",
        "connected",
        "direct",
        "filing",
        "flat",
        "holding",
        "naming",
        "noting",
        "present",
        "processing",
        "recognition",
        "recognizing",
        "registering",
        "silent",
        "soft",
        "staying",
        "steady",
        "still",
        "tracking",
        "unhurried",
        "watching",
        "wordless",
        "wrecked",
        # Paired-fragment companions (often the X in "Not X.")
        "accusation",
        "defense",
        "dismissal",
        "mocking",
        "pity",
        "question",
        "teasing",
        "testing",
    }
)

_A_BEAT_RE = re.compile(
    r"(?<![\w'])A beat(?:\s+of\s+\w+)?\.(?=\s|$|[\"')])",
)
_NOT_X_Y_RE = re.compile(
    r"(?<![\w'])Not\s+[A-Za-z]+\.\s+[A-Z][A-Za-z]+\.(?=\s|$|[\"')])",
)
_TAG_FRAGMENT_RE = re.compile(
    r"(?<![\w'])([A-Z][a-z]+)\.(?=\s|$|[\"')])",
)
_INVALID_TTS_CLOSING_TAG_RE = re.compile(r"</\s*[A-Za-z][^>\n]{0,80}>")
_INVALID_TTS_BRACKET_CLOSING_RE = re.compile(r"\[/\s*emotive\s*\]", re.IGNORECASE)
_EMOTIVE_MARKER_RE = re.compile(r"\[emotive:[^\]\r\n]{1,80}\]", re.IGNORECASE)


def _drop_if_banned(match: re.Match[str]) -> str:
    return "" if match.group(1).lower() in _BANNED_TAG_WORDS else match.group(0)


def strip_invalid_tts_closing_tags(text: object) -> str:
    """Remove closing tags emitted for standalone TTS/emotive markers."""
    if not isinstance(text, str) or not text:
        return text if isinstance(text, str) else ""
    text = _INVALID_TTS_CLOSING_TAG_RE.sub("", text)
    return _INVALID_TTS_BRACKET_CLOSING_RE.sub("", text)


def normalize_leading_emotive_dialogue(text: object) -> str:
    """Move unquoted emotive markers inside a missing opening dialogue quote."""
    if not isinstance(text, str) or not text:
        return text if isinstance(text, str) else ""

    def _is_escaped(pos: int) -> bool:
        backslashes = 0
        i = pos - 1
        while i >= 0 and text[i] == "\\":
            backslashes += 1
            i -= 1
        return backslashes % 2 == 1

    def _has_closing_quote_same_line(start: int) -> bool:
        i = start
        while i < len(text):
            ch = text[i]
            if ch in "\r\n":
                return False
            if ch == '"' and not _is_escaped(i):
                return True
            i += 1
        return False

    out: list[str] = []
    in_quote = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '"' and not _is_escaped(i):
            in_quote = not in_quote
            out.append(ch)
            i += 1
            continue
        marker_match = _EMOTIVE_MARKER_RE.match(text, i)
        if marker_match and not in_quote and _has_closing_quote_same_line(marker_match.end()):
            out.append('"')
            out.append(marker_match.group(0))
            in_quote = True
            i = marker_match.end()
            continue
        out.append(ch)
        i += 1

    return "".join(out)


def _sanitize_unquoted(text: str) -> str:
    if not text:
        return text
    text = _A_BEAT_RE.sub("", text)
    text = _NOT_X_Y_RE.sub("", text)
    text = _TAG_FRAGMENT_RE.sub(_drop_if_banned, text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def sanitize_prose(text: object) -> str:
    """Strip banned LLM-tic patterns outside of quoted dialogue.

    Quoted spans (text inside ``"..."``) are preserved verbatim.
    Returns the input unchanged if it isn't a string.
    """
    if not isinstance(text, str) or not text:
        return text if isinstance(text, str) else ""
    text = normalize_leading_emotive_dialogue(text)
    text = strip_invalid_tts_closing_tags(text)
    if '"' not in text:
        out = _sanitize_unquoted(text)
    else:
        pieces: list[str] = []
        n = len(text)
        last = 0
        in_quote = False
        i = 0
        while i < n:
            if text[i] == '"':
                if not in_quote:
                    pieces.append(_sanitize_unquoted(text[last:i]))
                    last = i
                    in_quote = True
                else:
                    pieces.append(text[last : i + 1])
                    last = i + 1
                    in_quote = False
            i += 1
        if in_quote:
            pieces.append(text[last:])
        else:
            pieces.append(_sanitize_unquoted(text[last:]))
        out = "".join(pieces)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r" +\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def sanitize_scene_output(scene_output: object) -> object:
    """Apply :func:`sanitize_prose` to every beat's ``text`` field in-place."""
    if not isinstance(scene_output, dict):
        return scene_output
    beats = scene_output.get("beats")
    if not isinstance(beats, list):
        return scene_output
    for beat in beats:
        if not isinstance(beat, dict):
            continue
        beat["text"] = sanitize_prose(beat.get("text"))
    return scene_output
