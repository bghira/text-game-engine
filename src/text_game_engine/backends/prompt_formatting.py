from __future__ import annotations

import re

from .base import ChatMessage

_PROMPT_SECTION_RE = re.compile(r"^([A-Z][A-Z0-9_]+):(?:\s*(.*))?$")


def build_structured_system_instructions(
    *,
    base_instructions: str,
    system_prompt: str,
) -> str:
    parts: list[str] = []
    base = str(base_instructions or "").strip()
    if base:
        parts.append(base)
    parts.append(
        "<output_contract>\n"
        "- Follow the SYSTEM_INSTRUCTIONS block exactly.\n"
        "- Return only the answer requested by the prompt.\n"
        "- If the prompt requires a strict format, output only that format.\n"
        "</output_contract>"
    )
    parts.append(
        "<verbosity_controls>\n"
        "- Prefer concise, information-dense writing.\n"
        "- Avoid repeating the user's request.\n"
        "</verbosity_controls>"
    )
    parts.append(
        "<tool_boundary>\n"
        "- This call is a text-completion request, not an autonomous coding task.\n"
        "- Do not claim to inspect files, run commands, or use tools unless the prompt explicitly requires it.\n"
        "</tool_boundary>"
    )
    prompt = str(system_prompt or "").strip()
    lower_prompt = prompt.lower()
    if prompt and (
        "json" in lower_prompt
        or "reasoning" in lower_prompt
        or "first key" in lower_prompt
    ):
        parts.append(
            "<structured_output_contract>\n"
            "- If SYSTEM_INSTRUCTIONS requires JSON, output exactly one JSON object and nothing else.\n"
            "- Never omit a required key just because it feels internal.\n"
            "- If SYSTEM_INSTRUCTIONS requires a reasoning field, include reasoning in every final JSON response.\n"
            "- If SYSTEM_INSTRUCTIONS specifies key order, preserve that order in the final JSON.\n"
            "</structured_output_contract>"
        )
    if prompt:
        parts.append(f"<system_instructions>\n{prompt}\n</system_instructions>")
    return "\n\n".join(part.strip() for part in parts if part and part.strip()).strip()


def build_codex_structured_system_instructions(
    *,
    base_instructions: str,
    system_prompt: str,
) -> str:
    parts: list[str] = []
    base = str(base_instructions or "").strip()
    if base:
        parts.append(base)
    parts.append(
        "<output_contract>\n"
        "- Follow the SYSTEM_INSTRUCTIONS block exactly.\n"
        "- Return only the answer requested by the prompt.\n"
        "- If the prompt requires a strict format, output only that format.\n"
        "</output_contract>"
    )
    parts.append(
        "<verbosity_controls>\n"
        "- Prefer concise, information-dense writing.\n"
        "- Avoid repeating the user's request.\n"
        "</verbosity_controls>"
    )
    parts.append(
        "<tool_boundary>\n"
        "- This call is a text-completion request, not an autonomous coding task.\n"
        "- Do not claim to inspect files, run commands, or use tools unless the prompt explicitly requires it.\n"
        "</tool_boundary>"
    )
    prompt = str(system_prompt or "").strip()
    lower_prompt = prompt.lower()
    if prompt and (
        "json" in lower_prompt
        or "reasoning" in lower_prompt
        or "first key" in lower_prompt
    ):
        parts.append(
            "<structured_output_contract>\n"
            "- If SYSTEM_INSTRUCTIONS requires JSON, output exactly one JSON object and nothing else.\n"
            "- Never omit a required key just because it feels internal.\n"
            "- If SYSTEM_INSTRUCTIONS requires a reasoning field, include reasoning in every final JSON response.\n"
            "- If SYSTEM_INSTRUCTIONS specifies key order, preserve that order in the final JSON.\n"
            "</structured_output_contract>"
        )
    if prompt:
        parts.append(
            f"<system_instructions>\n{_wrap_examples_for_claude(prompt)}\n</system_instructions>"
        )
    return "\n\n".join(part.strip() for part in parts if part and part.strip()).strip()


def build_claude_structured_system_instructions(
    *,
    base_instructions: str,
    system_prompt: str,
) -> str:
    parts: list[str] = []
    base = str(base_instructions or "").strip()
    if base:
        parts.append(base)
    parts.append(
        "<output_contract>\n"
        "- Follow the SYSTEM_INSTRUCTIONS block exactly.\n"
        "- Return only the answer requested by the prompt.\n"
        "- If the prompt requires a strict format, output only that format.\n"
        "</output_contract>"
    )
    parts.append(
        "<verbosity_controls>\n"
        "- Prefer concise, information-dense writing.\n"
        "- Avoid repeating the user's request.\n"
        "</verbosity_controls>"
    )
    parts.append(
        "<tool_boundary>\n"
        "- This call is a text-completion request, not an autonomous coding task.\n"
        "- Do not claim to inspect files, run commands, or use tools unless the prompt explicitly requires it.\n"
        "</tool_boundary>"
    )
    prompt = str(system_prompt or "").strip()
    lower_prompt = prompt.lower()
    if prompt and (
        "json" in lower_prompt
        or "reasoning" in lower_prompt
        or "first key" in lower_prompt
    ):
        parts.append(
            "<structured_output_contract>\n"
            "- If SYSTEM_INSTRUCTIONS requires JSON, output exactly one JSON object and nothing else.\n"
            "- Never omit a required key just because it feels internal.\n"
            "- If SYSTEM_INSTRUCTIONS requires a reasoning field, include reasoning in every final JSON response.\n"
            "- If SYSTEM_INSTRUCTIONS specifies key order, preserve that order in the final JSON.\n"
            "</structured_output_contract>"
        )
    if prompt:
        parts.append(
            f"<system_instructions>\n{_wrap_examples_for_claude(prompt)}\n</system_instructions>"
        )
    return "\n\n".join(part.strip() for part in parts if part and part.strip()).strip()


def build_structured_user_prompt(messages: list[ChatMessage]) -> str:
    non_system = [message for message in messages if message.role != "system"]
    if len(non_system) == 1 and non_system[0].role == "user":
        return f"<user_request>\n{non_system[0].content.strip()}\n</user_request>".strip()

    lines = [
        "<conversation>",
        "<response_rule>Respond to the latest user message while using earlier messages only as relevant context.</response_rule>",
    ]
    for message in non_system:
        role = str(message.role).strip().lower() or "user"
        lines.append(f"<{role}_message>")
        lines.append(str(message.content or "").strip())
        lines.append(f"</{role}_message>")
    lines.append("</conversation>")
    return "\n".join(lines).strip()


def build_codex_structured_user_prompt(messages: list[ChatMessage]) -> str:
    non_system = [message for message in messages if message.role != "system"]
    if len(non_system) == 1 and non_system[0].role == "user":
        wrapped = _wrap_examples_for_claude(non_system[0].content)
        return f"<user_request>\n{wrapped}\n</user_request>".strip()
    lines = [
        "<conversation>",
        "<response_rule>Respond to the latest user message while using earlier messages only as relevant context.</response_rule>",
    ]
    for message in non_system:
        role = str(message.role).strip().lower() or "user"
        lines.append(f"<{role}_message>")
        lines.append(_wrap_examples_for_claude(str(message.content or "").strip()))
        lines.append(f"</{role}_message>")
    lines.append("</conversation>")
    return "\n".join(lines).strip()


def build_claude_structured_user_prompt(messages: list[ChatMessage]) -> str:
    non_system = [message for message in messages if message.role != "system"]
    if len(non_system) == 1 and non_system[0].role == "user":
        wrapped = _wrap_prompt_sections_as_xml(non_system[0].content)
        return f"<user_request>\n{wrapped}\n</user_request>".strip()
    return build_structured_user_prompt(messages)


def _wrap_prompt_sections_as_xml(text: str) -> str:
    raw_lines = str(text or "").splitlines()
    if not raw_lines:
        return ""

    blocks: list[tuple[str, list[str]]] = []
    current_tag = "free_text"
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_tag, current_lines
        if current_lines:
            blocks.append((current_tag, current_lines[:]))
        current_tag = "free_text"
        current_lines = []

    for line in raw_lines:
        match = _PROMPT_SECTION_RE.match(line)
        if match:
            flush()
            current_tag = _section_tag_name(match.group(1))
            first_line = str(match.group(2) or "").strip()
            current_lines = [first_line] if first_line else []
            continue
        current_lines.append(line)
    flush()

    if not blocks:
        return str(text or "").strip()

    out: list[str] = []
    for tag, lines in blocks:
        content = _wrap_examples_for_claude("\n".join(lines).strip())
        if not content:
            out.append(f"<{tag} />")
            continue
        out.append(f"<{tag}>")
        out.append(content)
        out.append(f"</{tag}>")
    return "\n".join(out).strip()


def _section_tag_name(key: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(key or "").strip().lower()).strip("_")
    return text or "section"


def _wrap_examples_for_claude(text: str) -> str:
    lines = str(text or "").splitlines()
    if not lines:
        return ""
    out: list[str] = []
    example_buf: list[str] = []

    def flush() -> None:
        if not example_buf:
            return
        content = "\n".join(example_buf).strip()
        if content:
            out.append("<example>")
            out.append(content)
            out.append("</example>")
        example_buf.clear()

    for raw_line in lines:
        line = str(raw_line or "")
        if _is_example_line(line):
            example_buf.append(line)
            continue
        flush()
        out.append(line)
    flush()
    return "\n".join(out).strip()


def _is_example_line(line: str) -> bool:
    text = str(line or "").strip()
    if not text:
        return False
    if text.startswith(("Example:", "Examples:", "NOT:", "Output format:")):
        return True
    if text.startswith("{") and text.endswith("}"):
        return True
    if text.startswith('{"tool_call"'):
        return True
    return False
