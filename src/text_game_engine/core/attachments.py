from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, Protocol, Sequence

from .tokens import glm_token_count


class AttachmentLike(Protocol):
    filename: str | None
    size: int | None

    async def read(self) -> bytes:
        ...


class TextCompletionPort(Protocol):
    async def complete(
        self,
        system_prompt: str,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> str | None:
        ...


ProgressCallback = Callable[[str], Awaitable[None] | None]


@dataclass(frozen=True)
class AttachmentProcessingConfig:
    attachment_max_bytes: int = 500_000
    attachment_chunk_tokens: int = 50_000
    attachment_model_ctx_tokens: int = 200_000
    attachment_prompt_overhead_tokens: int = 6_000
    attachment_response_reserve_tokens: int = 90_000
    attachment_summary_max_tokens: int = 90_000
    attachment_max_parallel: int = 4
    attachment_guard_token: str = "--COMPLETED SUMMARY--"
    attachment_max_chunks: int = 8


async def extract_attachment_text(
    attachments: Sequence[AttachmentLike] | None,
    *,
    config: AttachmentProcessingConfig | None = None,
    logger: logging.Logger | None = None,
) -> Optional[str]:
    """Return text from first ``.txt`` attachment, error string, or ``None``.

    Returns ``ERROR:File too large (...)`` on size violation to preserve
    existing call-site behavior from the original Zork emulator.
    """
    cfg = config or AttachmentProcessingConfig()
    log = logger or logging.getLogger(__name__)
    if not attachments:
        return None

    txt_att = None
    for att in attachments:
        if att.filename and att.filename.lower().endswith(".txt"):
            txt_att = att
            break
    if txt_att is None:
        return None

    if txt_att.size and txt_att.size > cfg.attachment_max_bytes:
        size_kb = txt_att.size // 1024
        limit_kb = cfg.attachment_max_bytes // 1024
        return f"ERROR:File too large ({size_kb}KB, limit {limit_kb}KB)"

    try:
        raw = await txt_att.read()
    except Exception as exc:
        log.warning("Attachment read failed: %s", exc)
        return "ERROR:Could not read attached `.txt` file. Please re-upload and try again."
    if not raw:
        return "ERROR:Attached `.txt` file is empty."

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    text = text.strip()
    return text if text else "ERROR:Attached `.txt` file is empty."


async def extract_attachment_texts(
    attachments: Sequence[AttachmentLike] | None,
    *,
    config: AttachmentProcessingConfig | None = None,
    logger: logging.Logger | None = None,
) -> list[tuple[AttachmentLike, str | None]]:
    """Return text/errored status for each ``.txt`` attachment, preserving order."""
    cfg = config or AttachmentProcessingConfig()
    log = logger or logging.getLogger(__name__)
    if not attachments:
        return []

    out: list[tuple[AttachmentLike, str | None]] = []
    for att in attachments:
        if not getattr(att, "filename", None):
            continue
        filename = str(att.filename or "").lower()
        if not filename.endswith(".txt"):
            continue

        if att.size and att.size > cfg.attachment_max_bytes:
            size_kb = att.size // 1024
            limit_kb = cfg.attachment_max_bytes // 1024
            out.append(
                (
                    att,
                    f"ERROR:File too large ({size_kb}KB, limit {limit_kb}KB)",
                )
            )
            continue
        try:
            raw = await att.read()
        except Exception as exc:
            log.warning("Attachment read failed: %s", exc)
            out.append(
                (
                    att,
                    "ERROR:Could not read attached `.txt` file. Please re-upload and try again.",
                )
            )
            continue

        if not raw:
            out.append((att, "ERROR:Attached `.txt` file is empty."))
            continue

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
        text = text.strip()
        out.append((att, text if text else "ERROR:Attached `.txt` file is empty."))

    return out


class AttachmentTextProcessor:
    """Token-aware text chunking/summarization utility.

    Mirrors the original Zork attachment summarization flow and constants.
    """

    def __init__(
        self,
        completion: TextCompletionPort,
        *,
        token_count: Callable[[str], int] = glm_token_count,
        config: AttachmentProcessingConfig | None = None,
        logger: logging.Logger | None = None,
    ):
        self._completion = completion
        self._token_count = token_count
        self._config = config or AttachmentProcessingConfig()
        self._logger = logger or logging.getLogger(__name__)

    @staticmethod
    def _is_header_line(line: str) -> bool:
        stripped = str(line or "").strip()
        if not stripped:
            return False
        if re.match(r"^#{1,6}\s+\S", stripped):
            return True
        return bool(re.match(r"^[A-Z0-9][A-Z0-9 _/\-()&'.]{1,80}:\s*$", stripped))

    @staticmethod
    def _is_indented_line(line: str) -> bool:
        raw = str(line or "").rstrip("\n")
        return bool(re.match(r"^(?:\t+|\s{4,})\S", raw))

    def _split_structural_blocks(self, text: str) -> list[str]:
        clean = str(text or "").strip()
        if not clean:
            return []
        lines = clean.splitlines()
        blocks: list[str] = []
        current: list[str] = []

        def flush_current() -> None:
            if not current:
                return
            block = "\n".join(current).strip()
            current.clear()
            if block:
                blocks.append(block)

        for raw_line in lines:
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped:
                flush_current()
                continue
            if self._is_header_line(line):
                flush_current()
                blocks.append(stripped)
                continue
            if self._is_indented_line(raw_line):
                flush_current()
                blocks.append(line)
                continue
            current.append(line)
        flush_current()
        return blocks or [clean]

    def _hard_wrap_text(self, text: str, target_tokens: int) -> list[str]:
        clean = str(text or "").strip()
        if not clean:
            return []
        chars_per_tok = max(len(clean) / max(self._token_count(clean), 1), 1.0)
        target_chars = max(512, int(target_tokens * chars_per_tok))
        out: list[str] = []
        start = 0
        length = len(clean)
        while start < length:
            end = min(length, start + target_chars)
            if end < length:
                window = clean[start:end]
                breakpoints = [
                    window.rfind("\n\n"),
                    window.rfind("\n"),
                    window.rfind("    "),
                    window.rfind("\t"),
                    window.rfind(" "),
                ]
                best_break = max(breakpoints)
                if best_break > max(256, target_chars // 3):
                    end = start + best_break
            piece = clean[start:end].strip()
            if piece:
                out.append(piece)
            start = max(end, start + 1)
            while start < length and clean[start].isspace():
                start += 1
        return out

    def _pack_token_bounded_chunks(
        self,
        segments: Sequence[str],
        *,
        target_chunk_tokens: int,
    ) -> list[str]:
        packed: list[str] = []
        current: list[str] = []

        def flush_current() -> None:
            if not current:
                return
            block = "\n\n".join(current).strip()
            current.clear()
            if block:
                packed.append(block)

        for segment in segments:
            piece = str(segment or "").strip()
            if not piece:
                continue
            piece_tokens = self._token_count(piece)
            if piece_tokens > target_chunk_tokens:
                flush_current()
                packed.extend(self._hard_wrap_text(piece, target_chunk_tokens))
                continue
            if not current:
                current.append(piece)
                continue
            candidate = "\n\n".join([*current, piece]).strip()
            if self._token_count(candidate) <= target_chunk_tokens:
                current.append(piece)
                continue
            flush_current()
            current.append(piece)
        flush_current()
        return packed

    def _chunk_text(self, text: str) -> tuple[list[str], int, int, float, int]:
        clean = str(text or "").strip()
        if not clean:
            return [], 0, 0, 0.0, 0
        cfg = self._config
        total_tokens = self._token_count(clean)
        target_chunk_tokens = max(
            max(1, int(cfg.attachment_chunk_tokens)),
            total_tokens // max(1, int(cfg.attachment_max_chunks)),
        )
        chars_per_tok = len(clean) / max(total_tokens, 1)
        chunk_char_target = max(1, int(target_chunk_tokens * chars_per_tok))
        structural_blocks = self._split_structural_blocks(clean)
        chunks = self._pack_token_bounded_chunks(
            structural_blocks,
            target_chunk_tokens=target_chunk_tokens,
        )
        if not chunks:
            chunks = self._hard_wrap_text(clean, target_chunk_tokens)
        return chunks, total_tokens, target_chunk_tokens, chars_per_tok, chunk_char_target

    def _fallback_summary(self, text: str) -> str:
        clean = str(text or "").strip()
        if not clean:
            return ""
        chunks, _, _, _, _ = self._chunk_text(clean)
        if not chunks:
            return ""
        selected = chunks[:6]
        lines = ["Fallback extraction from uploaded source text (automated summary failed):"]
        for idx, chunk in enumerate(selected, start=1):
            snippet = " ".join(str(chunk or "").split())
            if len(snippet) > 1200:
                snippet = snippet[:1200].rsplit(" ", 1)[0].strip() + "..."
            lines.append(f"[Excerpt {idx}/{len(selected)}] {snippet}")
        result = "\n\n".join(lines).strip()
        if len(result) > 9000:
            result = result[:9000].rsplit(" ", 1)[0].strip() + "..."
        return result

    async def summarise_long_text(
        self,
        text: str,
        *,
        progress: ProgressCallback | None = None,
        summary_instructions: str | None = None,
    ) -> str:
        cfg = self._config
        budget_tokens = (
            cfg.attachment_model_ctx_tokens
            - cfg.attachment_prompt_overhead_tokens
            - cfg.attachment_response_reserve_tokens
        )
        min_chunk_tokens = cfg.attachment_chunk_tokens
        max_parallel = cfg.attachment_max_parallel
        guard = cfg.attachment_guard_token

        total_tokens = self._token_count(text)
        chunks, _, target_chunk_tokens, chars_per_tok, chunk_char_target = self._chunk_text(
            text
        )

        if not chunks:
            return ""

        if len(chunks) == 1 and self._token_count(chunks[0]) <= budget_tokens:
            return chunks[0]

        total = len(chunks)
        self._logger.info(
            "ATTACHMENT SUMMARISE text_len=%s total_tokens=%s chunk_char_target=%s total_chunks=%s",
            len(text),
            total_tokens,
            chunk_char_target,
            total,
        )
        await self._notify(progress, f"Summarising uploaded file... [0/{total}]")

        summary_max_tokens = min(
            cfg.attachment_summary_max_tokens,
            max(8_000, target_chunk_tokens // 2),
        )
        instruction_text = " ".join(str(summary_instructions or "").strip().split())[:600]
        summarise_system = (
            "Summarise the following text passage for a text-adventure campaign. "
            "Preserve all character names, plot points, locations, and key events. "
            f"Be detailed but concise. End with the exact line: {guard}"
        )
        if instruction_text:
            summarise_system = (
                f"{summarise_system}\n"
                "Additional user instruction for this summary:\n"
                f"{instruction_text}"
            )

        async def _summarise_chunk(chunk_text: str) -> str:
            try:
                result = await self._completion.complete(
                    summarise_system,
                    chunk_text,
                    max_tokens=summary_max_tokens,
                    temperature=0.3,
                )
                result = (result or "").strip()
                if guard not in result:
                    self._logger.warning("Guard token missing, retrying chunk")
                    result = await self._completion.complete(
                        summarise_system,
                        chunk_text,
                        max_tokens=summary_max_tokens,
                        temperature=0.3,
                    )
                    result = (result or "").strip()
                    if guard not in result:
                        self._logger.warning("Guard token still missing, accepting as-is")
                return result.replace(guard, "").strip()
            except Exception as exc:
                self._logger.warning("Chunk summarisation failed: %s", exc)
                return ""

        summaries: list[str] = []
        processed = 0
        for batch_start in range(0, total, max_parallel):
            batch = chunks[batch_start : batch_start + max_parallel]
            tasks = [_summarise_chunk(chunk) for chunk in batch]
            results = await asyncio.gather(*tasks)
            summaries.extend(results)
            processed += len(batch)
            await self._notify(progress, f"Summarising uploaded file... [{processed}/{total}]")

        summaries = [summary for summary in summaries if summary]
        if not summaries:
            self._logger.error("All chunk summaries failed")
            fallback = self._fallback_summary(text)
            if fallback:
                self._logger.warning(
                    "ATTACHMENT SUMMARY FALLBACK text_len=%s fallback_chars=%s",
                    len(text),
                    len(fallback),
                )
            if fallback:
                await self._notify(progress, "Summary model failed - using direct source excerpts fallback.")
            else:
                await self._notify(progress, "Summary failed - continuing without attachment.")
            return fallback

        joined = "\n\n".join(summaries)
        joined_tokens = self._token_count(joined)
        if joined_tokens <= budget_tokens:
            self._logger.info(
                "ATTACHMENT SUMMARY DONE tokens=%s chars=%s (within budget)",
                joined_tokens,
                len(joined),
            )
            file_kb = len(text) // 1024
            await self._notify(progress, f"Summary complete. ({joined_tokens} tokens from {file_kb}KB file)")
            return joined

        num_summaries = len(summaries)
        target_tokens_per = budget_tokens // num_summaries
        target_chars_per = int(target_tokens_per * chars_per_tok)

        summary_tok_counts = [self._token_count(summary) for summary in summaries]
        indexed = sorted(
            enumerate(summaries),
            key=lambda pair: summary_tok_counts[pair[0]],
            reverse=True,
        )
        to_condense = [
            (index, summary)
            for index, summary in indexed
            if summary_tok_counts[index] > target_tokens_per
        ]

        if to_condense:
            condense_total = len(to_condense)
            condense_done = 0
            await self._notify(progress, f"Condensing summaries... [0/{condense_total}]")

            async def _condense(index: int, summary_text: str) -> tuple[int, str]:
                condense_system = (
                    f"Condense this summary to roughly {target_tokens_per} tokens "
                    f"(~{target_chars_per} characters) "
                    "while preserving all character names, plot points, and locations. "
                    f"End with: {guard}"
                )
                try:
                    result = await self._completion.complete(
                        condense_system,
                        summary_text,
                        max_tokens=min(
                            cfg.attachment_summary_max_tokens,
                            max(2_048, target_tokens_per + 256),
                        ),
                        temperature=0.2,
                    )
                    result = (result or "").strip()
                    if guard not in result:
                        self._logger.warning("Guard token missing in condensation, accepting as-is")
                    return index, result.replace(guard, "").strip()
                except Exception as exc:
                    self._logger.warning("Condensation failed: %s", exc)
                    return index, summary_text

            for batch_start in range(0, len(to_condense), max_parallel):
                batch = to_condense[batch_start : batch_start + max_parallel]
                tasks = [_condense(index, summary) for index, summary in batch]
                results = await asyncio.gather(*tasks)
                for index, condensed in results:
                    if condensed:
                        summaries[index] = condensed
                condense_done += len(batch)
                await self._notify(progress, f"Condensing summaries... [{condense_done}/{condense_total}]")

        joined = "\n\n".join(summaries)
        joined_tokens = self._token_count(joined)
        if joined_tokens > budget_tokens:
            max_chars = int(budget_tokens * chars_per_tok * 0.9)
            if len(joined) > max_chars:
                suffix = "... [truncated]"
                joined = joined[: max_chars - len(suffix)] + suffix
                joined_tokens = self._token_count(joined)

        self._logger.info(
            "ATTACHMENT SUMMARY DONE tokens=%s chars=%s chunks=%s condensed=%s",
            joined_tokens,
            len(joined),
            total,
            len(to_condense) if to_condense else 0,
        )
        file_kb = len(text) // 1024
        await self._notify(progress, f"Summary complete. ({joined_tokens} tokens from {file_kb}KB file)")
        return joined

    async def _notify(self, callback: ProgressCallback | None, message: str) -> None:
        if callback is None:
            return
        try:
            maybe = callback(message)
            if asyncio.iscoroutine(maybe):
                await maybe
        except Exception:
            return
