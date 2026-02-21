from __future__ import annotations

import asyncio
import logging
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
    attachment_chunk_tokens: int = 2_000
    attachment_model_ctx_tokens: int = 200_000
    attachment_prompt_overhead_tokens: int = 6_000
    attachment_response_reserve_tokens: int = 4_000
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
        return None
    if not raw:
        return None

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    text = text.strip()
    return text if text else None


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

    async def summarise_long_text(
        self,
        text: str,
        *,
        progress: ProgressCallback | None = None,
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
        target_chunk_tokens = max(min_chunk_tokens, total_tokens // cfg.attachment_max_chunks)
        chars_per_tok = len(text) / max(total_tokens, 1)
        chunk_char_target = int(target_chunk_tokens * chars_per_tok)

        paragraphs = text.split("\n\n")
        chunks: list[str] = []
        current_chunk: list[str] = []
        current_len = 0
        for para in paragraphs:
            para_len = len(para)
            if current_len + para_len + 2 > chunk_char_target and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_len = para_len
            else:
                current_chunk.append(para)
                current_len += para_len + 2
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

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

        summary_max_tokens = min(1500, max(800, target_chunk_tokens // 4))
        summarise_system = (
            "Summarise the following text passage for a text-adventure campaign. "
            "Preserve all character names, plot points, locations, and key events. "
            f"Be detailed but concise. End with the exact line: {guard}"
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
            await self._notify(progress, "Summary failed - continuing without attachment.")
            return ""

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
                        max_tokens=target_tokens_per + 50,
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

