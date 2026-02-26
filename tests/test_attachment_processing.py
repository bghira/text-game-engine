from __future__ import annotations

import asyncio

from text_game_engine.core.attachments import (
    AttachmentProcessingConfig,
    AttachmentTextProcessor,
    extract_attachment_text,
)


class StubAttachment:
    def __init__(self, filename: str, data: bytes, size: int | None = None):
        self.filename = filename
        self._data = data
        self.size = len(data) if size is None else size

    async def read(self) -> bytes:
        return self._data


class StubCompletion:
    def __init__(self):
        self.calls: list[tuple[str, str, int, float]] = []

    async def complete(
        self,
        system_prompt: str,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> str:
        self.calls.append((system_prompt, prompt, max_tokens, temperature))
        if "Condense this summary" in system_prompt:
            return "condensed summary --COMPLETED SUMMARY--"
        return f"summary of {len(prompt)} chars --COMPLETED SUMMARY--"


def test_extract_attachment_text_happy_path_utf8():
    async def run_test():
        out = await extract_attachment_text([StubAttachment("notes.txt", "hello world".encode("utf-8"))])
        assert out == "hello world"

    asyncio.run(run_test())


def test_extract_attachment_text_too_large_error_contract():
    async def run_test():
        cfg = AttachmentProcessingConfig(attachment_max_bytes=10)
        out = await extract_attachment_text([StubAttachment("notes.txt", b"0123456789ab")], config=cfg)
        assert out == "ERROR:File too large (0KB, limit 0KB)"

    asyncio.run(run_test())


def test_extract_attachment_text_latin1_fallback():
    async def run_test():
        raw = "cafe\xe9".encode("latin-1")
        out = await extract_attachment_text([StubAttachment("story.txt", raw)])
        assert out == "cafe\xe9"

    asyncio.run(run_test())


def test_summarise_long_text_bypasses_completion_for_small_single_chunk():
    async def run_test():
        completion = StubCompletion()
        processor = AttachmentTextProcessor(
            completion=completion,
            token_count=lambda text: len(text.split()),
            config=AttachmentProcessingConfig(
                attachment_chunk_tokens=100,
                attachment_model_ctx_tokens=10_000,
                attachment_prompt_overhead_tokens=0,
                attachment_response_reserve_tokens=0,
                attachment_max_chunks=8,
            ),
        )
        text = "small text that fits"
        out = await processor.summarise_long_text(text)
        assert out == text
        assert completion.calls == []

    asyncio.run(run_test())


def test_summarise_long_text_chunks_and_condenses_when_over_budget():
    async def run_test():
        completion = StubCompletion()
        progress: list[str] = []
        processor = AttachmentTextProcessor(
            completion=completion,
            token_count=lambda text: max(len(text.split()), 1),
            config=AttachmentProcessingConfig(
                attachment_chunk_tokens=4,
                attachment_model_ctx_tokens=20,
                attachment_prompt_overhead_tokens=5,
                attachment_response_reserve_tokens=5,
                attachment_max_parallel=2,
                attachment_max_chunks=2,
            ),
        )
        text = (
            "alpha beta gamma delta epsilon zeta eta theta\n\n"
            "iota kappa lambda mu nu xi omicron pi\n\n"
            "rho sigma tau upsilon phi chi psi omega"
        )
        out = await processor.summarise_long_text(text, progress=progress.append)
        assert "summary complete." in progress[-1].lower()
        assert out
        assert any("Summarise the following text passage" in call[0] for call in completion.calls)
        assert any("Condense this summary" in call[0] for call in completion.calls)

    asyncio.run(run_test())

