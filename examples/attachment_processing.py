from __future__ import annotations

import asyncio

from text_game_engine.core.attachments import (
    AttachmentProcessingConfig,
    AttachmentTextProcessor,
    extract_attachment_text,
)


class InMemoryAttachment:
    def __init__(self, filename: str, data: bytes):
        self.filename = filename
        self._data = data
        self.size = len(data)

    async def read(self) -> bytes:
        return self._data


class StubCompletion:
    async def complete(
        self,
        system_prompt: str,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
    ) -> str:
        if "Condense this summary" in system_prompt:
            return "Condensed campaign notes with key names and events. --COMPLETED SUMMARY--"
        first = prompt.strip().splitlines()[0][:120]
        return f"Summary: {first} --COMPLETED SUMMARY--"


async def main() -> None:
    raw_text = (
        "Captain Mirel reaches the frost gate at dawn.\n\n"
        "The scouts report that the river crossing is blocked by ice and broken carts.\n\n"
        "A courier arrives with a warning that the city gate closes on Day 4.\n\n"
        "The group debates whether to risk the bridge or take the forest road."
    )
    attachment = InMemoryAttachment("chronicle.txt", raw_text.encode("utf-8"))

    extracted = await extract_attachment_text([attachment])
    if extracted is None:
        print("No text extracted.")
        return
    if extracted.startswith("ERROR:"):
        print(extracted)
        return

    processor = AttachmentTextProcessor(
        completion=StubCompletion(),
        token_count=lambda text: max(1, len(text.split())),
        config=AttachmentProcessingConfig(
            attachment_chunk_tokens=6,
            attachment_model_ctx_tokens=30,
            attachment_prompt_overhead_tokens=6,
            attachment_response_reserve_tokens=6,
            attachment_max_parallel=2,
            attachment_max_chunks=3,
        ),
    )

    async def progress(message: str) -> None:
        print("[progress]", message)

    summary = await processor.summarise_long_text(extracted, progress=progress)
    print("\nFinal summary:\n")
    print(summary)


if __name__ == "__main__":
    asyncio.run(main())
