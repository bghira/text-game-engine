from __future__ import annotations

from typing import Any, Protocol


class TextCompletionPort(Protocol):
    async def complete(
        self,
        system_prompt: str,
        prompt: str,
        *,
        temperature: float = 0.8,
        max_tokens: int = 2048,
    ) -> str | None:
        ...


class TimerEffectsPort(Protocol):
    async def edit_timer_line(
        self,
        channel_id: str,
        message_id: str,
        replacement: str,
    ) -> None:
        ...

    async def emit_timed_event(
        self,
        campaign_id: str,
        channel_id: str,
        actor_id: str | None,
        narration: str,
    ) -> None:
        ...


class MemorySearchPort(Protocol):
    def search(
        self,
        query: str,
        campaign_id: str,
        top_k: int = 5,
    ) -> list[tuple[int, str, str, float]]:
        ...

    def delete_turns_after(self, campaign_id: str, turn_id: int) -> int:
        ...


class IMDBLookupPort(Protocol):
    def search(self, query: str, max_results: int = 3) -> list[dict]:
        ...

    def enrich(self, results: list[dict]) -> list[dict]:
        ...

    def fetch_details(self, imdb_id: str) -> dict:
        ...


class MediaGenerationPort(Protocol):
    def gpu_worker_available(self) -> bool:
        ...

    async def enqueue_scene_generation(
        self,
        *,
        actor_id: str,
        prompt: str,
        model: str,
        reference_images: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        channel_id: str | None = None,
    ) -> bool:
        ...

    async def enqueue_avatar_generation(
        self,
        *,
        actor_id: str,
        prompt: str,
        model: str,
        metadata: dict[str, Any] | None = None,
        channel_id: str | None = None,
    ) -> bool:
        ...
