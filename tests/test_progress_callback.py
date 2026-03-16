"""Tests for the progress callback contract."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from text_game_engine.tool_aware_llm import _notify_progress


class TestNotifyProgress:
    """Verify _notify_progress handles sync/async callbacks and exceptions."""

    def test_async_callback_awaited(self):
        async def run():
            cb = AsyncMock()
            await _notify_progress(cb, "thinking", {"key": "val"})
            cb.assert_awaited_once_with("thinking", {"key": "val"})

        asyncio.run(run())

    def test_sync_callback_accepted(self):
        async def run():
            cb = MagicMock(return_value=None)
            await _notify_progress(cb, "tool_call", {"tool": "memory_search"})
            cb.assert_called_once_with("tool_call", {"tool": "memory_search"})

        asyncio.run(run())

    def test_none_callback_is_noop(self):
        async def run():
            await _notify_progress(None, "thinking")

        asyncio.run(run())

    def test_metadata_defaults_to_none(self):
        async def run():
            cb = AsyncMock()
            await _notify_progress(cb, "writing")
            cb.assert_awaited_once_with("writing", None)

        asyncio.run(run())

    def test_sync_callback_exception_swallowed(self):
        async def run():
            cb = MagicMock(side_effect=RuntimeError("boom"))
            await _notify_progress(cb, "thinking")

        asyncio.run(run())

    def test_async_callback_exception_swallowed(self):
        async def run():
            cb = AsyncMock(side_effect=RuntimeError("boom"))
            await _notify_progress(cb, "refining")

        asyncio.run(run())

    def test_callback_returning_coroutine_awaited(self):
        async def run():
            awaited = False

            async def _coro():
                nonlocal awaited
                awaited = True

            def sync_returning_coro(phase, meta):
                return _coro()

            await _notify_progress(sync_returning_coro, "tool_call", {"tool": "sms_read"})
            assert awaited

        asyncio.run(run())
