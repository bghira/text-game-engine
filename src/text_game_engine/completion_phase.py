from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

PHASE_RESEARCH = "research"
PHASE_NARRATION = "narration"

_PHASE: ContextVar[str] = ContextVar("tge_completion_phase", default=PHASE_NARRATION)


def current_phase() -> str:
    return _PHASE.get()


@contextmanager
def phase(name: str) -> Iterator[None]:
    token = _PHASE.set(str(name or PHASE_NARRATION))
    try:
        yield
    finally:
        _PHASE.reset(token)
