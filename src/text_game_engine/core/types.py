from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class TimerInstruction:
    delay_seconds: int
    event_text: str
    interruptible: bool = True
    interrupt_action: Optional[str] = None


@dataclass
class GiveItemInstruction:
    item: str
    to_actor_id: Optional[str] = None
    to_discord_mention: Optional[str] = None


@dataclass
class LLMTurnOutput:
    narration: str
    state_update: dict[str, Any] = field(default_factory=dict)
    summary_update: Optional[str] = None
    xp_awarded: int = 0
    player_state_update: dict[str, Any] = field(default_factory=dict)
    scene_image_prompt: Optional[str] = None
    timer_instruction: Optional[TimerInstruction] = None
    character_updates: dict[str, Any] = field(default_factory=dict)
    give_item: Optional[GiveItemInstruction] = None


@dataclass
class TurnContext:
    campaign_id: str
    actor_id: str
    session_id: Optional[str]
    action: str
    campaign_state: dict[str, Any]
    campaign_summary: str
    campaign_characters: dict[str, Any]
    player_state: dict[str, Any]
    player_level: int
    player_xp: int
    recent_turns: list[dict[str, Any]]
    start_row_version: int
    now: datetime


@dataclass
class ResolveTurnInput:
    campaign_id: str
    actor_id: str
    action: str
    session_id: Optional[str] = None


@dataclass
class ResolveTurnResult:
    status: str
    narration: Optional[str] = None
    scene_image_prompt: Optional[str] = None
    timer_instruction: Optional[TimerInstruction] = None
    conflict_reason: Optional[str] = None


@dataclass
class RewindResult:
    status: str
    target_turn_id: Optional[int] = None
    deleted_turns: int = 0
    reason: Optional[str] = None
