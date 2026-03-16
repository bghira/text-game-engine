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
    interrupt_scope: str = "global"


@dataclass
class GiveItemInstruction:
    item: str
    to_actor_id: Optional[str] = None
    to_discord_mention: Optional[str] = None


@dataclass
class DiceCheckOutcome:
    narration: str
    state_update: dict[str, Any] = field(default_factory=dict)
    player_state_update: dict[str, Any] = field(default_factory=dict)
    xp_awarded: int = 0


@dataclass
class DiceCheckRequest:
    attribute: str
    dc: int
    context: str
    on_success: DiceCheckOutcome
    on_failure: DiceCheckOutcome


@dataclass
class DiceCheckResult:
    attribute: str
    attribute_value: int
    dc: int
    roll: int
    modifier: int
    total: int
    success: bool
    context: str


@dataclass
class PuzzleTrigger:
    puzzle_type: str
    context: str
    difficulty: str = "medium"


@dataclass
class MinigameChallenge:
    game_type: str
    opponent_slug: str
    stakes: str = ""


@dataclass
class LLMTurnOutput:
    narration: str
    reasoning: Optional[str] = None
    scene_output: dict[str, Any] | None = None
    state_update: dict[str, Any] = field(default_factory=dict)
    summary_update: Optional[str] = None
    xp_awarded: int = 0
    player_state_update: dict[str, Any] = field(default_factory=dict)
    other_player_state_updates: dict[str, dict[str, Any]] = field(default_factory=dict)
    co_located_player_slugs: list[str] = field(default_factory=list)
    turn_visibility: dict[str, Any] | None = None
    scene_image_prompt: Optional[str] = None
    timer_instruction: Optional[TimerInstruction] = None
    character_updates: dict[str, Any] = field(default_factory=dict)
    give_item: Optional[GiveItemInstruction] = None
    dice_check: Optional[DiceCheckRequest] = None
    puzzle_trigger: Optional[PuzzleTrigger] = None
    minigame_challenge: Optional[MinigameChallenge] = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


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
    record_player_turn: bool = True
    allow_timer_instruction: bool = True


@dataclass
class ResolveTurnResult:
    status: str
    narration: Optional[str] = None
    scene_image_prompt: Optional[str] = None
    timer_instruction: Optional[TimerInstruction] = None
    give_item: Optional[dict[str, Any]] = None
    conflict_reason: Optional[str] = None
    dice_result: Optional[DiceCheckResult] = None
    active_puzzle: Optional[dict[str, Any]] = None
    active_minigame: Optional[dict[str, Any]] = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RewindResult:
    status: str
    target_turn_id: Optional[int] = None
    deleted_turns: int = 0
    reason: Optional[str] = None
