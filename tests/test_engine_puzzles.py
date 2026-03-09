"""Tests for engine integration of dice, puzzle, and minigame systems."""

from __future__ import annotations

from dataclasses import asdict
from unittest.mock import MagicMock

from text_game_engine.core.engine import GameEngine
from text_game_engine.core.minigames import MinigameEngine, MinigameState
from text_game_engine.core.puzzles import PuzzleEngine, PuzzleState
from text_game_engine.core.types import (
    DiceCheckOutcome,
    DiceCheckRequest,
    DiceCheckResult,
    LLMTurnOutput,
    MinigameChallenge,
    PuzzleTrigger,
    ResolveTurnInput,
    TurnContext,
)


class TestResolveDiceCheckIntegration:
    def test_dice_check_success_merges_into_output(self):
        llm_output = LLMTurnOutput(
            narration="You try to force the gate.",
            state_update={"gate": "closed"},
            player_state_update={},
            xp_awarded=0,
        )
        llm_output.dice_check = DiceCheckRequest(
            attribute="strength",
            dc=5,
            context="forcing the gate",
            on_success=DiceCheckOutcome(
                narration="The gate swings open!",
                state_update={"gate": "open"},
                player_state_update={"location": "courtyard"},
                xp_awarded=5,
            ),
            on_failure=DiceCheckOutcome(
                narration="The gate won't budge.",
            ),
        )
        campaign_state = {}
        result_output, dice_result = GameEngine._resolve_dice_check(
            llm_output, {"strength": 20}, campaign_state
        )
        # With strength 20, roll 1 + 20 = 21 >= DC 5 => always success
        assert dice_result.success
        assert "Success!" in result_output.narration
        assert "The gate swings open!" in result_output.narration
        assert result_output.state_update.get("gate") == "open"
        assert result_output.player_state_update.get("location") == "courtyard"
        assert result_output.xp_awarded == 5
        assert "_last_dice_check" in campaign_state

    def test_dice_check_failure_merges(self):
        llm_output = LLMTurnOutput(
            narration="You attempt the impossible.",
            state_update={},
            player_state_update={},
            xp_awarded=0,
        )
        llm_output.dice_check = DiceCheckRequest(
            attribute="wisdom",
            dc=100,
            context="impossible task",
            on_success=DiceCheckOutcome(narration="Miracle!"),
            on_failure=DiceCheckOutcome(narration="You fail completely.", xp_awarded=1),
        )
        campaign_state = {}
        result_output, dice_result = GameEngine._resolve_dice_check(
            llm_output, {"wisdom": 0}, campaign_state
        )
        assert not dice_result.success
        assert "Failure." in result_output.narration
        assert "You fail completely." in result_output.narration
        assert result_output.xp_awarded == 1

    def test_dice_check_from_dict(self):
        """Test that dict-based dice_check (as would come from JSON) works."""
        llm_output = LLMTurnOutput(narration="Test.", state_update={}, xp_awarded=0)
        llm_output.dice_check = {
            "attribute": "dexterity",
            "dc": 5,
            "context": "dodge",
            "on_success": {"narration": "Dodged!", "xp_awarded": 2},
            "on_failure": {"narration": "Hit!"},
        }
        campaign_state = {}
        result_output, dice_result = GameEngine._resolve_dice_check(
            llm_output, {"dexterity": 15}, campaign_state
        )
        assert dice_result is not None
        assert dice_result.attribute == "dexterity"

    def test_no_dice_check(self):
        llm_output = LLMTurnOutput(narration="Normal turn.")
        campaign_state = {}
        result_output, dice_result = GameEngine._resolve_dice_check(
            llm_output, {}, campaign_state
        )
        assert dice_result is None
        assert result_output.narration == "Normal turn."


class TestProcessPuzzleTrigger:
    def test_puzzle_trigger_creates_state(self):
        llm_output = LLMTurnOutput(narration="A sphinx blocks your path.")
        llm_output.puzzle_trigger = PuzzleTrigger(
            puzzle_type="riddle",
            context="sphinx encounter",
            difficulty="medium",
        )
        campaign_state = {}
        GameEngine._process_puzzle_trigger(llm_output, campaign_state)
        assert "_active_puzzle" in campaign_state
        assert campaign_state["_active_puzzle"]["puzzle_type"] == "riddle"
        # Question should be appended to narration
        assert campaign_state["_active_puzzle"]["question"] in llm_output.narration

    def test_puzzle_trigger_from_dict(self):
        llm_output = LLMTurnOutput(narration="A locked door.")
        llm_output.puzzle_trigger = {
            "puzzle_type": "math",
            "context": "combination lock",
            "difficulty": "easy",
        }
        campaign_state = {}
        GameEngine._process_puzzle_trigger(llm_output, campaign_state)
        assert "_active_puzzle" in campaign_state
        assert campaign_state["_active_puzzle"]["puzzle_type"] == "math"

    def test_no_trigger(self):
        llm_output = LLMTurnOutput(narration="Normal turn.")
        campaign_state = {}
        GameEngine._process_puzzle_trigger(llm_output, campaign_state)
        assert "_active_puzzle" not in campaign_state


class TestProcessMinigameChallenge:
    def test_minigame_challenge_creates_state(self):
        llm_output = LLMTurnOutput(narration="The innkeeper challenges you to a game.")
        llm_output.minigame_challenge = MinigameChallenge(
            game_type="tic_tac_toe",
            opponent_slug="innkeeper",
            stakes="the old map",
        )
        campaign_state = {}
        GameEngine._process_minigame_challenge(llm_output, campaign_state)
        assert "_active_minigame" in campaign_state
        assert campaign_state["_active_minigame"]["game_type"] == "tic_tac_toe"
        # Board should be in narration
        assert "A" in llm_output.narration  # column headers

    def test_minigame_from_dict(self):
        llm_output = LLMTurnOutput(narration="Roll the dice!")
        llm_output.minigame_challenge = {
            "game_type": "dice_duel",
            "opponent_slug": "gambler",
            "stakes": "gold",
        }
        campaign_state = {}
        GameEngine._process_minigame_challenge(llm_output, campaign_state)
        assert "_active_minigame" in campaign_state

    def test_unknown_game_type_ignored(self):
        llm_output = LLMTurnOutput(narration="Play chess!")
        llm_output.minigame_challenge = MinigameChallenge(
            game_type="chess",
            opponent_slug="wizard",
            stakes="kingdom",
        )
        campaign_state = {}
        GameEngine._process_minigame_challenge(llm_output, campaign_state)
        assert "_active_minigame" not in campaign_state

    def test_no_challenge(self):
        llm_output = LLMTurnOutput(narration="Normal turn.")
        campaign_state = {}
        GameEngine._process_minigame_challenge(llm_output, campaign_state)
        assert "_active_minigame" not in campaign_state


class TestPreLLMPuzzleMinigame:
    def test_puzzle_answer_validation(self):
        puzzle = PuzzleState(
            puzzle_type="riddle",
            context="test",
            difficulty="easy",
            question="What has hands but can't clap?",
            answer="clock",
            accept_patterns=["a clock"],
            hints=["tick tock"],
            max_attempts=3,
        )
        context = TurnContext(
            campaign_id="c1",
            actor_id="a1",
            session_id=None,
            action="clock",
            campaign_state={"_active_puzzle": puzzle.to_dict()},
            campaign_summary="",
            campaign_characters={},
            player_state={},
            player_level=1,
            player_xp=0,
            recent_turns=[],
            start_row_version=1,
            now=None,
        )
        turn_input = ResolveTurnInput(campaign_id="c1", actor_id="a1", action="clock")
        GameEngine._pre_llm_puzzle_minigame(context, turn_input)
        result = context.campaign_state.get("_puzzle_result")
        assert result is not None
        assert result["correct"] is True
        assert result["solved"] is True

    def test_puzzle_wrong_answer(self):
        puzzle = PuzzleState(
            puzzle_type="riddle",
            context="test",
            difficulty="easy",
            question="test",
            answer="clock",
            max_attempts=3,
        )
        context = TurnContext(
            campaign_id="c1",
            actor_id="a1",
            session_id=None,
            action="wrong answer",
            campaign_state={"_active_puzzle": puzzle.to_dict()},
            campaign_summary="",
            campaign_characters={},
            player_state={},
            player_level=1,
            player_xp=0,
            recent_turns=[],
            start_row_version=1,
            now=None,
        )
        turn_input = ResolveTurnInput(campaign_id="c1", actor_id="a1", action="wrong answer")
        GameEngine._pre_llm_puzzle_minigame(context, turn_input)
        result = context.campaign_state.get("_puzzle_result")
        assert result is not None
        assert result["correct"] is False

    def test_puzzle_hint(self):
        puzzle = PuzzleState(
            puzzle_type="riddle",
            context="test",
            difficulty="easy",
            question="test",
            answer="clock",
            hints=["tick tock", "on the wall"],
            max_attempts=3,
        )
        context = TurnContext(
            campaign_id="c1",
            actor_id="a1",
            session_id=None,
            action="hint",
            campaign_state={"_active_puzzle": puzzle.to_dict()},
            campaign_summary="",
            campaign_characters={},
            player_state={},
            player_level=1,
            player_xp=0,
            recent_turns=[],
            start_row_version=1,
            now=None,
        )
        turn_input = ResolveTurnInput(campaign_id="c1", actor_id="a1", action="hint")
        GameEngine._pre_llm_puzzle_minigame(context, turn_input)
        result = context.campaign_state.get("_puzzle_result")
        assert "hint" in result
        assert result["hint"] == "tick tock"

    def test_puzzle_give_up(self):
        puzzle = PuzzleState(
            puzzle_type="riddle",
            context="test",
            difficulty="easy",
            question="test",
            answer="clock",
            max_attempts=3,
        )
        context = TurnContext(
            campaign_id="c1",
            actor_id="a1",
            session_id=None,
            action="give up",
            campaign_state={"_active_puzzle": puzzle.to_dict()},
            campaign_summary="",
            campaign_characters={},
            player_state={},
            player_level=1,
            player_xp=0,
            recent_turns=[],
            start_row_version=1,
            now=None,
        )
        turn_input = ResolveTurnInput(campaign_id="c1", actor_id="a1", action="give up")
        GameEngine._pre_llm_puzzle_minigame(context, turn_input)
        result = context.campaign_state.get("_puzzle_result")
        assert result["failed"] is True

    def test_minigame_move_validation(self):
        challenge = MinigameChallenge(
            game_type="tic_tac_toe",
            opponent_slug="npc",
            stakes="pride",
        )
        game = MinigameEngine.new_game(challenge)
        context = TurnContext(
            campaign_id="c1",
            actor_id="a1",
            session_id=None,
            action="center",
            campaign_state={"_active_minigame": game.to_dict()},
            campaign_summary="",
            campaign_characters={},
            player_state={},
            player_level=1,
            player_xp=0,
            recent_turns=[],
            start_row_version=1,
            now=None,
        )
        turn_input = ResolveTurnInput(campaign_id="c1", actor_id="a1", action="center")
        GameEngine._pre_llm_puzzle_minigame(context, turn_input)
        result = context.campaign_state.get("_minigame_result")
        assert result is not None
        assert result["valid"] is True

    def test_minigame_forfeit(self):
        challenge = MinigameChallenge(
            game_type="tic_tac_toe",
            opponent_slug="npc",
            stakes="gold",
        )
        game = MinigameEngine.new_game(challenge)
        context = TurnContext(
            campaign_id="c1",
            actor_id="a1",
            session_id=None,
            action="forfeit",
            campaign_state={"_active_minigame": game.to_dict()},
            campaign_summary="",
            campaign_characters={},
            player_state={},
            player_level=1,
            player_xp=0,
            recent_turns=[],
            start_row_version=1,
            now=None,
        )
        turn_input = ResolveTurnInput(campaign_id="c1", actor_id="a1", action="forfeit")
        GameEngine._pre_llm_puzzle_minigame(context, turn_input)
        result = context.campaign_state.get("_minigame_result")
        assert result["finished"] is True
        mg = MinigameState.from_dict(context.campaign_state["_active_minigame"])
        assert mg.status == "npc_won"

    def test_no_active_puzzle_or_minigame(self):
        context = TurnContext(
            campaign_id="c1",
            actor_id="a1",
            session_id=None,
            action="look around",
            campaign_state={},
            campaign_summary="",
            campaign_characters={},
            player_state={},
            player_level=1,
            player_xp=0,
            recent_turns=[],
            start_row_version=1,
            now=None,
        )
        turn_input = ResolveTurnInput(campaign_id="c1", actor_id="a1", action="look around")
        GameEngine._pre_llm_puzzle_minigame(context, turn_input)
        assert "_puzzle_result" not in context.campaign_state
        assert "_minigame_result" not in context.campaign_state
