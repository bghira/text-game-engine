"""Tests for the dice engine module."""

from __future__ import annotations

import random

from text_game_engine.core.dice import (
    attribute_modifier,
    format_dice_result,
    resolve_dice_check,
    roll,
    roll_d20,
    skill_check,
)
from text_game_engine.core.types import DiceCheckOutcome, DiceCheckRequest, DiceCheckResult


class TestRollD20:
    def test_range(self):
        results = {roll_d20() for _ in range(500)}
        assert results.issubset(set(range(1, 21)))
        assert len(results) > 5  # should see variety

    def test_always_in_range(self):
        for _ in range(200):
            r = roll_d20()
            assert 1 <= r <= 20


class TestRoll:
    def test_single_die(self):
        results = roll(6, 1)
        assert len(results) == 1
        assert 1 <= results[0] <= 6

    def test_multiple_dice(self):
        results = roll(6, 5)
        assert len(results) == 5
        for r in results:
            assert 1 <= r <= 6


class TestAttributeModifier:
    def test_positive(self):
        assert attribute_modifier(5) == 5

    def test_zero(self):
        assert attribute_modifier(0) == 0

    def test_negative_clamped(self):
        assert attribute_modifier(-3) == 0


class TestSkillCheck:
    def test_with_known_seed(self):
        random.seed(42)
        result = skill_check(3, 10)
        assert isinstance(result, DiceCheckResult)
        assert result.attribute_value == 3
        assert result.modifier == 3
        assert result.dc == 10
        assert 1 <= result.roll <= 20
        assert result.total == result.roll + result.modifier
        assert result.success == (result.total >= result.dc)

    def test_high_attribute_helps(self):
        # With modifier 20, even roll=1 gives total=21, passing DC 20
        random.seed(0)
        result = skill_check(20, 20)
        assert result.modifier == 20
        assert result.total >= 21  # minimum roll 1 + 20

    def test_dc_zero_always_succeeds(self):
        for _ in range(50):
            result = skill_check(0, 0)
            assert result.success


class TestResolveDiceCheck:
    def test_selects_success_outcome(self):
        request = DiceCheckRequest(
            attribute="strength",
            dc=1,  # DC 1 means almost always succeed
            context="lifting a pebble",
            on_success=DiceCheckOutcome(
                narration="You lift it easily.",
                state_update={"pebble": "lifted"},
                xp_awarded=5,
            ),
            on_failure=DiceCheckOutcome(
                narration="You fail to lift it.",
            ),
        )
        result, outcome = resolve_dice_check(request, {"strength": 10})
        assert result.success
        assert outcome.narration == "You lift it easily."
        assert outcome.xp_awarded == 5

    def test_selects_failure_outcome(self):
        request = DiceCheckRequest(
            attribute="dexterity",
            dc=100,  # DC 100 means always fail
            context="impossible dodge",
            on_success=DiceCheckOutcome(narration="Dodged!"),
            on_failure=DiceCheckOutcome(narration="You fail."),
        )
        result, outcome = resolve_dice_check(request, {"dexterity": 5})
        assert not result.success
        assert outcome.narration == "You fail."

    def test_missing_attribute_modifier_zero(self):
        request = DiceCheckRequest(
            attribute="charisma",
            dc=15,
            context="persuasion",
            on_success=DiceCheckOutcome(narration="Success."),
            on_failure=DiceCheckOutcome(narration="Failure."),
        )
        result, _ = resolve_dice_check(request, {"strength": 10})
        assert result.attribute_value == 0
        assert result.modifier == 0
        assert result.attribute == "charisma"


class TestFormatDiceResult:
    def test_success_format(self):
        result = DiceCheckResult(
            attribute="strength",
            attribute_value=3,
            dc=15,
            roll=12,
            modifier=3,
            total=15,
            success=True,
            context="forcing the rusted gate",
        )
        text = format_dice_result(result)
        assert "\u2680 Strength check (DC 15)" in text
        assert "rolled 12 + 3 = 15" in text
        assert "Success!" in text

    def test_failure_format(self):
        result = DiceCheckResult(
            attribute="dexterity",
            attribute_value=2,
            dc=18,
            roll=7,
            modifier=2,
            total=9,
            success=False,
            context="dodging",
        )
        text = format_dice_result(result)
        assert "Dexterity check (DC 18)" in text
        assert "rolled 7 + 2 = 9" in text
        assert "Failure." in text

    def test_empty_attribute(self):
        result = DiceCheckResult(
            attribute="",
            attribute_value=0,
            dc=10,
            roll=5,
            modifier=0,
            total=5,
            success=False,
            context="test",
        )
        text = format_dice_result(result)
        assert "Skill check" in text
