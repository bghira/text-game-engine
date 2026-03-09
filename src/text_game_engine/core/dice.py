"""Dice engine — pure functions for skill checks and dice rolls."""

from __future__ import annotations

import random

from .types import DiceCheckOutcome, DiceCheckRequest, DiceCheckResult


def roll_d20() -> int:
    """Roll a single d20."""
    return random.randint(1, 20)


def roll(sides: int = 20, count: int = 1) -> list[int]:
    """Roll *count* dice with *sides* sides each."""
    return [random.randint(1, sides) for _ in range(count)]


def attribute_modifier(attribute_value: int) -> int:
    """Return the modifier for an attribute value.

    The attribute IS the bonus — a strength of 3 gives +3.
    """
    return max(0, int(attribute_value))


def skill_check(attribute_value: int, dc: int) -> DiceCheckResult:
    """Perform a d20 skill check against *dc* using *attribute_value* as modifier."""
    die = roll_d20()
    mod = attribute_modifier(attribute_value)
    total = die + mod
    return DiceCheckResult(
        attribute="",
        attribute_value=attribute_value,
        dc=dc,
        roll=die,
        modifier=mod,
        total=total,
        success=total >= dc,
        context="",
    )


def resolve_dice_check(
    request: DiceCheckRequest,
    player_attributes: dict[str, int],
) -> tuple[DiceCheckResult, DiceCheckOutcome]:
    """Roll against *request*, selecting the success/failure outcome."""
    attr_val = int(player_attributes.get(request.attribute, 0) or 0)
    result = skill_check(attr_val, request.dc)
    result.attribute = request.attribute
    result.context = request.context
    outcome = request.on_success if result.success else request.on_failure
    return result, outcome


def format_dice_result(result: DiceCheckResult) -> str:
    """Render a one-line mechanical summary.

    Example: ``\u2680 Strength check (DC 15): rolled 12 + 3 = 15 \u2014 Success!``
    """
    label = "Success!" if result.success else "Failure."
    attr_label = result.attribute.replace("_", " ").title() if result.attribute else "Skill"
    return (
        f"\u2680 {attr_label} check (DC {result.dc}): "
        f"rolled {result.roll} + {result.modifier} = {result.total} "
        f"\u2014 {label}"
    )
