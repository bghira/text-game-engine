"""Puzzle engine — deterministic puzzle generation and harness validation."""

from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass, field
from typing import Any

from .types import PuzzleTrigger

# ---------------------------------------------------------------------------
# Action verbs used by the heuristic to distinguish puzzle answers from
# normal gameplay actions.
# ---------------------------------------------------------------------------
_ACTION_VERBS = frozenset(
    {
        "go",
        "walk",
        "run",
        "move",
        "take",
        "get",
        "grab",
        "pick",
        "drop",
        "put",
        "open",
        "close",
        "use",
        "attack",
        "fight",
        "hit",
        "kill",
        "talk",
        "speak",
        "say",
        "ask",
        "look",
        "examine",
        "inspect",
        "search",
        "enter",
        "exit",
        "leave",
        "climb",
        "jump",
        "swim",
        "throw",
        "push",
        "pull",
        "turn",
        "read",
        "eat",
        "drink",
        "sleep",
        "wait",
        "inventory",
        "equip",
        "cast",
    }
)


# ---------------------------------------------------------------------------
# Riddle bank — curated per difficulty tier
# ---------------------------------------------------------------------------
_RIDDLE_BANK: dict[str, list[dict[str, Any]]] = {
    "easy": [
        {
            "question": "I have hands but I can't clap. What am I?",
            "answer": "clock",
            "accept_patterns": ["a clock", "watch", "a watch"],
            "hints": ["I hang on walls.", "I tell you something about the day.", "Tick tock."],
        },
        {
            "question": "I have a head and a tail but no body. What am I?",
            "answer": "coin",
            "accept_patterns": ["a coin", "penny", "a penny"],
            "hints": ["You might flip me.", "I'm found in pockets.", "I'm currency."],
        },
        {
            "question": "What has keys but no locks?",
            "answer": "piano",
            "accept_patterns": ["a piano", "keyboard", "a keyboard"],
            "hints": ["I make music.", "I have black and white parts.", "You press my keys."],
        },
        {
            "question": "What has a neck but no head?",
            "answer": "bottle",
            "accept_patterns": ["a bottle"],
            "hints": ["I can hold liquid.", "I'm often made of glass.", "You drink from me."],
        },
        {
            "question": "What can you catch but not throw?",
            "answer": "cold",
            "accept_patterns": ["a cold", "the cold"],
            "hints": ["You might stay in bed with me.", "I'm not a ball.", "Achoo!"],
        },
        {
            "question": "What gets wetter the more it dries?",
            "answer": "towel",
            "accept_patterns": ["a towel"],
            "hints": ["You use me after bathing.", "I'm made of cloth.", "I hang in bathrooms."],
        },
        {
            "question": "What has an eye but cannot see?",
            "answer": "needle",
            "accept_patterns": ["a needle"],
            "hints": ["I'm used with thread.", "I'm very sharp.", "Tailors use me."],
        },
        {
            "question": "I go up but never come down. What am I?",
            "answer": "age",
            "accept_patterns": ["your age", "my age"],
            "hints": ["Everyone has me.", "I increase with time.", "Birthdays mark me."],
        },
        {
            "question": "What has teeth but cannot bite?",
            "answer": "comb",
            "accept_patterns": ["a comb"],
            "hints": ["You use me on your head.", "I help with tangles.", "I'm found on a dresser."],
        },
        {
            "question": "What can travel around the world while staying in a corner?",
            "answer": "stamp",
            "accept_patterns": ["a stamp", "postage stamp", "a postage stamp"],
            "hints": ["I go on envelopes.", "The post office sells me.", "I'm small and sticky."],
        },
    ],
    "medium": [
        {
            "question": "I have cities but no houses, forests but no trees, water but no fish. What am I?",
            "answer": "map",
            "accept_patterns": ["a map"],
            "hints": ["I represent something larger.", "Explorers carry me.", "I show you where to go."],
        },
        {
            "question": "The more you take, the more you leave behind. What am I?",
            "answer": "footsteps",
            "accept_patterns": ["footstep", "steps", "footprints", "footprint"],
            "hints": ["Think about walking.", "You leave these in sand.", "They're on the ground behind you."],
        },
        {
            "question": "I speak without a mouth and hear without ears. I have no body but I come alive with wind. What am I?",
            "answer": "echo",
            "accept_patterns": ["an echo"],
            "hints": ["Mountains know me well.", "I repeat what you say.", "I'm a reflection of sound."],
        },
        {
            "question": "What can fill a room but takes up no space?",
            "answer": "light",
            "accept_patterns": ["sunlight", "darkness"],
            "hints": ["Flip a switch.", "I travel very fast.", "Without me you can't see."],
        },
        {
            "question": "I have lakes with no water, mountains with no stone, and cities with no buildings. What am I?",
            "answer": "map",
            "accept_patterns": ["a map"],
            "hints": ["I'm flat.", "Cartographers make me.", "I represent the world."],
        },
        {
            "question": "What is seen in the middle of March and April that can't be seen at the beginning or end of either month?",
            "answer": "r",
            "accept_patterns": ["the letter r", "letter r"],
            "hints": ["Think about the words, not the calendar.", "It's a single character.", "Look at the spelling."],
        },
        {
            "question": "What has a heart that doesn't beat?",
            "answer": "artichoke",
            "accept_patterns": ["an artichoke"],
            "hints": ["I'm a vegetable.", "My heart is the best part to eat.", "I have layered leaves."],
        },
        {
            "question": "I can be cracked, made, told, and played. What am I?",
            "answer": "joke",
            "accept_patterns": ["a joke", "jokes"],
            "hints": ["I make people laugh.", "Comedians deal in me.", "I have a punchline."],
        },
        {
            "question": "What disappears as soon as you say its name?",
            "answer": "silence",
            "accept_patterns": ["the silence"],
            "hints": ["It's the absence of something.", "Libraries prefer me.", "Speaking destroys me."],
        },
        {
            "question": "What has words but never speaks?",
            "answer": "book",
            "accept_patterns": ["a book", "books"],
            "hints": ["You find me in libraries.", "I have pages.", "Authors create me."],
        },
    ],
    "hard": [
        {
            "question": "I am not alive, but I grow; I don't have lungs, but I need air; I don't have a mouth, but water kills me. What am I?",
            "answer": "fire",
            "accept_patterns": ["a fire", "flame", "a flame"],
            "hints": ["I'm hot.", "I consume fuel.", "Firefighters fight me."],
        },
        {
            "question": "I can be long or short, grown or bought, painted or left bare, round or square. What am I?",
            "answer": "nails",
            "accept_patterns": ["nail", "a nail", "fingernails", "fingernail"],
            "hints": ["You have them on your body.", "Hammers work with one kind of me.", "Manicurists know me well."],
        },
        {
            "question": "What English word has three consecutive double letters?",
            "answer": "bookkeeper",
            "accept_patterns": ["book-keeper", "book keeper"],
            "hints": ["Think about someone who manages records.", "oo, kk, ee.", "It's a profession."],
        },
        {
            "question": "I turn once, what is out will not get in. I turn again, what is in will not get out. What am I?",
            "answer": "key",
            "accept_patterns": ["a key", "lock", "a lock"],
            "hints": ["I work with doors.", "I'm made of metal.", "You carry me on a ring."],
        },
        {
            "question": "The person who makes it, sells it. The person who buys it never uses it. The person who uses it never knows they're using it. What is it?",
            "answer": "coffin",
            "accept_patterns": ["a coffin", "casket", "a casket"],
            "hints": ["It's related to death.", "It's made of wood or metal.", "You lie in it."],
        },
        {
            "question": "What word in the English language does the following: the first two letters signify a male, the first three letters signify a female, the first four letters signify a great one, while the entire word signifies a great woman?",
            "answer": "heroine",
            "accept_patterns": ["a heroine"],
            "hints": ["He, her, hero...", "The answer is a person.", "She saves the day."],
        },
        {
            "question": "I have branches but no fruit, trunk, or leaves. What am I?",
            "answer": "bank",
            "accept_patterns": ["a bank"],
            "hints": ["I deal with money.", "I have multiple locations.", "People open accounts with me."],
        },
        {
            "question": "Forward I am heavy, but backward I am not. What am I?",
            "answer": "ton",
            "accept_patterns": ["a ton"],
            "hints": ["I'm a unit of measurement.", "Spell me backward.", "I measure weight."],
        },
        {
            "question": "What is it that given one, you'll have either two or none?",
            "answer": "choice",
            "accept_patterns": ["a choice", "option", "an option"],
            "hints": ["It's abstract.", "You make me every day.", "Decisions involve me."],
        },
        {
            "question": "I am always hungry, I must always be fed. The finger I touch will soon turn red. What am I?",
            "answer": "fire",
            "accept_patterns": ["a fire", "flame"],
            "hints": ["I consume things.", "I produce heat.", "Water puts me out."],
        },
    ],
}


# ---------------------------------------------------------------------------
# PuzzleState dataclass
# ---------------------------------------------------------------------------
@dataclass
class PuzzleState:
    """Serializable puzzle state stored in campaign_state['_active_puzzle']."""

    puzzle_type: str
    context: str
    difficulty: str
    question: str
    answer: str
    accept_patterns: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)
    attempts: int = 0
    max_attempts: int = 3
    hints_used: int = 0
    solved: bool = False
    failed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "puzzle_type": self.puzzle_type,
            "context": self.context,
            "difficulty": self.difficulty,
            "question": self.question,
            "answer": self.answer,
            "accept_patterns": list(self.accept_patterns),
            "hints": list(self.hints),
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "hints_used": self.hints_used,
            "solved": self.solved,
            "failed": self.failed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PuzzleState:
        return cls(
            puzzle_type=str(data.get("puzzle_type", "")),
            context=str(data.get("context", "")),
            difficulty=str(data.get("difficulty", "medium")),
            question=str(data.get("question", "")),
            answer=str(data.get("answer", "")),
            accept_patterns=list(data.get("accept_patterns") or []),
            hints=list(data.get("hints") or []),
            attempts=int(data.get("attempts", 0)),
            max_attempts=int(data.get("max_attempts", 3)),
            hints_used=int(data.get("hints_used", 0)),
            solved=bool(data.get("solved", False)),
            failed=bool(data.get("failed", False)),
        )


# ---------------------------------------------------------------------------
# Puzzle type generators
# ---------------------------------------------------------------------------

def _generate_riddle(trigger: PuzzleTrigger) -> PuzzleState:
    """Select a riddle from the curated bank using the context as seed."""
    difficulty = trigger.difficulty if trigger.difficulty in _RIDDLE_BANK else "medium"
    bank = _RIDDLE_BANK[difficulty]
    # Deterministic selection based on context hash
    idx = int(hashlib.sha256(trigger.context.encode()).hexdigest(), 16) % len(bank)
    entry = bank[idx]
    return PuzzleState(
        puzzle_type="riddle",
        context=trigger.context,
        difficulty=difficulty,
        question=entry["question"],
        answer=entry["answer"].lower().strip(),
        accept_patterns=[p.lower().strip() for p in entry.get("accept_patterns", [])],
        hints=list(entry.get("hints", [])),
    )


def _generate_math(trigger: PuzzleTrigger) -> PuzzleState:
    """Generate a deterministic math puzzle."""
    seed = int(hashlib.sha256(trigger.context.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    difficulty = trigger.difficulty

    if difficulty == "easy":
        a, b = rng.randint(2, 20), rng.randint(2, 20)
        op = rng.choice(["+", "-", "*"])
        if op == "+":
            answer = a + b
            question = f"What is {a} + {b}?"
        elif op == "-":
            a, b = max(a, b), min(a, b)
            answer = a - b
            question = f"What is {a} - {b}?"
        else:
            a, b = rng.randint(2, 12), rng.randint(2, 12)
            answer = a * b
            question = f"What is {a} x {b}?"
        hints = [
            f"This is a simple {op} problem.",
            f"The answer is between {answer - 5} and {answer + 5}.",
            f"The first digit is {str(answer)[0]}.",
        ]
    elif difficulty == "hard":
        # Word problem with two unknowns
        x = rng.randint(3, 15)
        y = rng.randint(3, 15)
        total = x + y
        diff = abs(x - y)
        question = (
            f"A merchant has two crates. Together they hold {total} items. "
            f"The larger crate holds {diff} more items than the smaller. "
            f"How many items are in the larger crate?"
        )
        answer = max(x, y)
        hints = [
            "Set up two equations: a + b = total, a - b = difference.",
            f"The smaller crate has {min(x, y)} items.",
            f"Add the total and difference, then divide by 2.",
        ]
    else:  # medium
        a = rng.randint(5, 25)
        b = rng.randint(2, 10)
        c = rng.randint(1, 15)
        answer = a * b + c
        question = f"What is {a} x {b} + {c}?"
        hints = [
            "Multiply first, then add.",
            f"{a} x {b} = {a * b}.",
            f"The answer is {answer}.",
        ]

    return PuzzleState(
        puzzle_type="math",
        context=trigger.context,
        difficulty=difficulty,
        question=question,
        answer=str(answer),
        accept_patterns=[],
        hints=hints,
    )


def _generate_sequence(trigger: PuzzleTrigger) -> PuzzleState:
    """Generate a number sequence puzzle."""
    seed = int(hashlib.sha256(trigger.context.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    difficulty = trigger.difficulty

    if difficulty == "easy":
        start = rng.randint(1, 10)
        step = rng.randint(2, 5)
        seq = [start + step * i for i in range(5)]
        answer = start + step * 5
        question = f"What comes next in the sequence? {', '.join(str(n) for n in seq)}, ?"
        hints = [
            "Look for a constant difference between numbers.",
            f"Each number increases by {step}.",
            f"The last shown number is {seq[-1]}.",
        ]
    elif difficulty == "hard":
        # Fibonacci-variant
        a, b = rng.randint(1, 5), rng.randint(1, 5)
        seq = [a, b]
        for _ in range(4):
            seq.append(seq[-1] + seq[-2])
        answer = seq[-1] + seq[-2]
        seq_display = seq[:5]
        question = f"What comes next in the sequence? {', '.join(str(n) for n in seq_display)}, {seq[5]}, ?"
        hints = [
            "Each number is the sum of the two before it.",
            f"The pattern starts with {a} and {b}.",
            f"{seq[5]} + {seq[4]} = ?",
        ]
    else:  # medium
        start = rng.randint(1, 5)
        ratio = rng.randint(2, 3)
        seq = [start * (ratio ** i) for i in range(5)]
        answer = start * (ratio ** 5)
        question = f"What comes next in the sequence? {', '.join(str(n) for n in seq)}, ?"
        hints = [
            "Look for a constant ratio between numbers.",
            f"Each number is multiplied by {ratio}.",
            f"The last shown number is {seq[-1]}.",
        ]

    return PuzzleState(
        puzzle_type="sequence",
        context=trigger.context,
        difficulty=difficulty,
        question=question,
        answer=str(answer),
        accept_patterns=[],
        hints=hints,
    )


def _generate_cipher(trigger: PuzzleTrigger) -> PuzzleState:
    """Generate a Caesar cipher puzzle."""
    seed = int(hashlib.sha256(trigger.context.encode()).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    difficulty = trigger.difficulty

    phrases = [
        "the treasure is hidden",
        "seek the golden key",
        "trust no one here",
        "danger lurks ahead",
        "the answer lies within",
        "follow the ancient path",
        "look beyond the veil",
        "time reveals all secrets",
    ]
    phrase = rng.choice(phrases)

    if difficulty == "easy":
        shift = rng.randint(1, 3)
    elif difficulty == "hard":
        shift = rng.randint(8, 13)
    else:  # medium
        shift = rng.randint(4, 7)

    encrypted = ""
    for ch in phrase:
        if ch.isalpha():
            base = ord("a") if ch.islower() else ord("A")
            encrypted += chr((ord(ch) - base + shift) % 26 + base)
        else:
            encrypted += ch

    question = (
        f"Decode this Caesar cipher (each letter has been shifted forward): {encrypted}"
    )
    hints_list = [
        f"The shift is {shift} positions forward.",
        f"The first word decodes to '{phrase.split()[0]}'.",
        f"Try shifting each letter back by {shift}.",
    ]

    return PuzzleState(
        puzzle_type="cipher",
        context=trigger.context,
        difficulty=difficulty,
        question=question,
        answer=phrase.lower().strip(),
        accept_patterns=[],
        hints=hints_list,
    )


_GENERATORS: dict[str, Any] = {
    "riddle": _generate_riddle,
    "math": _generate_math,
    "sequence": _generate_sequence,
    "cipher": _generate_cipher,
}


# ---------------------------------------------------------------------------
# PuzzleEngine — static interface
# ---------------------------------------------------------------------------
class PuzzleEngine:
    @staticmethod
    def generate(trigger: PuzzleTrigger) -> PuzzleState:
        """Create a new puzzle from a trigger."""
        gen = _GENERATORS.get(trigger.puzzle_type, _generate_riddle)
        return gen(trigger)

    @staticmethod
    def validate_answer(state: PuzzleState, player_input: str) -> tuple[bool, str]:
        """Check *player_input* against the puzzle answer.

        Returns ``(correct, feedback_message)``.
        """
        if state.solved or state.failed:
            return False, "This puzzle is already finished."

        normalized = player_input.strip().lower()

        # For math/sequence puzzles, try numeric comparison
        if state.puzzle_type in ("math", "sequence"):
            try:
                if abs(float(normalized) - float(state.answer)) < 0.01:
                    state.solved = True
                    return True, "Correct!"
            except (ValueError, TypeError):
                pass

        # String comparison
        if normalized == state.answer or normalized in state.accept_patterns:
            state.solved = True
            return True, "Correct!"

        state.attempts += 1
        remaining = state.max_attempts - state.attempts
        if remaining <= 0:
            state.failed = True
            return False, f"Wrong. The answer was: {state.answer}. The puzzle is failed."
        return False, f"That's not right. {remaining} attempt{'s' if remaining != 1 else ''} remaining."

    @staticmethod
    def get_hint(state: PuzzleState) -> str | None:
        """Return the next unused hint, or ``None`` if all used."""
        if state.hints_used >= len(state.hints):
            return None
        hint = state.hints[state.hints_used]
        state.hints_used += 1
        return hint

    @staticmethod
    def render_prompt_section(state: PuzzleState) -> str:
        """Build the ACTIVE_PUZZLE prompt injection section."""
        lines = [
            "ACTIVE_PUZZLE:",
            f"  type: {state.puzzle_type}",
            f"  question: \"{state.question}\"",
            f"  attempts: {state.attempts}/{state.max_attempts}",
            f"  hint_available: {str(state.hints_used < len(state.hints)).lower()}",
            "  instruction: \"Player is attempting a puzzle. If their action looks like an answer, "
            "the harness has already checked — see PUZZLE_RESULT if present. "
            "Narrate the outcome accordingly.\"",
        ]
        return "\n".join(lines)

    @staticmethod
    def is_puzzle_attempt(state: PuzzleState, player_input: str) -> bool:
        """Heuristic: is *player_input* a puzzle answer attempt?

        Short input (<=60 chars) with no obvious action verbs → puzzle attempt.
        Special keywords (hint, give up) are also handled.
        """
        stripped = player_input.strip().lower()

        # Special keywords always count
        if stripped in ("hint", "puzzle hint", "give up", "abandon puzzle"):
            return True

        # Too long → probably a normal action
        if len(stripped) > 60:
            return False

        # Check for action verbs at the start
        first_word = stripped.split()[0] if stripped.split() else ""
        if first_word in _ACTION_VERBS:
            return False

        return True
