"""Tests for the puzzle engine module."""

from __future__ import annotations

from text_game_engine.core.puzzles import PuzzleEngine, PuzzleState
from text_game_engine.core.types import PuzzleTrigger


class TestPuzzleGeneration:
    def test_riddle_easy(self):
        trigger = PuzzleTrigger(puzzle_type="riddle", context="a sphinx blocks the way", difficulty="easy")
        state = PuzzleEngine.generate(trigger)
        assert state.puzzle_type == "riddle"
        assert state.difficulty == "easy"
        assert state.question
        assert state.answer
        assert not state.solved
        assert not state.failed
        assert state.attempts == 0

    def test_riddle_medium(self):
        trigger = PuzzleTrigger(puzzle_type="riddle", context="ancient door puzzle", difficulty="medium")
        state = PuzzleEngine.generate(trigger)
        assert state.puzzle_type == "riddle"
        assert state.question

    def test_riddle_hard(self):
        trigger = PuzzleTrigger(puzzle_type="riddle", context="wizard's final test", difficulty="hard")
        state = PuzzleEngine.generate(trigger)
        assert state.question
        assert len(state.hints) > 0

    def test_math_easy(self):
        trigger = PuzzleTrigger(puzzle_type="math", context="merchant math", difficulty="easy")
        state = PuzzleEngine.generate(trigger)
        assert state.puzzle_type == "math"
        assert state.answer.lstrip("-").isdigit()

    def test_math_medium(self):
        trigger = PuzzleTrigger(puzzle_type="math", context="lock combination", difficulty="medium")
        state = PuzzleEngine.generate(trigger)
        assert state.answer

    def test_math_hard(self):
        trigger = PuzzleTrigger(puzzle_type="math", context="two crate problem", difficulty="hard")
        state = PuzzleEngine.generate(trigger)
        assert "crate" in state.question.lower() or state.answer.lstrip("-").isdigit()

    def test_sequence_easy(self):
        trigger = PuzzleTrigger(puzzle_type="sequence", context="number lock", difficulty="easy")
        state = PuzzleEngine.generate(trigger)
        assert state.puzzle_type == "sequence"
        assert state.answer

    def test_sequence_medium(self):
        trigger = PuzzleTrigger(puzzle_type="sequence", context="geometric puzzle", difficulty="medium")
        state = PuzzleEngine.generate(trigger)
        assert state.answer

    def test_sequence_hard(self):
        trigger = PuzzleTrigger(puzzle_type="sequence", context="fibonacci altar", difficulty="hard")
        state = PuzzleEngine.generate(trigger)
        assert state.answer

    def test_cipher_easy(self):
        trigger = PuzzleTrigger(puzzle_type="cipher", context="coded message", difficulty="easy")
        state = PuzzleEngine.generate(trigger)
        assert state.puzzle_type == "cipher"
        assert state.answer
        assert state.question != state.answer  # cipher text != plaintext

    def test_cipher_hard(self):
        trigger = PuzzleTrigger(puzzle_type="cipher", context="spy cipher", difficulty="hard")
        state = PuzzleEngine.generate(trigger)
        assert state.answer

    def test_deterministic_selection(self):
        """Same context should produce same riddle."""
        trigger = PuzzleTrigger(puzzle_type="riddle", context="test context", difficulty="easy")
        s1 = PuzzleEngine.generate(trigger)
        s2 = PuzzleEngine.generate(trigger)
        assert s1.question == s2.question
        assert s1.answer == s2.answer

    def test_unknown_type_defaults_to_riddle(self):
        trigger = PuzzleTrigger(puzzle_type="unknown_type", context="test", difficulty="easy")
        state = PuzzleEngine.generate(trigger)
        assert state.puzzle_type == "riddle"


class TestPuzzleValidation:
    def _make_state(self) -> PuzzleState:
        return PuzzleState(
            puzzle_type="riddle",
            context="test",
            difficulty="easy",
            question="What has hands but can't clap?",
            answer="clock",
            accept_patterns=["a clock", "watch"],
            hints=["hint1", "hint2"],
            max_attempts=3,
        )

    def test_correct_answer(self):
        state = self._make_state()
        correct, feedback = PuzzleEngine.validate_answer(state, "clock")
        assert correct
        assert state.solved
        assert "Correct" in feedback

    def test_correct_accept_pattern(self):
        state = self._make_state()
        correct, feedback = PuzzleEngine.validate_answer(state, "a clock")
        assert correct
        assert state.solved

    def test_case_insensitive(self):
        state = self._make_state()
        correct, _ = PuzzleEngine.validate_answer(state, "CLOCK")
        assert correct

    def test_wrong_answer_increments_attempts(self):
        state = self._make_state()
        correct, feedback = PuzzleEngine.validate_answer(state, "wrong")
        assert not correct
        assert state.attempts == 1
        assert "remaining" in feedback.lower()

    def test_max_attempts_triggers_failure(self):
        state = self._make_state()
        PuzzleEngine.validate_answer(state, "wrong1")
        PuzzleEngine.validate_answer(state, "wrong2")
        correct, feedback = PuzzleEngine.validate_answer(state, "wrong3")
        assert not correct
        assert state.failed
        assert "failed" in feedback.lower()

    def test_already_solved(self):
        state = self._make_state()
        state.solved = True
        correct, feedback = PuzzleEngine.validate_answer(state, "clock")
        assert not correct
        assert "finished" in feedback.lower()

    def test_math_numeric_validation(self):
        state = PuzzleState(
            puzzle_type="math",
            context="test",
            difficulty="easy",
            question="What is 5 + 3?",
            answer="8",
            max_attempts=3,
        )
        correct, _ = PuzzleEngine.validate_answer(state, "8")
        assert correct

    def test_math_float_tolerance(self):
        state = PuzzleState(
            puzzle_type="math",
            context="test",
            difficulty="easy",
            question="test",
            answer="8",
            max_attempts=3,
        )
        correct, _ = PuzzleEngine.validate_answer(state, "8.00")
        assert correct


class TestPuzzleHints:
    def test_hints_progress(self):
        state = PuzzleState(
            puzzle_type="riddle",
            context="test",
            difficulty="easy",
            question="test",
            answer="test",
            hints=["hint1", "hint2", "hint3"],
        )
        assert PuzzleEngine.get_hint(state) == "hint1"
        assert state.hints_used == 1
        assert PuzzleEngine.get_hint(state) == "hint2"
        assert state.hints_used == 2
        assert PuzzleEngine.get_hint(state) == "hint3"
        assert state.hints_used == 3
        assert PuzzleEngine.get_hint(state) is None

    def test_no_hints(self):
        state = PuzzleState(
            puzzle_type="riddle",
            context="test",
            difficulty="easy",
            question="test",
            answer="test",
            hints=[],
        )
        assert PuzzleEngine.get_hint(state) is None


class TestIsPuzzleAttempt:
    def test_short_text_is_attempt(self):
        state = PuzzleState(
            puzzle_type="riddle",
            context="test",
            difficulty="easy",
            question="test",
            answer="clock",
        )
        assert PuzzleEngine.is_puzzle_attempt(state, "clock")
        assert PuzzleEngine.is_puzzle_attempt(state, "a clock")
        assert PuzzleEngine.is_puzzle_attempt(state, "42")

    def test_action_verb_is_not_attempt(self):
        state = PuzzleState(
            puzzle_type="riddle",
            context="test",
            difficulty="easy",
            question="test",
            answer="clock",
        )
        assert not PuzzleEngine.is_puzzle_attempt(state, "look around the room")
        assert not PuzzleEngine.is_puzzle_attempt(state, "take the key")
        assert not PuzzleEngine.is_puzzle_attempt(state, "go north")

    def test_special_keywords(self):
        state = PuzzleState(
            puzzle_type="riddle",
            context="test",
            difficulty="easy",
            question="test",
            answer="clock",
        )
        assert PuzzleEngine.is_puzzle_attempt(state, "hint")
        assert PuzzleEngine.is_puzzle_attempt(state, "puzzle hint")
        assert PuzzleEngine.is_puzzle_attempt(state, "give up")
        assert PuzzleEngine.is_puzzle_attempt(state, "abandon puzzle")

    def test_long_text_is_not_attempt(self):
        state = PuzzleState(
            puzzle_type="riddle",
            context="test",
            difficulty="easy",
            question="test",
            answer="clock",
        )
        long_text = "I would like to carefully examine the mechanism on the wall and try to figure out what makes it tick"
        assert not PuzzleEngine.is_puzzle_attempt(state, long_text)


class TestPuzzleSerialization:
    def test_round_trip(self):
        state = PuzzleState(
            puzzle_type="cipher",
            context="spy note",
            difficulty="hard",
            question="Decode: khoor",
            answer="hello",
            accept_patterns=["hello world"],
            hints=["shift 3"],
            attempts=1,
            max_attempts=3,
            hints_used=0,
        )
        d = state.to_dict()
        restored = PuzzleState.from_dict(d)
        assert restored.puzzle_type == "cipher"
        assert restored.answer == "hello"
        assert restored.attempts == 1
        assert restored.hints == ["shift 3"]


class TestRenderPromptSection:
    def test_render(self):
        state = PuzzleState(
            puzzle_type="riddle",
            context="sphinx",
            difficulty="medium",
            question="What walks on four legs?",
            answer="human",
            hints=["biology"],
            attempts=1,
            max_attempts=3,
        )
        text = PuzzleEngine.render_prompt_section(state)
        assert "ACTIVE_PUZZLE:" in text
        assert "riddle" in text
        assert "1/3" in text
        assert "What walks on four legs?" in text
