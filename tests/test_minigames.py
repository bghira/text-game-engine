"""Tests for the minigame framework module."""

from __future__ import annotations

from text_game_engine.core.minigames import MinigameEngine, MinigameState
from text_game_engine.core.types import MinigameChallenge


class TestTicTacToe:
    def _new_game(self) -> MinigameState:
        challenge = MinigameChallenge(
            game_type="tic_tac_toe",
            opponent_slug="tavern-keeper",
            stakes="a free drink",
        )
        return MinigameEngine.new_game(challenge)

    def test_new_game_state(self):
        state = self._new_game()
        assert state.game_type == "tic_tac_toe"
        assert state.opponent_slug == "tavern-keeper"
        assert state.status == "player_turn"
        assert state.board is not None
        assert len(state.board) == 3
        assert all(len(row) == 3 for row in state.board)

    def test_valid_move(self):
        state = self._new_game()
        valid, msg = MinigameEngine.player_move(state, "A1")
        assert valid
        assert state.board[0][0] == "X"

    def test_invalid_position(self):
        state = self._new_game()
        valid, msg = MinigameEngine.player_move(state, "Z9")
        assert not valid
        assert "Invalid" in msg

    def test_position_taken(self):
        state = self._new_game()
        MinigameEngine.player_move(state, "center")
        # NPC will have moved somewhere too; try the same spot
        valid, msg = MinigameEngine.player_move(state, "center")
        assert not valid
        assert "taken" in msg.lower()

    def test_npc_auto_plays(self):
        state = self._new_game()
        MinigameEngine.player_move(state, "A1")
        # NPC should have placed a mark somewhere
        npc_marks = sum(
            1 for r in range(3) for c in range(3) if state.board[r][c] == "O"
        )
        # Game might be over (player_won/draw in edge cases), but usually NPC has moved
        assert npc_marks >= 1 or MinigameEngine.is_finished(state)

    def test_draw_detection(self):
        state = self._new_game()
        # Fill the board to force a draw/win (minimax NPC won't let player win easily)
        # Just play until the game ends
        moves = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]
        for move in moves:
            if MinigameEngine.is_finished(state):
                break
            MinigameEngine.player_move(state, move)
        assert MinigameEngine.is_finished(state)

    def test_render_board(self):
        state = self._new_game()
        rendered = MinigameEngine.render_board(state)
        assert "A" in rendered
        assert "B" in rendered
        assert "C" in rendered

    def test_is_game_move(self):
        state = self._new_game()
        assert MinigameEngine.is_game_move(state, "A1")
        assert MinigameEngine.is_game_move(state, "center")
        assert MinigameEngine.is_game_move(state, "5")
        assert MinigameEngine.is_game_move(state, "top left")
        assert not MinigameEngine.is_game_move(state, "look around")

    def test_number_positions(self):
        state = self._new_game()
        valid, msg = MinigameEngine.player_move(state, "5")
        assert valid
        assert state.board[1][1] == "X"  # center


class TestNim:
    def _new_game(self) -> MinigameState:
        challenge = MinigameChallenge(
            game_type="nim",
            opponent_slug="old-hermit",
            stakes="passage through the cave",
        )
        return MinigameEngine.new_game(challenge)

    def test_new_game_state(self):
        state = self._new_game()
        assert state.game_type == "nim"
        assert state.board == [3, 5, 7]
        assert state.status == "player_turn"

    def test_valid_move(self):
        state = self._new_game()
        valid, msg = MinigameEngine.player_move(state, "take 2 from pile 1")
        assert valid
        assert state.board[0] == 1

    def test_invalid_move_format(self):
        state = self._new_game()
        valid, msg = MinigameEngine.player_move(state, "remove three stones")
        assert not valid

    def test_take_too_many(self):
        state = self._new_game()
        valid, msg = MinigameEngine.player_move(state, "take 10 from pile 1")
        assert not valid
        assert "only has" in msg.lower()

    def test_invalid_pile(self):
        state = self._new_game()
        valid, msg = MinigameEngine.player_move(state, "take 1 from pile 5")
        assert not valid

    def test_play_to_completion(self):
        state = self._new_game()
        for _ in range(50):
            if MinigameEngine.is_finished(state):
                break
            # Take 1 from first non-empty pile
            for i, p in enumerate(state.board):
                if p > 0:
                    MinigameEngine.player_move(state, f"take 1 from pile {i+1}")
                    break
        assert MinigameEngine.is_finished(state)

    def test_is_game_move(self):
        state = self._new_game()
        assert MinigameEngine.is_game_move(state, "take 2 from pile 1")
        assert not MinigameEngine.is_game_move(state, "look around")

    def test_render(self):
        state = self._new_game()
        rendered = MinigameEngine.render_board(state)
        assert "Pile 1" in rendered
        assert "Pile 2" in rendered
        assert "Pile 3" in rendered


class TestDiceDuel:
    def _new_game(self) -> MinigameState:
        challenge = MinigameChallenge(
            game_type="dice_duel",
            opponent_slug="gambler",
            stakes="10 gold",
        )
        return MinigameEngine.new_game(challenge)

    def test_new_game_state(self):
        state = self._new_game()
        assert state.game_type == "dice_duel"
        assert state.extra.get("best_of") == 3

    def test_roll(self):
        state = self._new_game()
        valid, msg = MinigameEngine.player_move(state, "roll")
        assert valid
        assert "rolled" in msg.lower()

    def test_invalid_move(self):
        state = self._new_game()
        valid, msg = MinigameEngine.player_move(state, "attack")
        assert not valid

    def test_play_to_completion(self):
        state = self._new_game()
        for _ in range(20):
            if MinigameEngine.is_finished(state):
                break
            MinigameEngine.player_move(state, "roll")
        assert MinigameEngine.is_finished(state)

    def test_is_game_move(self):
        state = self._new_game()
        assert MinigameEngine.is_game_move(state, "roll")
        assert MinigameEngine.is_game_move(state, "roll dice")
        assert not MinigameEngine.is_game_move(state, "fight")

    def test_render(self):
        state = self._new_game()
        rendered = MinigameEngine.render_board(state)
        assert "Dice Duel" in rendered


class TestCoinFlip:
    def _new_game(self) -> MinigameState:
        challenge = MinigameChallenge(
            game_type="coin_flip",
            opponent_slug="fortune-teller",
            stakes="a reading",
        )
        return MinigameEngine.new_game(challenge)

    def test_new_game_state(self):
        state = self._new_game()
        assert state.game_type == "coin_flip"

    def test_heads(self):
        state = self._new_game()
        valid, msg = MinigameEngine.player_move(state, "heads")
        assert valid
        assert "coin" in msg.lower()

    def test_tails(self):
        state = self._new_game()
        valid, msg = MinigameEngine.player_move(state, "tails")
        assert valid

    def test_invalid(self):
        state = self._new_game()
        valid, msg = MinigameEngine.player_move(state, "edge")
        assert not valid

    def test_is_game_move(self):
        state = self._new_game()
        assert MinigameEngine.is_game_move(state, "heads")
        assert MinigameEngine.is_game_move(state, "tails")
        assert not MinigameEngine.is_game_move(state, "run away")

    def test_completes(self):
        state = self._new_game()
        # Best of 1, so one call should finish
        MinigameEngine.player_move(state, "heads")
        assert MinigameEngine.is_finished(state)


class TestMinigameQuit:
    def test_forfeit(self):
        challenge = MinigameChallenge(
            game_type="tic_tac_toe",
            opponent_slug="npc",
            stakes="pride",
        )
        state = MinigameEngine.new_game(challenge)
        assert MinigameEngine.is_game_move(state, "forfeit")
        assert MinigameEngine.is_game_move(state, "quit game")
        assert MinigameEngine.is_game_move(state, "resign")


class TestMinigameSerialization:
    def test_round_trip(self):
        challenge = MinigameChallenge(
            game_type="tic_tac_toe",
            opponent_slug="merchant",
            stakes="rare gem",
        )
        state = MinigameEngine.new_game(challenge)
        MinigameEngine.player_move(state, "center")
        d = state.to_dict()
        restored = MinigameState.from_dict(d)
        assert restored.game_type == "tic_tac_toe"
        assert restored.board[1][1] == "X"
        assert restored.turn_count == state.turn_count


class TestMinigameUnknownType:
    def test_unknown_raises(self):
        challenge = MinigameChallenge(
            game_type="chess",
            opponent_slug="wizard",
            stakes="kingdom",
        )
        try:
            MinigameEngine.new_game(challenge)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestRenderPromptSection:
    def test_render(self):
        challenge = MinigameChallenge(
            game_type="tic_tac_toe",
            opponent_slug="barkeep",
            stakes="free ale",
        )
        state = MinigameEngine.new_game(challenge)
        text = MinigameEngine.render_prompt_section(state)
        assert "ACTIVE_MINIGAME:" in text
        assert "tic_tac_toe" in text
        assert "barkeep" in text
        assert "free ale" in text
