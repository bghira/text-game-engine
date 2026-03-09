"""Minigame framework — harness-managed game state machines."""

from __future__ import annotations

import copy
import random
import re
from dataclasses import dataclass, field
from typing import Any

from .types import MinigameChallenge

# ---------------------------------------------------------------------------
# MinigameState dataclass
# ---------------------------------------------------------------------------

@dataclass
class MinigameState:
    """Serializable minigame state in campaign_state['_active_minigame']."""

    game_type: str
    opponent_slug: str
    stakes: str
    board: Any = None
    player_mark: str = "X"
    npc_mark: str = "O"
    status: str = "player_turn"  # player_turn, npc_turn, player_won, npc_won, draw
    turn_count: int = 0
    max_turns: int = 50
    # Extra state for multi-round games
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_type": self.game_type,
            "opponent_slug": self.opponent_slug,
            "stakes": self.stakes,
            "board": self.board,
            "player_mark": self.player_mark,
            "npc_mark": self.npc_mark,
            "status": self.status,
            "turn_count": self.turn_count,
            "max_turns": self.max_turns,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MinigameState:
        return cls(
            game_type=str(data.get("game_type", "")),
            opponent_slug=str(data.get("opponent_slug", "")),
            stakes=str(data.get("stakes", "")),
            board=data.get("board"),
            player_mark=str(data.get("player_mark", "X")),
            npc_mark=str(data.get("npc_mark", "O")),
            status=str(data.get("status", "player_turn")),
            turn_count=int(data.get("turn_count", 0)),
            max_turns=int(data.get("max_turns", 50)),
            extra=dict(data.get("extra") or {}),
        )


# ---------------------------------------------------------------------------
# Tic-Tac-Toe
# ---------------------------------------------------------------------------

_TTT_POSITION_MAP: dict[str, tuple[int, int]] = {
    "a1": (0, 0), "b1": (0, 1), "c1": (0, 2),
    "a2": (1, 0), "b2": (1, 1), "c2": (1, 2),
    "a3": (2, 0), "b3": (2, 1), "c3": (2, 2),
    "1": (0, 0), "2": (0, 1), "3": (0, 2),
    "4": (1, 0), "5": (1, 1), "6": (1, 2),
    "7": (2, 0), "8": (2, 1), "9": (2, 2),
    "top left": (0, 0), "top center": (0, 1), "top right": (0, 2),
    "top middle": (0, 1),
    "middle left": (1, 0), "center": (1, 1), "middle": (1, 1),
    "middle right": (1, 2),
    "bottom left": (2, 0), "bottom center": (2, 1), "bottom right": (2, 2),
    "bottom middle": (2, 1),
}


def _ttt_new_board() -> list[list[None]]:
    return [[None, None, None] for _ in range(3)]


def _ttt_winner(board: list[list]) -> str | None:
    """Return 'X', 'O', or None."""
    for row in board:
        if row[0] and row[0] == row[1] == row[2]:
            return row[0]
    for col in range(3):
        if board[0][col] and board[0][col] == board[1][col] == board[2][col]:
            return board[0][col]
    if board[0][0] and board[0][0] == board[1][1] == board[2][2]:
        return board[0][0]
    if board[0][2] and board[0][2] == board[1][1] == board[2][0]:
        return board[0][2]
    return None


def _ttt_full(board: list[list]) -> bool:
    return all(board[r][c] is not None for r in range(3) for c in range(3))


def _ttt_available(board: list[list]) -> list[tuple[int, int]]:
    return [(r, c) for r in range(3) for c in range(3) if board[r][c] is None]


def _ttt_minimax(board: list[list], mark: str, other: str, is_maximizing: bool) -> int:
    winner = _ttt_winner(board)
    if winner == mark:
        return 1
    if winner == other:
        return -1
    if _ttt_full(board):
        return 0
    if is_maximizing:
        best = -2
        for r, c in _ttt_available(board):
            board[r][c] = mark
            best = max(best, _ttt_minimax(board, mark, other, False))
            board[r][c] = None
        return best
    else:
        best = 2
        for r, c in _ttt_available(board):
            board[r][c] = other
            best = min(best, _ttt_minimax(board, mark, other, True))
            board[r][c] = None
        return best


def _ttt_npc_move(board: list[list], npc_mark: str, player_mark: str) -> tuple[int, int] | None:
    """Minimax-based NPC move for tic-tac-toe."""
    avail = _ttt_available(board)
    if not avail:
        return None
    best_score = -2
    best_move = avail[0]
    for r, c in avail:
        board[r][c] = npc_mark
        score = _ttt_minimax(board, npc_mark, player_mark, False)
        board[r][c] = None
        if score > best_score:
            best_score = score
            best_move = (r, c)
    return best_move


def _ttt_parse_move(move_input: str) -> tuple[int, int] | None:
    normalized = move_input.strip().lower()
    if normalized in _TTT_POSITION_MAP:
        return _TTT_POSITION_MAP[normalized]
    return None


def _ttt_render(board: list[list]) -> str:
    def cell(v):
        return v if v else "."
    lines = ["     A   B   C"]
    for i, row in enumerate(board):
        lines.append(f"  {i+1}  {cell(row[0])} | {cell(row[1])} | {cell(row[2])}")
        if i < 2:
            lines.append("    ---+---+---")
    return "\n".join(lines)


def _ttt_player_move(state: MinigameState, move_input: str) -> tuple[bool, str]:
    pos = _ttt_parse_move(move_input)
    if pos is None:
        return False, "Invalid position. Use A1-C3, 1-9, or descriptions like 'center'."

    r, c = pos
    board = state.board
    if board[r][c] is not None:
        return False, "That position is already taken."

    board[r][c] = state.player_mark
    state.turn_count += 1

    winner = _ttt_winner(board)
    if winner == state.player_mark:
        state.status = "player_won"
        return True, "You win!"
    if _ttt_full(board):
        state.status = "draw"
        return True, "It's a draw!"

    # NPC move
    npc_pos = _ttt_npc_move(board, state.npc_mark, state.player_mark)
    if npc_pos:
        board[npc_pos[0]][npc_pos[1]] = state.npc_mark
        state.turn_count += 1

    winner = _ttt_winner(board)
    if winner == state.npc_mark:
        state.status = "npc_won"
        return True, "The opponent wins."
    if _ttt_full(board):
        state.status = "draw"
        return True, "It's a draw!"

    state.status = "player_turn"
    return True, "Move accepted."


def _ttt_is_move(player_input: str) -> bool:
    return _ttt_parse_move(player_input) is not None


# ---------------------------------------------------------------------------
# Nim
# ---------------------------------------------------------------------------

def _nim_new_board() -> list[int]:
    return [3, 5, 7]


def _nim_render(piles: list[int]) -> str:
    lines = ["Piles:"]
    for i, count in enumerate(piles):
        stones = "O " * count if count > 0 else "(empty)"
        lines.append(f"  Pile {i+1}: {stones.strip()} ({count})")
    lines.append("Take stones: 'take N from pile M'")
    lines.append("Rule: The player who takes the LAST stone LOSES.")
    return "\n".join(lines)


def _nim_parse_move(move_input: str) -> tuple[int, int] | None:
    """Parse 'take N from pile M' → (pile_index, count)."""
    m = re.match(r"take\s+(\d+)\s+from\s+pile\s+(\d+)", move_input.strip().lower())
    if m:
        count = int(m.group(1))
        pile = int(m.group(2)) - 1  # 0-indexed
        return pile, count
    return None


def _nim_xor(piles: list[int]) -> int:
    result = 0
    for p in piles:
        result ^= p
    return result


def _nim_npc_move(piles: list[int]) -> tuple[int, int]:
    """Optimal nim strategy using XOR."""
    xor_val = _nim_xor(piles)
    if xor_val == 0:
        # Losing position — take 1 from first non-empty pile
        for i, p in enumerate(piles):
            if p > 0:
                return i, 1
    # Find a pile to reduce to make XOR = 0
    for i, p in enumerate(piles):
        target = p ^ xor_val
        if target < p:
            take = p - target
            # In misere, if all other piles are 0 or 1, leave exactly 1
            others = [piles[j] for j in range(len(piles)) if j != i]
            if all(x <= 1 for x in others):
                non_empty_others = sum(1 for x in others if x == 1)
                # Leave odd number of 1-piles total
                if non_empty_others % 2 == 0:
                    take = p  # take all
                else:
                    take = p - 1 if p > 1 else p  # leave 1
            return i, take
    # Fallback
    for i, p in enumerate(piles):
        if p > 0:
            return i, 1
    return 0, 0


def _nim_is_over(piles: list[int]) -> bool:
    return sum(piles) == 0


def _nim_player_move(state: MinigameState, move_input: str) -> tuple[bool, str]:
    parsed = _nim_parse_move(move_input)
    if parsed is None:
        return False, "Invalid move. Use 'take N from pile M' (e.g. 'take 2 from pile 1')."

    pile_idx, count = parsed
    piles = state.board
    if pile_idx < 0 or pile_idx >= len(piles):
        return False, f"Invalid pile. Choose pile 1-{len(piles)}."
    if count < 1:
        return False, "You must take at least 1 stone."
    if count > piles[pile_idx]:
        return False, f"Pile {pile_idx + 1} only has {piles[pile_idx]} stones."

    piles[pile_idx] -= count
    state.turn_count += 1

    # Check if player took the last stone (player loses in misere)
    if _nim_is_over(piles):
        state.status = "npc_won"
        return True, "You took the last stone. You lose!"

    # NPC move
    npc_pile, npc_count = _nim_npc_move(piles)
    if npc_count > 0:
        piles[npc_pile] -= npc_count
        state.turn_count += 1

    if _nim_is_over(piles):
        state.status = "player_won"
        return True, f"The opponent took the last stone from pile {npc_pile + 1}. You win!"

    state.status = "player_turn"
    return True, f"Opponent takes {npc_count} from pile {npc_pile + 1}."


def _nim_is_move(player_input: str) -> bool:
    return _nim_parse_move(player_input) is not None


# ---------------------------------------------------------------------------
# Dice Duel
# ---------------------------------------------------------------------------

def _dice_duel_new_extra() -> dict[str, Any]:
    return {"player_wins": 0, "npc_wins": 0, "rounds_played": 0, "best_of": 3, "dice_count": 3}


def _dice_duel_render(state: MinigameState) -> str:
    extra = state.extra
    lines = [
        f"Dice Duel — Best of {extra.get('best_of', 3)}",
        f"  You: {extra.get('player_wins', 0)} wins  |  Opponent: {extra.get('npc_wins', 0)} wins",
        f"  Rounds played: {extra.get('rounds_played', 0)}",
        "",
        "Type 'roll' to roll your dice!",
    ]
    return "\n".join(lines)


def _dice_duel_player_move(state: MinigameState, move_input: str) -> tuple[bool, str]:
    if move_input.strip().lower() != "roll":
        return False, "Type 'roll' to roll your dice."

    extra = state.extra
    dice_count = extra.get("dice_count", 3)
    best_of = extra.get("best_of", 3)
    wins_needed = (best_of // 2) + 1

    player_dice = [random.randint(1, 6) for _ in range(dice_count)]
    npc_dice = [random.randint(1, 6) for _ in range(dice_count)]
    player_total = sum(player_dice)
    npc_total = sum(npc_dice)

    extra["rounds_played"] = extra.get("rounds_played", 0) + 1
    state.turn_count += 1

    result_line = (
        f"You rolled: {player_dice} = {player_total}\n"
        f"Opponent rolled: {npc_dice} = {npc_total}"
    )

    if player_total > npc_total:
        extra["player_wins"] = extra.get("player_wins", 0) + 1
        result_line += "\nYou win this round!"
    elif npc_total > player_total:
        extra["npc_wins"] = extra.get("npc_wins", 0) + 1
        result_line += "\nOpponent wins this round!"
    else:
        result_line += "\nTie! No point awarded."

    if extra.get("player_wins", 0) >= wins_needed:
        state.status = "player_won"
        result_line += f"\n\nYou win the duel {extra['player_wins']}-{extra['npc_wins']}!"
    elif extra.get("npc_wins", 0) >= wins_needed:
        state.status = "npc_won"
        result_line += f"\n\nOpponent wins the duel {extra['npc_wins']}-{extra['player_wins']}!"
    else:
        state.status = "player_turn"

    return True, result_line


def _dice_duel_is_move(player_input: str) -> bool:
    return player_input.strip().lower() in ("roll", "roll dice", "roll the dice")


# ---------------------------------------------------------------------------
# Coin Flip
# ---------------------------------------------------------------------------

def _coin_flip_new_extra() -> dict[str, Any]:
    return {"player_wins": 0, "npc_wins": 0, "rounds_played": 0, "best_of": 1}


def _coin_flip_render(state: MinigameState) -> str:
    extra = state.extra
    best_of = extra.get("best_of", 1)
    lines = [
        f"Coin Flip — Best of {best_of}",
        f"  You: {extra.get('player_wins', 0)}  |  Opponent: {extra.get('npc_wins', 0)}",
        "",
        "Call it: 'heads' or 'tails'",
    ]
    return "\n".join(lines)


def _coin_flip_player_move(state: MinigameState, move_input: str) -> tuple[bool, str]:
    call = move_input.strip().lower()
    if call not in ("heads", "tails"):
        return False, "Call 'heads' or 'tails'."

    extra = state.extra
    best_of = extra.get("best_of", 1)
    wins_needed = (best_of // 2) + 1

    result = random.choice(["heads", "tails"])
    extra["rounds_played"] = extra.get("rounds_played", 0) + 1
    state.turn_count += 1

    if call == result:
        extra["player_wins"] = extra.get("player_wins", 0) + 1
        msg = f"The coin shows {result}. You called it!"
    else:
        extra["npc_wins"] = extra.get("npc_wins", 0) + 1
        msg = f"The coin shows {result}. You called {call} — wrong!"

    if extra.get("player_wins", 0) >= wins_needed:
        state.status = "player_won"
        msg += " You win!"
    elif extra.get("npc_wins", 0) >= wins_needed:
        state.status = "npc_won"
        msg += " You lose!"
    else:
        state.status = "player_turn"

    return True, msg


def _coin_flip_is_move(player_input: str) -> bool:
    return player_input.strip().lower() in ("heads", "tails")


# ---------------------------------------------------------------------------
# Game type registry
# ---------------------------------------------------------------------------

_GAME_INIT = {
    "tic_tac_toe": lambda: {"board": _ttt_new_board()},
    "nim": lambda: {"board": _nim_new_board()},
    "dice_duel": lambda: {"board": None, "extra": _dice_duel_new_extra()},
    "coin_flip": lambda: {"board": None, "extra": _coin_flip_new_extra()},
}

_GAME_PLAYER_MOVE = {
    "tic_tac_toe": _ttt_player_move,
    "nim": _nim_player_move,
    "dice_duel": _dice_duel_player_move,
    "coin_flip": _coin_flip_player_move,
}

_GAME_RENDER = {
    "tic_tac_toe": lambda s: _ttt_render(s.board),
    "nim": lambda s: _nim_render(s.board),
    "dice_duel": _dice_duel_render,
    "coin_flip": _coin_flip_render,
}

_GAME_IS_MOVE = {
    "tic_tac_toe": _ttt_is_move,
    "nim": _nim_is_move,
    "dice_duel": _dice_duel_is_move,
    "coin_flip": _coin_flip_is_move,
}


# ---------------------------------------------------------------------------
# MinigameEngine — static interface
# ---------------------------------------------------------------------------

class MinigameEngine:
    @staticmethod
    def new_game(challenge: MinigameChallenge) -> MinigameState:
        """Create a new minigame instance from a challenge."""
        init = _GAME_INIT.get(challenge.game_type)
        if init is None:
            raise ValueError(f"Unknown game type: {challenge.game_type}")
        init_data = init()
        return MinigameState(
            game_type=challenge.game_type,
            opponent_slug=challenge.opponent_slug,
            stakes=challenge.stakes,
            board=init_data.get("board"),
            extra=init_data.get("extra", {}),
        )

    @staticmethod
    def player_move(state: MinigameState, move_input: str) -> tuple[bool, str]:
        """Process a player move. Returns (valid, message).

        If valid, the move is applied and NPC auto-plays.
        """
        handler = _GAME_PLAYER_MOVE.get(state.game_type)
        if handler is None:
            return False, f"Unknown game type: {state.game_type}"
        return handler(state, move_input)

    @staticmethod
    def render_board(state: MinigameState) -> str:
        """Render ASCII art for the current game state."""
        renderer = _GAME_RENDER.get(state.game_type)
        if renderer is None:
            return f"[{state.game_type} — no renderer]"
        return renderer(state)

    @staticmethod
    def render_prompt_section(state: MinigameState) -> str:
        """Build the ACTIVE_MINIGAME prompt injection section."""
        board_text = MinigameEngine.render_board(state)
        lines = [
            "ACTIVE_MINIGAME:",
            f"  type: {state.game_type}",
            f"  opponent: {state.opponent_slug}",
            f"  stakes: \"{state.stakes}\"" if state.stakes else "  stakes: none",
            f"  status: {state.status}",
            f"  board: |",
        ]
        for line in board_text.split("\n"):
            lines.append(f"    {line}")
        lines.append(
            "  instruction: \"Render the board in narration. "
            "Narrate the game action. Ask for the player's next move.\""
        )
        return "\n".join(lines)

    @staticmethod
    def is_game_move(state: MinigameState, player_input: str) -> bool:
        """Heuristic: does input look like a valid move for this game type?"""
        stripped = player_input.strip().lower()
        # Universal quit commands
        if stripped in ("quit game", "forfeit", "quit", "resign"):
            return True
        checker = _GAME_IS_MOVE.get(state.game_type)
        if checker is None:
            return False
        return checker(stripped)

    @staticmethod
    def is_finished(state: MinigameState) -> bool:
        """Check if the game is in a terminal state."""
        return state.status in ("player_won", "npc_won", "draw")
