"""
Read-only Connect 4 tactical evaluator for benchmark scoring.

Does NOT modify arena/board.py. Operates only on deepcopy()'d Board
instances borrowed from arena.board, using its existing move()/wins()/
legal_moves() methods.

IMPORTANT — honesty about what this is:
This is a shallow heuristic minimax (alpha-beta, default depth 6) with a
center-weighted potential-line evaluation function. It is NOT a perfect
Connect 4 solver (a perfect solver requires either a full opening-book
search or bitboard negamax with much greater depth/optimization). Treat
"engine top choice" as "reasonable tactical opinion at limited depth",
not as ground truth optimal play. It IS strong enough to reliably catch
the two things asked for explicitly: immediate wins and immediate blocks,
since both are depth-1/depth-2 tactical facts, not heuristic judgments.
"""
import copy
from arena.board import Board, cols, RED, YELLOW, EMPTY


def _immediate_win(board: Board, col_index: int, player: int) -> bool:
    """Does playing col_index at the current board state win immediately for `player`?"""
    if board.height(col_index) >= 6:
        return False
    sim = copy.deepcopy(board)
    sim.player = player  # force the mover, independent of whose turn it "really" is
    sim.move(col_index)
    return sim.winner == player


def immediate_wins(board: Board, player: int) -> list:
    """All legal columns (letters) that would win immediately for `player`."""
    return [
        cols[x] for x in range(7)
        if board.height(x) < 6 and _immediate_win(board, x, player)
    ]


def blocking_moves(board: Board, player: int) -> list:
    """
    Columns (letters) that block an immediate opponent win.
    Defined as: the opponent has at least one column that would win for them
    right now, and playing THIS column occupies that same column (denying it).
    """
    opponent = -1 * player
    threats = immediate_wins(board, opponent)
    return threats  # the blocking move(s) are exactly the threat column(s)


# ---------------------------------------------------------------------------
# Shallow heuristic minimax, for "is this one of the engine's top choices"
# ---------------------------------------------------------------------------

_CENTER_WEIGHTS = [3, 4, 5, 7, 5, 4, 3]  # column preference, center-weighted


def _score_position(board: Board, player: int) -> float:
    """
    Heuristic static evaluation from `player`'s perspective.
    Counts open 4-cell windows weighted by how many of the player's/opponent's
    pieces already occupy them, plus a center-column preference.
    """
    opponent = -1 * player
    score = 0.0

    # Center column preference
    for x in range(7):
        for y in range(6):
            if board.cells[y][x] == player:
                score += _CENTER_WEIGHTS[x] * 0.1
            elif board.cells[y][x] == opponent:
                score -= _CENTER_WEIGHTS[x] * 0.1

    def window_score(cells4, who):
        count_p = cells4.count(who)
        count_o = cells4.count(-1 * who)
        count_e = cells4.count(EMPTY)
        if count_o > 0 and count_p > 0:
            return 0  # blocked window, worthless to either side
        if count_p == 4:
            return 1000
        if count_p == 3 and count_e == 1:
            return 40
        if count_p == 2 and count_e == 2:
            return 8
        if count_p == 1 and count_e == 3:
            return 1
        if count_o == 4:
            return -1000
        if count_o == 3 and count_e == 1:
            return -40  # symmetric with the count_p==3 case above — negamax's
            # recursive negation requires score(board,A) == -score(board,-A);
            # an asymmetric weight here (e.g. -45) breaks that invariant.
        if count_o == 2 and count_e == 2:
            return -8
        if count_o == 1 and count_e == 3:
            return -1
        return 0

    # horizontal
    for y in range(6):
        for x in range(4):
            window = [board.cells[y][x + i] for i in range(4)]
            score += window_score(window, player)
    # vertical
    for x in range(7):
        for y in range(3):
            window = [board.cells[y + i][x] for i in range(4)]
            score += window_score(window, player)
    # diagonal /
    for y in range(3):
        for x in range(4):
            window = [board.cells[y + i][x + i] for i in range(4)]
            score += window_score(window, player)
    # diagonal \
    for y in range(3):
        for x in range(4):
            window = [board.cells[y + 3 - i][x + i] for i in range(4)]
            score += window_score(window, player)

    return score


def _negamax(board: Board, depth: int, alpha: float, beta: float, player: int) -> float:
    winner = board.wins()
    if winner:
        # Prefer faster wins / slower losses: no explicit depth bonus needed at
        # this shallow depth, but sign must match `player`'s perspective.
        return 1_000_000 if winner == player else -1_000_000

    legal = [x for x in range(7) if board.height(x) < 6]
    if not legal or depth == 0:
        return _score_position(board, player)

    best = float("-inf")
    for x in legal:
        sim = copy.deepcopy(board)
        sim.player = player
        sim.move(x)
        val = -_negamax(sim, depth - 1, -beta, -alpha, -1 * player)
        best = max(best, val)
        alpha = max(alpha, val)
        if alpha >= beta:
            break
    return best


def rank_moves(board: Board, player: int, depth: int = 6) -> list:
    """
    Return legal columns (letters) ranked best-to-worst for `player`,
    using shallow minimax. Ties broken by center-column preference.
    """
    legal = [x for x in range(7) if board.height(x) < 6]
    scored = []
    for x in legal:
        sim = copy.deepcopy(board)
        sim.player = player
        sim.move(x)
        val = -_negamax(sim, depth - 1, float("-inf"), float("inf"), -1 * player)
        scored.append((cols[x], val))
    scored.sort(key=lambda t: (-t[1], -_CENTER_WEIGHTS[cols.index(t[0])]))
    return scored  # list of (column_letter, score), best first


def evaluate_move(board: Board, player: int, move_col: str, top_n: int = 2, depth: int = 6) -> dict:
    """
    Full tactical evaluation of a single proposed move for `player` on `board`.
    board is NOT mutated (all analysis uses deep copies internally).

    Returns:
      is_legal            bool
      wins_immediately     bool
      blocks_opponent_win  bool
      is_top_choice        bool  (within top_n of the engine's ranking)
      engine_rank          int or None (1-indexed rank of this move; None if illegal)
      engine_top_moves     list of (column, score) — the engine's own ranking, for reference
      opponent_threats     list of columns where the opponent could win immediately (context)
    """
    result = {
        "is_legal": False,
        "wins_immediately": False,
        "blocks_opponent_win": False,
        "is_top_choice": False,
        "engine_rank": None,
        "engine_top_moves": None,
        "opponent_threats": [],
    }

    if move_col not in cols:
        return result

    col_index = cols.find(move_col)
    if board.height(col_index) >= 6:
        return result

    result["is_legal"] = True
    result["wins_immediately"] = _immediate_win(board, col_index, player)
    threats = blocking_moves(board, player)
    result["opponent_threats"] = threats
    result["blocks_opponent_win"] = move_col in threats

    ranking = rank_moves(board, player, depth=depth)
    result["engine_top_moves"] = ranking
    for i, (letter, _score) in enumerate(ranking):
        if letter == move_col:
            result["engine_rank"] = i + 1
            break
    if result["engine_rank"] is not None:
        result["is_top_choice"] = result["engine_rank"] <= top_n

    return result
