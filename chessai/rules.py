"""Fast game-end detection, equivalent to board.outcome(claim_draw=True).

python-chess's repetition check unwinds and replays the move stack, which dominated search time.
A draw by repetition is claimable once a position has occurred twice and a legal move would repeat
it a third time, which takes at least 7 reversible plies since the last capture or pawn move, so
the check is skipped while the halfmove clock is below that.
"""
import chess


def terminal_value(board: chess.Board):
    """None if the game goes on, else the value for the player to move: -1.0 if mated, 0.0 if drawn."""
    if board.is_checkmate():
        return -1.0
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    clock = board.halfmove_clock
    if clock >= 99 and board.can_claim_fifty_moves():   # also claimable one ply early
        return 0.0
    if clock >= 7 and board.can_claim_threefold_repetition():
        return 0.0
    return None
