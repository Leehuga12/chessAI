"""Move <-> policy index, plus the legal-move mask.

Uses the AlphaZero layout: 64 from-squares x 73 move types = 4672 outputs.
  planes 0-55   "queen moves": 8 directions x 7 distances (also covers castling
                and queen promotions)
  planes 56-63  knight moves
  planes 64-72  underpromotions: (knight, bishop, rook) x (capture left, push,
                capture right)
Squares are taken from the mover's point of view (see encoding.relative_square),
so the same index means the same thing for white and black.
"""
import chess
import numpy as np

from .encoding import relative_square

NUM_MOVE_PLANES = 73
POLICY_SIZE = 64 * NUM_MOVE_PLANES

# (dx, dy) = (file change, rank change)
QUEEN_DIRECTIONS = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))
KNIGHT_JUMPS = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
UNDERPROMOTIONS = (chess.KNIGHT, chess.BISHOP, chess.ROOK)


def move_to_index(move: chess.Move, board: chess.Board) -> int:
    turn = board.turn
    frm = relative_square(move.from_square, turn)
    to = relative_square(move.to_square, turn)
    dx = (to & 7) - (frm & 7)
    dy = (to >> 3) - (frm >> 3)

    if move.promotion is not None and move.promotion != chess.QUEEN:
        plane = 64 + UNDERPROMOTIONS.index(move.promotion) * 3 + (dx + 1)
    elif (dx, dy) in KNIGHT_JUMPS:
        plane = 56 + KNIGHT_JUMPS.index((dx, dy))
    else:
        distance = max(abs(dx), abs(dy))
        direction = (dx // distance, dy // distance)
        plane = QUEEN_DIRECTIONS.index(direction) * 7 + (distance - 1)
    return frm * NUM_MOVE_PLANES + plane


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Inverse of move_to_index. Returns a move that may not be legal; check against the mask."""
    frm, plane = divmod(index, NUM_MOVE_PLANES)
    turn = board.turn
    promotion = None

    if plane >= 64:
        piece_idx, dx = divmod(plane - 64, 3)
        dx -= 1
        dy = 1
        promotion = UNDERPROMOTIONS[piece_idx]
    elif plane >= 56:
        dx, dy = KNIGHT_JUMPS[plane - 56]
    else:
        direction, distance = divmod(plane, 7)
        dx, dy = (QUEEN_DIRECTIONS[direction][0] * (distance + 1),
                  QUEEN_DIRECTIONS[direction][1] * (distance + 1))

    to_file = (frm & 7) + dx
    to_rank = (frm >> 3) + dy
    if not (0 <= to_file < 8 and 0 <= to_rank < 8):
        raise ValueError(f"index {index} points off the board")
    to = to_rank * 8 + to_file

    from_sq = relative_square(frm, turn)
    to_sq = relative_square(to, turn)

    # A pawn reaching the last rank with a queen-like move is a queen promotion.
    if promotion is None:
        piece = board.piece_at(from_sq)
        if piece is not None and piece.piece_type == chess.PAWN and to_rank == 7:
            promotion = chess.QUEEN
    return chess.Move(from_sq, to_sq, promotion)


def legal_move_mask(board: chess.Board) -> np.ndarray:
    mask = np.zeros(POLICY_SIZE, dtype=bool)
    for move in board.legal_moves:
        mask[move_to_index(move, board)] = True
    return mask
