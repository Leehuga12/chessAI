"""Board -> tensor encoding.

The board is always shown from the side to move's point of view: when black is
to move the board is mirrored top-to-bottom and colours are swapped, so the
network only ever has to learn "my pieces move up the board".

Planes (channels-first, shape (18, 8, 8)):
  0-5    side-to-move pieces: pawn, knight, bishop, rook, queen, king
  6-11   opponent pieces, same order
  12     side to move (1.0 if white, 0.0 if black)
  13-14  side-to-move castling rights: kingside, queenside
  15-16  opponent castling rights: kingside, queenside
  17     en passant target square (a single 1.0, if any)
"""
import chess
import numpy as np

NUM_PLANES = 18
PIECE_ORDER = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)


def relative_square(square: int, turn: bool) -> int:
    """Square as seen from the mover's side (mirror ranks when black is to move)."""
    return square if turn == chess.WHITE else square ^ 56


def encode_board(board: chess.Board) -> np.ndarray:
    turn = board.turn
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        rel = relative_square(square, turn)
        offset = 0 if piece.color == turn else 6
        planes[offset + PIECE_ORDER.index(piece.piece_type), rel >> 3, rel & 7] = 1.0

    if turn == chess.WHITE:
        planes[12] = 1.0

    opponent = not turn
    planes[13] = float(board.has_kingside_castling_rights(turn))
    planes[14] = float(board.has_queenside_castling_rights(turn))
    planes[15] = float(board.has_kingside_castling_rights(opponent))
    planes[16] = float(board.has_queenside_castling_rights(opponent))

    if board.ep_square is not None:
        rel = relative_square(board.ep_square, turn)
        planes[17, rel >> 3, rel & 7] = 1.0

    return planes
