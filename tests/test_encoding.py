import chess
import numpy as np

from chessai import NUM_PLANES, encode_board


def test_shape_and_dtype():
    planes = encode_board(chess.Board())
    assert planes.shape == (NUM_PLANES, 8, 8)
    assert planes.dtype == np.float32


def test_start_position_piece_counts():
    planes = encode_board(chess.Board())
    expected = [8, 2, 2, 2, 1, 1]
    assert [int(planes[i].sum()) for i in range(6)] == expected
    assert [int(planes[6 + i].sum()) for i in range(6)] == expected


def test_start_position_flags():
    planes = encode_board(chess.Board())
    assert planes[12].min() == 1.0          # white to move
    assert all(planes[i].min() == 1.0 for i in range(13, 17))  # all castling rights
    assert planes[17].sum() == 0


def test_own_pawns_on_second_rank_for_both_colours():
    board = chess.Board()
    assert encode_board(board)[0][1].sum() == 8
    board.push_san("e4")  # now black to move; black's own pawns should also appear on row 1
    planes = encode_board(board)
    assert planes[0][1].sum() == 8
    assert planes[12].max() == 0.0          # black to move


def test_black_view_is_mirrored_white_view():
    # After 1.e4 the white pawn on e4 is an opponent pawn seen from black's side,
    # and it should appear on the mirrored rank (e4 -> row 4 from black's view).
    board = chess.Board()
    board.push_san("e4")
    planes = encode_board(board)
    assert planes[6][4][4] == 1.0
    assert planes[6][6].sum() == 7


def test_castling_rights_removed():
    board = chess.Board()
    for san in ("e4", "e5", "Ke2"):
        board.push_san(san)
    planes = encode_board(board)  # black to move; white lost both rights
    assert planes[13].min() == 1.0 and planes[14].min() == 1.0   # black (mover) keeps rights
    assert planes[15].max() == 0.0 and planes[16].max() == 0.0   # white (opponent) lost them


def test_en_passant_plane():
    board = chess.Board()
    for san in ("e4", "a6", "e5", "d5"):
        board.push_san(san)
    planes = encode_board(board)  # white to move, ep square d6 -> row 5, col 3
    assert planes[17].sum() == 1.0
    assert planes[17][5][3] == 1.0
