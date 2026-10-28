import random

import chess
import pytest

from chessai import POLICY_SIZE, index_to_move, legal_move_mask, move_to_index


def test_policy_size():
    assert POLICY_SIZE == 4672


def test_start_position_has_20_legal_moves():
    mask = legal_move_mask(chess.Board())
    assert mask.shape == (POLICY_SIZE,)
    assert mask.sum() == 20


def test_black_start_reply_has_20_legal_moves():
    board = chess.Board()
    board.push_san("e4")
    assert legal_move_mask(board).sum() == 20


def test_same_index_for_equivalent_white_and_black_moves():
    white = chess.Board()
    black = chess.Board()
    black.push_san("a3")
    assert move_to_index(chess.Move.from_uci("e2e4"), white) == \
        move_to_index(chess.Move.from_uci("e7e5"), black)


def test_round_trip_on_special_moves():
    cases = [
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),   # castling
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1c1"),
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8g8"),
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q"),        # queen promotion
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8n"),        # underpromotions
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8r"),
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8b"),
        ("1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7b8q"),      # promotion capture
        ("4k3/8/8/8/8/8/p7/4K3 b - - 0 1", "a2a1n"),        # black underpromotion
        ("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3", "e5d6"),  # en passant
    ]
    for fen, uci in cases:
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves, (fen, uci)
        assert index_to_move(move_to_index(move, board), board) == move, (fen, uci)


def test_underpromotions_get_distinct_indices():
    board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    indices = {move_to_index(chess.Move.from_uci(f"a7a8{p}"), board) for p in "qrbn"}
    assert len(indices) == 4


def test_off_board_index_raises():
    with pytest.raises(ValueError):
        index_to_move(0 * 73 + 28, chess.Board())  # a1, south direction -> off the board


@pytest.mark.parametrize("seed", range(5))
def test_random_games_round_trip_and_unique(seed):
    rng = random.Random(seed)
    board = chess.Board()
    for _ in range(200):
        if board.is_game_over():
            break
        moves = list(board.legal_moves)
        indices = [move_to_index(m, board) for m in moves]
        assert len(set(indices)) == len(moves)                       # no collisions
        assert all(0 <= i < POLICY_SIZE for i in indices)
        assert legal_move_mask(board).sum() == len(moves)            # mask matches
        for m, i in zip(moves, indices):
            assert index_to_move(i, board) == m                      # round trip
        board.push(rng.choice(moves))
