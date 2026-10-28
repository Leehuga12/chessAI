import random

import chess

from chessai.rules import terminal_value


def reference(board):
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    return 0.0 if outcome.winner is None else -1.0


def test_matches_python_chess_on_random_games():
    rng = random.Random(0)
    checked = 0
    for _ in range(60):
        board = chess.Board()
        for _ in range(250):
            assert terminal_value(board) == reference(board), board.fen()
            checked += 1
            if board.is_game_over(claim_draw=True):
                break
            board.push(rng.choice(list(board.legal_moves)))
    assert checked > 3000


def test_threefold_repetition_is_detected_exactly_when_claimable():
    board = chess.Board()
    for san in ["Nf3", "Nf6", "Ng1", "Ng8"] * 2:
        assert terminal_value(board) == reference(board)
        board.push_san(san)
    assert board.can_claim_threefold_repetition()
    assert terminal_value(board) == 0.0


def test_repetition_after_a_capture_does_not_count():
    # shuffling after an irreversible move; check agreement at every step
    board = chess.Board()
    for san in ["e4", "e5", "Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1", "Ng8"]:
        board.push_san(san)
        assert terminal_value(board) == reference(board)


def test_fifty_move_rule_and_mates():
    assert terminal_value(chess.Board("8/8/4k3/8/8/4K3/4P3/8 w - - 100 80")) == 0.0
    assert terminal_value(chess.Board("8/8/4k3/8/8/4K3/4P3/8 w - - 98 80")) is None
    assert terminal_value(chess.Board("R5k1/5ppp/8/8/8/8/8/4K3 b - - 0 1")) == -1.0   # mated
    assert terminal_value(chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")) == 0.0       # stalemate
    assert terminal_value(chess.Board("8/8/4k3/8/8/4K3/8/8 w - - 0 1")) == 0.0        # K vs K


def test_matches_python_chess_when_shuffling_pieces():
    # prefer quiet piece moves so repetitions and long reversible stretches are common
    rng = random.Random(1)
    repetitions = 0
    for _ in range(80):
        board = chess.Board()
        for _ in range(160):
            assert terminal_value(board) == reference(board), board.fen()
            if board.is_game_over(claim_draw=True):
                repetitions += board.can_claim_threefold_repetition()
                break
            moves = list(board.legal_moves)
            quiet = [m for m in moves if not board.is_capture(m)
                     and board.piece_type_at(m.from_square) in (chess.KNIGHT, chess.ROOK, chess.KING)]
            board.push(rng.choice(quiet or moves))
    assert repetitions > 0   # the scenario was actually exercised
