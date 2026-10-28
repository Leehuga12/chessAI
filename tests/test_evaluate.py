import math

import chess
import pytest
import torch

from chessai.evaluate import (DEFAULT_ENGINE, OPENINGS, RandomOpponent, elo_error_margin, estimate_elo,
                              evaluate, make_opponent, opening_board, play_eval_game)
from chessai.network import ChessNet


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return ChessNet(blocks=1, channels=16).eval()


def test_even_score_gives_opponent_elo():
    assert estimate_elo(0.5, 20, 1500) == pytest.approx(1500)


def test_elo_rises_with_score_and_stays_finite_at_extremes():
    assert estimate_elo(0.75, 40, 1500) > 1500 > estimate_elo(0.25, 40, 1500)
    assert math.isfinite(estimate_elo(0.0, 10, 1500)) and math.isfinite(estimate_elo(1.0, 10, 1500))
    assert estimate_elo(0.76, 100, 1500) == pytest.approx(1500 + 400 * math.log10(0.76 / 0.24), abs=10)


def test_margin_shrinks_with_more_games():
    assert elo_error_margin(30, 20, 50) < elo_error_margin(3, 2, 5)


def test_margin_finite_when_all_results_equal():
    assert math.isfinite(elo_error_margin(0, 0, 10)) and elo_error_margin(0, 0, 10) > 0
    assert math.isfinite(elo_error_margin(10, 0, 0)) and elo_error_margin(10, 0, 0) > 0


def test_openings_are_legal_and_distinct():
    fens = {opening_board(i).fen() for i in range(len(OPENINGS))}
    assert len(fens) == len(OPENINGS)
    assert all(opening_board(i).ply() == 4 for i in range(len(OPENINGS)))


def test_unknown_opponent_spec():
    with pytest.raises(ValueError):
        make_opponent("magnus")


def test_game_against_random_returns_valid_result(net):
    r = play_eval_game(net, RandomOpponent(1), True, 0, "cpu", simulations=2)
    assert r in (0.0, 0.5, 1.0)


def test_evaluate_counts_add_up_and_is_repeatable(net):
    a = evaluate(net, RandomOpponent(3), 2, "cpu", 2)
    assert a["wins"] + a["draws"] + a["losses"] == 2
    assert 0.0 <= a["score"] <= 1.0
    assert evaluate(net, RandomOpponent(3), 2, "cpu", 2) == a   # same net, same opponent seed


@pytest.mark.skipif(not DEFAULT_ENGINE.exists(), reason="Stockfish not downloaded")
def test_stockfish_opponent_plays_legal_moves():
    for spec in ("sf-skill0", "sf-elo1320"):
        opp = make_opponent(spec)
        try:
            board = chess.Board()
            for _ in range(6):
                move = opp.move(board)
                assert move in board.legal_moves
                board.push(move)
        finally:
            opp.close()
