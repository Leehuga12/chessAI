import chess
import numpy as np
import pytest
import torch

from chessai import POLICY_SIZE, encode_board, legal_move_mask, move_to_index
from chessai.mcts import MCTS
from chessai.network import ChessNet
from chessai.selfplay import ReplayBuffer, play_game


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return ChessNet(blocks=1, channels=16).eval()


def test_network_output_shapes(net):
    x = torch.from_numpy(np.stack([encode_board(chess.Board())] * 3))
    policy, value = net(x)
    assert policy.shape == (3, POLICY_SIZE)
    assert value.shape == (3,)
    assert value.abs().max() <= 1.0


def test_search_visits_only_legal_moves_and_restores_board(net):
    board = chess.Board()
    fen = board.fen()
    visits = MCTS(net, "cpu", simulations=20, rng=np.random.default_rng(0)).search(board, add_noise=True)
    assert board.fen() == fen
    assert visits.sum() == 20
    assert not visits[~legal_move_mask(board)].any()


def test_search_finds_mate_in_one(net):
    # Back-rank mate: Ra8#
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1")
    visits = MCTS(net, "cpu", simulations=60, rng=np.random.default_rng(0)).search(board)
    mate = move_to_index(chess.Move.from_uci("a1a8"), board)
    assert visits.argmax() == mate


def test_search_refuses_finished_game(net):
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")  # stalemate
    with pytest.raises(ValueError):
        MCTS(net, "cpu", simulations=5).search(board)


def test_self_play_game_produces_valid_samples(net):
    samples, result = play_game(net, "cpu", simulations=4, rng=np.random.default_rng(1))
    assert result in (-1, 0, 1) and len(samples) > 0
    for s in samples:
        assert s.planes.shape == (18, 8, 8)
        assert abs(s.probs.sum() - 1.0) < 1e-4
        assert s.value in (-1.0, 0.0, 1.0)
    if result != 0:  # winner's positions are +1, loser's -1, alternating by ply
        assert samples[0].value == -samples[1].value


def test_replay_buffer_batch(net):
    samples, _ = play_game(net, "cpu", simulations=2, rng=np.random.default_rng(2))
    buf = ReplayBuffer(1000)
    buf.add(samples)
    planes, policy, values = buf.sample_batch(8)
    assert planes.shape[1:] == (18, 8, 8) and policy.shape[1] == POLICY_SIZE
    assert np.allclose(policy.sum(axis=1), 1.0, atol=1e-4)
    assert len(values) == len(planes)
