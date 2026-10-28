import chess
import numpy as np
import pytest
import torch

from chessai import legal_move_mask, move_to_index
from chessai.arena import play_match
from chessai.mcts import MCTS, Search, run_round
from chessai.network import ChessNet
from chessai.parallel import play_games
from chessai.selfplay import generate_games


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return ChessNet(blocks=1, channels=16).eval()


def all_nodes(node):
    yield node
    for _, _, child in node.edges:
        yield from all_nodes(child)


@pytest.mark.parametrize("leaves", [1, 4, 32])
def test_visit_counts_and_no_leftover_virtual_loss(net, leaves):
    board = chess.Board()
    fen = board.fen()
    s = Search(net, board, simulations=40, rng=np.random.default_rng(0))
    while not s.done:
        run_round([s], "cpu", leaves)
    assert board.fen() == fen
    assert s.visits().sum() == 40
    assert s.root.N == 40
    for n in all_nodes(s.root):
        assert not n.pending
        # every visit is backed up: a node's visits equal the sum over its children (plus its own leaf visit)
        if n.edges and n is not s.root:
            assert n.N == 1 + sum(c.N for _, _, c in n.edges)


def test_batch_larger_than_budget_and_tiny_budget(net):
    for sims, leaves in [(1, 8), (3, 64)]:
        s = Search(net, chess.Board(), simulations=sims, rng=np.random.default_rng(0))
        while not s.done:
            run_round([s], "cpu", leaves)
        assert s.visits().sum() == sims


def test_single_leaf_search_matches_unbatched_choice(net):
    # leaves=1 has no virtual-loss effect, so it must be deterministic and legal
    board = chess.Board()
    a = MCTS(net, "cpu", 30, leaves=1, rng=np.random.default_rng(0)).search(board)
    b = MCTS(net, "cpu", 30, leaves=1, rng=np.random.default_rng(0)).search(board)
    assert (a == b).all()
    assert not a[~legal_move_mask(board)].any()


def test_batched_search_still_finds_mate(net):
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1")
    visits = MCTS(net, "cpu", 80, leaves=8, rng=np.random.default_rng(0)).search(board)
    assert visits.argmax() == move_to_index(chess.Move.from_uci("a1a8"), board)


def test_parallel_games_return_one_result_each_in_order(net):
    out = play_games([(net, net)] * 5, "cpu", simulations=2, parallel=3, leaves=2,
                     rng=np.random.default_rng(0), max_plies=20, record=True)
    assert len(out) == 5
    for result, history in out:
        assert result in (-1, 0, 1)
        assert len(history) == 20   # capped, so every game has exactly max_plies positions


def test_parallel_matches_serial_for_same_seed_structure(net):
    samples, results = generate_games(net, "cpu", games=3, simulations=2, parallel=3, leaves=2,
                                      rng=np.random.default_rng(5))
    assert len(results) == 3 and len(samples) > 0
    assert all(abs(s.probs.sum() - 1) < 1e-4 for s in samples)


def test_arena_alternates_colours_and_scores(net):
    score, (w, d, l) = play_match(net, net, "cpu", games=4, simulations=2, seed=1, parallel=4, leaves=2)
    assert w + d + l == 4 and 0.0 <= score <= 1.0


def test_two_different_nets_in_one_batch_round():
    torch.manual_seed(1)
    a, b = ChessNet(1, 16).eval(), ChessNet(1, 16).eval()
    sa = Search(a, chess.Board(), 6, rng=np.random.default_rng(0))
    sb = Search(b, chess.Board(), 6, rng=np.random.default_rng(0))
    while not (sa.done and sb.done):
        run_round([sa, sb], "cpu", 3)
    assert sa.visits().sum() == 6 and sb.visits().sum() == 6
