"""Search speed benchmark: simulations/sec for a single search and for parallel self-play.

    python benchmark.py --blocks 6 --channels 64
    python benchmark.py --device cpu --simulations 50 --searches 4 --games-per-slot 1
"""
import argparse
import time

import chess
import numpy as np
import torch

from chessai.mcts import MCTS
from chessai.network import ChessNet
from chessai.parallel import play_games


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def sample_boards(count, plies, seed):
    """Positions reached by random play, so the benchmark is not just the opening."""
    rng = np.random.default_rng(seed)
    boards = []
    while len(boards) < count:
        board = chess.Board()
        for _ in range(plies):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
        if not board.is_game_over():
            boards.append(board)
    return boards


def bench_single(net, device, boards, simulations, leaves):
    mcts = MCTS(net, device, simulations, leaves=leaves, rng=np.random.default_rng(0))
    mcts.search(boards[0].copy())  # warm-up
    sync(device)
    start = time.perf_counter()
    for b in boards:
        mcts.search(b.copy())
    sync(device)
    return len(boards) * simulations / (time.perf_counter() - start)


def bench_parallel(net, device, parallel, games, simulations, leaves, plies):
    rng = np.random.default_rng(0)
    play_games([(net, net)] * parallel, device, simulations, parallel, leaves, rng,
               max_plies=2, add_noise=True, record=True)  # warm-up
    sync(device)
    start = time.perf_counter()
    out = play_games([(net, net)] * games, device, simulations, parallel, leaves, rng,
                     max_plies=plies, add_noise=True, record=True)
    sync(device)
    moves = sum(len(h) for _, h in out)
    return moves * simulations / (time.perf_counter() - start)


def ints(text):
    return [int(x) for x in text.split(",")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--simulations", type=int, default=100, help="MCTS simulations per move")
    p.add_argument("--single-leaves", type=ints, default=[1, 8, 32], help="comma list, single-search test")
    p.add_argument("--searches", type=int, default=8, help="positions searched per single-search measurement")
    p.add_argument("--parallel", type=ints, default=[8, 16, 32, 64], help="comma list, self-play test")
    p.add_argument("--leaves", type=int, default=8, help="leaves per step in the self-play test")
    p.add_argument("--plies", type=int, default=20, help="plies per game in the self-play test")
    p.add_argument("--games-per-slot", type=int, default=2, help="games = parallel x this")
    args = p.parse_args()

    torch.manual_seed(0)
    net = ChessNet(args.blocks, args.channels).to(args.device).eval()
    print(f"device={args.device}  blocks={args.blocks}  channels={args.channels}  "
          f"params={sum(x.numel() for x in net.parameters()):,}  simulations/move={args.simulations}")

    boards = sample_boards(args.searches, 12, seed=1)
    print("\nsingle search (simulations/sec)")
    for leaves in args.single_leaves:
        print(f"  leaves={leaves:<3} {bench_single(net, args.device, boards, args.simulations, leaves):8.0f}")

    print(f"\nparallel self-play, leaves={args.leaves}, {args.plies} plies/game (simulations/sec)")
    for parallel in args.parallel:
        games = parallel * args.games_per_slot
        rate = bench_parallel(net, args.device, parallel, games, args.simulations, args.leaves, args.plies)
        print(f"  parallel={parallel:<3} games={games:<4} {rate:8.0f}")


if __name__ == "__main__":
    main()
