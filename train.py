"""AlphaZero-style training loop: self-play -> train -> gate.

Tiny CPU smoke test:
    python train.py --iterations 2 --games 2 --simulations 8 --train-steps 5 --arena-games 2

Continue a stopped or finished run (--iterations is the total to reach, so the same command can be re-run):
    python train.py --iterations 200 --out-dir runs/run1 ...        # first time
    python train.py --iterations 200 --resume runs/run1 ...         # after a stop; nets, optimizer, log and buffer carry on
"""
import argparse
import copy
import csv
import time
from pathlib import Path

import numpy as np
import torch

from chessai.arena import play_match
from chessai.network import ChessNet, load_checkpoint, save_checkpoint
from chessai.selfplay import ReplayBuffer, generate_games
from chessai.training import (STATE_FILE, load_buffer, load_state, save_buffer, save_state,
                              train_steps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=2, help="total iterations to reach (not additional ones)")
    p.add_argument("--games", type=int, default=2, help="self-play games per iteration")
    p.add_argument("--simulations", type=int, default=8, help="MCTS simulations per move")
    p.add_argument("--train-steps", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--arena-games", type=int, default=2)
    p.add_argument("--parallel", type=int, default=16, help="games played at once (shared network calls)")
    p.add_argument("--leaves", type=int, default=8, help="leaves evaluated together per search step")
    p.add_argument("--gate", type=float, default=0.55, help="score needed to replace the best net")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--buffer", type=int, default=100_000)
    p.add_argument("--out-dir", default="runs/default")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init-from", metavar="CKPT", help="start a fresh run from this checkpoint (e.g. runs/pretrain/pretrained.pt)")
    p.add_argument("--resume", metavar="DIR", help="continue the run saved in DIR (network size comes from the save)")
    p.add_argument("--buffer-every", type=int, default=10,
                   help="save the replay buffer every N iterations (it is large); state is saved every iteration")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.resume or args.out_dir)
    fields = ["iteration", "seconds", "selfplay_games", "white_wins", "black_wins", "draws",
              "samples", "loss", "policy_loss", "value_loss", "arena_score", "promoted"]
    log_path = out / "train_log.csv"

    if args.resume:
        if not (out / STATE_FILE).exists():
            p.error(f"nothing to resume: {out / STATE_FILE} does not exist")
        net, best, optimizer, start_it, elapsed = load_state(out, device, args.lr)
        buffer, buffer_it = load_buffer(out, args.buffer)
        print(f"resumed {out} at iteration {start_it}; " + (
            f"replay buffer restored from iteration {buffer_it} ({len(buffer)} samples)" if buffer_it is not None
            else "no saved replay buffer, starting with an empty one"))
        if (args.blocks, args.channels) != (net.config["blocks"], net.config["channels"]):
            print(f"note: using the saved network size {net.config}, not --blocks/--channels")
        if start_it >= args.iterations:
            print(f"already at iteration {start_it}; raise --iterations to continue")
            return
    else:
        if (out / STATE_FILE).exists():
            p.error(f"{out} already holds a run; use --resume {out} to continue it or pick a new --out-dir")
        out.mkdir(parents=True, exist_ok=True)
        if args.init_from:
            net, _ = load_checkpoint(args.init_from, device)
            net.eval()
            print(f"starting from {args.init_from} (network size {net.config})")
        else:
            net = ChessNet(args.blocks, args.channels).to(device).eval()
        best = copy.deepcopy(net)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
        buffer = ReplayBuffer(args.buffer)
        start_it, elapsed = 0, 0.0
        save_checkpoint(best, out / "iter_000.pt")
        save_checkpoint(best, out / "best.pt", {"iteration": 0})
        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, fields).writeheader()
    # a resumed run must not replay the same games, so the seed includes the iteration it starts from
    torch.manual_seed(args.seed + start_it)
    rng = np.random.default_rng([args.seed, start_it])

    start = time.time()
    print(f"device={device}  params={sum(x.numel() for x in net.parameters()):,}")
    for it in range(start_it + 1, args.iterations + 1):
        samples, results = generate_games(best, device, args.games, args.simulations,
                                          args.parallel, args.leaves, rng)
        buffer.add(samples)
        tally = {r: results.count(r) for r in (1, -1, 0)}

        loss, pl, vl = map(float, train_steps(net, buffer, optimizer, args.train_steps, args.batch_size, device))
        score, _ = play_match(net, best, device, args.arena_games, args.simulations, seed=it,
                              parallel=args.parallel, leaves=args.leaves)
        promoted = score >= args.gate
        if promoted:
            best = copy.deepcopy(net)
            save_checkpoint(best, out / f"iter_{it:03d}.pt", {"iteration": it})
            save_checkpoint(best, out / "best.pt", {"iteration": it})

        row = dict(iteration=it, seconds=round(elapsed + time.time() - start, 1), selfplay_games=args.games,
                   white_wins=tally[1], black_wins=tally[-1], draws=tally[0], samples=len(buffer),
                   loss=round(loss, 4), policy_loss=round(pl, 4), value_loss=round(vl, 4),
                   arena_score=round(score, 3), promoted=int(promoted))
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fields).writerow(row)
        print(row)
        save_state(out, net, best, optimizer, it, elapsed + time.time() - start)
        if it % args.buffer_every == 0 or it == args.iterations:
            save_buffer(out, buffer, it)


if __name__ == "__main__":
    main()
