"""AlphaZero-style training loop: self-play -> train -> gate.

Tiny CPU smoke test:
    python train.py --iterations 2 --games 2 --simulations 8 --train-steps 5 --arena-games 2
"""
import argparse
import copy
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from chessai.arena import play_match
from chessai.network import ChessNet, save_checkpoint
from chessai.selfplay import ReplayBuffer, generate_games


def train_steps(net, buffer, optimizer, steps, batch_size, device):
    net.train()
    totals = np.zeros(3)
    for _ in range(steps):
        planes, policy, values = buffer.sample_batch(batch_size)
        planes, policy, values = (torch.from_numpy(a).to(device) for a in (planes, policy, values))
        logits, pred = net(planes)
        policy_loss = -(policy * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        value_loss = F.mse_loss(pred, values)
        loss = policy_loss + value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        totals += (loss.item(), policy_loss.item(), value_loss.item())
    net.eval()
    return totals / max(steps, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=2)
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
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    net = ChessNet(args.blocks, args.channels).to(device).eval()
    best = copy.deepcopy(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    buffer = ReplayBuffer(args.buffer)
    save_checkpoint(best, out / "iter_000.pt")
    save_checkpoint(best, out / "best.pt", {"iteration": 0})

    log_path = out / "train_log.csv"
    fields = ["iteration", "seconds", "selfplay_games", "white_wins", "black_wins", "draws",
              "samples", "loss", "policy_loss", "value_loss", "arena_score", "promoted"]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fields).writeheader()

    start = time.time()
    print(f"device={device}  params={sum(x.numel() for x in net.parameters()):,}")
    for it in range(1, args.iterations + 1):
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

        row = dict(iteration=it, seconds=round(time.time() - start, 1), selfplay_games=args.games,
                   white_wins=tally[1], black_wins=tally[-1], draws=tally[0], samples=len(buffer),
                   loss=round(loss, 4), policy_loss=round(pl, 4), value_loss=round(vl, 4),
                   arena_score=round(score, 3), promoted=int(promoted))
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fields).writerow(row)
        print(row)


if __name__ == "__main__":
    main()
