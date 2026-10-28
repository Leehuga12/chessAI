"""Evaluate checkpoints against fixed reference opponents and append to a results CSV.

Example:
    python evaluate.py runs/default/iter_000.pt runs/default/best.pt \
        --opponents random sf-skill0 sf-elo1320 --games 16 --simulations 50
"""
import argparse
import csv
import re
import time
from pathlib import Path

import torch

from chessai.evaluate import DEFAULT_ENGINE, elo_error_margin, estimate_elo, evaluate, make_opponent
from chessai.network import load_checkpoint

FIELDS = ["time", "checkpoint", "iteration", "opponent", "opponent_elo", "games", "wins", "draws",
          "losses", "score", "est_elo", "elo_margin", "simulations"]


def iteration_of(path: Path, extra: dict) -> int:
    if "iteration" in extra:
        return int(extra["iteration"])
    m = re.search(r"iter_(\d+)", path.name)
    return int(m.group(1)) if m else -1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--opponents", nargs="+", default=["random", "sf-skill0", "sf-elo1320"])
    p.add_argument("--games", type=int, default=16)
    p.add_argument("--simulations", type=int, default=50)
    p.add_argument("--engine", default=str(DEFAULT_ENGINE))
    p.add_argument("--out", default="results/results.csv")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists()

    for spec in args.opponents:
        opponent = make_opponent(spec, args.engine)
        try:
            for ckpt in args.checkpoints:
                path = Path(ckpt)
                net, extra = load_checkpoint(path, device)
                net.eval()
                r = evaluate(net, opponent, args.games, device, args.simulations)
                elo = margin = ""
                if opponent.elo is not None:
                    elo = round(estimate_elo(r["score"], r["games"], opponent.elo))
                    margin = round(elo_error_margin(r["wins"], r["draws"], r["losses"]))
                row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "checkpoint": str(path),
                       "iteration": iteration_of(path, extra), "opponent": spec,
                       "opponent_elo": opponent.elo or "", "games": r["games"], "wins": r["wins"],
                       "draws": r["draws"], "losses": r["losses"], "score": round(r["score"], 3),
                       "est_elo": elo, "elo_margin": margin, "simulations": args.simulations}
                with open(out, "a", newline="") as f:
                    w = csv.DictWriter(f, FIELDS)
                    if new_file:
                        w.writeheader()
                        new_file = False
                    w.writerow(row)
                print(row)
        finally:
            opponent.close()


if __name__ == "__main__":
    main()
