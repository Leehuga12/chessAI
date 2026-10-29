"""Head-to-head match between two nets: the clearest answer to "is the new net actually better?".

    python compare_nets.py runs/pretrain_big/pretrained.pt runs/pretrain/pretrained.pt --games 100 --simulations 100

Plays GAMES games, alternating colours, with a few random opening moves so the games differ. Prints the score of the first net with a
95% interval; if the interval does not include 50%, one net is genuinely stronger. 100 games gives an interval of about +-7 points,
400 games about +-3.5. Unlike the Stockfish ladder (8 fixed openings), these games do not repeat.
"""
import argparse
import math

import torch

from chessai.arena import play_match
from chessai.network import load_checkpoint


def score_interval(wins, draws, losses):
    """Score of the first net and a 95% interval (normal approximation, each game independent)."""
    n = wins + draws + losses
    p = (wins + 0.5 * draws) / n
    var = (wins * (1 - p) ** 2 + draws * (0.5 - p) ** 2 + losses * p ** 2) / n
    half = 1.96 * math.sqrt(var / n)
    return p, p - half, p + half


def main():
    p = argparse.ArgumentParser()
    p.add_argument("first", help="checkpoint to test")
    p.add_argument("second", help="checkpoint to test against")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--simulations", type=int, default=100, help="search simulations per move for both nets")
    p.add_argument("--parallel", type=int, default=50, help="games played at once")
    p.add_argument("--leaves", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    a, _ = load_checkpoint(args.first, device)
    b, _ = load_checkpoint(args.second, device)
    a.eval()
    b.eval()
    score, (w, d, l) = play_match(a, b, device, args.games, args.simulations, seed=args.seed, parallel=args.parallel, leaves=args.leaves)
    _, lo, hi = score_interval(w, d, l)
    verdict = "first net is stronger" if lo > 0.5 else "second net is stronger" if hi < 0.5 else "no clear difference"
    print(f"{args.first}\n  vs {args.second}\n  {args.games} games, {args.simulations} simulations: +{w} ={d} -{l}  "
          f"score {100 * score:.1f}%  (95% interval {100 * lo:.1f}% to {100 * hi:.1f}%)  -> {verdict}")


if __name__ == "__main__":
    main()
