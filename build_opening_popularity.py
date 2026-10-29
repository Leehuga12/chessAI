"""Count how often strong players reach each named opening, so the UI's Random opening picks realistic lines.

    python build_opening_popularity.py "data/raw_elite/Lichess Elite Database/lichess_elite_2019-12.pgn"
    python build_opening_popularity.py games.pgn --max-games 100000 --min-count 20

Writes chessai/data/openings/popularity.json ({opening id: games}); openings seen fewer than --min-count times are left out.
"""
import argparse
import json
from pathlib import Path

from chessai.openings import DEFAULT_DIR, Openings, count_openings


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pgn", nargs="+")
    p.add_argument("--out", default=str(Path(DEFAULT_DIR) / "popularity.json"))
    p.add_argument("--plies", type=int, default=16, help="look this many half-moves into each game")
    p.add_argument("--min-rating", type=int, default=2200, help="both players at least this")
    p.add_argument("--min-count", type=int, default=20, help="drop openings seen fewer times than this")
    p.add_argument("--max-games", type=int, default=None)
    args = p.parse_args()

    counts = count_openings(args.pgn, Openings(), args.plies, args.min_rating, args.max_games)
    kept = {k: v for k, v in sorted(counts.items(), key=lambda kv: -kv[1]) if v >= args.min_count}
    Path(args.out).write_text(json.dumps(kept, indent=0), encoding="utf-8")
    total = sum(counts.values())
    print(f"{total:,} games reached a named opening; {len(kept)} openings kept (of {len(counts)} seen) -> {args.out}")
    for k, v in list(kept.items())[:8]:
        print(f"  {v:>7,}  {k}")


if __name__ == "__main__":
    main()
