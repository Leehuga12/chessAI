"""Build the supervised warm-start dataset from PGN files.

    python prepare_data.py games.pgn --out-dir data/elite --min-elo 2300
    python prepare_data.py lichess_elite_2024-01.zip other.pgn --max-games 200000

Writes train_*.npz / val_*.npz shards and prepare_stats.json to --out-dir (git-ignored). Reads .pgn, .zip and
(with `pip install zstandard`) .pgn.zst. Then train with pretrain.py.
"""
import argparse

from chessai.pgn_data import prepare


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pgn", nargs="+", help="PGN files (.pgn, .zip, .pgn.zst)")
    p.add_argument("--out-dir", default="data/prepared")
    p.add_argument("--min-elo", type=int, default=2200, help="both players must be rated at least this")
    p.add_argument("--min-plies", type=int, default=16, help="skip games shorter than this many half-moves")
    p.add_argument("--max-games", type=int, default=None, help="stop after this many accepted games")
    p.add_argument("--val-every", type=int, default=50, help="every Nth accepted game goes to the validation set")
    p.add_argument("--shard-size", type=int, default=200_000, help="positions per output file")
    args = p.parse_args()

    stats = prepare(args.pgn, args.out_dir, args.min_elo, args.min_plies, args.max_games, args.val_every, args.shard_size)
    print(f"games read {stats['games_read']:,}, used {stats['games_used']:,}")
    for reason, n in sorted(stats["rejected"].items(), key=lambda kv: -kv[1]):
        print(f"  skipped {n:,}: {reason}")
    print(f"positions: {stats['train_positions']:,} train, {stats['val_positions']:,} validation -> {args.out_dir}")


if __name__ == "__main__":
    main()
