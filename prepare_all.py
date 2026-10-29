"""Prepare many PGN files at once, one process per file (each file is read on a single core, so this is the fast way).

    python prepare_all.py "data/raw_elite/Lichess Elite Database/lichess_elite_2019-*.pgn" --out-root data/months --max-games 100000
    python consolidate_data.py data/months/* --out-dir data/combined

A file whose output folder already has prepare_stats.json is skipped, so an interrupted run can simply be started again.
Quote the pattern (the program expands it) so the shell does not.
"""
import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from chessai.pgn_data import prepare


def one(job):
    path, out, min_elo, min_plies, max_games = job
    stats = prepare([path], out, min_elo=min_elo, min_plies=min_plies, max_games=max_games, log=lambda *_: None)
    return path, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("patterns", nargs="+", help="PGN files or quoted glob patterns (.pgn, .zip, .pgn.zst)")
    p.add_argument("--out-root", default="data/months", help="one sub-folder per input file is made here")
    p.add_argument("--min-elo", type=int, default=2200)
    p.add_argument("--min-plies", type=int, default=16)
    p.add_argument("--max-games", type=int, default=None, help="per file")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = p.parse_args()

    files = sorted({f for pat in args.patterns for f in glob.glob(pat)})
    if not files:
        sys.exit("no files match " + " ".join(args.patterns))
    jobs = []
    for f in files:
        out = Path(args.out_root) / Path(f).stem
        if (out / "prepare_stats.json").exists():
            print(f"skip (already prepared): {out}")
        else:
            jobs.append((f, str(out), args.min_elo, args.min_plies, args.max_games))
    start = time.time()
    total = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(one, j) for j in jobs]):
            path, s = fut.result()
            total += s["train_positions"] + s["val_positions"]
            print(f"{Path(path).name}: {s['games_used']:,} games, {s['train_positions'] + s['val_positions']:,} positions", flush=True)
    print(f"done: {len(jobs)} files, {total:,} positions in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
