"""Merge prepared data folders into one folder that pretrain.py can read straight from disk.

    python consolidate_data.py data/elite_2019-* data/elite_2020-* --out-dir data/combined
    python pretrain.py data/combined --blocks 10 --channels 128 --amp

Use this when the data is too big to hold in memory (see chessai/pgn_data.py: consolidate). Writes train_/val_ *.bin and meta.json.
"""
import argparse

from chessai.pgn_data import consolidate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_dirs", nargs="+", help="folders written by prepare_data.py")
    p.add_argument("--out-dir", default="data/combined")
    args = p.parse_args()
    meta = consolidate(args.data_dirs, args.out_dir)
    print(f"-> {args.out_dir}: {meta['train']:,} train, {meta['val']:,} validation positions")


if __name__ == "__main__":
    main()
