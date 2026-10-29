"""One-page picture of a PGN training set: who played, how strong, how the games ended, how long they were.

    python report_data.py "data/raw_elite/Lichess Elite Database/lichess_elite_2019-12.pgn" --out results/data_report.png
    python report_data.py games.pgn --min-elo 2200 --sample 10000

Reads only the headers of every game (fast), and parses the moves of a random sample to get game lengths.
Pass the same --min-elo you gave prepare_data.py so the picture shows what the net actually trains on.
"""
import argparse
import re
import textwrap
from collections import Counter
from pathlib import Path

import chess.pgn
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from chessai.pgn_data import _text_streams  # noqa: E402

WHITE, DRAW, BLACK = "#e8e8e8", "#e9b44c", "#3b4252"
ACCENT = "#2f6fdb"


def scan(paths, min_elo):
    """Per-game header data (ratings, results, time control, opening) for every accepted game."""
    lows, highs, results, controls, openings = [], [], Counter(), Counter(), Counter()
    seen = skipped = 0
    for path in paths:
        with _text_streams(path) as streams:
            for stream in streams:
                while True:
                    h = chess.pgn.read_headers(stream)
                    if h is None:
                        break
                    seen += 1
                    try:
                        w, b = int(h["WhiteElo"]), int(h["BlackElo"])
                    except (KeyError, ValueError):
                        skipped += 1
                        continue
                    if min(w, b) < min_elo or h.get("Result") not in ("1-0", "0-1", "1/2-1/2"):
                        skipped += 1
                        continue
                    lows.append(min(w, b))
                    highs.append(max(w, b))
                    results[h["Result"]] += 1
                    m = re.search(r"Rated (\w+)", h.get("Event", ""))
                    controls[m.group(1) if m else "Other"] += 1
                    if h.get("Opening"):
                        openings[re.split(r"[:,]", h["Opening"])[0].strip()] += 1
    return dict(lows=np.array(lows), highs=np.array(highs), results=results, controls=controls, openings=openings,
                seen=seen, skipped=skipped)


def sample_lengths(paths, min_elo, n, rng):
    """Plies per game for roughly n games, taken at random from the accepted ones."""
    lengths = []
    for path in paths:
        with _text_streams(path) as streams:
            for stream in streams:
                while len(lengths) < n:
                    game = chess.pgn.read_game(stream)
                    if game is None:
                        break
                    h = game.headers
                    try:
                        if min(int(h["WhiteElo"]), int(h["BlackElo"])) < min_elo or h.get("Result") not in ("1-0", "0-1", "1/2-1/2"):
                            continue
                    except (KeyError, ValueError):
                        continue
                    if rng.random() < 0.5:   # thin the stream a little so the sample is not only the file's first games
                        lengths.append(sum(1 for _ in game.mainline_moves()))
    return np.array(lengths)


def draw_report(data, lengths, title, out_path):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.titlesize": 12})
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    n = len(data["lows"])
    res = data["results"]
    positions = int(lengths.mean() * n) if len(lengths) else 0

    # 1. summary text
    ax = axes[0, 0]
    ax.axis("off")
    lines = [("Games used", f"{n:,}"), ("Games read", f"{data['seen']:,}"),
             ("Left out", f"{data['skipped']:,}"),
             ("Positions (one per move)", f"~{positions:,}" if positions else "-"),
             ("Stronger player, median", f"{int(np.median(data['highs']))}"),
             ("Weaker player, median", f"{int(np.median(data['lows']))}"),
             ("Typical game length", f"{int(np.median(lengths)) // 2} moves" if len(lengths) else "-")]
    ax.set_title("What this data is", loc="left")
    for i, (k, v) in enumerate(lines):
        y = 0.9 - i * 0.12
        ax.text(0.0, y, k, color="#555", transform=ax.transAxes, va="center")
        ax.text(0.62, y, v, fontweight="bold", transform=ax.transAxes, va="center")

    # 2. ratings
    ax = axes[0, 1]
    lo = int(data["lows"].min() // 50 * 50)
    hi = int(data["highs"].max() // 50 * 50 + 100)
    bins = np.arange(lo, hi, 50)
    ax.hist(data["highs"], bins=bins, color=ACCENT, alpha=0.85, label="stronger player")
    ax.hist(data["lows"], bins=bins, color="#e07b39", alpha=0.85, label="weaker player")
    ax.set(title="How strong the players are", xlabel="Lichess rating", ylabel="games")
    ax.legend(frameon=False)

    # 3. results
    ax = axes[0, 2]
    counts = [res["1-0"], res["1/2-1/2"], res["0-1"]]
    bars = ax.bar(["White wins", "Draws", "Black wins"], counts, color=[WHITE, DRAW, BLACK], edgecolor="#888")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c, f"{100 * c / n:.1f}%", ha="center", va="bottom", fontweight="bold")
    score = (res["1-0"] + 0.5 * res["1/2-1/2"]) / n
    ax.set(title=f"How games ended  (White scores {100 * score:.1f}%)", ylabel="games")
    ax.set_ylim(0, max(counts) * 1.12)

    # 4. time controls
    ax = axes[1, 0]
    items = data["controls"].most_common(6)
    ax.barh([k for k, _ in items][::-1], [v for _, v in items][::-1], color=ACCENT)
    for i, (k, v) in enumerate(items[::-1]):
        ax.text(v, i, f" {100 * v / n:.0f}%", va="center")
    ax.set(title="Time control", xlabel="games")

    # 5. length
    ax = axes[1, 1]
    if len(lengths):
        ax.hist(np.clip(lengths, 0, 200), bins=np.arange(0, 205, 5), color=ACCENT)
        ax.axvline(np.median(lengths), color="#e07b39", lw=2)
        ax.text(np.median(lengths) + 2, ax.get_ylim()[1] * 0.92, f"median {int(np.median(lengths))}", color="#e07b39", fontweight="bold")
    ax.set(title="How long games last", xlabel="half-moves (one move by one side)", ylabel="games (sample)")

    # 6. openings
    ax = axes[1, 2]
    items = data["openings"].most_common(10)
    ax.barh([k[:30] for k, _ in items][::-1], [v for _, v in items][::-1], color=ACCENT)
    ax.set(title="Most common openings", xlabel="games")

    blitz = 100 * data["controls"].get("Blitz", 0) / n
    note = ("Read with care: ratings are Lichess ratings (a different scale from FIDE or chess.com). "
            + (f"{blitz:.0f}% of these are blitz games, played fast, so they contain more mistakes than slow games. " if blitz > 50 else "")
            + "Games shorter than 16 half-moves are dropped when the data is prepared.")
    fig.suptitle(title, fontsize=15, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, 0.005, textwrap.fill(note, 190), fontsize=9.5, color="#555")
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return dict(games=n, positions=positions, white_score=score, median_length=int(np.median(lengths)) if len(lengths) else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pgn", nargs="+")
    p.add_argument("--min-elo", type=int, default=2200)
    p.add_argument("--sample", type=int, default=10_000, help="games to parse for the length chart")
    p.add_argument("--out", default="results/data_report.png")
    p.add_argument("--title", default=None)
    args = p.parse_args()

    rng = np.random.default_rng(0)
    data = scan(args.pgn, args.min_elo)
    lengths = sample_lengths(args.pgn, args.min_elo, args.sample, rng)
    title = args.title or f"Training data: {', '.join(Path(x).name for x in args.pgn[:2])}"
    info = draw_report(data, lengths, title, args.out)
    print(f"{info['games']:,} games, ~{info['positions']:,} positions, White scores {100 * info['white_score']:.1f}% -> {args.out}")


if __name__ == "__main__":
    main()
