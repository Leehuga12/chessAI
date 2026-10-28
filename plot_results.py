"""Chart the evaluation results: score per opponent, and estimated Elo, against training iteration.

    python plot_results.py results/results.csv results/improvement.png
"""
import csv
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(csv_path="results/results.csv", png_path="results/improvement.png"):
    scores = defaultdict(dict)    # opponent -> {iteration: score}   (latest row wins)
    elos = defaultdict(dict)      # opponent -> {iteration: (elo, margin)}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            it = int(row["iteration"])
            scores[row["opponent"]][it] = float(row["score"])
            if row["est_elo"]:
                elos[row["opponent"]][it] = (float(row["est_elo"]), float(row["elo_margin"] or 0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for opp, series in sorted(scores.items()):
        its = sorted(series)
        ax1.plot(its, [series[i] * 100 for i in its], marker="o", label=opp)
    ax1.set(title="Score vs fixed opponents", xlabel="training iteration", ylabel="score (%)", ylim=(-2, 102))
    ax1.legend()
    ax1.grid(alpha=0.3)

    for opp, series in sorted(elos.items()):
        its = sorted(series)
        ax2.errorbar(its, [series[i][0] for i in its], yerr=[series[i][1] for i in its],
                     marker="o", capsize=3, label=opp)
    ax2.set(title="Estimated Elo (95% margin)", xlabel="training iteration", ylabel="Elo")
    if elos:
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "no Elo-rated opponents evaluated yet", ha="center", transform=ax2.transAxes)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    print("wrote", png_path)


if __name__ == "__main__":
    main(*sys.argv[1:3])
