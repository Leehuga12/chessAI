"""Readable charts of a pretraining run. Works on a run that is still going.

    python plot_pretrain.py runs/pretrain --data-dir data/elite_2019-12 --out results/pretrain_report.png

Reads pretrain_log.csv from the run folder. --data-dir is optional; with it the value chart shows the
"always predict a draw" baseline computed from the held-out games.
"""
import argparse
import csv
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, GREEN, GREY = "#2f6fdb", "#e07b39", "#2e9e6b", "#888888"
UNIFORM_LOSS = math.log(4672)   # the loss of guessing evenly over every move slot


def read_log(run_dir):
    rows = list(csv.DictReader(open(Path(run_dir) / "pretrain_log.csv", newline="")))
    f = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else None
    train = [{k: f(r, k) for k in ("step", "epoch", "seconds", "lr", "loss", "policy_loss", "value_loss")}
             for r in rows if r["loss"] != ""]
    val = [{k: f(r, k) for k in ("step", "epoch", "seconds", "val_top1", "val_top3", "val_policy_loss", "val_value_mse")}
           for r in rows if r["val_top1"] != ""]
    return train, val


def smooth(y, k=5):
    y = np.asarray(y, dtype=float)
    if len(y) < k:
        return y
    pad = np.concatenate([np.full(k - 1, y[0]), y])
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def draw(train, val, out_path, baseline_value_mse=None, title="Pretraining progress"):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.titlesize": 12})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    step = np.array([r["step"] for r in train])

    # 1. policy loss
    ax = axes[0, 0]
    ax.plot(step, [r["policy_loss"] for r in train], color=BLUE, alpha=0.25, lw=1)
    ax.plot(step, smooth([r["policy_loss"] for r in train]), color=BLUE, lw=2.5, label="training (smoothed)")
    if val:
        ax.plot([r["step"] for r in val], [r["val_policy_loss"] for r in val], "o-", color=ORANGE, lw=2, ms=7,
                label="held-out games")
    top = max(r["policy_loss"] for r in train)
    low = min([r['policy_loss'] for r in train] + [r['val_policy_loss'] for r in val]) * 0.85
    ax.set_ylim(low, top * 1.08)
    ax.set(title="Is it learning which move strong players choose?", xlabel="training step",
           ylabel="move prediction error (lower is better)")
    ax.legend(frameon=False, loc="upper right")
    if val and train:
        gap = val[-1]["val_policy_loss"] - np.mean([r["policy_loss"] for r in train[-5:]])
        ax.text(0.98, 0.62, f"held-out minus training: {gap:+.2f}\n(near zero = learning real patterns;\nlarge and growing = memorising)",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#555")

    # 2. accuracy: grouped bars per finished epoch
    ax = axes[0, 1]
    if len(val) > 6:
        xs = [r["step"] for r in val]
        ax.plot(xs, [100 * r["val_top1"] for r in val], "o-", color=BLUE, lw=2.5, ms=4, label="top-1: its first choice was the move played")
        ax.plot(xs, [100 * r["val_top3"] for r in val], "o-", color=GREEN, lw=2.5, ms=4, label="top-3: the move played was in its top three")
        for key, col, dy in (("val_top1", BLUE, -14), ("val_top3", GREEN, 7)):
            ax.annotate(f"{100 * val[-1][key]:.1f}%", (xs[-1], 100 * val[-1][key]), textcoords="offset points", xytext=(0, dy),
                        ha="center", fontsize=10, fontweight="bold", color=col)
        ax.set_ylim(0, 105)
        ax.axhline(3, color=GREY, ls="--", lw=1)
        ax.text(xs[0], 4.5, "random legal moves would score about 3%", fontsize=9, color="#555")
        ax.set_xlabel("training step")
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.1), fontsize=9)
    elif val:
        x = np.arange(len(val))
        w = 0.38
        b1 = ax.bar(x - w / 2, [100 * r["val_top1"] for r in val], w, color=BLUE, label="top-1: its first choice was the move played")
        b3 = ax.bar(x + w / 2, [100 * r["val_top3"] for r in val], w, color=GREEN, label="top-3: the move played was in its top three")
        for bars in (b1, b3):
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{bar.get_height():.1f}%", ha="center", fontsize=10, fontweight="bold")
        ax.set_xticks(x, [f"epoch {r['epoch']:g}" for r in val])
        ax.set_xlim(-0.7, max(len(val), 3) - 0.3)
        ax.set_ylim(0, 105)
        ax.axhline(3, color=GREY, ls="--", lw=1)
        ax.text(max(len(val), 3) - 0.35, 4.5, "random legal moves would score about 3%", ha="right", fontsize=9, color="#555")
        ax.legend(frameon=False, loc="upper left", fontsize=9, bbox_to_anchor=(0, 1.0), ncol=1)
    else:
        ax.text(0.5, 0.5, "No finished epoch yet", transform=ax.transAxes, ha="center", color=GREY)
    ax.set(title="How often it picks the same move as a strong player", ylabel="% of held-out positions")

    # 3. value error
    ax = axes[1, 0]
    ax.plot(step, smooth([r["value_loss"] for r in train]), color=BLUE, lw=2.5, label="training (smoothed)")
    if val:
        ax.plot([r["step"] for r in val], [r["val_value_mse"] for r in val], "o-", color=ORANGE, lw=2, ms=7, label="held-out games")
    if baseline_value_mse is not None:
        ax.axhline(baseline_value_mse, color=GREY, ls="--")
        ax.text(step.min(), baseline_value_mse, " always predicting a draw", color=GREY, va="bottom", fontsize=9)
    ax.set(title="Can it tell who is winning?", xlabel="training step", ylabel="result prediction error (lower is better)")
    ax.legend(frameon=False)

    # 4. learning rate + summary
    ax = axes[1, 1]
    ax.plot(step, [r["lr"] for r in train], color=GREEN, lw=2.5)
    ax.set(title="Learning rate and run summary", xlabel="training step", ylabel="learning rate")
    last = train[-1]
    lines = [f"steps: {int(last['step']):,}   checkpoints: {len(val)}", f"time: {last['seconds'] / 60:.1f} min"]
    if val:
        lines += [f"held-out top-1: {100 * val[-1]['val_top1']:.1f}%   top-3: {100 * val[-1]['val_top3']:.1f}%",
                  f"held-out value error: {val[-1]['val_value_mse']:.3f}" + (f"  (draw baseline {baseline_value_mse:.3f})" if baseline_value_mse else "")]
    ax.text(0.03, 0.05, "\n".join(lines), transform=ax.transAxes, ha="left", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round", fc="#f4f6fa", ec="#cfd6e4"))

    fig.suptitle(title, fontsize=15, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--data-dir", nargs="+", default=None, help="prepared data folder(s), for the value baseline")
    p.add_argument("--out", default="results/pretrain_report.png")
    args = p.parse_args()
    train, val = read_log(args.run_dir)
    if not train:
        raise SystemExit("the log has no training rows yet")
    baseline = None
    if args.data_dir:
        from chessai.pgn_data import load_split
        baseline = float(np.mean(load_split(args.data_dir, "val")["values"].astype(np.float32) ** 2))
    draw(train, val, args.out, baseline, title=f"Pretraining progress (run: {Path(args.run_dir).name})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
