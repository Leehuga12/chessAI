"""Supervised warm start: teach the net to predict the moves and results of strong games.

    python prepare_data.py games.pgn --out-dir data/elite
    python pretrain.py data/elite --out-dir runs/pretrain --epochs 3 --blocks 6 --channels 64
    python pretrain.py data/a data/b data/c --blocks 10 --channels 128 --amp --save-every 20000     # several folders, bigger net
    python pretrain.py data/a data/b data/c --resume runs/pretrain --epochs 6                         # --epochs is the total to reach

Writes pretrained.pt (every --save-every steps and every epoch; same format as the other checkpoints, so evaluate.py,
serve.py and train.py --init-from can load it), pretrain_log.csv and pretrain_state.pt (for --resume, which continues from the
last save, even in the middle of an epoch) to --out-dir.
Logged: policy loss, value loss, and on a fixed sample of held-out games the top-1 / top-3 move accuracy (over all 4672 move
slots, illegal moves not masked, so slightly pessimistic) and value error.
"""
import argparse
import csv
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from chessai.network import ChessNet, load_checkpoint, save_checkpoint
from chessai.pgn_data import load_split, unpack_planes

LOG_FIELDS = ["step", "epoch", "seconds", "lr", "loss", "policy_loss", "value_loss",
              "val_top1", "val_top3", "val_policy_loss", "val_value_mse"]


def lr_at(step, total, base, warmup=100, floor=0.05):
    """Linear warm-up, then cosine decay down to floor * base."""
    if step < warmup:
        return base * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(total - warmup, 1))
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))


def batch_tensors(data, idx, device):
    planes = torch.from_numpy(unpack_planes(data["packed"][idx])).to(device)
    moves = torch.from_numpy(data["moves"][idx].astype(np.int64)).to(device)
    values = torch.from_numpy(data["values"][idx].astype(np.float32)).to(device)
    return planes, moves, values


def validation_sample(n_total, limit, seed=0):
    """The same random positions every time, so validation numbers can be compared across the run."""
    if n_total <= limit:
        return np.arange(n_total)
    return np.sort(np.random.default_rng(seed).choice(n_total, size=limit, replace=False))


@torch.inference_mode()
def validate(net, data, device, sample, batch_size=1024):
    net.eval()
    n = len(sample)
    top1 = top3 = 0
    pol = val = 0.0
    for start in range(0, n, batch_size):
        planes, moves, values = batch_tensors(data, sample[start:start + batch_size], device)
        logits, pred = net(planes)
        logits = logits.float()
        pol += F.cross_entropy(logits, moves, reduction="sum").item()
        val += F.mse_loss(pred.float(), values, reduction="sum").item()
        top = logits.topk(3, dim=1).indices
        top1 += (top[:, 0] == moves).sum().item()
        top3 += (top == moves[:, None]).any(dim=1).sum().item()
    net.train()
    return dict(val_top1=top1 / n, val_top3=top3 / n, val_policy_loss=pol / n, val_value_mse=val / n)


def atomic_save(obj, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_dirs", nargs="+", help="folder(s) written by prepare_data.py")
    p.add_argument("--out-dir", default="runs/pretrain")
    p.add_argument("--resume", metavar="DIR", help="continue the run saved in DIR")
    p.add_argument("--init-from", metavar="CKPT", help="start a NEW run from this net's weights (fresh optimizer and learning-rate schedule), "
                   "e.g. to keep improving an existing net on more data; its size overrides --blocks/--channels")
    p.add_argument("--epochs", type=int, default=3, help="total epochs to reach")
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--value-weight", type=float, default=1.0, help="weight of the value loss against the policy loss")
    p.add_argument("--val-positions", type=int, default=50_000)
    p.add_argument("--log-every", type=int, default=200, help="steps between log lines")
    p.add_argument("--save-every", type=int, default=0,
                   help="also validate, save pretrained.pt and save resume state every N steps (0 = only at the end of each epoch)")
    p.add_argument("--max-steps", type=int, default=None, help="stop early (testing, speed checks)")
    p.add_argument("--amp", action="store_true", help="bfloat16 mixed precision on the GPU (about 1.5-1.8x faster on nets of 8+ blocks)")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.resume or args.out_dir)
    log_path = out / "pretrain_log.csv"
    state_path = out / "pretrain_state.pt"

    t_load = time.time()
    train, val = load_split(args.data_dirs, "train"), load_split(args.data_dirs, "val")
    sample = validation_sample(len(val["moves"]), args.val_positions)
    if isinstance(train["packed"], np.memmap):   # read the file once, in order, so the first epoch's random reads hit the cache
        t_warm = time.time()
        for i in range(0, len(train["packed"]), 4_000_000):
            train["packed"][i:i + 4_000_000].max()
        print(f"read the data file once to warm the cache ({time.time() - t_warm:.0f}s)", flush=True)
    n = len(train["moves"])
    steps_per_epoch = n // args.batch_size
    if steps_per_epoch == 0:
        raise SystemExit(f"{n} training positions is fewer than one batch of {args.batch_size}")
    total_steps = args.max_steps or steps_per_epoch * args.epochs

    skip = 0   # batches of the current epoch that were already done before a resume
    if args.resume:
        if not state_path.exists():
            p.error(f"nothing to resume: {state_path} does not exist")
        state = torch.load(state_path, map_location=device)
        net = ChessNet(**state["config"]).to(device)
        net.load_state_dict(state["net"])
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, step, elapsed, skip = state["epoch"], state["step"], state["elapsed"], state.get("epoch_batches", 0)
        print(f"resumed {out} after epoch {start_epoch}, batch {skip} (step {step})")
        if start_epoch >= args.epochs:
            print("already at the requested number of epochs; raise --epochs to continue")
            return
    else:
        if state_path.exists():
            p.error(f"{out} already holds a run; use --resume {out} or pick a new --out-dir")
        out.mkdir(parents=True, exist_ok=True)
        if args.init_from:
            net, _ = load_checkpoint(args.init_from, device)
            print(f"starting from the weights in {args.init_from} ({net.config})")
        else:
            net = ChessNet(args.blocks, args.channels).to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        start_epoch, step, elapsed = 0, 0, 0.0
        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, LOG_FIELDS).writeheader()

    print(f"device={device}  params={sum(x.numel() for x in net.parameters()):,}  "
          f"train={n:,} positions ({steps_per_epoch:,} steps/epoch)  val={len(val['moves']):,} "
          f"(validating on {len(sample):,})  loaded in {time.time() - t_load:.0f}s", flush=True)
    start = time.time()
    run_loss = np.zeros(3)
    run_count = 0

    def write_row(row):
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, LOG_FIELDS).writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})

    def checkpoint(epochs_done, batches_done, progress, lr):
        """Validate, then save the net and the full resume state. progress = epochs completed, fractional mid-epoch."""
        metrics = validate(net, val, device, sample)
        write_row(dict(step=step, epoch=round(progress, 3), seconds=elapsed + time.time() - start, lr=lr, loss="", policy_loss="",
                       value_loss="", **metrics))
        print(f"  held-out at {progress:.2f} epochs: top-1 {metrics['val_top1']:.1%}  top-3 {metrics['val_top3']:.1%}  "
              f"value error {metrics['val_value_mse']:.3f}", flush=True)
        save_checkpoint(net.eval(), out / "pretrained.pt", {"pretrained": True, "epoch": round(progress, 2), "steps": step, **metrics})
        net.train()
        atomic_save({"config": net.config, "net": net.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epochs_done,
                     "epoch_batches": batches_done, "step": step, "elapsed": elapsed + time.time() - start}, state_path)

    stopped = False
    net.train()
    for epoch in range(start_epoch, args.epochs):
        order = np.random.default_rng([args.seed, epoch]).permutation(n)   # the same order after a resume
        first = skip if epoch == start_epoch else 0
        for b in range(first, steps_per_epoch):
            lr = lr_at(step, total_steps, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            planes, moves, values = batch_tensors(train, order[b * args.batch_size:(b + 1) * args.batch_size], device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.amp and device == "cuda"):
                logits, pred = net(planes)
            policy_loss = F.cross_entropy(logits.float(), moves)
            value_loss = F.mse_loss(pred.float(), values)
            loss = policy_loss + args.value_weight * value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            run_loss += (loss.item(), policy_loss.item(), value_loss.item())
            run_count += 1
            if step % args.log_every == 0 or step == total_steps:
                avg = run_loss / run_count
                spent = elapsed + time.time() - start
                row = dict(step=step, epoch=epoch + 1, seconds=spent, lr=lr, loss=avg[0], policy_loss=avg[1], value_loss=avg[2])
                row.update({k: "" for k in LOG_FIELDS if k.startswith("val_")})
                write_row(row)
                left = (total_steps - step) * (time.time() - start) / max(step - (state["step"] if args.resume else 0), 1)
                print(f"step {step:>8,}/{total_steps:,}  epoch {epoch + 1}  loss {avg[0]:.3f}  policy {avg[1]:.3f}  value {avg[2]:.3f}  "
                      f"lr {lr:.2e}  ~{left / 60:.0f} min left", flush=True)
                run_loss[:] = 0
                run_count = 0
            if args.max_steps and step >= args.max_steps:   # an early stop still leaves a usable, resumable checkpoint
                checkpoint(epoch, b + 1, epoch + (b + 1) / steps_per_epoch, lr)
                stopped = True
                break
            if args.save_every and step % args.save_every == 0 and b + 1 < steps_per_epoch:
                checkpoint(epoch, b + 1, epoch + (b + 1) / steps_per_epoch, lr)
        if stopped:
            break
        checkpoint(epoch + 1, 0, epoch + 1, lr)
        print(f"epoch {epoch + 1} done", flush=True)


if __name__ == "__main__":
    main()
