"""One round of gradient steps on the replay buffer."""
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .network import ChessNet
from .selfplay import ReplayBuffer


def train_steps(net, buffer, optimizer, steps, batch_size, device):
    net.train()
    totals = np.zeros(3)
    for _ in range(steps):
        planes, policy, values = buffer.sample_batch(batch_size)
        planes, policy, values = (torch.from_numpy(a).to(device) for a in (planes, policy, values))
        logits, pred = net(planes)
        policy_loss = -(policy * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        value_loss = F.mse_loss(pred, values)
        loss = policy_loss + value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        totals += (loss.item(), policy_loss.item(), value_loss.item())
    net.eval()
    return totals / max(steps, 1)


# ---- resume support -------------------------------------------------------------------------------
STATE_FILE = "state.pt"       # nets + optimizer + iteration, written after every iteration
BUFFER_FILE = "buffer.pkl"    # replay buffer, written every few iterations (it is large)


def _atomic(path, write):
    """Write to a temp file then rename, so a crash mid-write cannot corrupt the previous save."""
    tmp = Path(str(path) + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def save_state(out, net, best, optimizer, iteration, elapsed):
    _atomic(Path(out) / STATE_FILE, lambda p: torch.save(
        {"config": net.config, "net": net.state_dict(), "best": best.state_dict(),
         "optimizer": optimizer.state_dict(), "iteration": iteration, "elapsed": elapsed}, p))


def load_state(out, device, lr):
    """Returns (net, best, optimizer, iteration, elapsed_seconds). The learning rate comes from `lr`."""
    data = torch.load(Path(out) / STATE_FILE, map_location=device)
    net, best = (ChessNet(**data["config"]).to(device) for _ in range(2))
    net.load_state_dict(data["net"])
    best.load_state_dict(data["best"])
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    optimizer.load_state_dict(data["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = lr
    return net.eval(), best.eval(), optimizer, data["iteration"], data["elapsed"]


def save_buffer(out, buffer, iteration):
    def write(p):
        with open(p, "wb") as f:
            pickle.dump({"iteration": iteration, "samples": list(buffer.data)}, f, protocol=pickle.HIGHEST_PROTOCOL)
    _atomic(Path(out) / BUFFER_FILE, write)


def load_buffer(out, capacity):
    """Returns (ReplayBuffer, iteration it was saved at), or (empty ReplayBuffer, None) if there is no file."""
    buffer = ReplayBuffer(capacity)
    path = Path(out) / BUFFER_FILE
    if not path.exists():
        return buffer, None
    with open(path, "rb") as f:   # our own file, written by save_buffer
        data = pickle.load(f)
    buffer.add(data["samples"])
    return buffer, data["iteration"]
