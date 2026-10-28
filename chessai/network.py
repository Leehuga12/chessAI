"""Residual policy+value network.

Input  (B, 18, 8, 8) from encoding.encode_board.
Output policy logits (B, 4672) laid out as from_square * 73 + move_plane (see moves.py),
       value (B,) in [-1, 1] from the side to move's point of view.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import NUM_PLANES
from .moves import NUM_MOVE_PLANES, POLICY_SIZE


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class ChessNet(nn.Module):
    def __init__(self, blocks: int = 6, channels: int = 64):
        super().__init__()
        self.config = {"blocks": blocks, "channels": channels}
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.tower = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, NUM_MOVE_PLANES, 1),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        h = self.tower(self.stem(x))
        policy = self.policy_head(h)                       # (B, 73, 8, 8)
        policy = policy.permute(0, 2, 3, 1).reshape(-1, POLICY_SIZE)
        value = self.value_head(h).squeeze(-1)
        return policy, value


def save_checkpoint(net: ChessNet, path, extra=None):
    torch.save({"config": net.config, "state": net.state_dict(), "extra": extra or {}}, path)


def load_checkpoint(path, device="cpu"):
    data = torch.load(path, map_location=device)
    net = ChessNet(**data["config"]).to(device)
    net.load_state_dict(data["state"])
    return net, data.get("extra", {})
