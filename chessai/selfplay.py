"""Self-play game generation and the replay buffer."""
import random
from collections import deque
from dataclasses import dataclass

import chess
import numpy as np

from .moves import POLICY_SIZE
from .parallel import MAX_PLIES, TEMPERATURE_PLIES, pick_move, play_games  # noqa: F401


@dataclass
class Sample:
    planes: np.ndarray    # uint8 (18, 8, 8); every plane is binary
    indices: np.ndarray   # int16, moves with visits
    probs: np.ndarray     # float32, visit distribution over `indices`
    value: float          # game result from the mover's point of view


def _to_samples(result, history):
    samples = []
    for planes, idx, probs, mover in history:
        z = 0.0 if result == 0 else (1.0 if (result == 1) == (mover == chess.WHITE) else -1.0)
        samples.append(Sample(planes, idx, probs, z))
    return samples


def generate_games(net, device, games, simulations, parallel=16, leaves=8, rng=None):
    """Self-play `games` games, `parallel` at a time. Returns (samples, results) where results
    holds +1 / -1 / 0 (white win / black win / draw) per game."""
    out = play_games([(net, net)] * games, device, simulations, parallel, leaves, rng,
                     add_noise=True, record=True)
    samples = []
    for result, history in out:
        samples.extend(_to_samples(result, history))
    return samples, [r for r, _ in out]


def play_game(net, device, simulations: int, rng=None, leaves=8):
    """Self-play one game. Returns (samples, result)."""
    samples, results = generate_games(net, device, 1, simulations, 1, leaves, rng)
    return samples, results[0]


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.data = deque(maxlen=capacity)

    def __len__(self):
        return len(self.data)

    def add(self, samples):
        self.data.extend(samples)

    def sample_batch(self, batch_size: int):
        """Returns (planes, policy_targets, values) as float32 numpy arrays."""
        batch = random.sample(self.data, min(batch_size, len(self.data)))
        planes = np.stack([s.planes for s in batch]).astype(np.float32)
        policy = np.zeros((len(batch), POLICY_SIZE), dtype=np.float32)
        for row, s in enumerate(batch):
            policy[row, s.indices] = s.probs
        values = np.array([s.value for s in batch], dtype=np.float32)
        return planes, policy, values
