"""Head-to-head matches between two networks (used for gating, and later for evaluation)."""
import numpy as np

from .parallel import MAX_PLIES, play_games

OPENING_PLIES = 4   # a few sampled moves so repeated games are not identical


def play_match(net_a, net_b, device, games, simulations, seed=0, parallel=16, leaves=8):
    """Plays `games` games alternating colours, `parallel` at a time.

    Returns net_a's score in [0, 1] (win = 1, draw = 0.5) and (wins, draws, losses) for net_a."""
    rng = np.random.default_rng(seed)
    specs = [(net_a, net_b) if g % 2 == 0 else (net_b, net_a) for g in range(games)]
    out = play_games(specs, device, simulations, parallel, leaves, rng, max_plies=MAX_PLIES,
                     temperature_plies=OPENING_PLIES)
    wins = draws = losses = 0
    for g, (r, _) in enumerate(out):
        r_a = r if g % 2 == 0 else -r
        wins += r_a == 1
        losses += r_a == -1
        draws += r_a == 0
    return (wins + 0.5 * draws) / max(games, 1), (wins, draws, losses)
