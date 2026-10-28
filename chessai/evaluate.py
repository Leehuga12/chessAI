"""Fixed-opponent evaluation: play a network against a reference opponent and score it.

Opponent specs (the reference opponents never change between evaluations):
  random        uniformly random legal moves (win rate only, no Elo claimed)
  sf-skill0     Stockfish at Skill Level 0 (win rate only)
  sf-elo<N>     Stockfish with UCI_LimitStrength and UCI_Elo=N (N >= 1320). Nominal Elo,
                used to turn a score into an estimated Elo.

Stockfish searches a fixed node budget (not a time limit), so results do not depend on
how fast the machine is.
"""
import math
import os
import random
from pathlib import Path

import chess
import chess.engine
import numpy as np

from .mcts import MCTS
from .moves import index_to_move

MAX_PLIES = 300
STOCKFISH_NODES = 2000
DEFAULT_ENGINE = Path(__file__).resolve().parent.parent / "tools" / "stockfish" / "stockfish" / \
    "stockfish-windows-x86-64-universal.exe"

# Short fixed opening lines (SAN), so every evaluation starts from the same positions.
OPENINGS = [
    "e4 e5 Nf3 Nc6",
    "e4 c5 Nf3 d6",
    "d4 d5 c4 e6",
    "d4 Nf6 c4 g6",
    "e4 e6 d4 d5",
    "c4 e5 Nc3 Nf6",
    "Nf3 d5 g3 Nf6",
    "e4 c6 d4 d5",
]


def opening_board(index: int) -> chess.Board:
    board = chess.Board()
    for san in OPENINGS[index % len(OPENINGS)].split():
        board.push_san(san)
    return board


class RandomOpponent:
    elo = None

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def move(self, board):
        return self.rng.choice(list(board.legal_moves))

    def close(self):
        pass


class StockfishOpponent:
    def __init__(self, engine_path, elo=None, skill=None, nodes=STOCKFISH_NODES):
        self.elo = elo
        self.nodes = nodes
        self.engine = chess.engine.SimpleEngine.popen_uci(os.path.abspath(engine_path))
        if elo is not None:
            self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
        elif skill is not None:
            self.engine.configure({"Skill Level": skill})

    def move(self, board):
        return self.engine.play(board, chess.engine.Limit(nodes=self.nodes)).move

    def close(self):
        self.engine.quit()


def make_opponent(spec: str, engine_path=DEFAULT_ENGINE):
    if spec == "random":
        return RandomOpponent()
    if spec == "sf-skill0":
        return StockfishOpponent(engine_path, skill=0, nodes=1)
    if spec.startswith("sf-elo"):
        return StockfishOpponent(engine_path, elo=int(spec[len("sf-elo"):]))
    raise ValueError(f"unknown opponent spec: {spec}")


def play_eval_game(net, opponent, net_is_white, opening_index, device, simulations):
    """Returns the network's result: 1 win, 0.5 draw, 0 loss."""
    mcts = MCTS(net, device, simulations, rng=np.random.default_rng(0))
    board = opening_board(opening_index)
    net_color = chess.WHITE if net_is_white else chess.BLACK
    while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
        if board.turn == net_color:
            board.push(index_to_move(int(mcts.search(board).argmax()), board))
        else:
            board.push(opponent.move(board))
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.5
    return 1.0 if outcome.winner == net_color else 0.0


def evaluate(net, opponent, games, device, simulations):
    """Alternates colours; game g uses opening g // 2 for both colours."""
    wins = draws = losses = 0
    for g in range(games):
        r = play_eval_game(net, opponent, g % 2 == 0, g // 2, device, simulations)
        wins += r == 1.0
        draws += r == 0.5
        losses += r == 0.0
    return {"games": games, "wins": wins, "draws": draws, "losses": losses,
            "score": (wins + 0.5 * draws) / max(games, 1)}


def estimate_elo(score: float, games: int, opponent_elo: float) -> float:
    """Elo implied by a match score, with +0.5 / +1 smoothing so 0% and 100% stay finite.
    With few games this is very noisy; see elo_error_margin."""
    s = (score * games + 0.5) / (games + 1)
    return opponent_elo + 400.0 * math.log10(s / (1.0 - s))


def elo_error_margin(wins: int, draws: int, losses: int) -> float:
    """Approximate 95% margin (+/- Elo): a Wilson interval on the score (draws count as half),
    converted to Elo. Stays finite when every game has the same result."""
    n = wins + draws + losses
    if n < 1:
        return float("inf")
    s = (wins + 0.5 * draws) / n
    z = 1.96
    centre = (s + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(s * (1 - s) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    lo, hi = min(max(centre - half, 0.001), 0.999), min(max(centre + half, 0.001), 0.999)
    f = lambda x: 400.0 * math.log10(x / (1.0 - x))
    return (f(hi) - f(lo)) / 2.0
