"""Play many games at once so their leaf evaluations share network calls."""
from collections import deque

import chess
import numpy as np

from .encoding import encode_board
from .mcts import Search, run_round
from .moves import index_to_move
from .rules import terminal_value

MAX_PLIES = 200          # games longer than this are scored as draws
TEMPERATURE_PLIES = 30   # sample moves in proportion to visits early, play the best move after


def pick_move(visits: np.ndarray, ply: int, rng, temperature_plies: int = TEMPERATURE_PLIES) -> int:
    if ply < temperature_plies:
        return int(rng.choice(len(visits), p=visits / visits.sum()))
    return int(visits.argmax())


class _Game:
    def __init__(self, index, white_net, black_net):
        self.index = index
        self.nets = {chess.WHITE: white_net, chess.BLACK: black_net}
        self.board = chess.Board()
        self.history = []   # (planes uint8, indices int16, probs float32, mover)
        self.search = None


def play_games(specs, device, simulations, parallel=16, leaves=8, rng=None, max_plies=MAX_PLIES,
               temperature_plies=TEMPERATURE_PLIES, add_noise=False, record=False):
    """specs: list of (white_net, black_net). Up to `parallel` games run at a time.

    Returns a list, in spec order, of (result, history): result is +1 white win, -1 black win,
    0 draw; history is only filled in when record=True."""
    rng = rng or np.random.default_rng()
    results = [None] * len(specs)
    waiting = deque(range(len(specs)))
    active = []

    while waiting or active:
        while waiting and len(active) < parallel:
            i = waiting.popleft()
            active.append(_Game(i, *specs[i]))

        still_active = []
        for g in active:
            value = terminal_value(g.board)
            if value is not None or g.board.ply() >= max_plies:
                if value == -1.0:   # side to move is mated
                    result = -1 if g.board.turn == chess.WHITE else 1
                else:
                    result = 0
                results[g.index] = (result, g.history)
            else:
                if g.search is None:
                    g.search = Search(g.nets[g.board.turn], g.board, simulations,
                                      add_noise=add_noise, rng=rng)
                still_active.append(g)
        active = still_active
        if not active:
            continue

        run_round([g.search for g in active], device, leaves)

        for g in active:
            if g.search.done:
                visits = g.search.visits()
                if record:
                    nz = np.nonzero(visits)[0]
                    g.history.append((encode_board(g.board).astype(np.uint8), nz.astype(np.int16),
                                      (visits[nz] / visits[nz].sum()).astype(np.float32), g.board.turn))
                idx = pick_move(visits, g.board.ply(), rng, temperature_plies)
                g.board.push(index_to_move(idx, g.board))
                g.search = None
    return results
