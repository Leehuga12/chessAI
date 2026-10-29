"""Opening names, and opening lines to start a game from.

Data: the Lichess chess-openings collection (CC0 / public domain), in chessai/data/openings/{a..e}.tsv with
columns eco, name, pgn. A position is named by the deepest known opening it matches, and the match is by
position, so a transposition into a known line (1.Nf3 c5 2.e4 = 1.e4 c5 2.Nf3) gets the same name.
"""
import csv
import json
import random
import re
from pathlib import Path

import chess
import numpy as np

DEFAULT_DIR = Path(__file__).parent / "data" / "openings"
MOVE_NUMBER = re.compile(r"\d+\.+")


class Openings:
    def __init__(self, folder=DEFAULT_DIR):
        self.entries = []       # dict(id, eco, name, san (list), plies), in file order
        self.by_id = {}
        self.by_position = {}   # position key -> (entry, plies of the line that reaches it); the longest line wins
        for path in sorted(Path(folder).glob("*.tsv")):
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f, delimiter="\t"):
                    self._add(row.get("eco", ""), row.get("name", ""), row.get("pgn", ""))
        # how often strong players reach each line (built from game data by build_opening_popularity.py); optional
        pop = Path(folder) / "popularity.json"
        self.popularity = json.loads(pop.read_text(encoding="utf-8")) if pop.exists() else {}

    def _add(self, eco, name, pgn):
        san = MOVE_NUMBER.sub(" ", pgn).split()
        if not (eco and name and san):
            return
        board = chess.Board()
        try:
            for move in san:
                board.push_san(move)
        except ValueError:
            return   # a line python-chess cannot replay is not usable
        oid = f"{eco} {name}"
        entry = self.by_id.get(oid)
        if entry is None:   # the first line listed under a name is the one offered for starting a game
            entry = dict(id=oid, eco=eco, name=name, san=san, plies=len(san))
            self.entries.append(entry)
            self.by_id[oid] = entry
        # every line, including alternate move orders of the same name, names the position it reaches
        key = self._key(board)
        known = self.by_position.get(key)
        if known is None or len(san) > known[1]:
            self.by_position[key] = (entry, len(san))

    @staticmethod
    def _key(board):
        return board.epd()   # pieces, side to move, castling, en passant: no move counters

    def name_for(self, board):
        """The opening this exact position belongs to, or None."""
        hit = self.by_position.get(self._key(board))
        return hit[0] if hit else None

    def lookup(self, opening_id):
        return self.by_id.get(opening_id)

    def random_entry(self, rng=None, min_plies=3, max_plies=12):
        """A random opening to start a game from. With popularity data, lines are drawn in proportion to the square root
        of how often strong players play them (mainstream lines are likely, rare ones still possible); without it, uniformly
        among lines of moderate length."""
        rng = rng or random
        weights = None
        pool = [e for e in self.entries if min_plies <= e["plies"] <= max_plies and e["id"] in self.popularity]
        if pool:
            weights = [self.popularity[e["id"]] ** 0.5 for e in pool]
        else:
            pool = [e for e in self.entries if min_plies <= e["plies"] <= max_plies] or self.entries
        if not pool:
            return None
        if hasattr(rng, "integers"):   # numpy generator
            if weights is None:
                return pool[int(rng.integers(len(pool)))]
            w = np.asarray(weights, dtype=float)
            return pool[int(rng.choice(len(pool), p=w / w.sum()))]
        return rng.choices(pool, weights=weights)[0] if weights else rng.choice(pool)

    def listing(self):
        """For the page's search box."""
        return [dict(id=e["id"], eco=e["eco"], name=e["name"], plies=e["plies"], games=self.popularity.get(e["id"], 0))
                for e in self.entries]


def count_openings(paths, openings, max_plies=16, min_rating=2200, max_games=None, log=print):
    """For each game, the deepest named opening reached within the first `max_plies` half-moves; returns {opening id: games}."""
    from .pgn_data import read_games   # imported here so loading opening names never needs the PGN reader
    counts = {}
    games = 0
    for game in read_games(paths):
        h = game.headers
        try:
            if min(int(h["WhiteElo"]), int(h["BlackElo"])) < min_rating:
                continue
        except (KeyError, ValueError):
            continue
        board = game.board()
        found = None
        for ply, move in enumerate(game.mainline_moves()):
            if ply >= max_plies:
                break
            board.push(move)
            hit = openings.name_for(board)
            if hit:
                found = hit
        if found:
            counts[found["id"]] = counts.get(found["id"], 0) + 1
        games += 1
        if games % 20000 == 0:
            log(f"counted {games:,} games")
        if max_games and games >= max_games:
            break
    return counts
