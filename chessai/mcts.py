"""Monte Carlo tree search guided by the network (AlphaZero style, PUCT), with batched evaluation.

Every node stores W from the point of view of the player to move *at that node*, so a parent
scores a child with -child.W / child.N.

To keep a GPU busy the search collects several leaves before asking the network about them.
While a leaf is waiting, a "virtual loss" makes its path look bad so the next descent goes
elsewhere. `Search` is a step-wise object (request leaves -> evaluate -> fulfill) so that many
searches, from many games at once, can share one network call (see parallel.py).
"""
import math

import chess
import numpy as np
import torch

from .encoding import encode_board
from .moves import POLICY_SIZE, move_to_index
from .rules import terminal_value as _terminal_value

VIRTUAL_LOSS = 1


class Node:
    __slots__ = ("prior", "N", "W", "edges", "expanded", "terminal_value", "pending")

    def __init__(self, prior: float):
        self.prior = prior
        self.N = 0
        self.W = 0.0
        self.edges = []             # list of (policy_index, move, child)
        self.expanded = False
        self.terminal_value = None  # value for the player to move, if the game is over here
        self.pending = False        # waiting for a network evaluation


class Request:
    """A leaf waiting for the network."""
    __slots__ = ("node", "path", "planes", "moves", "idxs", "is_root")

    def __init__(self, node, path, planes, moves, idxs, is_root=False):
        self.node, self.path, self.planes = node, path, planes
        self.moves, self.idxs, self.is_root = moves, idxs, is_root


@torch.inference_mode()
def evaluate_batch(net, planes, device):
    """planes: list of (18, 8, 8) arrays. Returns (logits (N, POLICY_SIZE), values (N,)) as numpy."""
    x = torch.from_numpy(np.stack(planes)).to(device)
    logits, values = net(x)
    return logits.float().cpu().numpy(), values.float().cpu().numpy()


def _make_request(node, path, board, is_root=False):
    moves = list(board.legal_moves)
    idxs = np.fromiter((move_to_index(m, board) for m in moves), dtype=np.int64, count=len(moves))
    return Request(node, path, encode_board(board), moves, idxs, is_root)


class Search:
    def __init__(self, net, board: chess.Board, simulations: int, c_puct=1.5, add_noise=False,
                 dirichlet_alpha=0.3, noise_fraction=0.25, rng=None):
        if _terminal_value(board) is not None:
            raise ValueError("cannot search a finished game")
        self.net = net
        self.board = board
        self.simulations = simulations
        self.c_puct = c_puct
        self.add_noise = add_noise
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_fraction = noise_fraction
        self.rng = rng or np.random.default_rng()
        self.root = Node(1.0)
        self.started = 0     # simulations whose descent has begun
        self.completed = 0   # simulations fully backed up
        self._root_requested = False

    @property
    def done(self) -> bool:
        return self.completed >= self.simulations

    def _select(self, node: Node):
        sqrt_n = math.sqrt(max(node.N, 1))
        best, best_score = None, -1e9
        for edge in node.edges:
            child = edge[2]
            q = -child.W / child.N if child.N else 0.0
            score = q + self.c_puct * child.prior * sqrt_n / (1 + child.N)
            if score > best_score:
                best, best_score = edge, score
        return best

    def request(self, max_leaves: int):
        """Descend until `max_leaves` leaves need the network (or the simulation budget is used).
        Terminal leaves are resolved on the spot."""
        if not self._root_requested:
            self._root_requested = True
            self.root.pending = True
            return [_make_request(self.root, [self.root], self.board, is_root=True)]
        if not self.root.expanded:
            return []
        requests = []
        while self.started < self.simulations and len(requests) < max_leaves:
            node, path, pushed = self.root, [self.root], 0
            node.N += VIRTUAL_LOSS
            node.W += VIRTUAL_LOSS
            while node.expanded and node.terminal_value is None:
                _, move, child = self._select(node)
                self.board.push(move)
                pushed += 1
                node = child
                path.append(node)
                node.N += VIRTUAL_LOSS
                node.W += VIRTUAL_LOSS

            if node.pending:                      # another descent already queued this leaf
                for n in path:
                    n.N -= VIRTUAL_LOSS
                    n.W -= VIRTUAL_LOSS
                for _ in range(pushed):
                    self.board.pop()
                break

            self.started += 1
            if not node.expanded:
                node.terminal_value = _terminal_value(self.board)
                if node.terminal_value is not None:
                    node.expanded = True
            if node.terminal_value is not None:
                self._backup(path, node.terminal_value)
            else:
                node.pending = True
                requests.append(_make_request(node, path, self.board))
            for _ in range(pushed):
                self.board.pop()
        return requests

    def fulfill(self, requests, logits, values):
        """logits: (len(requests), POLICY_SIZE) array; values: (len(requests),)."""
        for req, row, value in zip(requests, logits, values):
            legal = row[req.idxs]
            priors = np.exp(legal - legal.max())
            priors /= priors.sum()
            node = req.node
            node.edges = [(int(i), m, Node(float(p))) for i, m, p in zip(req.idxs, req.moves, priors)]
            node.expanded = True
            node.pending = False
            if req.is_root:
                if self.add_noise:
                    noise = self.rng.dirichlet([self.dirichlet_alpha] * len(node.edges))
                    for k, (_, _, child) in enumerate(node.edges):
                        child.prior = (1 - self.noise_fraction) * child.prior + self.noise_fraction * noise[k]
            else:
                self._backup(req.path, float(value))

    def _backup(self, path, value):
        """Replace each node's virtual loss with the real result (N already counted the visit)."""
        for n in reversed(path):
            n.W += value - VIRTUAL_LOSS
            value = -value
        self.completed += 1

    def visits(self) -> np.ndarray:
        out = np.zeros(POLICY_SIZE, dtype=np.float32)
        for i, _, child in self.root.edges:
            out[i] = child.N
        return out


def run_round(searches, device, max_leaves: int):
    """One network call per distinct net: gather leaves from every unfinished search, evaluate
    them together, hand the results back."""
    groups = {}
    for s in searches:
        if s.done:
            continue
        reqs = s.request(max_leaves)
        if reqs:
            groups.setdefault(id(s.net), (s.net, []))[1].append((s, reqs))
    for net, items in groups.values():
        planes = [r.planes for _, reqs in items for r in reqs]
        logits, values = evaluate_batch(net, planes, device)
        k = 0
        for s, reqs in items:
            s.fulfill(reqs, logits[k:k + len(reqs)], values[k:k + len(reqs)])
            k += len(reqs)


class MCTS:
    """Convenience wrapper: run one search to completion."""

    def __init__(self, net, device="cpu", simulations=50, c_puct=1.5, dirichlet_alpha=0.3,
                 noise_fraction=0.25, rng=None, leaves=8):
        self.net = net
        self.device = device
        self.simulations = simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_fraction = noise_fraction
        self.rng = rng or np.random.default_rng()
        self.leaves = leaves

    def new_search(self, board, add_noise=False) -> Search:
        return Search(self.net, board, self.simulations, self.c_puct, add_noise,
                      self.dirichlet_alpha, self.noise_fraction, self.rng)

    def search(self, board: chess.Board, add_noise: bool = False) -> np.ndarray:
        """Returns visit counts as a (POLICY_SIZE,) array. The board is restored before returning."""
        s = self.new_search(board, add_noise)
        while not s.done:
            run_round([s], self.device, self.leaves)
        return s.visits()
