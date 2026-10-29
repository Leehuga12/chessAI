"""Engine behind the local web UI (serve.py).

`run()` owns the networks and must be called on the main thread: on Windows (torch 2.14 + CUDA 13) the
first CUDA convolution crashes the process with an access violation when it runs on a worker thread.
It either plays the net against itself one move at a time
(watch mode, optionally learning from the finished games) or plays against a human (play mode).
The HTTP side only reads `state()` and queues commands, so the page never blocks on the GPU.
"""
import copy
import csv
import io
import json
import queue
import re
import threading
import time
import traceback
from pathlib import Path

import chess
import numpy as np
import torch

from .arena import play_match
from .encoding import encode_board
from .mcts import Search, run_round
from .moves import index_to_move
from .network import ChessNet, load_checkpoint, save_checkpoint
from .openings import Openings
from .parallel import MAX_PLIES, TEMPERATURE_PLIES, pick_move
from .rules import terminal_value
from .selfplay import ReplayBuffer, _to_samples
from .training import train_steps

LEAVES = 8
MISTAKE_DROP = 0.3   # a move is a "mistake" if the mover's value estimate falls by more than this
CANDIDATES = 6
RECENT_GAMES = 40    # results kept for the form strip
SUMMARY_HOLD = 6.0   # seconds a finished Watch-mode game waits for the summary to be read, unless dismissed
RESULT_TEXT = {1: "1-0", -1: "0-1", 0: "1/2-1/2"}
GROUP_PRETRAINED, GROUP_SELFPLAY, GROUP_BOTH = "Pretrained on strong games", "Self-play", "Pretrained, then improved by self-play"
NET_GROUPS = (GROUP_PRETRAINED, GROUP_BOTH, GROUP_SELFPLAY)

DEFAULTS = dict(mode="selfplay", speed=1, learning=False, paused=False, summary=True, opening="none", human_color="white", simulations=50,
                train_steps=20, batch_size=64, lr=1e-3, arena_every=10, arena_games=6, gate=0.55)
# name -> (cast, low, high) for numbers, or a tuple of allowed values
LIMITS = dict(simulations=(int, 1, 2000), train_steps=(int, 1, 5000), batch_size=(int, 8, 4096),
              lr=(float, 1e-6, 1.0), arena_every=(int, 1, 1000), arena_games=(int, 2, 200),
              gate=(float, 0.0, 1.0), speed=(1, 4, 16, 64), mode=("selfplay", "play"),
              human_color=("white", "black"))
CURVE_FIELDS = ["iteration", "game", "loss", "policy_loss", "value_loss", "arena_score", "promoted"]


def validate(changes: dict) -> dict:
    """Keep only known settings, coerced and range-checked. Raises ValueError on a bad value."""
    out = {}
    for key, value in changes.items():
        if key in ("learning", "paused", "summary"):
            out[key] = bool(value)
        elif key == "opening":   # "none", "random" or an opening id; set_settings checks the id exists
            out[key] = str(value)[:200]
        elif key in LIMITS:
            spec = LIMITS[key]
            if isinstance(spec[0], type):
                cast, lo, hi = spec
                v = cast(value)
                if not lo <= v <= hi:
                    raise ValueError(f"{key} must be between {lo} and {hi}")
                out[key] = v
            else:
                if value not in spec:
                    raise ValueError(f"{key} must be one of {spec}")
                out[key] = value
    return out


class Live:
    def __init__(self, out_dir="runs/ui", blocks=6, channels=64, device=None, simulations=50, seed=0, init_from=None,
                 runs_root=None, start_net=None, openings_dir=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.runs_root = Path(runs_root) if runs_root else self.out.parent   # nets can only be loaded from under here
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.lock = threading.RLock()
        self.cmds = queue.Queue()
        self.t0 = time.time()
        self.session = int(self.t0)
        self.openings = Openings(openings_dir) if openings_dir else Openings()
        self._records = {}     # win/draw/loss record per net
        self._net_cache = {}   # checkpoint peeks for the net picker
        self._nets = []        # last scan of the runs folder, served to the page
        self._scan_asked = 0.0

        self.settings = dict(DEFAULTS, simulations=simulations)
        self.status = "starting"
        self.games = []
        self.game_id = 0
        best_path = self.out / "best.pt"
        if best_path.exists():
            best, extra = load_checkpoint(best_path, self.device)
            promotions, source = int(extra.get("promotions", 0)), best_path
        elif init_from:   # a fresh UI run that starts from a pretrained net
            best, _ = load_checkpoint(init_from, self.device)
            promotions, source = 0, Path(init_from)
        else:
            best, promotions, source = ChessNet(blocks, channels).to(self.device), 0, best_path
        self._install(best, promotions, self.out, source)
        self._load_record()
        self._nets = self.list_nets()
        target = self._start_target(start_net)
        if target:
            self._switch_net(self._resolve(target))   # also starts the first game
        else:
            self.new_game()

    @staticmethod
    def _label(path):
        """Path as shown and sent to the page: relative to the working folder when possible."""
        p = Path(path).resolve()
        try:
            return p.relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            return p.as_posix()

    def _install(self, best, promotions, save_dir, source):
        """Make `best` the playing net. Training progress for it is saved in save_dir, never over the source file."""
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best = best.eval()
        self.promotions = promotions
        best_path = self.save_dir / "best.pt"
        if not best_path.exists():   # save the starting net too, so there is always a checkpoint to evaluate
            save_checkpoint(self.best, best_path, {"promotions": promotions, "iteration": 0})
        latest_path = self.save_dir / "latest.pt"
        if latest_path.exists():     # the net being trained, saved after every round
            self.net, _ = load_checkpoint(latest_path, self.device)
            self.net.eval()
        else:
            self.net = copy.deepcopy(self.best).eval()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=DEFAULTS["lr"], weight_decay=1e-4)
        self.buffer = ReplayBuffer(100_000)
        self.params = sum(p.numel() for p in self.net.parameters())
        self.net_key = self._label(self.save_dir)     # identifies the record and learning curve
        self.source_label = self._label(source)
        src = Path(source)
        self.net_name = f"{src.parent.name}/{src.name}"
        self.curve = self._load_curve()
        self.games_since_arena = 0
        self.last_arena = self.curve_last_arena()

    # ---- game state -------------------------------------------------------------------------
    def new_game(self):
        with self.lock:
            self.game_id += 1
            self.game_mode = self.settings["mode"]
            self.human = chess.WHITE if self.settings["human_color"] == "white" else chess.BLACK
            self.board = chess.Board()
            self.moves = []
            self.history = []
            self.result = None
            self.reason = None
            self.ack = False        # summary dismissed
            self.over_at = None     # when the finished game was fully processed
            self.candidates = []
            self.eval_white = None
            self.last_q = None
            self.last_move = None
            self.mistakes = {"white": 0, "black": 0}
            self.opening = None
            self._play_book()
            self.status = "thinking" if self.game_mode == "selfplay" else "your move"

    def _play_book(self):
        """Play the chosen opening line (or a random one) before the game proper, as moves marked 'book'."""
        choice = self.settings["opening"]
        if choice == "none":
            return
        entry = self.openings.random_entry(self.rng) if choice == "random" else self.openings.lookup(choice)
        if not entry:
            return
        board = self.board
        for san in entry["san"]:
            move = board.parse_san(san)
            e = dict(ply=board.ply() + 1, side="white" if board.turn else "black", san=board.san(move), uci=move.uci(),
                     by="book", e=None, q=None, share=None, drop=None)
            board.push(move)
            e["fen"] = board.fen()
            e["op"] = self._opening_name(board)
            self.moves.append(e)
            self.last_move = move.uci()
        self.opening = {"eco": entry["eco"], "name": entry["name"]}

    def _opening_name(self, board):
        info = self.openings.name_for(board)
        return {"eco": info["eco"], "name": info["name"]} if info else None

    def _think(self, board, noise):
        s = Search(self.best, board.copy(), self.settings["simulations"], add_noise=noise, rng=self.rng)
        while not s.done:
            run_round([s], self.device, LEAVES)
        edges = s.root.edges
        total = max(sum(c.N for _, _, c in edges), 1)
        value = -sum(c.W for _, _, c in edges) / total   # from the side to move
        rows = sorted(((c.N, m, c) for _, m, c in edges), key=lambda r: -r[0])[:CANDIDATES]
        cands = [dict(san=board.san(m), uci=m.uci(), share=n / total,
                      q=(-c.W / c.N if c.N else 0.0), prior=c.prior) for n, m, c in rows]
        return s.visits(), cands, float(value)

    def _apply(self, move, searched=None):
        """Play `move`. searched = (visits, candidates, value) when the AI chose it."""
        with self.lock:
            board = self.board
            mover = board.turn
            before = board.copy()
            entry = dict(ply=board.ply() + 1, side="white" if mover else "black", san=board.san(move),
                         uci=move.uci(), by="ai" if searched else "human", e=None, q=None,
                         share=None, drop=None)
            if searched:
                visits, cands, value = searched
                entry["e"] = value
                chosen = next((c for c in cands if c["uci"] == move.uci()), None)
                if chosen:
                    entry.update(q=chosen["q"], share=chosen["share"])
                self.candidates = cands
                self.eval_white = value if mover == chess.WHITE else -value
                nz = np.nonzero(visits)[0]
                self.history.append((encode_board(before).astype(np.uint8), nz.astype(np.int16),
                                     (visits[nz] / visits[nz].sum()).astype(np.float32), mover))
            elif self.last_q is not None:
                entry["e"] = -self.last_q   # value the AI's reply gave us, from the human's side
            prev = self.moves[-1] if self.moves else None
            if prev and prev["e"] is not None and entry["e"] is not None:
                prev["drop"] = prev["e"] + entry["e"]
                if prev["drop"] > MISTAKE_DROP:
                    self.mistakes[prev["side"]] += 1
            self.last_q = entry["q"]
            self.moves.append(entry)
            board.push(move)
            entry["fen"] = board.fen()   # position after this move, so the page can replay moves one by one
            entry["op"] = self._opening_name(board)   # the opening this position is named, if any
            if entry["op"]:
                self.opening = entry["op"]
            self.last_move = move.uci()
            v = terminal_value(board)
            if v is not None or board.ply() >= MAX_PLIES:
                mated_white = board.turn == chess.WHITE
                result = (-1 if mated_white else 1) if v == -1.0 else 0
            else:
                return
        self._finish(result)

    def _finish(self, result):
        with self.lock:
            self.result = RESULT_TEXT[result]
            outcome = self.board.outcome(claim_draw=True)
            self.reason = outcome.termination.name.replace("_", " ").lower() if outcome else "move limit reached"
            game = dict(id=self.game_id, session=self.session, mode=self.game_mode, result=self.result,
                        plies=len(self.moves), mistakes=dict(self.mistakes),
                        human="white" if self.human == chess.WHITE else "black", net=self.net_key, opening=self.opening, moves=self.moves)
            self.games.append(game)
            with open(self.out / "games.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(game) + "\n")
            self._tally(self.game_mode, self.result, game["human"])
            learn = self.game_mode == "selfplay" and self.settings["learning"]
            self.status = "game over"
        if learn:
            self.buffer.add(_to_samples(result, self.history))
            self._learn()
        with self.lock:
            self.over_at = time.time()

    # ---- learning -----------------------------------------------------------------------------
    def _load_curve(self):
        path = self.save_dir / "train_log.csv"
        if not path.exists():
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, CURVE_FIELDS).writeheader()
            return []
        with open(path, newline="") as f:
            return [{k: (float(v) if v != "" else None) for k, v in row.items()} for row in csv.DictReader(f)]

    def curve_last_arena(self):
        scored = [r["arena_score"] for r in self.curve if r.get("arena_score") is not None]
        return scored[-1] if scored else None

    def _learn(self):
        s = self.settings
        if len(self.buffer) < s["batch_size"]:
            return
        with self.lock:
            self.status = "training"
        for group in self.optimizer.param_groups:
            group["lr"] = s["lr"]
        loss, pl, vl = (float(x) for x in train_steps(self.net, self.buffer, self.optimizer,
                                                      s["train_steps"], s["batch_size"], self.device))
        save_checkpoint(self.net, self.save_dir / "latest.pt", {"iteration": len(self.curve) + 1})
        row = dict(iteration=len(self.curve) + 1, game=self.game_id, loss=loss, policy_loss=pl,
                   value_loss=vl, arena_score=None, promoted=None)
        self.games_since_arena += 1
        if self.games_since_arena >= s["arena_every"]:
            with self.lock:
                self.status = "gating (new net vs best)"
            score, _ = play_match(self.net, self.best, self.device, s["arena_games"], s["simulations"],
                                  seed=row["iteration"])
            promoted = score >= s["gate"]
            if promoted:
                self.best = copy.deepcopy(self.net).eval()
                self.promotions += 1
                save_checkpoint(self.best, self.save_dir / "best.pt",
                                {"promotions": self.promotions, "iteration": row["iteration"]})
            row.update(arena_score=score, promoted=int(promoted))
            self.last_arena = score
            self.games_since_arena = 0
        with self.lock:
            self.curve.append(row)
            with open(self.save_dir / "train_log.csv", "a", newline="") as f:
                csv.DictWriter(f, CURVE_FIELDS).writerow({k: ("" if v is None else v) for k, v in row.items()})
            self.status = "game over"

    # ---- commands -----------------------------------------------------------------------------
    def set_settings(self, changes):
        clean = validate(changes)
        with self.lock:
            if clean.get("opening", "none") not in ("none", "random") and not self.openings.lookup(clean["opening"]):
                raise ValueError("unknown opening")
            restart = any(k in clean and clean[k] != self.settings[k] for k in ("mode", "human_color", "opening"))
            self.settings.update(clean)
        if restart:
            self.cmds.put({"type": "new_game"})
        return clean

    def human_move(self, uci):
        with self.lock:
            if self.game_mode != "play" or self.result or self.board.turn != self.human:
                raise ValueError("not your turn")
            move = chess.Move.from_uci(uci)
            piece = self.board.piece_at(move.from_square)
            if piece and piece.piece_type == chess.PAWN and chess.square_rank(move.to_square) in (0, 7) \
                    and not move.promotion:
                move.promotion = chess.QUEEN   # always queens; under-promotion is not offered
            if move not in self.board.legal_moves:
                raise ValueError("illegal move")
        self.cmds.put({"type": "move", "move": move})

    def _handle(self, cmd):
        if cmd["type"] == "new_game":
            self.new_game()
        elif cmd["type"] == "move":
            with self.lock:
                ok = self.game_mode == "play" and not self.result and self.board.turn == self.human
            if ok:
                self._apply(cmd["move"])
        elif cmd["type"] == "scan_nets":
            nets = self.list_nets()
            with self.lock:
                self._nets = nets
        elif cmd["type"] == "load_net":
            try:
                self._switch_net(cmd["path"])
            except Exception:
                traceback.print_exc()
                with self.lock:
                    self.status = "could not load that net (see server console)"

    # ---- win / draw / loss record ---------------------------------------------------------------
    def _rec(self, key):
        return self._records.setdefault(key, {"counts": {"play": {"win": 0, "draw": 0, "loss": 0},
                                                         "selfplay": {"white": 0, "draw": 0, "black": 0}},
                                              "recent": {"play": [], "selfplay": []}})

    @property
    def counts(self):
        return self._rec(self.net_key)["counts"]

    @property
    def recent(self):
        return self._rec(self.net_key)["recent"]

    def _tally(self, mode, result, human, key=None):
        """Count a finished game. Play mode is from the human's side; Watch mode is White / Black."""
        if mode == "play":
            outcome = "draw" if result == "1/2-1/2" else ("win" if (result == "1-0") == (human == "white") else "loss")
        else:
            outcome = {"1-0": "white", "0-1": "black"}.get(result, "draw")
        rec = self._rec(key or self.net_key)
        rec["counts"][mode][outcome] += 1
        rec["recent"][mode] = (rec["recent"][mode] + [outcome])[-RECENT_GAMES:]

    def _load_record(self):
        """Rebuild the records from every game saved in games.jsonl (games from before nets were tracked belong to this folder's net)."""
        path = self.out / "games.jsonl"
        if not path.exists():
            return
        default_key = self._label(self.out)
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    g = json.loads(line)
                    self._tally(g["mode"], g["result"], g.get("human", "white"), g.get("net", default_key))
                except (ValueError, KeyError):
                    continue   # a damaged line should not stop the page from starting

    # ---- net picker -----------------------------------------------------------------------------
    def _classify(self, p, extra):
        """(group, text) for a net worth offering in the picker, or None to hide the file.
        Offered: pretrained nets, and the best net of each self-play line. Hidden: training copies (latest.pt),
        numbered checkpoints, evaluation snapshots, untrained nets, and a branch's best.pt until self-play has improved it."""
        untrained = not extra.get("promotions") and not extra.get("iteration")
        if p.name == "pretrained.pt" and extra.get("pretrained"):
            return GROUP_PRETRAINED, f"{extra.get('epoch')} epochs, {100 * extra.get('val_top1', 0):.0f}% of moves match strong players"
        if p.name != "best.pt":
            return None
        n = extra.get("promotions", 0)
        if "branches" in p.parts:   # a picked net trained further by self-play: only interesting once it improved
            return None if untrained else (GROUP_BOTH, f"{n} improvements")
        if untrained and p.parent.resolve() != self.out.resolve():
            return None
        return GROUP_SELFPLAY, ("untrained start" if untrained else f"{n} improvements over its start")

    def _peek(self, p):
        """Describe one checkpoint file, or None if it is not a net worth offering (cached until the file changes)."""
        st = p.stat()
        key = (str(p), st.st_mtime_ns, st.st_size)
        if key not in self._net_cache:
            info = None
            try:
                data = torch.load(p, map_location="cpu")
                if isinstance(data, dict) and "config" in data and "state" in data:
                    shown = self._classify(p, data.get("extra", {}))
                    if shown:
                        c = data["config"]
                        info = dict(path=self._label(p), mtime=st.st_mtime, group=shown[0],
                                    label=f"{shown[1]}  ({p.parent.name}, {c['blocks']}x{c['channels']})")
            except Exception:
                info = None
            self._net_cache[key] = info
        return self._net_cache[key]

    def nets_listing(self):
        """For the web thread: the last scan, plus a request for a fresh one. Never touches torch itself,
        because on this setup any torch call off the main thread crashes the process."""
        with self.lock:
            listing = list(self._nets)
            stale = time.time() - self._scan_asked > 3
            if stale:
                self._scan_asked = time.time()
        if stale:
            self.cmds.put({"type": "scan_nets"})
        return listing

    def list_nets(self):
        """The nets worth offering under the runs folder, grouped, newest first within a group.
        Main thread only (it reads checkpoints with torch)."""
        found = []
        for p in self.runs_root.rglob("*.pt"):
            if p.name == "state.pt" or p.name.endswith("_state.pt"):
                continue
            info = self._peek(p)
            if info:
                found.append(info)
        order = {g: i for i, g in enumerate(NET_GROUPS)}
        return sorted(found, key=lambda d: (order[d["group"]], -d["mtime"]))

    def _resolve(self, path):
        """Absolute path of a net file; raises ValueError unless it is a .pt file under the runs folder."""
        p = Path(path)
        p = (p if p.is_absolute() else Path.cwd() / p).resolve()
        if p.suffix != ".pt" or not p.is_file() or self.runs_root.resolve() not in p.parents:
            raise ValueError("not a net checkpoint inside the runs folder")
        return p

    def _start_target(self, start_net):
        """Which net the UI opens on: None keeps this folder's own net, 'auto' picks the newest pretrained net."""
        if not start_net or start_net == "selfplay":
            return None
        if start_net == "auto":
            pretrained = [n for n in self._nets if n["group"] == GROUP_PRETRAINED]
            return pretrained[0]["path"] if pretrained else None
        return start_net

    def request_net(self, path):
        """Check the path (called from the web thread, so no torch here) and queue a switch.
        The worker opens the file; a file that is not a net shows up as an error status there."""
        p = self._resolve(path)
        self.cmds.put({"type": "load_net", "path": p})
        return self._label(p)

    def _switch_net(self, p):
        """Play (and train) a different net. Its progress is saved in its own folder, so no file of any other net is overwritten."""
        with self.lock:
            self.status = "loading net"
        out, branches = self.out.resolve(), (self.out / "branches").resolve()
        parent = p.parent.resolve()
        if parent == out or parent == branches or branches in parent.parents:
            save_dir, use = parent, p                    # one of this UI's own nets: use it where it is
        else:
            slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{parent.name}_{p.stem}")
            save_dir = self.out / "branches" / slug
            resume = save_dir / "best.pt"                # picked this net before: carry on from that branch
            use = resume if resume.exists() else p
        best, extra = load_checkpoint(use, self.device)
        self._install(best, int(extra.get("promotions", 0)), save_dir, p)
        self.new_game()
        self.cmds.put({"type": "scan_nets"})   # the new branch folder now holds checkpoints

    def _summary_hold(self):
        """Seconds left that a finished game should stay on screen for its summary (0 if none)."""
        with self.lock:
            if not (self.settings["summary"] and self.result and not self.ack and self.over_at):
                return 0.0
            return max(0.0, SUMMARY_HOLD - (time.time() - self.over_at))

    def ack_summary(self):
        with self.lock:
            self.ack = True

    # ---- worker -------------------------------------------------------------------------------
    def _step(self):
        """Advance one move if there is something to do. Returns seconds to wait afterwards."""
        with self.lock:
            mode, result, turn, human = self.game_mode, self.result, self.board.turn, self.human
            speed, learning = self.settings["speed"], self.settings["learning"]
            paused = self.settings["paused"]
        if mode == "selfplay":
            if paused:
                return 0.1
            delay = 1.0 / speed
            if result:
                if self._summary_hold() > 0:
                    return 0.1
                self.new_game()
                return delay
            t = time.time()
            searched = self._think(self.board, noise=learning)
            visits = searched[0]
            idx = pick_move(visits, self.board.ply(), self.rng, TEMPERATURE_PLIES)
            self._apply(index_to_move(idx, self.board), searched)
            return max(0.0, delay - (time.time() - t))
        if result or turn == human:
            return 0.1
        with self.lock:
            self.status = "thinking"
        searched = self._think(self.board, noise=False)
        self._apply(index_to_move(int(searched[0].argmax()), self.board), searched)
        with self.lock:
            if not self.result:
                self.status = "your move"
        return 0.0

    def run(self):
        wait = 0.0
        while True:
            try:
                self._handle(self.cmds.get(timeout=wait))
                wait = 0.0
                continue
            except queue.Empty:
                pass
            try:
                wait = self._step()
            except Exception:
                traceback.print_exc()
                with self.lock:
                    self.status = "error (see server console)"
                wait = 2.0

    # ---- read side ----------------------------------------------------------------------------
    def state(self, stockfish_csv="results/results.csv"):
        with self.lock:
            board, s = self.board, self.settings
            legal = {}
            if self.game_mode == "play" and not self.result and board.turn == self.human:
                for m in board.legal_moves:
                    legal.setdefault(chess.square_name(m.from_square), []).append(chess.square_name(m.to_square))
            if self.game_mode == "play":
                c = self.counts["play"]
                games = sum(c.values())
                win_pct, win_label = (100 * c["loss"] / games if games else None), "AI vs you"   # your losses are the AI's wins
            else:
                win_pct = None if self.last_arena is None else 100 * self.last_arena
                win_label = "arena"
            elapsed = int(time.time() - self.t0)
            return dict(
                settings=dict(s), status="paused" if s["paused"] and self.game_mode == "selfplay" else self.status,
                game=dict(id=self.game_id, mode=self.game_mode, fen=board.fen(), last=self.last_move,
                          turn="white" if board.turn else "black", ply=board.ply(), result=self.result, reason=self.reason, opening=self.opening,
                          hold_left=(self._summary_hold() if self.game_mode == "selfplay" else 0.0),
                          moves=list(self.moves), mistakes=dict(self.mistakes),
                          human="white" if self.human == chess.WHITE else "black"),
                candidates=list(self.candidates), eval_white=self.eval_white, legal=legal,
                stats=dict(win_pct=win_pct, win_label=win_label, mistakes=dict(self.mistakes),
                           moves=len(self.moves), runtime=f"{elapsed // 3600:02d}:{elapsed % 3600 // 60:02d}:{elapsed % 60:02d}"),
                info=dict(device=self.device, blocks=self.net.config["blocks"], channels=self.net.config["channels"],
                          params=self.params, buffer=len(self.buffer), leaves=LEAVES, promotions=self.promotions,
                          games_played=len(self.games), mistake_drop=MISTAKE_DROP),
                record=dict(counts={k: dict(v) for k, v in self.counts.items()}, recent={k: list(v) for k, v in self.recent.items()}),
                net=dict(name=self.net_name, source=self.source_label, key=self.net_key),
                curve=list(self.curve[-500:]),
                games=[{k: g[k] for k in ("id", "mode", "result", "plies", "mistakes")} for g in self.games],
                stockfish=read_stockfish(stockfish_csv),
            )

    def game_moves(self, game_id):
        with self.lock:
            if game_id == self.game_id:
                return list(self.moves)
            for g in self.games:
                if g["id"] == game_id:
                    return list(g["moves"])
        return None

    def games_json(self):
        with self.lock:
            cur = dict(id=self.game_id, session=self.session, mode=self.game_mode, result=self.result,
                       plies=len(self.moves), mistakes=dict(self.mistakes), moves=list(self.moves))
            return json.dumps(self.games + ([cur] if self.moves and not self.result else []), indent=1)

    def games_csv(self):
        cols = ["game", "mode", "result", "ply", "side", "by", "move", "uci", "value_before", "value_after_choice",
                "visit_share", "value_drop"]
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(cols)
        with self.lock:
            all_games = self.games + ([dict(id=self.game_id, mode=self.game_mode, result=None, moves=self.moves)]
                                      if self.moves and not self.result else [])
            for g in all_games:
                for m in g["moves"]:
                    w.writerow([g["id"], g["mode"], g["result"] or "", m["ply"], m["side"], m["by"], m["san"], m["uci"],
                                m["e"], m["q"], m["share"], m["drop"]])
        return buf.getvalue()


def read_stockfish(path):
    """Latest row per (opponent, checkpoint) from the evaluate.py results file, if there is one."""
    p = Path(path)
    if not p.exists():
        return []
    latest = {}
    with open(p, newline="") as f:
        for row in csv.DictReader(f):
            latest[(row["opponent"], row["checkpoint"])] = dict(
                opponent=row["opponent"], checkpoint=f"{Path(row['checkpoint'].replace(chr(92), '/')).parent.name}/{Path(row['checkpoint'].replace(chr(92), '/')).stem}", iteration=row["iteration"],
                score=row["score"], est_elo=row["est_elo"], margin=row["elo_margin"], games=row["games"])
    return list(latest.values())
