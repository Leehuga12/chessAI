"""Turn PGN game files into training positions for the supervised warm start.

Every position of every accepted game becomes one sample: the board (18 binary planes, bit-packed to
144 bytes), the index of the move the player chose (the policy target, same 4672 slots as the net),
and the game result from the mover's point of view (+1 / 0 / -1, the value target).

Reads plain .pgn, .zip (every .pgn inside) and, if the optional `zstandard` package is installed, .pgn.zst.
Games are kept only if both players have a rating at or above `min_elo`, so a file of mixed games can be filtered.
"""
import contextlib
import io
import json
import zipfile
from pathlib import Path

import chess
import chess.pgn
import numpy as np

from .encoding import encode_board
from .moves import move_to_index

RESULT_VALUE = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}
PACKED_BYTES = 18 * 8 * 8 // 8   # 144


@contextlib.contextmanager
def _text_streams(path):
    """Yields a list of text streams for one input file."""
    path = Path(path)
    name = path.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            members = [n for n in zf.namelist() if n.lower().endswith(".pgn")]
            with contextlib.ExitStack() as stack:
                yield [stack.enter_context(io.TextIOWrapper(zf.open(n), encoding="utf-8", errors="replace"))
                       for n in members]
    elif name.endswith(".zst"):
        try:
            import zstandard
        except ImportError:
            raise SystemExit(f"{path.name} is zstd-compressed: run  pip install zstandard  or unzip it first") from None
        with open(path, "rb") as raw:
            reader = zstandard.ZstdDecompressor().stream_reader(raw)
            yield [io.TextIOWrapper(reader, encoding="utf-8", errors="replace")]
    else:
        with open(path, encoding="utf-8", errors="replace") as f:
            yield [f]


def read_games(paths):
    """Yields chess.pgn.Game objects from every file, skipping games python-chess could not parse cleanly."""
    for path in paths:
        with _text_streams(path) as streams:
            for stream in streams:
                while True:
                    game = chess.pgn.read_game(stream)
                    if game is None:
                        break
                    if not game.errors:
                        yield game


def rejection_reason(game, min_elo, min_plies):
    """None if the game should be used, else a short reason."""
    h = game.headers
    if h.get("Variant", "Standard") not in ("Standard", "From Position") or "FEN" in h:
        return "not a standard game from the start position"
    if h.get("Result") not in RESULT_VALUE:
        return "no decisive or drawn result"
    try:
        if min(int(h["WhiteElo"]), int(h["BlackElo"])) < min_elo:
            return "rating below minimum"
    except (KeyError, ValueError):
        return "missing rating"
    if sum(1 for _ in game.mainline_moves()) < min_plies:
        return "too short"
    return None


def game_samples(game):
    """Yields (packed_planes uint8[144], move_index int, value int) for every position in the game."""
    result = RESULT_VALUE[game.headers["Result"]]
    board = game.board()
    for move in game.mainline_moves():
        packed = np.packbits(encode_board(board).astype(np.uint8).reshape(-1))
        z = result if board.turn == chess.WHITE else -result
        yield packed, move_to_index(move, board), z
        board.push(move)


class ShardWriter:
    """Collects samples and writes them as numbered .npz shards (uncompressed, so loading is fast)."""

    def __init__(self, out_dir, prefix, shard_size):
        self.out_dir, self.prefix, self.shard_size = Path(out_dir), prefix, shard_size
        self.packed, self.moves, self.values = [], [], []
        self.shards = 0
        self.total = 0

    def add(self, packed, move, value):
        self.packed.append(packed)
        self.moves.append(move)
        self.values.append(value)
        self.total += 1
        if len(self.moves) >= self.shard_size:
            self.flush()

    def flush(self):
        if not self.moves:
            return
        np.savez(self.out_dir / f"{self.prefix}_{self.shards:03d}.npz", packed=np.stack(self.packed),
                 moves=np.array(self.moves, dtype=np.int16), values=np.array(self.values, dtype=np.int8))
        self.shards += 1
        self.packed, self.moves, self.values = [], [], []


def prepare(paths, out_dir, min_elo=2200, min_plies=16, max_games=None, val_every=50, shard_size=200_000,
            log=print, progress_every=2000):
    """Reads `paths`, writes train_*.npz and val_*.npz shards plus prepare_stats.json into out_dir.

    Every `val_every`-th accepted game goes to the validation set (whole games, so no position leaks across)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    writers = {"train": ShardWriter(out, "train", shard_size), "val": ShardWriter(out, "val", shard_size)}
    stats = {"games_read": 0, "games_used": 0, "rejected": {}, "min_elo": min_elo, "min_plies": min_plies}
    for game in read_games(paths):
        stats["games_read"] += 1
        reason = rejection_reason(game, min_elo, min_plies)
        if reason:
            stats["rejected"][reason] = stats["rejected"].get(reason, 0) + 1
            continue
        writer = writers["val" if stats["games_used"] % val_every == 0 else "train"]
        stats["games_used"] += 1
        for packed, move, value in game_samples(game):
            writer.add(packed, move, value)
        if stats["games_read"] % progress_every == 0:
            log(f"read {stats['games_read']:,} games, used {stats['games_used']:,}, "
                f"{writers['train'].total + writers['val'].total:,} positions")
        if max_games and stats["games_used"] >= max_games:
            break
    for w in writers.values():
        w.flush()
    stats["train_positions"], stats["val_positions"] = writers["train"].total, writers["val"].total
    (out / "prepare_stats.json").write_text(json.dumps(stats, indent=1))
    return stats


def _shards(data_dirs, split):
    dirs = [data_dirs] if isinstance(data_dirs, (str, Path)) else list(data_dirs)
    files = [f for d in dirs for f in sorted(Path(d).glob(f"{split}_*.npz"))]
    if not files:
        raise FileNotFoundError(f"no {split}_*.npz shards in {', '.join(map(str, dirs))}; run prepare_data.py first")
    return files


def consolidate(data_dirs, out_dir, log=print):
    """Merge many prepared folders into one folder of flat files that can be read straight from disk (see load_split).

    Needed for datasets too large to hold in memory: on Windows the memory a program may reserve is capped (RAM plus page
    file), but a file mapped read-only is not charged to that cap, and the operating system keeps the hot parts in its cache."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {}
    for split in ("train", "val"):
        n = 0
        with open(out / f"{split}_packed.bin", "wb") as pf, open(out / f"{split}_moves.bin", "wb") as mf,                 open(out / f"{split}_values.bin", "wb") as vf:
            for f in _shards(data_dirs, split):
                with np.load(f) as z:
                    pf.write(np.ascontiguousarray(z["packed"], dtype=np.uint8).tobytes())
                    mf.write(np.ascontiguousarray(z["moves"], dtype=np.int16).tobytes())
                    vf.write(np.ascontiguousarray(z["values"], dtype=np.int8).tobytes())
                    n += len(z["moves"])
        meta[split] = n
        log(f"{split}: {n:,} positions")
    (out / "meta.json").write_text(json.dumps(meta))
    return meta


def load_split(data_dirs, split):
    """Returns dict(packed (N,144), moves (N,), values (N,)) for one or more prepared folders.

    A folder made by consolidate() is read straight from disk (packed stays on disk, memory-mapped; the small arrays are
    loaded). Otherwise every shard is loaded into memory, without ever holding two copies."""
    dirs = [data_dirs] if isinstance(data_dirs, (str, Path)) else list(data_dirs)
    if len(dirs) == 1 and (Path(dirs[0]) / "meta.json").exists():
        d = Path(dirs[0])
        n = json.loads((d / "meta.json").read_text())[split]
        return {"packed": np.memmap(d / f"{split}_packed.bin", dtype=np.uint8, mode="r", shape=(n, PACKED_BYTES)),
                "moves": np.fromfile(d / f"{split}_moves.bin", dtype=np.int16),
                "values": np.fromfile(d / f"{split}_values.bin", dtype=np.int8)}
    files = _shards(dirs, split)
    sizes = []
    for f in files:
        with np.load(f) as z:
            sizes.append(len(z["moves"]))
    total = sum(sizes)
    packed = np.empty((total, PACKED_BYTES), dtype=np.uint8)
    moves = np.empty(total, dtype=np.int16)
    values = np.empty(total, dtype=np.int8)
    pos = 0
    for f, n in zip(files, sizes):
        with np.load(f) as z:
            packed[pos:pos + n], moves[pos:pos + n], values[pos:pos + n] = z["packed"], z["moves"], z["values"]
        pos += n
    return {"packed": packed, "moves": moves, "values": values}


def unpack_planes(packed):
    """(B, 144) uint8 -> (B, 18, 8, 8) float32, exactly what encode_board returns."""
    return np.unpackbits(packed, axis=1).reshape(-1, 18, 8, 8).astype(np.float32)
