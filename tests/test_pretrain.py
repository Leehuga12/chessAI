import csv
import io
import sys
import zipfile
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # pretrain.py lives at the repo root
import pretrain  # noqa: E402

from chessai.encoding import encode_board  # noqa: E402
from chessai.moves import index_to_move  # noqa: E402
from chessai.network import load_checkpoint  # noqa: E402
from chessai.pgn_data import load_split, prepare, unpack_planes  # noqa: E402


def random_game(rng, white_elo, black_elo, plies=40):
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers.update({"WhiteElo": str(white_elo), "BlackElo": str(black_elo)})
    node = game
    for _ in range(plies):
        if board.is_game_over():
            break
        move = rng.choice(list(board.legal_moves))
        node = node.add_variation(move)
        board.push(move)
    game.headers["Result"] = board.result() if board.is_game_over() else rng.choice(["1-0", "0-1", "1/2-1/2"])
    return game


def write_pgn(path, games):
    with open(path, "w", encoding="utf-8") as f:
        for g in games:
            print(g, file=f, end="\n\n")


@pytest.fixture(scope="module")
def pgn_file(tmp_path_factory):
    rng = np.random.default_rng(3)
    games = [random_game(rng, 2500, 2400) for _ in range(24)] + [random_game(rng, 1500, 2600) for _ in range(3)]
    games.append(random_game(rng, 2500, 2500, plies=6))           # too short
    path = tmp_path_factory.mktemp("pgn") / "games.pgn"
    write_pgn(path, games)
    return path, games


def test_prepare_filters_and_counts(pgn_file, tmp_path):
    path, _ = pgn_file
    stats = prepare([path], tmp_path, min_elo=2200, min_plies=16, val_every=5, log=lambda *_: None)
    assert stats["games_read"] == 28 and stats["games_used"] == 24
    assert stats["rejected"] == {"rating below minimum": 3, "too short": 1}
    assert stats["train_positions"] + stats["val_positions"] == sum(
        len(list(g.mainline_moves())) for g in pgn_file[1][:24])
    assert stats["val_positions"] > 0 and (tmp_path / "prepare_stats.json").exists()


def test_samples_match_the_game_exactly(pgn_file, tmp_path):
    path, games = pgn_file
    prepare([path], tmp_path, min_elo=2200, val_every=5, log=lambda *_: None)
    val = load_split(tmp_path, "val")
    first = games[0]                                              # the first accepted game is the first validation game
    n = len(list(first.mainline_moves()))
    planes = unpack_planes(val["packed"][:n])
    board = first.board()
    result = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}[first.headers["Result"]]
    for i, move in enumerate(first.mainline_moves()):
        assert np.array_equal(planes[i], encode_board(board))      # bit-packing round-trips
        assert index_to_move(int(val["moves"][i]), board) == move  # the policy target is the move played
        assert val["values"][i] == (result if board.turn == chess.WHITE else -result)
        board.push(move)


def test_zip_input_matches_plain_pgn(pgn_file, tmp_path):
    path, _ = pgn_file
    zpath = tmp_path / "games.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.write(path, "games.pgn")
    a = prepare([path], tmp_path / "a", log=lambda *_: None)
    b = prepare([zpath], tmp_path / "b", log=lambda *_: None)
    assert a["train_positions"] == b["train_positions"] and a["games_used"] == b["games_used"]


def run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["pretrain.py", *args])
    pretrain.main()


def test_pretrain_learns_logs_validates_and_resumes(pgn_file, tmp_path, monkeypatch):
    path, _ = pgn_file
    data = tmp_path / "data"
    prepare([path], data, min_elo=2200, val_every=5, log=lambda *_: None)
    out = tmp_path / "run"
    common = ["--blocks", "1", "--channels", "8", "--batch-size", "32", "--device", "cpu", "--log-every", "5",
              "--val-positions", "200", "--lr", "3e-3"]
    run(monkeypatch, str(data), "--out-dir", str(out), "--epochs", "2", *common)

    rows = list(csv.DictReader(open(out / "pretrain_log.csv", newline="")))
    train_rows = [r for r in rows if r["loss"]]
    assert float(train_rows[-1]["policy_loss"]) < float(train_rows[0]["policy_loss"])   # it is learning
    val_rows = [r for r in rows if r["val_top1"]]
    assert len(val_rows) == 2 and 0 <= float(val_rows[-1]["val_top1"]) <= float(val_rows[-1]["val_top3"]) <= 1

    net, extra = load_checkpoint(out / "pretrained.pt")
    assert extra["pretrained"] and extra["epoch"] == 2 and net.config == {"blocks": 1, "channels": 8}

    run(monkeypatch, str(data), "--resume", str(out), "--epochs", "3", *common)
    rows = list(csv.DictReader(open(out / "pretrain_log.csv", newline="")))
    assert [r["epoch"] for r in rows if r["val_top1"]] == ["1", "2", "3"]
    assert torch.load(out / "pretrain_state.pt")["epoch"] == 3
    with pytest.raises(SystemExit):                               # a fresh run must not overwrite an existing one
        run(monkeypatch, str(data), "--out-dir", str(out), "--epochs", "1", *common)


def test_lr_schedule_warms_up_then_decays():
    base, total = 1e-3, 1000
    assert pretrain.lr_at(0, total, base) < pretrain.lr_at(99, total, base) <= base
    assert pretrain.lr_at(500, total, base) < pretrain.lr_at(100, total, base)
    assert pretrain.lr_at(total, total, base) == pytest.approx(base * 0.05)


def test_reports_render_without_error(pgn_file, tmp_path, monkeypatch):
    import report_data
    import plot_pretrain
    path, _ = pgn_file
    info = report_data.draw_report(report_data.scan([path], 2200),
                                   report_data.sample_lengths([path], 2200, 20, np.random.default_rng(0)),
                                   "test", tmp_path / "data.png")
    assert info["games"] == 25 and (tmp_path / "data.png").stat().st_size > 20_000   # 24 + the 6-ply game: the report works from headers, prepare_data also drops short games

    data = tmp_path / "data"
    prepare([path], data, min_elo=2200, val_every=5, log=lambda *_: None)
    out = tmp_path / "run"
    run(monkeypatch, str(data), "--out-dir", str(out), "--epochs", "2", "--blocks", "1", "--channels", "8", "--batch-size", "32",
        "--device", "cpu", "--log-every", "5", "--val-positions", "200")
    train, val = plot_pretrain.read_log(out)
    plot_pretrain.draw(train, val, tmp_path / "train.png", baseline_value_mse=0.5)
    assert (tmp_path / "train.png").stat().st_size > 20_000
    plot_pretrain.draw(train, [], tmp_path / "early.png")        # a run with no finished epoch yet must still draw


def test_load_split_combines_folders_and_validation_sample_is_fixed(pgn_file, tmp_path):
    path, _ = pgn_file
    a = prepare([path], tmp_path / "a", min_elo=2200, val_every=5, max_games=10, log=lambda *_: None)
    b = prepare([path], tmp_path / "b", min_elo=2200, val_every=5, max_games=6, log=lambda *_: None)
    both = load_split([tmp_path / "a", tmp_path / "b"], "train")
    assert len(both["moves"]) == a["train_positions"] + b["train_positions"]
    assert both["packed"].shape == (len(both["moves"]), 144)
    one = load_split(tmp_path / "a", "train")                                # a single folder still works
    assert np.array_equal(both["packed"][:len(one["moves"])], one["packed"])
    s1, s2 = pretrain.validation_sample(1000, 100), pretrain.validation_sample(1000, 100)
    assert np.array_equal(s1, s2) and len(set(s1)) == 100 and s1.max() < 1000
    assert len(pretrain.validation_sample(50, 100)) == 50


def test_resume_picks_up_in_the_middle_of_an_epoch(pgn_file, tmp_path, monkeypatch):
    path, _ = pgn_file
    data = tmp_path / "data"
    prepare([path], data, min_elo=2200, val_every=5, log=lambda *_: None)
    out = tmp_path / "run"
    common = ["--blocks", "1", "--channels", "8", "--batch-size", "8", "--device", "cpu", "--log-every", "5",
              "--val-positions", "100", "--save-every", "10"]
    train_n = len(load_split(data, "train")["moves"])
    steps_per_epoch = train_n // 8
    assert steps_per_epoch > 60                                              # the test needs room to stop mid-epoch
    run(monkeypatch, str(data), "--out-dir", str(out), "--epochs", "1", "--max-steps", "35", *common)
    state = torch.load(out / "pretrain_state.pt")
    assert state["step"] == 35 and state["epoch"] == 0 and state["epoch_batches"] == 35     # stopped inside epoch 1
    _, extra = load_checkpoint(out / "pretrained.pt")
    assert 0 < extra["epoch"] < 1                                            # a fractional epoch in the checkpoint

    run(monkeypatch, str(data), "--resume", str(out), "--epochs", "1", *common)
    state = torch.load(out / "pretrain_state.pt")
    assert state["step"] == steps_per_epoch and state["epoch"] == 1 and state["epoch_batches"] == 0   # no batch lost or repeated
    rows = list(csv.DictReader(open(out / "pretrain_log.csv", newline="")))
    steps = [int(r["step"]) for r in rows if r["loss"]]
    assert steps == sorted(steps) and len(steps) == len(set(steps))          # the log continues, it does not restart
    assert [float(r["epoch"]) for r in rows if r["val_top1"]][-1] == 1.0


def test_consolidated_data_reads_from_disk_and_matches_the_shards(pgn_file, tmp_path):
    from chessai.pgn_data import consolidate
    path, _ = pgn_file
    prepare([path], tmp_path / "a", min_elo=2200, val_every=5, max_games=10, log=lambda *_: None)
    prepare([path], tmp_path / "b", min_elo=2200, val_every=5, max_games=6, log=lambda *_: None)
    meta = consolidate([tmp_path / "a", tmp_path / "b"], tmp_path / "all", log=lambda *_: None)
    for split in ("train", "val"):
        in_memory = load_split([tmp_path / "a", tmp_path / "b"], split)
        on_disk = load_split(tmp_path / "all", split)
        assert isinstance(on_disk["packed"], np.memmap) and not isinstance(in_memory["packed"], np.memmap)
        assert meta[split] == len(in_memory["moves"]) == len(on_disk["moves"])
        for key in ("packed", "moves", "values"):
            assert np.array_equal(np.asarray(on_disk[key]), in_memory[key])


def test_pretrain_runs_on_consolidated_data(pgn_file, tmp_path, monkeypatch):
    from chessai.pgn_data import consolidate
    path, _ = pgn_file
    prepare([path], tmp_path / "a", min_elo=2200, val_every=5, log=lambda *_: None)
    consolidate([tmp_path / "a"], tmp_path / "all", log=lambda *_: None)
    out = tmp_path / "run"
    run(monkeypatch, str(tmp_path / "all"), "--out-dir", str(out), "--epochs", "1", "--blocks", "1", "--channels", "8",
        "--batch-size", "16", "--device", "cpu", "--log-every", "5", "--val-positions", "100")
    net, extra = load_checkpoint(out / "pretrained.pt")
    assert extra["epoch"] == 1 and net.config == {"blocks": 1, "channels": 8}


def test_compare_nets_reports_score_and_interval(tmp_path, monkeypatch, capsys):
    import compare_nets
    from chessai.network import ChessNet, save_checkpoint
    torch.manual_seed(0)
    for name in ("a", "b"):
        save_checkpoint(ChessNet(1, 8), tmp_path / f"{name}.pt", {})
    monkeypatch.setattr(sys, "argv", ["compare_nets.py", str(tmp_path / "a.pt"), str(tmp_path / "b.pt"), "--games", "4",
                                      "--simulations", "2", "--parallel", "4", "--device", "cpu"])
    compare_nets.main()
    out = capsys.readouterr().out
    assert "4 games" in out and "95% interval" in out
    # the interval maths: 60 wins, 20 draws, 20 losses -> score 70%, clearly above 50%
    p, lo, hi = compare_nets.score_interval(60, 20, 20)
    assert p == pytest.approx(0.70) and lo > 0.5 > 0 and hi < 0.85
    assert compare_nets.score_interval(10, 0, 10)[1] < 0.5 < compare_nets.score_interval(10, 0, 10)[2]   # an even match includes 50%


def test_prepare_all_runs_files_in_parallel_and_skips_finished_ones(pgn_file, tmp_path, monkeypatch, capsys):
    import prepare_all
    path, games = pgn_file
    second = tmp_path / "second.pgn"
    from test_pretrain import write_pgn  # noqa: F401  (this module's own helper)
    write_pgn(second, games[:12])
    root = tmp_path / "months"
    argv = ["prepare_all.py", str(path), str(second), "--out-root", str(root), "--workers", "2"]
    monkeypatch.setattr(sys, "argv", argv)
    prepare_all.main()
    assert (root / "games" / "prepare_stats.json").exists() and (root / "second" / "prepare_stats.json").exists()
    assert len(load_split([root / "games", root / "second"], "train")["moves"]) > 0
    stamp = (root / "games" / "prepare_stats.json").stat().st_mtime_ns
    capsys.readouterr()
    prepare_all.main()
    assert "skip (already prepared)" in capsys.readouterr().out
    assert (root / "games" / "prepare_stats.json").stat().st_mtime_ns == stamp          # untouched the second time


def test_pretrain_can_start_from_an_existing_net(pgn_file, tmp_path, monkeypatch):
    from chessai.network import ChessNet, save_checkpoint
    path, _ = pgn_file
    prepare([path], tmp_path / "data", min_elo=2200, val_every=5, log=lambda *_: None)
    torch.manual_seed(5)
    start = ChessNet(2, 16)
    save_checkpoint(start, tmp_path / "start.pt", {})
    out = tmp_path / "run"
    run(monkeypatch, str(tmp_path / "data"), "--out-dir", str(out), "--epochs", "1", "--blocks", "1", "--channels", "8",   # size flags are ignored
        "--batch-size", "16", "--device", "cpu", "--log-every", "5", "--val-positions", "100", "--lr", "1e-6", "--init-from", str(tmp_path / "start.pt"))
    net, extra = load_checkpoint(out / "pretrained.pt")
    assert net.config == {"blocks": 2, "channels": 16}                                # it took the starting net's size
    drift = max((a - b).abs().max().item() for a, b in zip(net.parameters(), start.parameters()))     # weights only (batch-norm statistics always move)
    assert 0 < drift < 0.05                                                            # trained a little from those weights, not from scratch
