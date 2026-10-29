import csv
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))   # train.py lives at the repo root
import train  # noqa: E402

from chessai.network import ChessNet  # noqa: E402
from chessai.selfplay import ReplayBuffer, Sample  # noqa: E402
from chessai.training import load_buffer, load_state, save_buffer, save_state  # noqa: E402

TINY = ["--games", "2", "--simulations", "2", "--train-steps", "2", "--batch-size", "8", "--blocks", "1",
        "--channels", "8", "--arena-games", "2", "--buffer-every", "1"]


def run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["train.py", *TINY, *args])
    train.main()


def rows(path):
    with open(path / "train_log.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_resume_continues_log_counter_and_state(monkeypatch, tmp_path):
    out = tmp_path / "r"
    run(monkeypatch, "--iterations", "2", "--out-dir", str(out))
    first = rows(out)
    assert [r["iteration"] for r in first] == ["1", "2"]
    assert (out / "state.pt").exists() and (out / "buffer.pkl").exists()

    run(monkeypatch, "--iterations", "4", "--resume", str(out))
    resumed = rows(out)
    assert [r["iteration"] for r in resumed] == ["1", "2", "3", "4"]
    assert resumed[:2] == first                                   # earlier rows untouched, header not repeated
    assert float(resumed[2]["seconds"]) >= float(resumed[1]["seconds"])
    assert int(resumed[2]["samples"]) > int(resumed[1]["samples"])   # the buffer was restored, not emptied
    assert torch.load(out / "state.pt")["iteration"] == 4


def test_resume_when_already_done_changes_nothing(monkeypatch, tmp_path, capsys):
    out = tmp_path / "r"
    run(monkeypatch, "--iterations", "1", "--out-dir", str(out))
    before = (out / "train_log.csv").read_text()
    run(monkeypatch, "--iterations", "1", "--resume", str(out))
    assert "already at iteration 1" in capsys.readouterr().out
    assert (out / "train_log.csv").read_text() == before


def test_fresh_run_refuses_to_overwrite_and_resume_needs_a_save(monkeypatch, tmp_path):
    out = tmp_path / "r"
    run(monkeypatch, "--iterations", "1", "--out-dir", str(out))
    with pytest.raises(SystemExit):
        run(monkeypatch, "--iterations", "2", "--out-dir", str(out))
    with pytest.raises(SystemExit):
        run(monkeypatch, "--iterations", "2", "--resume", str(tmp_path / "nothing"))


def test_state_round_trip_keeps_weights_and_optimizer(tmp_path):
    net = ChessNet(1, 8).eval()
    best = ChessNet(1, 8).eval()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    net(torch.zeros(2, 18, 8, 8))[0].sum().backward()
    opt.step()
    save_state(tmp_path, net, best, opt, 7, 12.5)
    net2, best2, opt2, it, elapsed = load_state(tmp_path, "cpu", 5e-4)
    assert (it, elapsed) == (7, 12.5)
    assert all(torch.equal(a, b) for a, b in zip(net.state_dict().values(), net2.state_dict().values()))
    assert all(torch.equal(a, b) for a, b in zip(best.state_dict().values(), best2.state_dict().values()))
    assert opt2.param_groups[0]["lr"] == 5e-4
    assert len(opt2.state) == len(opt.state) > 0


def test_buffer_round_trip_and_missing_file(tmp_path):
    import numpy as np
    buf = ReplayBuffer(10)
    buf.add([Sample(np.zeros((18, 8, 8), np.uint8), np.array([1, 2], np.int16), np.array([.5, .5], np.float32), 1.0)])
    save_buffer(tmp_path, buf, 3)
    loaded, it = load_buffer(tmp_path, 10)
    assert it == 3 and len(loaded) == 1 and loaded.data[0].value == 1.0
    empty, it = load_buffer(tmp_path / "none", 10)
    assert it is None and len(empty) == 0
