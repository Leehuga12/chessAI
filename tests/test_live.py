import chess
import numpy as np
import pytest
import torch

from chessai.live import Live, validate


@pytest.fixture
def live(tmp_path):
    return Live(tmp_path, blocks=1, channels=8, device="cpu", simulations=4)


def test_validate_rejects_bad_values_and_ignores_unknown():
    assert validate({"speed": 16, "learning": 0, "nonsense": 1}) == {"speed": 16, "learning": False}
    for bad in ({"speed": 3}, {"simulations": 0}, {"mode": "x"}, {"gate": 2}):
        with pytest.raises(ValueError):
            validate(bad)


def test_selfplay_game_is_logged_and_learns(live):
    live.set_settings({"learning": True, "batch_size": 8, "train_steps": 2, "arena_every": 1, "arena_games": 2, "simulations": 2, "summary": False})
    live._step()
    assert live.moves and live.candidates and live.eval_white is not None
    assert live.moves[-1]["fen"] == live.board.fen()   # the page replays moves from these
    for _ in range(400):
        if live.result:
            break
        live._step()
    assert live.result and len(live.games) == 1
    assert live.curve and live.curve[0]["arena_score"] is not None   # gated after the first game
    assert (live.out / "games.jsonl").read_text().count("\n") == 1
    assert live.games_csv().count("\n") == len(live.games[0]["moves"]) + 1
    live._step()                                                    # next step starts a fresh game
    assert live.game_id == 2 and not live.moves


def test_play_mode_human_then_ai_reply_and_illegal_move(live):
    live.set_settings({"mode": "play", "human_color": "white"})
    live._handle(live.cmds.get_nowait())                            # the new_game queued by the mode change
    assert live.game_mode == "play" and live.board.turn == chess.WHITE
    assert live._step() == 0.1                                      # waits for the human
    with pytest.raises(ValueError):
        live.human_move("e2e5")
    live.human_move("e2e4")
    live._handle(live.cmds.get_nowait())
    assert live.moves[0]["by"] == "human" and live.board.turn == chess.BLACK
    live._step()                                                    # AI replies
    assert live.moves[1]["by"] == "ai" and live.board.turn == chess.WHITE
    with pytest.raises(ValueError):
        live.human_move("a7a6")                                      # black's pawn, white to move


def test_mistake_counted_when_value_drops(live):
    live.new_game()
    def searched(value):
        visits = np.zeros(4672, dtype=np.float32); visits[0] = 1
        return visits, [], value
    live._apply(chess.Move.from_uci("e2e4"), searched(0.5))   # white thought it was +0.5
    live._apply(chess.Move.from_uci("e7e5"), searched(0.4))   # black now sees +0.4 for black: white lost 0.9
    assert live.moves[0]["drop"] == pytest.approx(0.9)
    assert live.mistakes == {"white": 1, "black": 0}


def test_pause_stops_selfplay_until_resumed(live):
    live._step()
    n = len(live.moves)
    live.set_settings({"paused": True})
    assert live._step() == 0.1 and len(live.moves) == n
    assert live.state()["status"] == "paused"
    live.set_settings({"paused": False})
    live._step()
    assert len(live.moves) == n + 1


def test_finished_watch_game_waits_for_summary_then_continues(live):
    live.set_settings({"simulations": 2, "learning": False})
    for _ in range(400):
        if live.result:
            break
        live._step()
    assert live.result and live.over_at
    gid = live.game_id
    assert live._step() == 0.1 and live.game_id == gid          # holding on the summary
    assert live.state()["game"]["hold_left"] > 0
    live.ack_summary()
    live._step()
    assert live.game_id == gid + 1                              # dismissed: next game starts


def test_summary_off_does_not_hold(live):
    live.set_settings({"simulations": 2, "learning": False, "summary": False})
    for _ in range(400):
        if live.result:
            break
        live._step()
    gid = live.game_id
    live._step()
    assert live.game_id == gid + 1


def test_checkpoints_are_saved_and_training_net_is_reloaded(tmp_path):
    live = Live(tmp_path, blocks=1, channels=8, device="cpu", simulations=2)
    assert (tmp_path / "best.pt").exists()                       # the starting net, so evaluate.py has something to load
    live.set_settings({"learning": True, "batch_size": 8, "train_steps": 2, "arena_every": 99, "summary": False})
    for _ in range(400):
        if live.curve:
            break
        live._step()
    assert (tmp_path / "latest.pt").exists()
    reloaded = Live(tmp_path, blocks=1, channels=8, device="cpu", simulations=2)
    assert all(torch.equal(a, b) for a, b in zip(live.net.state_dict().values(), reloaded.net.state_dict().values()))


def test_record_counts_from_each_side_and_survives_restart(tmp_path):
    live = Live(tmp_path, blocks=1, channels=8, device="cpu", simulations=2)
    live._tally("play", "1-0", "white")    # you won as White
    live._tally("play", "1-0", "black")    # you lost as Black
    live._tally("play", "0-1", "black")    # you won as Black
    live._tally("play", "1/2-1/2", "white")
    live._tally("selfplay", "0-1", "white")
    assert live.counts["play"] == {"win": 2, "draw": 1, "loss": 1}
    assert live.counts["selfplay"] == {"white": 0, "draw": 0, "black": 1}
    assert live.recent["play"] == ["win", "loss", "win", "draw"]

    live.set_settings({"simulations": 2, "learning": False, "summary": False})
    for _ in range(400):
        if live.result:
            break
        live._step()
    assert sum(live.counts["selfplay"].values()) == 2          # the finished game was counted
    again = Live(tmp_path, blocks=1, channels=8, device="cpu", simulations=2)
    assert sum(again.counts["selfplay"].values()) == 1          # rebuilt from games.jsonl: only the one game that was saved
    assert sum(again.counts["play"].values()) == 0
    assert live.state()["record"]["counts"]["selfplay"] == live.counts["selfplay"]


# ---- net picker ----------------------------------------------------------------------------------
@pytest.fixture
def picker(tmp_path):
    from chessai.network import ChessNet, save_checkpoint
    runs = tmp_path / "runs"
    (runs / "other").mkdir(parents=True)
    pre = runs / "other" / "pretrained.pt"
    save_checkpoint(ChessNet(2, 16), pre, {"pretrained": True, "epoch": 3, "val_top1": 0.48})
    torch.save({"net": {}, "optimizer": {}, "epoch": 3}, runs / "other" / "pretrain_state.pt")   # not a net
    torch.save({"x": 1}, runs / "other" / "junk.pt")                                                # not a net
    live = Live(runs / "ui", blocks=1, channels=8, device="cpu", simulations=2, runs_root=runs)
    return live, runs, pre


def run_queue(live):
    while not live.cmds.empty():
        live._handle(live.cmds.get_nowait())


def switch(live, path):
    live.request_net(path)
    run_queue(live)


def test_list_nets_offers_only_relevant_nets_in_groups(picker):
    from chessai.network import ChessNet, save_checkpoint
    live, runs, pre = picker
    # things that must be hidden: training copies, numbered checkpoints, snapshots, other untrained nets
    for name, extra in (("latest.pt", {"iteration": 31}), ("iter_000.pt", {}), ("trained_latest.pt", {"iteration": 75}),
                        ("random_init_control.pt", {"iteration": 0})):
        save_checkpoint(ChessNet(1, 8), runs / "other" / name, extra)
    (runs / "smoke").mkdir()
    save_checkpoint(ChessNet(1, 8), runs / "smoke" / "best.pt", {"iteration": 0})           # untrained leftover
    (runs / "ui" / "branches" / "x_y").mkdir(parents=True)
    save_checkpoint(ChessNet(1, 8), runs / "ui" / "branches" / "x_y" / "best.pt", {"promotions": 0, "iteration": 0})  # copy of its start
    live._nets = live.list_nets()
    shown = [(n["group"], n["path"].split("/")[-2] + "/" + n["path"].split("/")[-1]) for n in live.nets_listing()]
    assert shown == [("Pretrained on strong games", "other/pretrained.pt"), ("Self-play", "ui/best.pt")]
    label = {n["path"].split("/")[-1]: n["label"] for n in live.list_nets()}
    assert "3 epochs, 48% of moves match" in label["pretrained.pt"] and "2x16" in label["pretrained.pt"]

    # a branch's best.pt appears once self-play has improved it
    (runs / "ui" / "branches" / "p_q").mkdir()
    save_checkpoint(ChessNet(1, 8), runs / "ui" / "branches" / "p_q" / "best.pt", {"promotions": 3, "iteration": 40})
    groups = [n["group"] for n in live.list_nets()]
    assert groups == ["Pretrained on strong games", "Pretrained, then improved by self-play", "Self-play"]


def test_request_net_rejects_unsafe_paths(picker, tmp_path):
    live, runs, pre = picker
    outside = tmp_path / "evil.pt"
    torch.save({"config": {"blocks": 1, "channels": 8}, "state": {}, "extra": {}}, outside)
    for bad in (outside, runs / "missing.pt", runs / ".." / "evil.pt", runs / "other", runs / "other" / "notes.txt"):
        with pytest.raises(ValueError):
            live.request_net(bad)
    assert live.cmds.empty()


def test_a_file_that_is_not_a_net_is_refused_and_the_current_net_stays(picker):
    live, runs, pre = picker
    before = (live.net_key, live.net.config, live.game_id)
    switch(live, runs / "other" / "junk.pt")                  # a .pt file, but not a net
    assert "could not load" in live.status
    assert (live.net_key, live.net.config, live.game_id) == before


def test_web_thread_calls_never_use_torch(picker, monkeypatch):
    """On this setup torch crashes the process off the main thread, so the web-facing calls must not touch it."""
    live, runs, pre = picker
    def boom(*a, **k):
        raise AssertionError("torch.load called from a web-facing method")
    monkeypatch.setattr(torch, "load", boom)
    live._scan_asked = 0.0
    assert len(live.nets_listing()) == 2                      # the scan done at start-up
    live.request_net(pre)
    live.state()


def test_scan_command_refreshes_the_listing(picker):
    from chessai.network import ChessNet, save_checkpoint
    live, runs, pre = picker
    (runs / "newrun").mkdir()
    save_checkpoint(ChessNet(1, 8), runs / "newrun" / "best.pt", {"promotions": 4, "iteration": 9})
    assert len(live.nets_listing()) == 2                      # not seen yet
    run_queue(live)                                            # the worker handles the scan it was asked for
    assert any(n["path"].endswith("newrun/best.pt") and "4 improvements" in n["label"] for n in live.nets_listing())


def test_switching_nets_changes_the_net_and_never_touches_other_files(picker):
    live, runs, pre = picker
    ui_best, pre_bytes = (runs / "ui" / "best.pt").read_bytes(), pre.read_bytes()
    assert live.net.config == {"blocks": 1, "channels": 8}
    switch(live, pre)
    assert live.net.config == live.best.config == {"blocks": 2, "channels": 16}
    assert live.net_key.endswith("ui/branches/other_pretrained") and live.net_name == "other/pretrained.pt"
    assert live.state()["net"]["name"] == "other/pretrained.pt" and live.game_id == 2 and not live.moves

    live.set_settings({"simulations": 2, "learning": True, "batch_size": 8, "train_steps": 2, "arena_every": 99, "summary": False})
    for _ in range(400):
        if live.curve:
            break
        live._step()
    assert (runs / "ui" / "branches" / "other_pretrained" / "latest.pt").exists()   # progress goes to the branch folder
    assert not (runs / "ui" / "latest.pt").exists()                                  # the original UI net is untouched
    assert (runs / "ui" / "best.pt").read_bytes() == ui_best and pre.read_bytes() == pre_bytes


def test_records_and_curves_are_per_net_and_a_branch_resumes(picker):
    live, runs, pre = picker
    live._tally("play", "1-0", "white")                       # a win against the original net
    switch(live, pre)
    assert live.counts["play"] == {"win": 0, "draw": 0, "loss": 0}
    live._tally("play", "0-1", "white")                       # a loss against the pretrained net
    live.set_settings({"simulations": 2, "learning": True, "batch_size": 8, "train_steps": 2, "arena_every": 99, "summary": False})
    for _ in range(400):
        if live.curve:
            break
        live._step()
    rounds = len(live.curve)
    switch(live, runs / "ui" / "best.pt")                      # back to the original
    assert live.counts["play"] == {"win": 1, "draw": 0, "loss": 0} and live.curve == []
    switch(live, pre)                                          # picking the pretrained net again carries on
    assert live.counts["play"] == {"win": 0, "draw": 0, "loss": 1} and len(live.curve) == rounds
    assert (runs / "ui" / "branches" / "other_pretrained" / "best.pt").exists()


def test_games_remember_their_net_and_the_record_reloads_per_net(picker):
    live, runs, pre = picker
    switch(live, pre)
    live.set_settings({"simulations": 2, "learning": False, "summary": False})
    for _ in range(400):
        if live.result:
            break
        live._step()
    assert live.games[-1]["net"].endswith("branches/other_pretrained")
    again = Live(runs / "ui", blocks=1, channels=8, device="cpu", simulations=2, runs_root=runs)
    assert sum(again.counts["selfplay"].values()) == 0         # it starts on the original net, which has no games
    switch(again, pre)
    assert sum(again.counts["selfplay"].values()) == 1         # the pretrained net's game is found in games.jsonl


def test_ui_opens_on_the_newest_pretrained_net_by_default(picker):
    live, runs, pre = picker
    auto = Live(runs / "ui", blocks=1, channels=8, device="cpu", simulations=2, runs_root=runs, start_net="auto")
    assert auto.net_name == "other/pretrained.pt" and auto.best.config == {"blocks": 2, "channels": 16}
    assert auto.net_key.endswith("ui/branches/other_pretrained") and auto.game_id == 1
    own = Live(runs / "ui", blocks=1, channels=8, device="cpu", simulations=2, runs_root=runs, start_net="selfplay")
    assert own.net_name == "ui/best.pt" and own.best.config == {"blocks": 1, "channels": 8}
    plain = Live(runs / "ui", blocks=1, channels=8, device="cpu", simulations=2, runs_root=runs)
    assert plain.net_name == "ui/best.pt"                                   # the library default is unchanged
    named = Live(runs / "ui", blocks=1, channels=8, device="cpu", simulations=2, runs_root=runs, start_net=str(pre))
    assert named.net_name == "other/pretrained.pt"


def test_auto_falls_back_to_the_own_net_when_nothing_is_pretrained(tmp_path):
    runs = tmp_path / "runs"
    live = Live(runs / "ui", blocks=1, channels=8, device="cpu", simulations=2, runs_root=runs, start_net="auto")
    assert live.net_name == "ui/best.pt"
    with pytest.raises(ValueError):
        Live(runs / "ui", blocks=1, channels=8, device="cpu", simulations=2, runs_root=runs, start_net=str(tmp_path / "x.pt"))


def test_learning_is_off_by_default_and_nothing_trains_until_it_is_switched_on(live):
    assert live.settings["learning"] is False
    live.set_settings({"simulations": 2, "batch_size": 8, "train_steps": 2, "summary": False})
    for _ in range(400):
        if live.result:
            break
        live._step()
    assert live.result and live.curve == [] and len(live.buffer) == 0
    assert not (live.save_dir / "latest.pt").exists()          # no training copy was written either
