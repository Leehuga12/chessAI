import chess
import pytest

from chessai.live import Live


@pytest.fixture
def live(tmp_path):
    runs = tmp_path / "runs"
    lv = Live(runs / "ui", blocks=1, channels=8, device="cpu", simulations=2, runs_root=runs)
    if not lv.stockfish_ok:
        pytest.skip("Stockfish is not installed in tools/stockfish")
    yield lv
    lv.close()


def apply_queue(live):
    while not live.cmds.empty():
        live._handle(live.cmds.get_nowait())


def play_out(live, limit=400):
    for _ in range(limit):
        if live.result:
            return
        live._step()
    raise AssertionError("the game did not finish")


def test_watch_net_vs_stockfish_alternates_colours_records_per_elo_and_never_trains(live):
    live.set_settings({"opponent": "stockfish", "sf_elo": 1320, "net_color": "alternate", "learning": True,
                       "batch_size": 8, "summary": False})
    apply_queue(live)
    assert live.game_opponent == "stockfish" and live.net_white is True
    assert live.state()["game"]["players"]["black"] == "Stockfish ~1320"

    live._step()                                                       # the net (White) moves first
    live._step()                                                       # then Stockfish
    assert [m["by"] for m in live.moves[:2]] == ["ai", "stockfish"]
    play_out(live)
    assert live.result and {m["by"] for m in live.moves} == {"ai", "stockfish"}
    row = live.state()["record"]["vs_stockfish"]["elos"]["1320"]
    assert sum(row.values()) == 1                                      # recorded against this Elo, from the net's side
    assert len(live.buffer) == 0 and live.curve == []                  # learning is ON but Stockfish games never train
    assert live.history == []

    live._step()                                                       # next game: colours swap
    assert live.net_white is False and live.moves == []
    live._step()
    assert live.moves[0]["by"] == "stockfish"                          # Stockfish has White now


def test_play_vs_stockfish_does_not_use_the_net_at_all(live, monkeypatch):
    live.set_settings({"mode": "play", "opponent": "stockfish", "sf_elo": 1500, "human_color": "white", "summary": False})
    apply_queue(live)
    monkeypatch.setattr(live, "_think", lambda *a, **k: pytest.fail("the net must not be consulted when you play Stockfish"))
    assert live._step() == 0.1 and live.state()["legal"]               # waits for you
    live.human_move("e2e4")
    apply_queue(live)
    assert live.moves[0]["by"] == "human"
    assert live._step() == 0.0 and live.moves[1]["by"] == "stockfish"  # Stockfish replies at once
    assert live.board.turn == chess.WHITE and live.state()["legal"]
    assert live.state()["game"]["players"] == {"white": "You", "black": "Stockfish ~1500"}


def test_changing_the_elo_restarts_with_the_new_strength_and_validates(live):
    live.set_settings({"opponent": "stockfish", "sf_elo": 1700})
    apply_queue(live)
    live._step()
    before = live.game_id
    live.set_settings({"sf_elo": 2100})
    assert not live.cmds.empty()                                       # a new game is queued
    apply_queue(live)
    assert live.game_id == before + 1 and live.game_sf_elo == 2100
    for bad in ({"sf_elo": 1000}, {"sf_elo": 3500}, {"opponent": "gpt"}, {"net_color": "green"}):
        with pytest.raises(ValueError):
            live.set_settings(bad)


def test_results_against_stockfish_are_counted_from_the_right_side(live):
    live._tally("play", "1-0", "white", opponent="stockfish", sf_elo=1700)       # you were White and won
    live._tally("play", "1-0", "black", opponent="stockfish", sf_elo=1700)       # you were Black and lost
    live._tally("play", "1/2-1/2", "white", opponent="stockfish", sf_elo=1900)
    live._tally("selfplay", "0-1", "white", opponent="stockfish", sf_elo=1700, net_white=False)   # the net was Black and won
    live._tally("selfplay", "0-1", "white", opponent="stockfish", sf_elo=1700, net_white=True)    # the net was White and lost
    rec = live.state()["record"]
    assert rec["human_vs_stockfish"]["elos"] == {"1700": {"win": 1, "draw": 0, "loss": 1}, "1900": {"win": 0, "draw": 1, "loss": 0}}
    assert rec["vs_stockfish"]["elos"] == {"1700": {"win": 1, "draw": 0, "loss": 1}}
    assert rec["counts"]["selfplay"] == {"white": 0, "draw": 0, "black": 0}      # not mixed into the self-play record
    assert rec["human_vs_stockfish"]["recent"] == ["win", "loss", "draw"]


def test_stockfish_records_survive_a_restart(live, tmp_path):
    live.set_settings({"opponent": "stockfish", "sf_elo": 1320, "net_color": "white", "summary": False})
    apply_queue(live)
    play_out(live)
    runs = tmp_path / "runs"
    again = Live(runs / "ui", blocks=1, channels=8, device="cpu", simulations=2, runs_root=runs)
    try:
        assert again.state()["record"]["vs_stockfish"] == live.state()["record"]["vs_stockfish"]
        assert sum(again.counts["selfplay"].values()) == 0
    finally:
        again.close()


def test_without_stockfish_the_option_is_refused(tmp_path):
    runs = tmp_path / "runs"
    lv = Live(runs / "ui", blocks=1, channels=8, device="cpu", simulations=2, runs_root=runs, engine_path=tmp_path / "nope.exe")
    assert lv.stockfish_ok is False and lv.state()["engine"]["available"] is False
    with pytest.raises(ValueError):
        lv.set_settings({"opponent": "stockfish"})
    assert lv.state()["game"]["opponent"] == "self"


def test_all_strengths_are_offered_and_the_table_lists_only_stockfish_elo_rows(live, tmp_path):
    header = "time,checkpoint,iteration,opponent,opponent_elo,games,wins,draws,losses,score,est_elo,elo_margin,simulations\n"
    rows = ["t,runs/a/pretrained.pt,-1,random,,40,40,0,0,1.0,,,50",
            "t,runs/a/pretrained.pt,-1,sf-skill0,,40,30,5,5,0.8,,,50",
            "t,runs/a/pretrained.pt,-1,sf-elo1500,1500,40,20,10,10,0.6,1560,100,50",
            "t,runs/a/pretrained.pt,-1,sf-elo1500,1500,40,30,5,5,0.8,1700,100,200"]   # re-run later: the newest row wins
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(header + "\n".join(rows) + "\n")
    st = live.state(stockfish_csv=csv_path)
    assert st["engine"]["elos"] == [1320, 1500, 1700, 1900, 2100, 2300, 2500, 2700, 2900, 3190]   # the full list, whatever was measured
    assert [(r["opponent"], r["sims"], r["score"]) for r in st["stockfish"]] == [("sf-elo1500", "200", "0.8")]   # no random / skill0; sims shown
    assert live.state(stockfish_csv=tmp_path / "missing.csv")["stockfish"] == []
