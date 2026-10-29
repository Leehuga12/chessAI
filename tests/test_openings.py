import chess
import numpy as np
import pytest

from chessai.live import Live
from chessai.openings import DEFAULT_DIR, Openings

TSV = """eco\tname\tpgn
B00\tKing's Pawn Game\t1. e4
B20\tSicilian Defense\t1. e4 c5
B50\tSicilian Defense: Modern Variations\t1. e4 c5 2. Nf3 d6
B50\tSicilian Defense: Modern Variations\t1. Nf3 c5 2. e4 d6
C20\tKing's Pawn Game: Two Knights\t1. e4 e5 2. Nf3 Nc6
D06\tQueen's Gambit\t1. d4 d5 2. c4
A00\tPolish Opening\t1. b4
"""


@pytest.fixture
def small(tmp_path):
    (tmp_path / "a.tsv").write_text(TSV, encoding="utf-8")
    return tmp_path


def play(san):
    b = chess.Board()
    for m in san.split():
        b.push_san(m)
    return b


def test_names_follow_the_deepest_known_line_and_survive_transposition(small):
    o = Openings(small)
    assert o.name_for(play("e4"))["name"] == "King's Pawn Game"
    assert o.name_for(play("e4 c5"))["name"] == "Sicilian Defense"
    assert o.name_for(play("e4 c5 Nf3 d6"))["name"] == "Sicilian Defense: Modern Variations"
    assert o.name_for(play("Nf3 c5 e4 d6"))["name"] == "Sicilian Defense: Modern Variations"   # the other move order
    assert o.name_for(play("e4 c5 Nf3")) is None                                          # between two named lines
    assert o.name_for(play("g4")) is None
    assert len(o.entries) == 6                                                             # one entry per name, not per move order


def test_random_entry_respects_length_and_works_with_numpy(small):
    o = Openings(small)
    rng = np.random.default_rng(0)
    for _ in range(30):
        e = o.random_entry(rng, min_plies=3, max_plies=4)
        assert 3 <= e["plies"] <= 4
    assert Openings(small).random_entry(np.random.default_rng(1), min_plies=50, max_plies=60) is not None   # falls back to anything
    assert Openings(small / "missing").random_entry() is None


def test_the_shipped_data_loads_and_every_line_is_playable():
    o = Openings(DEFAULT_DIR)
    assert len(o.entries) > 3000
    assert o.name_for(play("e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6"))["name"].endswith("Najdorf Variation")
    assert o.lookup("B90 Sicilian Defense: Najdorf Variation")["plies"] >= 10


def live_with(small, tmp_path, **kw):
    return Live(tmp_path / "runs" / "ui", blocks=1, channels=8, device="cpu", simulations=2, openings_dir=small, **kw)


def test_game_starts_from_the_chosen_opening_with_book_moves(small, tmp_path):
    live = live_with(small, tmp_path)
    live.set_settings({"opening": "B50 Sicilian Defense: Modern Variations"})
    live._handle(live.cmds.get_nowait())                                   # the new game the choice triggers
    assert [m["san"] for m in live.moves] == ["e4", "c5", "Nf3", "d6"] and all(m["by"] == "book" for m in live.moves)
    assert live.board.turn == chess.WHITE and live.board.ply() == 4
    assert live.state()["game"]["opening"] == {"eco": "B50", "name": "Sicilian Defense: Modern Variations"}
    assert live.moves[1]["op"]["name"] == "Sicilian Defense"               # the name follows the moves
    live._step()                                                            # the AI plays on from the book position
    assert live.moves[4]["by"] == "ai" and len(live.moves) == 5
    assert live.moves[-1]["fen"] == live.board.fen()


def test_unknown_opening_is_rejected_and_none_means_start_from_the_beginning(small, tmp_path):
    live = live_with(small, tmp_path)
    with pytest.raises(ValueError):
        live.set_settings({"opening": "Z99 Nonsense"})
    live.set_settings({"opening": "none"})
    assert live.cmds.empty() and live.moves == []


def test_random_opening_changes_between_games_and_is_stored_with_the_game(small, tmp_path):
    live = live_with(small, tmp_path)
    live.set_settings({"opening": "random", "summary": False, "learning": False})
    live._handle(live.cmds.get_nowait())
    seen = set()
    for _ in range(25):
        live.new_game()
        assert live.moves and all(m["by"] == "book" for m in live.moves)
        seen.add(live.opening["name"])
    assert len(seen) > 1                                                    # it really is random
    for _ in range(600):
        if live.result:
            break
        live._step()
    assert live.result and live.games[-1]["opening"]["name"]                # the saved game records its opening


def test_play_mode_with_an_opening_asks_the_right_side_to_move(small, tmp_path):
    live = live_with(small, tmp_path)
    live.set_settings({"mode": "play", "human_color": "black", "opening": "B20 Sicilian Defense"})
    live._handle(live.cmds.get_nowait())
    assert live.board.turn == chess.WHITE                                  # book ended on Black's c5, so White (the AI) moves
    assert live._step() == 0.0 and live.moves[-1]["by"] == "ai"            # the AI replies at once
    assert live.board.turn == chess.BLACK and live.state()["legal"]        # now it is the human's turn


def test_random_uses_popularity_when_available_and_listing_reports_it(small):
    import json
    (small / "popularity.json").write_text(json.dumps({"B50 Sicilian Defense: Modern Variations": 900,
                                                       "C20 King's Pawn Game: Two Knights": 100}), encoding="utf-8")
    o = Openings(small)
    rng = np.random.default_rng(0)
    drawn = [o.random_entry(rng, min_plies=3, max_plies=12)["id"] for _ in range(400)]
    assert set(drawn) == {"B50 Sicilian Defense: Modern Variations", "C20 King's Pawn Game: Two Knights"}   # only lines with data
    assert drawn.count("B50 Sicilian Defense: Modern Variations") > drawn.count("C20 King's Pawn Game: Two Knights")  # 3:1 by square root
    games = {e["id"]: e["games"] for e in o.listing()}
    assert games["B50 Sicilian Defense: Modern Variations"] == 900 and games["B20 Sicilian Defense"] == 0


def test_the_shipped_popularity_data_only_names_real_openings():
    o = Openings(DEFAULT_DIR)
    assert o.popularity and all(k in o.by_id for k in o.popularity)
