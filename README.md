# chessAI

A chess player that learns from win/draw/loss outcomes via AlphaZero-style self-play.
See `PROJECT_BRIEF.md` for the goals and staged plan.

## Status

Steps 1-3 done: board encoding, move mask, network, MCTS, self-play, training loop with arena gating, Stockfish evaluation harness and chart.
Not built yet: web UI. Search is batched (see below); the GPU speedup has not been measured yet.

## Setup (Windows)

    python -m venv .venv
    .venv\Scripts\python -m pip install -r requirements.txt
    .venv\Scripts\python -m pip install -r requirements-cpu.txt     # work PC
    .venv\Scripts\python -m pip install -r requirements-gpu.txt     # home PC (RTX 5060)

See `TODO.md` for project status and next steps.

## Test

    .venv\Scripts\python -m pytest -q

## Layout

- `chessai/encoding.py` - board to an (18, 8, 8) tensor, always from the mover's point of view
- `chessai/moves.py` - move <-> index in a 4672-slot policy (64 squares x 73 move types), and the legal-move mask
- `tests/` - encoding and move-mask tests, including random-game round trips

## Train (tiny CPU smoke test)

    .venv\Scripts\python train.py --iterations 2 --games 2 --simulations 8 --train-steps 5 --arena-games 2

Writes checkpoints (`best.pt`, `iter_NNN.pt`) and `train_log.csv` to `runs/<out-dir>`.

## Evaluate and chart

Stockfish 19 (official release, GitHub `official-stockfish/Stockfish`) lives in `tools/stockfish/` and is used only as the measuring opponent.

    .venv\Scripts\python evaluate.py runs\default\iter_000.pt runs\default\best.pt --opponents random sf-skill0 sf-elo1320 --games 16 --simulations 50
    .venv\Scripts\python plot_results.py results\results.csv results\improvement.png

Opponents never change: `random`, `sf-skill0` (win rate only) and `sf-elo<N>` (N >= 1320, gives an estimated Elo).
Stockfish searches a fixed node budget, and each opponent plays from the same 8 fixed openings with colours alternated.
Caveats: Elo labels are Stockfish's nominal ratings, not a calibrated scale; with few games the 95% margin is large;
games are capped at 300 plies and counted as draws, which flatters very weak nets (an untrained net draws a random mover this way).

## Speed

Search evaluates several leaves per network call (virtual loss), and self-play/arena games run
`--parallel` at a time so their leaves share one call. Tune `--parallel` (games at once) and `--leaves`
(leaves per search step; keep it well below `--simulations`, since large batches make the search a little less sharp).
`chessai/rules.py` replaces python-chess's repetition check, which dominated run time; a test checks it agrees with
`board.outcome(claim_draw=True)` over thousands of positions.

Measured on the work PC (CPU only, 4 blocks x 64 channels, 50 simulations/move):
about 230 simulations/sec before, about 1,000 after (single search at 32 leaves per call ~1,100, parallel self-play ~1,060).
