# chessAI

A chess player that learns from win/draw/loss outcomes via AlphaZero-style self-play.
See `PROJECT_BRIEF.md` for the goals and staged plan.

## Status

Steps 1-3 done: board encoding, move mask, network, MCTS, self-play, training loop with arena gating, Stockfish evaluation harness and chart.
Local web UI built (`serve.py`). Search is batched (see below).

## Setup (Windows)

    python -m venv .venv
    .venv\Scripts\python -m pip install -r requirements.txt
    .venv\Scripts\python -m pip install -r requirements-cpu.txt     # work PC
    .venv\Scripts\python -m pip install -r requirements-gpu.txt     # home PC (RTX 5060)

See `TODO.md` for project status and next steps.

## Test

    .venv\Scripts\python -m pytest -q

## Web UI

    .venv\Scripts\python serve.py            # then open http://127.0.0.1:8000

Watch the net play itself (speed x1/x4/x16/x64, Learning on/off) or switch to Play and play it. Right side: candidate moves with
search shares, editable training settings, a per-game move log (CSV/JSON download), and the learning curve (training loss, arena score).
Everything is saved in `runs/ui` (`best.pt`, `train_log.csv`, `games.jsonl`); an existing `best.pt` is loaded on start.
Learning only runs in Watch mode and is off by default (the Learning button turns it on): self-play training did not improve the pretrained nets in tests. A move is a "mistake" when the net's value estimate for the mover drops by more than 0.3 (a heuristic, not ground truth).
The GPU loop runs on the main thread: on this setup (Windows, torch 2.14 + CUDA 13) a CUDA convolution on a worker thread crashes Python.

## Warm start from strong games (supervised)

The goal is to play well against strong players, which self-play alone will not reach on one GPU. So first teach the net to predict
the moves and results of strong human games, then refine with self-play.

    .venv\Scripts\python prepare_data.py games.zip --out-dir data\elite --min-elo 2300
    .venv\Scripts\python pretrain.py data\elite --out-dir runs\pretrain --epochs 3 --blocks 6 --channels 64
    .venv\Scripts\python serve.py --init-from runs\pretrain\pretrained.pt --out-dir runs\ui2        # play it / self-play from it
    .venv\Scripts\python train.py --init-from runs\pretrain\pretrained.pt --out-dir runs\run2 ...     # or train from it

`prepare_data.py` reads `.pgn`, `.zip` and (after `pip install zstandard`) `.pgn.zst`, keeps games where both players are rated at least
`--min-elo`, and writes bit-packed positions to `data/` (git-ignored; about 150 bytes per position). Every 50th game is held out for validation.
`pretrain.py` logs policy/value loss and, on the held-out games, top-1 and top-3 move accuracy (`pretrain_log.csv`); `--resume` continues a run
(`--epochs` is the total to reach; raising it on resume stretches the learning-rate schedule). Bigger nets: `--blocks` / `--channels`.

Scaling up (several months, a bigger net): `prepare_data.py` each month (one process per month runs in parallel), then merge them so the data can be read from disk instead of held in memory:

    .venv\Scripts\python consolidate_data.py data\elite_2019-* data\elite_2020-* --out-dir data\combined
    .venv\Scripts\python pretrain.py data\combined --out-dir runs\pretrain_big --epochs 3 --blocks 10 --channels 128 --batch-size 1024 --amp --save-every 10000

`--save-every` validates, saves `pretrained.pt` and the resume state every N steps, so `--resume` loses at most that much work. Training speed on the RTX 5070 (positions/s):
6x64 about 44,000; 8x96 31,000 (bf16); 10x128 24,000 (bf16); 12x128 21,000 (bf16). bf16 (`--amp`) is about 1.5-1.8x faster from 8 blocks up, no gain at 6x64.

Measured here on synthetic games (RTX 5070, 6 blocks x 64 channels, batch 512): data preparation about 33,000 positions/s on one CPU core,
training about 22,000 positions/s, so 10 million positions is roughly 8 minutes per epoch. Mixed precision (`--amp`) was slower on this small net.
Whether the warm-started net is actually stronger must still be measured with `evaluate.py`.

## How to train a better net

A net gets better from three things: more (and better) games to learn from, more capacity (a bigger net), and more training time. Search at play time
(`--simulations`) adds strength on top without any training. Always measure: a lower training error does not by itself mean a stronger player.

1. **Prepare more games.** One process per file, run in parallel (about 2 minutes per 50,000 games per file):

       .venv\Scripts\python prepare_all.py "data\raw_elite\Lichess Elite Database\lichess_elite_201*.pgn" "data\raw_elite\Lichess Elite Database\lichess_elite_2020-*.pgn" --out-root data\months --max-games 100000
       .venv\Scripts\python consolidate_data.py data\months\* --out-dir data\combined2

   `--max-games` caps each file (the biggest months have 360,000+ games). Positions take 144 bytes each on disk; roughly 85 positions per game.
   `--min-elo` raises the rating needed from both players (default 2200).

2. **Train.** Either keep improving the current net on the bigger data set (fastest, keeps what it has learned):

       .venv\Scripts\python pretrain.py data\combined2 --init-from runs\pretrain_big\pretrained.pt --lr 3e-4 --epochs 2 --amp --save-every 10000 --out-dir runs\pretrain_v2

   or train a bigger net from scratch (`--blocks 12 --channels 128`, or more channels; see the speed table above for the cost). Time per epoch is
   positions divided by the speed (10x128 with `--amp`: about 24,000 positions per second). A run can be stopped and continued with `--resume runs\pretrain_v2`.

3. **Measure.** The clearest test is a head-to-head match against the net you want to beat:

       .venv\Scripts\python compare_nets.py runs\pretrain_v2\pretrained.pt runs\pretrain_big\pretrained.pt --games 200 --simulations 100

   It prints the score with a 95% interval and says whether the difference is real. Also run `evaluate.py` for the Stockfish ladder (its 8 fixed openings make
   its error margins optimistic) and read the held-out move-match accuracy in `plot_pretrain.py`'s report.

4. **Use it.** `serve.py` opens on the newest pretrained net by default; the **Net** dropdown switches between nets.

Self-play fine-tuning (`train.py --init-from`) did not improve a pretrained net in tests (see TODO.md); more supervised data and a bigger net are the proven levers so far.

## Choosing a net in the UI

The UI opens on the newest pretrained net it finds under `runs/` (`--net auto`, the default); `--net selfplay` opens on the folder's own
self-play net instead, and `--net runs\pretrain\pretrained.pt` names one explicitly.

The **Net** dropdown at the top of the controls lists every saved net under `runs/` (pretrained, self-play, checkpoints) with a short description,
newest first. Picking one loads it as the playing net, in Watch or Play mode, without a restart. What happens to it:

- Its training progress is saved in its own folder, `runs/ui/branches/<folder>_<file>/`, so the file you picked is never overwritten.
- Its win/draw/loss record and learning curve are its own; switching back restores the other net's. Picking a net you used before carries on from its branch.
- The replay buffer starts empty on each switch (it holds games from a different net). Only files under `runs/` can be loaded.

## Openings

The UI names the opening as the game is played (shown under the board, and in the game summary), using the Lichess opening list
(`chessai/data/openings/*.tsv`, CC0). Names are found by position, so a transposition gets the same name.

The **Opening** box starts every new game from a chosen line: click it, type to search (for example "najdorf"), use the arrow keys and Enter or
click. **Random opening** (or the **Random** button) picks a line in proportion to the square root of how often strong players play it, using
`chessai/data/openings/popularity.json`, built from the training games with `build_opening_popularity.py`. **No opening** starts from the
beginning. The opening moves are played as "book" moves; the AI plays on from there, in Watch or Play mode. Choosing an opening starts a new game.

## Layout

- `chessai/encoding.py` - board to an (18, 8, 8) tensor, always from the mover's point of view
- `chessai/moves.py` - move <-> index in a 4672-slot policy (64 squares x 73 move types), and the legal-move mask
- `serve.py`, `ui/index.html`, `chessai/live.py` - local web UI, its page, and the engine behind it
- `benchmark.py` - search speed (simulations/sec)
- `prepare_all.py`, `consolidate_data.py`, `compare_nets.py` - prepare many months in parallel, merge them for disk-based training, head-to-head net comparison
- `prepare_data.py`, `pretrain.py`, `chessai/pgn_data.py` - PGN games to training positions, and the supervised warm start
- `tests/` - encoding and move-mask tests, including random-game round trips

## Train (tiny CPU smoke test)

    .venv\Scripts\python train.py --iterations 2 --games 2 --simulations 8 --train-steps 5 --arena-games 2

Writes checkpoints (`best.pt`, `iter_NNN.pt`) and `train_log.csv` to `runs/<out-dir>`.

Resume a stopped or finished run (`--iterations` is the total to reach, so you can re-run the same command with `--resume`):

    .venv\Scripts\python train.py --iterations 200 --resume runs\run1

`state.pt` (both nets, optimizer, iteration, elapsed time) is saved after every iteration; `buffer.pkl` (the replay buffer, about 1.8 KB per
position, so up to ~180 MB when full) every `--buffer-every` iterations (default 10). If a stop loses the last few iterations of buffer, resume
says which iteration the buffer came from. Network size comes from the save; a fresh run into a folder that already holds a run is refused.

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

Measured on the RTX 5070 (`python benchmark.py --blocks 6 --channels 64`, 100 simulations/move, 2026-10-29):

| test | simulations/sec |
|---|---|
| single search, 1 / 8 / 32 leaves | 764 / 3,756 / 5,899 |
| self-play, leaves 8, parallel 8 / 16 / 32 / 64 | 7,042 / 7,149 / 6,794 / 6,045 |
| self-play, parallel 16, leaves 4 / 8 / 16 / 32 | 6,735 / 7,149 / 7,642 / 7,570 |

Self-play plateaus near 7,000 regardless of `--parallel` or `--leaves`, so Python-side move generation and tree work are the limit, not the GPU.
Chosen: `--parallel 16 --leaves 8`. Single runs, so differences under about 10% are noise.
