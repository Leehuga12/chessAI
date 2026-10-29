# TODO

Read this first. It is the working state of the project, written for Claude and for the owner.
Goal and constraints live in `PROJECT_BRIEF.md`. How to run things lives in `README.md`.

## Current state (as of 2026-10-28)

- Done and tested (49 tests pass): board encoding, move mask, policy/value net, batched MCTS, parallel self-play,
  replay buffer, arena gating, training loop (`train.py`), Stockfish evaluation (`evaluate.py`), chart (`plot_results.py`).
- Never run at real scale. No evidence yet that the model learns. Do not claim improvement without numbers from `evaluate.py`.
- Work PC: CPU only, Python 3.14, venv at `.venv`. Home PC: RTX 5060, not set up yet.
- Search speed on the work PC CPU: about 1,000 simulations/sec (was 230). GPU speed is unmeasured.
- Repo: https://github.com/Leehuga12/chessAI (branch `main`). Commits use a noreply author identity set locally.

## Direction change (2026-10-29)

Goal confirmed with the owner: beat high-rated players/bots. Pure self-play on one GPU will not get there, so the plan is
supervised warm start from strong games (Lichess Elite database or similar) -> self-play refinement -> more search at play time.
Stockfish as a teacher (labelling positions) is an optional upgrade that needs an explicit yes from the owner (the brief says not otherwise).

- [x] Data pipeline (`prepare_data.py`) and pretraining (`pretrain.py`), tested on synthetic games; `--init-from` in `train.py` and `serve.py`
- [ ] Owner picks and downloads one month of strong games (ask before any large download; check the file size first)
- [ ] Run `prepare_data.py` then `pretrain.py` on it; evaluate `pretrained.pt` with `evaluate.py` against the untrained control and Stockfish rungs
- [ ] If the warm start helps: larger net, more months of data, then self-play from it (step 4 below, with `--init-from`)
- [x] Warm start measured (2026-10-29): 1 month of Lichess Elite (16M positions, 6x64 net, 3 epochs, 17 min): 48% top-1 / 77% top-3 move match on held-out
      games; roughly even with Stockfish "1700" (nominal) at 50 simulations. Self-play-only net: 0% against Stockfish 1320.
- [x] Self-play fine-tuning of the pretrained net does NOT help yet (15 iterations x 32 games, 100 simulations, gate 20 games): lr 1e-3 is harmful (mean arena
      0.31, 0 promotions); lr 1e-4 and 1e-5 stay level (mean arena 0.49 / 0.48, 3-4 promotions that are within chance). Head-to-head, 100 games vs the original:
      lr 1e-4 48.0% (95% interval 40.7 to 55.3), lr 1e-5 45.0% (37.0 to 53.0): no measurable gain. Too little self-play data/search to move a pretrained net.
- [x] Scaled supervised warm start (2026-10-29): 15 months of Lichess Elite (899k games, 72.7M training positions, 50k games/month cap except Dec 2019),
      10 blocks x 128 channels (3.1M params), bf16, batch 1024, 3 epochs, 150 min (`runs/pretrain_big/pretrained.pt`). Held-out move match: 55.5% top-1, 83.6% top-3
      (small net: 48.3 / 77.5 on a different, December-only validation set). Head-to-head vs the small pretrained net, 100 games: 82.0% (95% interval 75.4 to 88.6).
      Stockfish (nominal Elo, 2000 nodes, 40 games, 50 simulations): 1700 -> 80%, 1900 -> 74%, 2100 -> 59% (est. ~1930 / 2070 / 2160, margins 110-170, optimistic: only
      ~16 distinct games per 40 because of 8 fixed openings).
- [ ] Next: all 3.8M games (about 5x more data; needs `consolidate_data.py` data on disk, ~50 GB), a bigger net (12x128 or 15x192), more epochs; revisit self-play only
      with far more games, mixed-in strong-game data and a low lr
- [ ] Raise play-time simulations in the UI (the net plays noticeably stronger with more search)

## Conventions

- Run Python through the venv: `.venv\Scripts\python` (Windows).
- Tests: `.venv\Scripts\python -m pytest -q`. Keep them green before committing.
- Ask the owner before: large downloads, heavy installs, pushing to GitHub.
- Report negative results plainly. Verify improvement with harness numbers first.
- Stockfish lives in `tools/stockfish/` (git-ignored). Download: https://github.com/official-stockfish/Stockfish/releases (tag `sf_19`,
  file `stockfish-windows-x86-64-universal.zip`, about 81 MB), unzip so the exe is at
  `tools/stockfish/stockfish/stockfish-windows-x86-64-universal.exe`.

## Next steps (in order)

### 1. Set up the home PC (owner does most of this)
- [x] `git clone https://github.com/Leehuga12/chessAI.git`
- [x] Create venv, then `pip install -r requirements.txt` and `pip install -r requirements-gpu.txt`
- [x] Download Stockfish as described above
- [x] Check CUDA: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` must print True and the 5060
- [x] Run the tests; all should pass
- Done when: tests pass and `train.py` prints `device=cuda`.

### 2. Add `benchmark.py` and measure GPU speed
- [x] Add a committed `benchmark.py` (the scratch scripts used earlier are not in the repo): simulations/sec for a single search at
      leaves 1/8/32, and parallel self-play at parallel 8/16/32/64, for a given `--blocks --channels`
- [x] Run it on the 5060, record the numbers in `README.md` under Speed, pick the best `--parallel` and `--leaves`
- Done when: GPU numbers are in the README. Expect Python move generation to become the limit; if so, consider multiple
  self-play processes feeding one GPU, or moving encoding to vectorised code.

### 3. Add checkpoint resume to `train.py`
- [x] `--resume <dir>`: reload best/candidate net, optimizer, iteration counter, and append to the existing `train_log.csv`
- [x] Save the replay buffer periodically (or accept restarting with an empty one and say so)
- Needed before any run that takes more than a few hours.

### 4. First real training run
- [ ] Starting point (untested guess, tune after step 2):
      `python train.py --iterations 200 --games 64 --simulations 100 --parallel 32 --leaves 8 --blocks 6 --channels 64 --train-steps 200 --batch-size 256 --arena-games 20 --out-dir runs/run1`
- [ ] Watch `train_log.csv`: policy loss should fall, `promoted` should be 1 fairly often early on. If the gate never promotes, investigate
      before running longer (learning rate, train steps per iteration, buffer size).
- [ ] Known gap: no learning-rate schedule, no weight-decay tuning, no value/policy loss weighting. Add only if the run stalls.

### 5. Evaluate during training and chart it
- [ ] Every N iterations run `evaluate.py` on `best.pt` against `random`, `sf-skill0`, `sf-elo1320`, then higher Elo rungs as it improves
      (`sf-elo1500`, `sf-elo1700`, ...). Use 50 or more games per point; 16 games gives margins of hundreds of Elo.
- [ ] Always include the untrained net (`iter_000.pt`) as the control.
- [ ] Run `plot_results.py` to make `results/improvement.png`.
- [ ] Optional: make `evaluate()` play games in parallel (it is serial and will be the slow part).

### 6. Local web UI (Pong-in-a-dish style) - first version built, see README
- [ ] Small local server (FastAPI or Flask) plus one HTML page, no build step
- [ ] Board you can play against the model on (use `python-chess` for legality; the owner does not know the rules)
- [ ] Controls: speed (1x/4x/16x/64x) for watching self-play, Learning on/off toggle, New model button
- [ ] Live curves: loss, arena score, Elo vs Stockfish rungs
- [ ] Panel showing the model's top candidate moves (MCTS visit shares) and its win estimate
- Reference look: the owner's Pong prototype (dark theme, game left, learning curve right, stats row).

### 7. Final report
- [ ] Short write-up: what improved, by how much (with margins), what did not work, compute used. Put it in `RESULTS.md`.

### Optional, only if asked
- [ ] Stage 2: supervised warm start from Lichess games (needs a large download: ask first)
- [ ] Stage 4: "dish-style" outcome-only experiment, compared honestly against self-play

## Memory note (Windows)

The memory a program may reserve is capped at RAM + page file (here 32.6 GB + 2 GB). Loading 10.7 GB of data into RAM failed once even with 17 GB free, because
other programs had reserved the rest. `consolidate_data.py` writes flat files that `pretrain.py` maps read-only from disk (not counted against the cap).

## Known limitations (be honest about these in any report)

- Games are capped (200 plies in self-play and the arena, 300 in Stockfish evaluation) and scored as draws. An untrained net draws a random mover
  this way, so early scores against `random` flatter it. Judge progress on the Stockfish rungs.
- Stockfish Elo labels are nominal, not a calibrated scale. Stockfish searches a fixed node budget (2000), so absolute Elo differs from rated play.
- Batched search (virtual loss) is slightly less sharp than one-leaf search; keep `--leaves` well below `--simulations`.
- `chessai/rules.py` must keep matching `board.outcome(claim_draw=True)`; `tests/test_rules.py` checks this. python-chess treats a draw as
  claimable one ply early (repetition at clock 7, fifty-move at 99).
- Training and search have only been run for seconds on CPU. Hyperparameters are untuned guesses.

## Map of the code

- `chessai/encoding.py` board to (18, 8, 8); `chessai/moves.py` move index (4672) and legal mask
- `chessai/network.py` ResNet policy+value; `chessai/mcts.py` batched PUCT search; `chessai/parallel.py` many games at once
- `chessai/selfplay.py` samples and replay buffer; `chessai/arena.py` gating matches; `chessai/rules.py` fast game-end check
- `chessai/evaluate.py` opponents and Elo maths; `train.py`, `evaluate.py`, `plot_results.py` command-line entry points
