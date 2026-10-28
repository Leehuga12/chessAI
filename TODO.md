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
- [ ] `git clone https://github.com/Leehuga12/chessAI.git`
- [ ] Create venv, then `pip install -r requirements.txt` and `pip install -r requirements-gpu.txt`
- [ ] Download Stockfish as described above
- [ ] Check CUDA: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` must print True and the 5060
- [ ] Run the tests; all should pass
- Done when: tests pass and `train.py` prints `device=cuda`.

### 2. Add `benchmark.py` and measure GPU speed
- [ ] Add a committed `benchmark.py` (the scratch scripts used earlier are not in the repo): simulations/sec for a single search at
      leaves 1/8/32, and parallel self-play at parallel 8/16/32/64, for a given `--blocks --channels`
- [ ] Run it on the 5060, record the numbers in `README.md` under Speed, pick the best `--parallel` and `--leaves`
- Done when: GPU numbers are in the README. Expect Python move generation to become the limit; if so, consider multiple
  self-play processes feeding one GPU, or moving encoding to vectorised code.

### 3. Add checkpoint resume to `train.py`
- [ ] `--resume <dir>`: reload best/candidate net, optimizer, iteration counter, and append to the existing `train_log.csv`
- [ ] Save the replay buffer periodically (or accept restarting with an empty one and say so)
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

### 6. Local web UI (Pong-in-a-dish style)
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
