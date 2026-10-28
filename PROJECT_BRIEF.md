# Project brief: a chess model that learns and improves from win/loss feedback

Use this file as the starting prompt for a new project. It carries over what was learned from the "Pong in a Dish" prototype (`brain.js`, `index.html`, `test.js` in the Pong Game folder).

## Goal

Build a chess player that gets measurably stronger over time, learning mainly from game outcomes (win, draw, loss), and show the improvement with a clear curve.

## Background: what the Pong prototype taught

- Pong used a small spiking-neuron "dish". A hit played a calm, regular beat and a miss played random noise (inspired by the DishBrain experiments).
- With hit/miss as the only signal, my first attempt never converged. The weights random-walked. The delay between an action and its outcome, plus thousands of synapses to blame, made the signal too weak.
- It only worked once the miss step also told the network which direction was correct (error-guided learning). That is a supervised hint, not pure self-learning.
- Chess is much harder: one outcome arrives dozens of moves after the move that mattered. A plain "calm or noise" signal will not be enough, so use proper reinforcement learning (below).
- Lesson to keep: test improvement headlessly against a control, and report results honestly, including when something doesn't work.

## Approach (default plan)

Stage 1: Foundations
- Python, PyTorch, `python-chess`.
- Board encoding: 8x8x12 piece planes plus side to move, castling and en passant.
- A small policy+value network (residual conv net, about 6 blocks to start).
- A legal-move mask on the policy output.

Stage 2: Supervised warm start (optional but recommended)
- Train the network to predict moves from a public database of strong games (for example Lichess open database, filtered by rating).
- Report move-prediction accuracy and a quick Elo check against the reference opponent.

Stage 3: Self-play reinforcement learning (AlphaZero style)
- Self-play with MCTS guided by the network.
- Targets: the MCTS visit distribution for the policy and the game result for the value.
- Replay buffer, periodic training, and a gating step: a new network replaces the best one only if it beats it over N games.
- Win/draw/loss is the only game feedback. No hand-written strategy.

Stage 4: Optional experiment, "feedback only, dish-style"
- A separate mode that tries to learn with only an outcome signal and no engine or database guidance, to compare against Stage 3.
- Document honestly whether it improves.

## How to measure improvement

- Fix one reference opponent, for example Stockfish at a set skill level or node limit, and never change it.
- After every N training iterations, play a fixed number of games (alternating colors, fixed opening set) and log win rate and estimated Elo.
- Also keep a control: the untrained network, and the Stage 2 network if used.
- Output: a CSV of results plus a chart (Elo vs. training time), plus a final short report.

## Constraints and assumptions (confirm or change these)

- Hardware: assume a single laptop or desktop; use a GPU if available. Keep the first version small enough to train in hours, not weeks.
- Start with a smaller scale (fewer MCTS simulations, smaller network) and scale up only once the loop works.
- Build from scratch with PyTorch and `python-chess`, rather than wrapping an existing engine's learning code.
- Stockfish is used only as the measuring opponent, not as a teacher, unless Stage 2 or distillation is explicitly chosen.

## Open questions to settle first

1. Hardware: GPU or CPU only? Cloud budget?
2. Is the goal a strong player, or a clear demonstration of learning from outcomes only?
3. Is the Stage 2 supervised warm start allowed, or must it learn from self-play alone?
4. Should there be a simple UI (play against it in a browser), or is a command line fine?

## Deliverables

- Project structure with a README and a single command to train and to evaluate.
- Training and evaluation scripts, saved checkpoints, and logs.
- An improvement chart and a short results write-up (what improved, by how much, what did not work).
- Tests for board encoding, move masking, and the evaluation harness.

## Working rules

- Verify improvement with numbers from the evaluation harness before claiming it.
- Report failures and negative results plainly.
- Ask before downloading large datasets or installing heavy dependencies.
