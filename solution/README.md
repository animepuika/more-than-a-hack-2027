# FirstTrial: our first local experiment

This experiment asks whether handling missing ball data and avoiding brief team switches improves the official baseline. It does not train a machine-learning model or search for winning settings.

## Run it

Use Python 3.11 or newer. From the repository folder in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-first-trial.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m solution.run
```

The tactical ATD XML and TXT must be in `data/tactical/`. Alternatively use `--data-dir PATH`, where PATH contains the `tactical/` folder. Run `python -m solution.run --help` with the same environment for other options.

The first run writes predictions for all 199,319 frames, checks the official submission format, and scores the completed predictions against the public five-minute example. It writes everything under `data/runs/first_trial/`, which Git ignores.

Open `data/runs/first_trial/comparison.html` for the score table and possession timeline. Each method has a CSV under `predictions/` and a frame-by-frame explanation under `diagnostics/`. `manifest.json` records settings, package versions, and input/code/output hashes.

To generate another set without reading answer labels:

```powershell
.\.venv\Scripts\python.exe -m solution.run --out data/runs/first_trial_repeat --skip-score
```

## What we already have

| Input | Used now? | Purpose |
|---|---|---|
| Tactical ATD TXT | Yes | Ball/player positions and provided speeds, frame by frame |
| Tactical ATD XML | Yes | Team assignment of each tracking slot and frame rate |
| Published match metadata in `games.py` | Yes | Pitch dimensions, 25 fps, frame count, and half boundaries |
| Public possession labels | Evaluation only | Check the finished predictions on five minutes |
| Tactical homography | No | Relates pitch positions to video pixels; useful for visual checks |
| Tactical Smart Tagging | No | Possible event clues for a later experiment |
| Broadcast tracking/homography | No | Needs time alignment before it can supplement tactical data |
| Match video | No | Useful for reviewing mistakes; not required for this run |

The supplied track slots are anonymous track fragments. They are not 22 permanent player identities. Version 1 therefore uses team-level distances and robust speed summaries. It does not calculate named-player acceleration or guess identities.

## Four fixed comparisons

| Method | Main idea |
|---|---|
| `reference_smooth50` | Reproduce the official nearest-player baseline with its published smoothing |
| `gap_sequence` | Use clear nearby-team evidence; choose a sequence that penalizes rapid switches and carries evidence across missing observations |
| `gap_stop_sequence` | Add sustained out-of-bounds, stationary-ball-plus-slow-play, and slow-play-plus-long-gap clues |
| `gap_stop_sequence_x` | Replace only short internal missing-ball gaps with X when their surrounding decoded team is the same |

H = Barcelona, A = Athletic Club, D = dead/out of play, X = abstain. Missing ball data does not automatically mean D. Slow movement alone does not mean D. These stoppage clues are hypotheses that can be wrong, not confirmed fouls, cards, substitutions, or goals.

The sequence model uses minimum-cost decoding (Viterbi). A nearby team receives a lower cost; changing team incurs an extra cost. The costs are rules, not trained probabilities. Each half is decoded independently. Frames outside the published halves are X in our three methods. The reference is reproduced exactly.

X is an output policy, not a fourth possession state. For spell scoring the official evaluator fills X from previous non-X labels within its scoring blocks, including D; it also backfills leading X. Per-frame macro F1 counts a graded X as a miss. Adding X to an already smooth same-team gap therefore should not improve spell F1 and may reduce macro F1. We keep this comparison to measure that tradeoff explicitly.

## Where to change our code

| File | Responsibility |
|---|---|
| `config.json` | Declared distance, timing, speed, and switching settings |
| `data.py` | Strict streaming ATD reader; normalized coordinates converted to metres |
| `features.py` | Ball gaps, coverage, ball jumps, and potential stoppage clues |
| `predict.py` | Team evidence, sequence decoding, and optional X policy |
| `run.py` | Repeatable experiment, format checks, hashes, and official scorer invocation |
| `report.py` | Local score table and timeline, built after prediction is finished |
| `../tests/test_first_trial.py` | Checks using invented tracking examples |

Our implementation lives in `solution/`. The official starter, scorer, public labels, match configuration, and published baseline files stay unchanged. All supplied data and generated outputs stay in ignored `data/`. No command here commits or pushes.

## How to judge the result

1. Confirm frame alignment and valid output before comparing scores.
2. Compare all four results, including worse ones. The main metric is spell F1; macro F1 measures a different aspect.
3. Inspect the timeline and explanation files to find false stoppages, extra switches, and uncertain turnover times.
4. Review a few difficult clips if video is available and the challenge permits that use. Do not enter label-derived corrections into prediction code.
5. Improve one idea at a time and keep its result. Avoid repeatedly tuning against the same public five minutes.
6. Before broad model selection, obtain permitted additional labels or a separate validation period. Use time blocks or whole possessions, not randomly mixed neighboring frames.

Later comparisons could include a tree model with time context and a sequence decoder, or a temporal model if enough labels become available. Trying every model on this tiny sample would mostly test which one fits the example. Better evidence and validation come first.

Initial settings are engineering choices, not calibrated thresholds. Future observations are used, so this is an offline experiment. Timing of a turnover inside a fully missing interval remains uncertain. A slow missing-ball period can still be live play, and this may produce a false D. Scores on five minutes do not establish performance across the full match.
