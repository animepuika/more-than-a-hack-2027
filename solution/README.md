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

## Second experiment: team shape and movement

The first example and its aggregate scores are preserved in commit `7151778` and `results/first_trial_v1.json`. Run the second comparison with:

```powershell
.\.venv\Scripts\python.exe -m solution.run_v2
```

It also needs `data/tactical/BAR-ATH_tactical_smart_tagging.xml`. Outputs go to the separate ignored folder `data/runs/second_trial/`, leaving the first run intact. Use `--skip-score --out data/runs/second_trial_repeat` for a prediction-only repeat. The first experiment's command and settings still work.

| New feature | How we use it | Limitation |
|---|---|---|
| Team hull area | Area covered by each team's detected positions | Includes the goalkeeper and depends on tracking coverage |
| Team width, length, center, spread | Describe whether a team stretches or contracts | A wide shape does not automatically mean possession |
| Smoothed velocity | Position change over 0.04 seconds, smoothed over five velocity samples | Only a short continuous track fragment is trusted |
| Acceleration and braking | Changes in smoothed speed; fractions of tracks above/below burst thresholds | Undetected identity changes and noise remain possible |
| Movement toward a common area | Count tracks from both sides approaching a short-gap ball estimate | This can indicate a press or duel, not necessarily the owner |
| Automatic phase tags | Small possession priors from attacking/defending phases | These are automatic annotations, not manual confirmed events |

`context.py` computes these features and weak team prototypes. We learn what each team's shape/movement looks like from clear visible-ball proximity examples in that same half. Those proxy labels can be wrong. We sample at one frame per second and require enough examples from both teams for each feature; otherwise that feature contributes nothing. This is a small fitted classifier using noisy tracking-derived labels, not supervised training on the public answers.

`config_v2.json` declares the new settings before scoring. The six comparisons are the official baseline, the unchanged v1 approach, shape only, motion only, shape plus motion, and shape plus motion plus automatic tags. Extra clues are weak costs inside the same sequence decoder, not hard team overrides. The public possession labels are read only after all predictions and hashes are written.

Geometry removes very close duplicate detections and requires enough tracks from both teams. Movement resets at missing samples, implausible speed jumps, and half boundaries. A slot can still change identity without an obvious jump, so no named-player acceleration or role is claimed. Ball interpolation covers only short internal gaps with usable observations on both sides, plausible endpoint travel speed, and no confirmed stoppage clue. It does not recover the actual ball path.

Defending tags name the defending team, so their possession prior is inverted. Conflicting team votes cancel. Unknown teams, set-piece intervals, and goal intervals are excluded from team priors. We do not invent a manual foul/card/substitution log, and we do not label a whole restart interval D. The basic stoppage rules remain unchanged so we can measure the new team features separately.

The HTML report now includes percentage frame accuracy as well as F1 scores. `diagnostics/context_features.csv` records the collective features and costs locally. The second comparison reuses the same public sample seen in v1: it is exploratory evidence, not an independent validation result. Broadcast fusion, video review, named-player roles, and reliable event-based stoppage timing remain future work.

### Measured result on the public five minutes

| Method | Frame accuracy | Spell F1 | Macro F1 |
|---|---:|---:|---:|
| Official reference | 71.32% | 0.3256 | 0.6207 |
| V1 with stoppage clues | 71.20% | 0.3238 | 0.5642 |
| V2 shape | 73.61% | 0.3273 | 0.5806 |
| V2 movement | 69.28% | 0.3289 | 0.5475 |
| V2 shape + movement | 73.63% | 0.3273 | 0.5807 |
| V2 shape + movement + automatic tags | 71.71% | 0.2800 | 0.5688 |

Shape accounts for nearly all the observed frame-accuracy gain. Adding movement to shape gets only one extra frame correct out of 7,500; that is not evidence that the combination is better. The movement-only variant has the highest observed spell F1, while its frame accuracy is worse. Automatic tags produce more possession switches and a lower spell score in this implementation.

The official evaluator's 1,000-draw bootstrap gives broad overlapping spell-score ranges: approximately 0.209–0.455 for shape plus movement, 0.200–0.448 for v1, and 0.214–0.447 for the reference. These are not independent validation results or proof of a full-match improvement. Missed stoppages remain a major weakness, and the reference still has the highest macro F1.

All 31 behavior checks passed. A complete prediction-only repeat produced identical bytes and matching hashes for all six methods. The v1 and reference predictions also match their original files exactly. The local `reproducibility.json` and `uncertainty/scoreboard.csv` preserve those checks. V2 code is on `FirstTrial`; generated match outputs remain local.

## Third experiment: dead-ball candidates and spell duration

Run the fixed v3 comparison with:

```powershell
.\.venv\Scripts\python.exe -m solution.run_v3
```

It uses the tactical ATD and automatic Smart Tagging files. It adds two hypotheses: sustained low movement plus a stationary or missing ball can mark a dead-ball candidate, and minimum-duration decoding can prevent very short state changes. Set-piece tags only support a candidate when the tracking clues also agree. Each candidate is tested as a soft cost or a forced `D`, with and without the minimum-duration decoder. No public answer labels are read until every prediction has been written and hashed.

| Method | Frame accuracy | Spell F1 | Macro F1 | Predicted possessions |
|---|---:|---:|---:|---:|
| Official reference | 71.32% | 0.3256 | 0.6207 | 26 |
| V2 shape | 73.61% | 0.3273 | 0.5806 | 27 |
| V2 shape + movement | 73.63% | 0.3273 | 0.5807 | 27 |
| V3 soft dead-ball cost | 80.80% | 0.2979 | 0.7644 | 30 |
| V3 soft cost + duration | 80.75% | 0.2979 | 0.7638 | 30 |
| V3 forced dead-ball label | 81.96% | 0.2857 | 0.7926 | 32 |
| V3 forced label + duration | 81.95% | 0.2857 | 0.7925 | 32 |
| V3 soft cost + set-piece support | 80.72% | 0.3083 | 0.7635 | 31 |
| V3 soft + set-piece + duration | 80.61% | 0.3083 | 0.7624 | 31 |
| V3 forced label + set-piece support | 81.45% | 0.2745 | 0.7867 | 34 |
| V3 forced + set-piece + duration | 81.44% | 0.2745 | 0.7866 | 34 |

The highest v3 spell score is still below the v2 shape and movement result and the official reference. Every result uses the same small public example of 17 possessions. The 90% spell-score bootstrap ranges overlap broadly, so this run does not show that any one version is the full-match winner.

The new stoppage rules raised dead-ball recall from 7.3% in v2 shape plus movement to 45.3% with soft costs and 58.3% with forced labels. Forced labels also reduced dead-ball precision from 96.9% to 87.3%. The v3 variants predicted 30–34 possessions, compared with 17 in the answer, which points to extra breaks or team switches. This improves dead-ball frame detection while hurting possession-spell agreement.

The tested duration limits were 0.32 seconds for team states and 0.40 seconds for dead-ball states. They changed only 1–8 of the 7,500 graded frames and did not change spell F1 or the possession count. These settings add little beyond the existing sequence smoothing. Automatic set-piece support slightly raised the soft version's spell score, but it stayed below the reference and its bootstrap range overlaps the other methods.

All 40 behavior checks passed. A prediction-only repeat produced identical bytes for all 11 methods, and the v2 shape, v2 shape plus movement, and official reference outputs exactly match their saved versions. Full outputs, frame diagnostics, and the comparison page are in ignored `data/runs/third_trial/`; the safe aggregate score export is `results/first_trial_v3.json`. V3 is on `FirstTrial` in commit `90544bd`; generated match outputs remain local.

## Fourth experiment: player reach physics and an HMM

Install the added HMM dependency and run V4 with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-v4.txt
.\.venv\Scripts\python.exe -m solution.run_v4
```

V4 tests two ideas: **who could reach the ball soonest?** and **can a Hidden Markov Model (HMM) smooth frame-by-frame evidence?** It compares seven fixed methods, including the official reference, V2, and V3 results. It does not fit model settings to the public answer labels.

### Physics reach estimate

For each tracked player, the model uses their current position and smoothed movement direction. It lets them continue for a 0.7 second reaction time, then accelerate toward the ball at up to 7 m/s² until a 5 m/s cap. A logistic curve turns estimated arrival time into a chance of reaching the target within 1 second. Individual player chances are combined into relative Barcelona/Athletic reach scores, which enter the possession decoder only as a weak extra cost.

This is **inspired by Spearman's pitch-control model**, not a full reproduction. The current version leaves out ball-flight drag and trajectory choice, offside, individual player attributes, and full pitch-control integration. It also needs a ball coordinate to calculate reach. The tracking file had ball coordinates on 56.9% of frames; safe interpolation filled only 41 additional missing-ball frames. So this version cannot resolve the long gaps that make up most hard cases.

### HMM smoother

The HMM reads the raw Home/Away/Dead cost winner and its confidence level. Its states are H, A, and D. The transition settings favor staying in the same state, and the emission settings describe how much to trust strong or weak observations. We set those values by hand; we did not train them on the 17 public possession spells. Each half is decoded separately. In this run, the HMM made more switches than the V2 sequence decoder.

### Measured result on the public five minutes

| Method | Frame accuracy | Spell F1 | 5th–95th percentile range | Predicted possessions |
|---|---:|---:|---:|---:|
| Official reference | 71.32% | 0.3256 | 0.214–0.447 | 26 |
| V2 shape + movement | 73.63% | **0.3273** | 0.209–0.455 | 27 |
| V3 soft cost + set-piece support | 80.72% | 0.3083 | 0.200–0.417 | 31 |
| V4 pitch-control reach | 69.33% | 0.1517 | 0.097–0.214 | 41 |
| V4 HMM | 73.24% | 0.1949 | 0.123–0.267 | 61 |
| V4 pitch control + HMM | 67.31% | 0.0379 | 0.017–0.062 | 99 |
| V4 tagged stoppage + HMM | 80.13% | 0.1928 | 0.125–0.260 | 66 |

V2 shape + movement remains the strongest tested Spell F1 method at 0.3273. Every new V4 variant produced more predicted possession periods than the 17 in the public labels. These settings are too switchy for the main metric. Pitch control alone lowered both frame accuracy and Spell F1. Combining it with this HMM was worse still. Do not add either to the main approach yet.

The public-example bootstrap ranges are wide and overlap. The sample has only 17 labelled spells and has already been used in V1–V4, so it cannot choose a full-match winner. The next useful step is to review specific wrong switches, then validate any new settings on separate labelled possessions.

All 48 behavior tests passed, and a prediction-only repeat produced byte-identical files for all seven methods. The full run, scores, frame explanations, and comparison page are under ignored `data/runs/fourth_trial/`; the prediction-only repeat is in `data/runs/fourth_trial_repeat/`. The aggregate score export is `results/first_trial_v4.json`. The generated match outputs stay local; the repository contains the reusable source, settings, tests, and aggregate scores.

## Fifth experiment: abstain on calm missing-ball frames

Run the fixed V5 comparison with:

```powershell
.\.venv\Scripts\python.exe -m solution.run_v5
```

V5 tests whether writing `X` during missing-ball frames can keep the official spell scorer from treating uncertain guesses as possession changes. It compares abstention on every missing-ball frame with abstention only when player movement does not look like a scramble. The scramble signal is a proxy from anonymous track speeds and acceleration; it does not prove that a tackle, turnover, card, or injury happened.

### Measured result on the public five minutes

| Method | Accuracy | Spell F1 | Macro F1 | Predicted possessions |
|---|---:|---:|---:|---:|
| Official reference | 71.32% | 0.3256 | 0.6207 | 26 |
| V2 shape + movement | 73.63% | **0.3273** | 0.5807 | 27 |
| V3 tagged soft stoppage | 80.72% | 0.3083 | **0.7635** | 31 |
| V5 V2 + calm-gap X | 70.33% | 0.3273 | 0.5831 | 27 |
| V5 V3 + calm-gap X | 77.43% | 0.3083 | 0.7465 | 31 |

The V5 `X` policies did not raise Spell F1. Calm-gap `X` ties V2 on the public sample, while the tagged V3 parent remains below V2 on the ranking metric. We should keep V2 as the current public-sample Spell F1 leader; the test does not establish a full-match winner.

### Dead-ball audit

The user's review correctly pointed to dead-ball detection as a major gap. Across the 1,469 public `D` frames, V2 marked only 107 as dead: **7.3% recall**, with no false `D` frames. V3 tagged soft stoppage raised this to 666 correct `D` frames: **45.3% recall and 96.1% precision**, but still missed 803 dead-ball frames and added 27 false `D` frames. V5's abstention policies leave these dead-ball predictions unchanged.

This is the next modeling priority. First find whether play is live or stopped; only then decide which team has possession. Strong clues include a ball tracked outside the pitch, sustained ball stillness, collective movement slowing or gathering, automatic restart/goal tags, and a clear restart movement pattern. Low player speed alone is not enough: teams can move slowly while the ball is live. Use a time-aware state change with a short confirmation period and different thresholds for entering and leaving a stoppage, so one noisy frame cannot split a possession.

Cards, fights, and injuries can explain why a pause happens, but the scored label only needs to say whether the ball is in play. Coordinates can suggest a disruption (for example, a group clustering around one point or one player stopping while others gather), but they cannot reliably identify the cause. The supplied BAR-ATH Smart Tagging has 638 automatic tactical instances, including set-piece and goal tags; it is not a complete manual foul, card, or injury log. The tactical video can help review examples, while the current V5 algorithm uses tracking and Smart Tagging rather than visual video recognition.

### Candidate external training data

The competition's [rules](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md) permit external data when organisers can inspect it and the technical document explains it. They prohibit private full-match ground truth and any other source of corrected possession labels for this BAR-ATH fixture. A promising generic training source is Metrica's [public sample-data repository](https://github.com/metrica-sports/sample-data): it contains three anonymized matches with synchronized tracking and event data on a 105 × 68 m pitch. Event intervals can provide weak examples of live possession and ball-out/restart periods. This is a candidate for training generic movement/stoppage patterns, not a source of extra BAR-ATH answers. Its README asks users to acknowledge Metrica for public work.

The external data was **not used in the V5 scores**. The local environment could not resolve GitHub to download it, so no training result is claimed. Before adding it, we should test across separate sample matches and record the source and preprocessing so the result remains reproducible.

All 54 behavior tests passed, all seven outputs passed the official file-format validation, and a prediction-only repeat produced byte-identical files for all seven methods. Aggregate scores are in `results/first_trial_v5.json`; frame diagnostics and the comparison page are under ignored `data/runs/fifth_trial/`. No match data or generated frame-level outputs belong in the public repository.
