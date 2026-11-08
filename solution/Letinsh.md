# Team Letinsh — technical document

**More than a Hack 2027, Phase 1: ball possession in FC Barcelona vs Athletic Club**

| | |
|---|---|
| Submission | `Letinsh.csv`, 199,319 rows, passes `evaluation/validate_submission.py` |
| File sha256 | `504f10eac5833806f96962139095b51f76eea83769c28a4a531b57e54a1b71b7` |
| Code | <https://github.com/animepuika/more-than-a-hack-2027/tree/FirstTrial> (folder `solution/`) |
| Rebuild | `python -m solution.make_submission --out data/submission` |
| Public five minutes | Spell F1 0.612 (official smooth125 baseline 0.424, smooth50 0.326), scored once after the build |

## 1. Approach

Every frame gets **H**, **A**, **D** or **X**. The method is a chain of small models and fixed rules:

1. Find who controls the ball.
2. Follow the team in possession through long stretches where the tracking loses the ball.
3. Mark dead ball from player formations and from the TV broadcast cutting away.
4. Remove the false possession changes that cost the most points.

All models are trained on **public practice matches**, never on this match: 3 Metrica sample games and 7 DFL Bundesliga games. Every tuned setting is chosen on those matches or on SkillCorner's open data. The BAR-ATH data is only the input the finished method runs on. Nothing is labelled by hand, and the public answers were never used for training or for choosing settings.

**Why this design.** The public sample has only 17 possessions, and the organisers' own baselines show that a method can win those five minutes and lose the full match. Hand-written rules reached at most 0.41 there. So we built a practice bench instead:

- 10 complete matches with their own possession labels, about 3,700 possessions;
- copies of these matches that hide the ball and players the way real tracking loses them;
- leave-one-match-out testing;
- pass rules written down before each experiment.

Of more than 30 logged experiments, changes were kept only when they passed their rules. Any override is recorded in the history.

## 2. The pipeline

| Step | What it does | Trained / set on |
|---|---|---|
| 1. Tracking features | Ball and team positions, speeds, distance of each team's nearest player to the ball, ball gaps, clear-control evidence. Fixed rules give a first dead-ball guess: ball out of the pitch, a still ball with slow play, or slow play during a long ball gap. | Hand-set rules |
| 2. Turnover model | Gradient-boosted trees estimate P(Barcelona has the ball) per frame. Inputs: which team is close to the ball now, 0.4 s and 1 s before and after, who touched it last and touches it next, sudden changes of ball direction, team shape. A two-state Viterbi decoder turns this into possessions, with a fixed 0.5 s switch cost. | 3 Metrica matches, recorded and with 25% of the live ball hidden |
| 3. Ball-free gap model | During ball gaps of 2 s or more, the team in possession is inferred from players only: how far each team has pushed up, its defensive line, width and their changes. It names the right team on 84–89% of live frames of an unseen match. The gap is re-decoded, anchored to the team before it and the first confirmed controller after it. | 3 Metrica matches |
| 4. Dead ball from formations | Recognises restart shapes without using the ball: kick-off (everyone in their own half), corner taker at the flag, goal kick (attackers out of the box), throw-in taker on the touchline, a still wall, players slowing together. It is only trusted where at least 21 players are tracked for a full second. | 10 practice matches; threshold fixed on them |
| 5. Dead ball from the broadcast | TV directors cut to replays and close-ups when play stops. While the broadcast frame has no valid pitch view, the tactical frame is dead. The first 3 s and last 6 s of each cut are dropped, and dead ball ends when the players' 1-second mean speed passes 2 m/s. | Trims chosen on 20 SkillCorner matches |
| 6. Cleanup | A possession shorter than 1 s goes back to the team that had the ball. A dead piece shorter than 3 s inside one team's possession is removed. Dead stretches less than 4 s apart are joined. | Picked on practice copies, judged on separate copies |

Measured on practice matches the model had not seen:

- **Turnover model (3 Metrica matches):** Spell F1 0.574 → 0.511 as hidden live ball rises from 0% to 40%, against 0.415 → 0.297 for the official smooth50 baseline.
- **Dead-ball threshold:** finds 80% of dead ball at 93% precision, with 2.0% of clearly live play called dead.
- **Cleanup:** Spell F1 0.478 → 0.523, better in every bootstrap resample.

## 3. Possession rules we implemented

- **Possession** belongs to the team in control of the ball. Training labels: Metrica's event logs give the team of the latest controlling event (pass, carry, recovery, set piece, shot). The DFL provider gives possession per frame.
- **Touches and challenges.** A touch without control (a block, a deflection, a challenge that does not win the ball) is not a change of possession. On DFL, opposing touches shorter than 1 s inside a possession are relabelled to the team in possession. In the output, any possession shorter than 1 s is returned to the team that had the ball. In Metrica's labels, challenges, fouls and ball-outs never change the owner.
- **Loose balls.** The ball stays with the last team in control until the other team clearly controls it. The decoder charges a cost for every team switch, so a brief approach by an opponent does not flip possession.
- **Ball not tracked.** A missing ball is not treated as dead ball. In gaps of 2 s or more, the team is followed from player positions alone (step 3).
- **Restarts and stoppages.** Dead ball (**D**) runs from the stoppage until the restart:
  - recognised from restart formations, from broadcast cutaways, or from the ball leaving the pitch or lying still with slow play;
  - a cutaway's dead ball ends when the players speed up again;
  - a dead piece shorter than 3 s inside one team's possession is treated as noise.
  - Throw-ins, goal kicks, corners, free kicks and kick-offs are all **D** until the ball is played.
- **X** only outside the two halves. Inside the halves every frame gets H, A or D.

## 4. Assets used and dropped

**Used in the submission**

| Asset | How |
|---|---|
| Tactical tracking (ATD) | Main input: player and ball positions, team of each track |
| Broadcast homography | Only to detect when the broadcast shows no pitch view (step 5) |
| Smart Tagging | Not an input. Used only for automatic checks that need no answers (e.g. how many tagged set pieces our dead ball covers) |
| [Metrica sample data](https://github.com/metrica-sports/sample-data), 3 matches | Training possession and dead-ball models. Credited as the source asks |
| [DFL open data](https://springernature.figshare.com/articles/dataset/28196177), 7 matches (Bassek et al., 2025, *Scientific Data*, CC BY 4.0) | Training the dead-ball model; choosing its threshold |
| [SkillCorner open data](https://github.com/SkillCorner/opendata), 20 matches (MIT, © SkillCorner) | Ball-gap lengths for the practice copies; the cutaway trims |
| Python 3.13, numpy, pandas, scikit-learn, scipy | Everything above; versions pinned in `requirements-submission.txt` |

The match video and the tactical homography are **not** used by the submission.

**Tried and dropped** (details and numbers in `solution/HISTORY.md`):

- **Referee whistle from the broadcast sound:** a tone-pair detector, but its alignment with dead ball was only 1.4× chance.
- **Commentary speech recognition (Whisper):** most of the audio was not clear speech, and its cues failed checks in each half.
- **YOLO ball detection in the video** to fill tracking gaps:
  - adding recovered positions did not reliably improve possessions;
  - simulating partial recovery on practice matches showed that position errors of 1 m erase most of the gain;
  - a public ball detector, used as published, was within 1 m of the tracked ball in only 38% of frames.
- **Throw-in / goal-kick detector:** practice gain +0.01, but it failed its precision rules.
- **Smaller ideas, each too small or harmful:** a high-ball detector, filling tracking blind spots, a crowd/fight rule, touch-based switch timing, adapting the dead-ball model to this match, and more SkillCorner training for possession.

## 5. How we validated

- **Practice bench:** 10 full matches (3 Metrica, 7 DFL), always scored with the official Spell F1 on a match the model did not see.
- **Realistic copies:** the bench uses copies that hide the ball and drop players, because this match's tracking loses the ball in about 41% of frames.
- **Pass rules written first:** a minimum gain, wins in most matches, and a whole-match bootstrap. Failed experiments were kept in the log, not retried with new settings.
- **Automatic checks on BAR-ATH that need no answers:**
  - share of clearly live frames (a player controlling a ball moving faster than 2 m/s) called dead;
  - dead-ball share;
  - possession changes per live minute;
  - coverage of tagged set pieces.

  These detect bugs; they do not tune anything.
- **Public five minutes:** scored only after a version was frozen, as a sanity check.

## 6. Limitations

- **Short possessions are the main loss.** On practice matches, 75% of the possessions we miss last under 5 s, and possessions under 2 s are found only about 11% of the time. They are missed even with complete tracking, so better input alone does not fix them.
- **Too little dead ball:** we call 38% of the match dead; real matches have 40–50%. Throw-ins and goal kicks are hardest, especially when the broadcast stays on the pitch.
- **False dead ball:** about 6% of clearly live frames are called dead. This comes from the dead-ball threshold, which is fixed on practice matches and not adjusted to this match.
- **Different practice data:** Metrica's ball vanishes and DFL's ball freezes at stoppages, so the dead-ball model cannot learn from the ball; it uses formations only. Provider labels also define possession slightly differently from this challenge.
- **The full-match score is unknown.** The public sample is 17 possessions; differences below about 0.04 there are noise.
- **A stronger possession model that did not transfer.** We tested a bidirectional GRU that reads the whole sequence and was trained on all 10 practice matches (V34/V35). On unseen practice matches it raised Spell F1 from 0.566 to 0.638 and won all 10. On this match, however, it failed a check written before the build: on frames where one player clearly controls a moving ball, it agreed with that player's team 85.5% of the time, against 90.6% for the submitted method. The practice copies hide the ball only in short gaps, while this match's tracking often loses it for much longer, which is the likely cause. We kept the submitted method.
- **With more time:** practice copies with a broad range of ball-gap lengths so the sequence model can be used safely, better throw-in and goal-kick detection, and more practice matches with broadcast video.

## 7. Reproduction

1. Python 3.13: `python -m venv .venv`, then `pip install -r requirements-submission.txt`.
2. Data, kept outside Git under `data/`:
   - the challenge bundle (tactical tracking in `data/tactical/`, broadcast homography in `data/broadcast/`);
   - Metrica sample data in `data/external/metrica-sample-data/`;
   - DFL open data in `data/external/dfl-open/`.
3. Run `python -m solution.make_submission --out data/submission`. It takes about 30–60 minutes on a laptop:
   - retrains every model with fixed seeds;
   - writes `data/submission/Letinsh.csv`;
   - checks it with the official validator.
4. The settings measured on external data are stored in `solution/config_external.json`. To measure them again: download SkillCorner's open data and run `python -m solution.external_settings`, then `python -m solution.external_settings threshold`.

**Development note.** The code was written with the help of AI coding assistants (Claude Code and ChatGPT). No language model is part of the pipeline, and no labels were produced by hand or by an AI. Every label comes from the code above.
