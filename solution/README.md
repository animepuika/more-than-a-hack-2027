# Team Letinsh — possession for Barcelona vs Athletic Club

We label every one of the 199,319 frames of the tactical video as **H** (Barcelona has the ball), **A** (Athletic), **D** (dead ball) or **X** (outside the two halves). Everything is produced by code: nothing is hand-labelled, and the hidden full-match answers are never used. Following the organisers' rule clarification of November 2026, every trained or tuned setting comes from external data; our match is only the input the finished method runs on.

## Start here

| | |
|---|---|
| **Submission** | `Letinsh.csv`, made by **V33** (V20 + cleanup, all settings from external data) |
| **Rebuild it** | `python -m solution.make_submission --out data/submission` |
| **Expected file** | sha256 `504f10eac5833806f96962139095b51f76eea83769c28a4a531b57e54a1b71b7` |
| **Public five minutes** | Spell F1 **0.612** (official smooth125 baseline: 0.424). Scored once after the build, for information only |

The public sample is only 17 possessions, so differences of a few hundredths there are within noise. The full-match score is unknown until the final ranking.

## Rebuild the submission

1. **Python 3.13** (the verified build used 3.13.1 on Windows):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements-submission.txt
   ```

2. **Data**, all kept under the ignored `data/` folder, never in Git:
   - The challenge bundle for BAR-ATH: the tactical tracking (`data/tactical/`) and the broadcast homography (`data/broadcast/`). See `games.py`; `MTAH_DATA_DIR` can point elsewhere.
   - [Metrica sample data](https://github.com/metrica-sports/sample-data), 3 matches, in `data/external/metrica-sample-data/`.
   - [DFL open data](https://springernature.figshare.com/articles/dataset/28196177), 7 matches, in `data/external/dfl-open/` (about 2.5 GB).

3. **Run:**

   ```powershell
   .\.venv\Scripts\python.exe -m solution.make_submission --out data/submission
   ```

   This takes about 30–60 minutes on a laptop. It retrains every model from the practice data with fixed seeds, writes `data/submission/Letinsh.csv`, and checks the format with the official validator. The settings measured on external data are stored in `solution/config_external.json`. To measure them again, download SkillCorner's open data (below) and run `python -m solution.external_settings`, then `python -m solution.external_settings threshold`.

## How it works

1. **Features from tracking:** where the ball and both teams are, how fast they move, and who clearly controls the ball. Fixed hand-written rules give a first dead-ball guess.
2. **Who has the ball:** a model trained on 3 Metrica matches spots touches and turnovers. A second model guesses the team during gaps where the ball is not tracked (41% of frames), from the players' movement alone.
3. **Dead ball from player formations:** a model trained on 10 practice matches (Metrica + DFL) recognises restart shapes such as corners, throw-ins, kick-offs and free-kick walls. It is only trusted where at least 21 players are tracked. Its threshold is fixed on the practice matches: tested on each match after training on the other nine, at most 2% of truly live play with a moving ball is called dead.
4. **Dead ball from the TV broadcast:** directors cut to close-ups and replays when play stops. While the broadcast shows no pitch view, the frame is marked dead. The first 3 s and last 6 s of each cut are left out (the values SkillCorner's 20 open matches give), and play restarts when the players speed up again.
5. **Cleanup:** possession flips shorter than 1 s go back to the team that had the ball (a touch without control is not a change of possession). Dead-ball pieces shorter than 3 s inside one team's possession are removed, and dead-ball stretches less than 4 s apart are joined.

Every rule and setting was fixed before testing, on practice matches or by hand. The practice matches are also blanked out like real tracking (ball gaps taken from SkillCorner) so the models learn to cope with a missing ball. Automatic checks of our match that need no answers only look for bugs. The public five minutes were scored last, for information.

## Results

| Version | Public 5 min | Practice matches | Status |
|---|---:|---|---|
| Official smooth125 baseline | 0.424 | | Reference |
| V20 (dead ball from the broadcast) | 0.533 | | Earlier candidate |
| V20 + cleanup | 0.574 | Cleanup: +0.044 on separate practice test copies | Replaced: three settings were tuned on our match's own tracking |
| V20 + cleanup + video ball recovery | 0.581–0.595 | | Experimental, built on the replaced version; not used |
| **V33** (V20 + cleanup, external settings only) | **0.612** | Dead-ball threshold: 2% false dead, 80% of dead ball found, 93% precision on unseen practice matches | **Submission** |

**Where we still lose points** (practice matches): when the ball is clearly visible, the team is wrong in about 9% of frames. Most of these are within 2 s of a real possession change, where the switch is about half a second early or late. Restoring hidden ball and player positions on practice copies would gain about +0.08. That is why we tried recovering the ball from the video.

## Experimental: ball recovery from the video

A public ball detector (YOLO, used as published, never retrained on our video) searched the tactical video where the tracking lost the ball. It was built on the replaced V20 version and is not part of the submission. Details are in [HISTORY.md](HISTORY.md#experimental-video-recovery).

## Data used and credits

| Source | Used for | Licence |
|---|---|---|
| Challenge data: tactical tracking, broadcast homography, Smart Tagging | Inputs (the tags are used only for answer-free checks) | Challenge terms |
| [Metrica sample data](https://github.com/metrica-sports/sample-data) (3 matches) | Training possession and dead-ball models | See its repository (to be confirmed before submission) |
| [DFL open data](https://springernature.figshare.com/articles/dataset/28196177), Bassek et al. (2025), *Scientific Data* (7 matches) | Training the dead-ball model | CC BY 4.0 |
| [SkillCorner open data](https://github.com/SkillCorner/opendata) (20 A-League matches) | Ball-gap lengths for the training copies and the cutaway trims (`config_external.json`) | MIT, © SkillCorner |
| Roboflow sports ball detector (YOLOv8) | Experimental video recovery only | AGPL-3.0 notice; dataset CC BY 4.0 |

## Limitations

- The public sample is 17 possessions; the full-match score cannot be measured beforehand.
- We still call less dead ball than real matches have (about 38% against 40–50%). Throw-ins and goal kicks are the hardest.
- An answer-free check of our match finds 6.0% of clearly live frames called dead (the earlier version: 3.9%). This comes from the threshold now being fixed on practice matches instead of on our match. It was not adjusted, because that would again tune on our match.
- Practice data differs from our match: Metrica's ball vanishes and DFL's ball freezes at stoppages, so no model learns from ball behaviour after a stoppage.
- Tried and not used: the referee whistle and commentary audio, a throw-in/goal-kick detector, filling tracking blind spots, a high-ball detector, more SkillCorner training, and touch-based switch timing. Details are in [HISTORY.md](HISTORY.md).

## Repository layout

| Path | Contents |
|---|---|
| `solution/make_submission.py` | The one command that builds the submission |
| `solution/` | All method code; experiment runners are `run_v*.py` |
| `solution/results/` | Saved result summaries of each experiment |
| `solution/HISTORY.md` | Every experiment in order, with the results and decisions at the time |
| `tests/` | Unit tests: `python -m unittest discover -s tests` |
| `data/` (ignored) | Match data, practice data, runs and predictions; never committed |
