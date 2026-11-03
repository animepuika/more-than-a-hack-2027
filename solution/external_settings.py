"""Settings measured on outside data only (SkillCorner open data), for the rules of 1 November 2026.

The organisers' clarification: every trained or tuned setting must come from external data, never
from the provided game. Earlier versions measured three things on BAR-ATH's own tracking (no
answers, but still the provided game). This module re-measures them on SkillCorner's 20 open
A-League matches (MIT, (c) SkillCorner), whose broadcast tracking loses the ball and cuts away
from the pitch like BAR-ATH's video does. Definitions, fixed before measuring:

- Camera on the pitch: at least one player detected. Cutaway: no player detected.
- Ball gap lengths: runs of frames without a detected ball, inside labelled (non-X) camera-on
  stretches of one period, at 25 fps.
- Ball hidden shares: the share of live (H/A) and of dead (D) camera-on frames without a ball.
- Cutaway trims: from start trims 0-6 s and end trims 0-10 s (whole seconds; at least 2 s must
  remain, as before), keep the pairs where at most 2% of the kept cutaway frames are live (the
  same 2% false-dead budget used for dead ball elsewhere). Of those, pick the pair that keeps
  the most dead frames. Ties go to the smaller total trim.

Also fixed before measuring, on the 10 practice games (3 Metrica, 7 DFL, their own labels):
- Dead-ball start threshold: the V16 formation model is trained on 9 games as recorded (players
  complete; its inputs never use the ball) and predicts the 10th, for all 10. The threshold is
  raised, with calibrated_dead_mask unchanged, until at most 2% of the pooled frames that are
  truly live with the ball moving faster than 2 m/s are called dead. This replaces the old
  calibration on BAR-ATH's own tracking.

    python -m solution.external_settings            # SkillCorner settings
    python -m solution.external_settings threshold  # practice-game threshold
Both write into solution/config_external.json.
"""
import json
from pathlib import Path

import numpy as np

from .camera_cutaway import MIN_S
from .features import runs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'solution/config_external.json'
START_GRID, END_GRID, LIVE_BUDGET = range(0, 7), range(0, 11), 0.02


def load():
    """Saved settings: gap lengths (frames) and hidden shares for masking, and cutaway trims."""
    return json.loads(OUT.read_text())


def trim(mask, fps, start_s, end_s, min_s=MIN_S):
    out = np.zeros(len(mask), bool)
    for a, b in runs(mask):
        a2, b2 = a + int(round(start_s * fps)), b - int(round(end_s * fps))
        if b2 - a2 >= min_s * fps:
            out[a2:b2] = True
    return out


def measure():
    from .skillcorner_open import load_skillcorner_matches
    lengths, live, dead = [], np.zeros(2), np.zeros(2)
    grid = {(s, e): np.zeros(2) for s in START_GRID for e in END_GRID}
    names = []
    for sc in load_skillcorner_matches():
        m, labels = sc.match, np.asarray(sc.labels)
        fps = int(m.fps)
        names.append(m.name)
        keep = labels != 'X'
        camera = np.isfinite(m.player_xy_m[..., 0]).sum(axis=1) > 0
        no_ball = ~np.isfinite(m.ball_xy_m).all(axis=1)
        on = keep & camera
        for a, b in runs(on):
            for c, d in runs(no_ball[a:b]):
                lengths.append(int(d - c))
        is_live, is_dead = np.isin(labels, ['H', 'A']) & on, (labels == 'D') & on
        live += [(no_ball & is_live).sum(), is_live.sum()]
        dead += [(no_ball & is_dead).sum(), is_dead.sum()]
        cut = keep & ~camera
        for (s, e), totals in grid.items():
            kept = trim(cut, fps, s, e)
            totals += [(kept & np.isin(labels, ['H', 'A'])).sum(), (kept & (labels == 'D')).sum()]
        print('measured', m.name, flush=True)
    table = {f'{s},{e}': {'live_frames': int(v[0]), 'dead_frames': int(v[1]),
                          'live_share': float(v[0] / max(v.sum(), 1))} for (s, e), v in grid.items()}
    allowed = [(s, e) for (s, e), v in grid.items() if v.sum() and v[0] / v.sum() <= LIVE_BUDGET]
    start, end = max(allowed, key=lambda k: (grid[k][1], -(k[0] + k[1])))
    settings = {
        'source': 'SkillCorner open data, 20 A-League 2024/25 matches (MIT, (c) SkillCorner)',
        'matches': names,
        'ball_gap_lengths_frames': sorted(lengths),
        'ball_hidden_share_live': float(live[0] / live[1]),
        'ball_hidden_share_dead': float(dead[0] / dead[1]),
        'cutaway_trim_start_s': float(start), 'cutaway_trim_end_s': float(end),
        'cutaway_trim_rule': 'max dead frames kept with live share <= 2%; ties to the smaller total trim',
        'cutaway_trim_table': table,
    }
    OUT.write_text(json.dumps(settings) + '\n')
    print(json.dumps({k: v for k, v in settings.items() if k not in ('ball_gap_lengths_frames', 'cutaway_trim_table')}
                     | {'gaps': len(lengths), 'median_gap_s': float(np.median(lengths)) / 25}, indent=2))


def practice_sources():
    """The 10 practice games as (name, match, labels), Metrica first, as in run_bench10."""
    from .run_bench10 import load_sources
    return [(s['name'], s['match'], np.asarray(s['labels'])) for s in load_sources()]


def dead_model(tables, labels):
    from .deadball_formation import fit
    from .run_player_dropout import subset
    rows = [subset(t, l, ['H', 'A', 'D']) for t, l in zip(tables, labels)]
    return fit([t for t, _ in rows], [l for _, l in rows], stride=1)


def threshold():
    from .deadball_formation import calibrated_dead_mask, dead_probability
    from .deadball_v16 import dead_features
    MOVING, BUDGET = 2.0, 0.02
    quality = json.loads((ROOT/'solution/config_v7.json').read_text())
    games = practice_sources()
    tables = [dead_features(m, ball=False) for _, m, _ in games]
    probs, periods, live = [], [], []
    cache = ROOT/'data/runs/external_threshold'
    cache.mkdir(parents=True, exist_ok=True)
    for i, (name, m, labels) in enumerate(games):
        saved = cache/f'{name}.npy'  # held-out probabilities, so a rerun skips the training
        active = m.period > 0
        if saved.exists():
            p = np.load(saved)
        else:
            model = dead_model([t for j, t in enumerate(tables) if j != i], [g[2] for j, g in enumerate(games) if j != i])
            p = np.zeros(len(labels))
            p[active] = dead_probability(model, tables[i][active])
            np.save(saved, p)
        ball = np.asarray(m.ball_xy_m, float)
        speed = np.full(len(ball), np.nan)
        speed[1:-1] = np.linalg.norm(ball[2:] - ball[:-2], axis=1) * m.fps / 2
        moving = np.isin(labels, ['H', 'A']) & (np.nan_to_num(speed) > MOVING) & active
        probs += [p, np.zeros(1)]; periods += [m.period, np.zeros(1, int)]; live += [moving, np.zeros(1, bool)]
        print('held out', name, int(moving.sum()), flush=True)
    mask, start = calibrated_dead_mask(np.concatenate(probs), np.concatenate(periods), quality, np.concatenate(live), BUDGET)
    truth = np.concatenate([np.concatenate([g[2], ['X']]) for g in games])
    scored = np.isin(truth, ['H', 'A', 'D'])
    settings = load()
    settings.update({'dead_start_threshold': float(start), 'dead_threshold_rule':
                     'leave-one-game-out V16 on 10 practice games; <= 2% of truly-live moving-ball frames dead',
                     'dead_threshold_check': {'false_dead_on_moving_live': float(mask[np.concatenate(live)].mean()),
                                              'dead_recall': float(mask[truth == 'D'].mean()),
                                              'dead_precision': float((truth[mask & scored] == 'D').mean())}})
    OUT.write_text(json.dumps(settings) + '\n')
    print(json.dumps({k: settings[k] for k in ('dead_start_threshold', 'dead_threshold_check')}, indent=2))


if __name__ == '__main__':
    import sys
    threshold() if sys.argv[1:] == ['threshold'] else measure()
