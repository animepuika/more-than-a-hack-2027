"""Check the mixed-crowd stoppage detector: bench (must not hurt) and BAR-ATH (answer-free).

bench   : precision of crowd frames against Metrica D labels, and V13 + crowd D against V13
          at all four hiding levels (the detector uses players only, so one mask per match)
bar-ath : crowd intervals, overlap with V13's dead ball, false alarms on frames where a player
          clearly controls a moving ball; the mask is saved for later combination
No settings are tuned here. Outputs stay under data/; no commit or push.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import possession_metrics as pm
from .crowd_detector import (CROWD_MAX_SPEED_MPS, CROWD_MIN_PER_TEAM, CROWD_MIN_PLAYERS, CROWD_MIN_S,
                             CROWD_RADIUS_M, crowd_dead_mask)
from .features import runs
from .run import ROOT, sha256
from .run_v14 import MOVING_BALL_MPS, overlay

SETTINGS = {'radius_m': CROWD_RADIUS_M, 'min_players': CROWD_MIN_PLAYERS, 'min_per_team': CROWD_MIN_PER_TEAM,
            'max_speed_mps': CROWD_MAX_SPEED_MPS, 'min_s': CROWD_MIN_S}


def intervals(mask, fps):
    spans = [(b - a) / fps for a, b in runs(mask)]
    return {'count': len(spans), 'total_s': float(sum(spans)), 'median_s': float(np.median(spans)) if spans else None}


def bench(args, progress):
    from .external_possession import attach_teams, possession_labels
    from .external_samples import DEFAULT_SAMPLE_ROOT, load_metrica_samples
    from .run_external import pooled
    from .run_gap_model import LEVELS, paired_difference

    matches = attach_teams(load_metrica_samples(DEFAULT_SAMPLE_ROOT), DEFAULT_SAMPLE_ROOT)
    names = [m.name for m in matches]
    labels = {m.name: possession_labels(m.events, m.period)[0] for m in matches}
    crowd, per_match = {}, {}
    for m in matches:
        crowd[m.name] = crowd_dead_mask(m.player_xy_m, m.home_slots, m.period, m.pitch, m.fps)
        graded = labels[m.name] != 'X'
        hit = crowd[m.name] & graded
        per_match[m.name] = {'intervals': intervals(crowd[m.name], m.fps),
                             'precision_vs_D_labels': float((labels[m.name][hit] == 'D').mean()) if hit.any() else None}
        progress(f"{m.name}: {per_match[m.name]}")
    read = lambda path: pd.read_csv(path)['label'].to_numpy(dtype='<U1')
    results = {}
    for level in LEVELS:
        tag = 'live' + level.zfill(2)
        preds = {n: {'v13_combined': read(args.v13_bench/'predictions'/tag/n/'v13_combined.csv')} for n in names}
        for m in matches:
            preds[m.name]['v13_plus_crowd'] = overlay(preds[m.name]['v13_combined'], crowd[m.name], m.period > 0)
        gt = pooled(labels[n] for n in names)
        joined = {k: pooled(preds[n][k] for n in names) for k in ['v13_plus_crowd', 'v13_combined']}
        scores = {k: float(pm.evaluate(pm.label_codes(gt), pm.label_codes(v))['spell_f1_mean']) for k, v in joined.items()}
        results[level] = {'spell_f1': scores,
                          'crowd_minus_v13': paired_difference(gt, joined['v13_plus_crowd'], joined['v13_combined'])}
        progress(f"level {level}%: V13+crowd {scores['v13_plus_crowd']:.4f}  V13 {scores['v13_combined']:.4f}  "
                 f"diff {results[level]['crowd_minus_v13']['mean']:+.4f}")
    (args.out/'crowd_bench_results.json').write_text(json.dumps(
        {'created_utc': datetime.now(timezone.utc).isoformat(), 'settings': SETTINGS, 'per_match': per_match,
         'bench_results': results, 'source_sha256': sha256(ROOT/'solution/crowd_detector.py')}, indent=2)+'\n')


def bar_ath(args, progress):
    from games import GAME, GAMES, asset_paths, half_video_blocks
    from .data import load_tracking
    from .run_v6 import _period_keys

    n, fps, pitch = (GAMES[GAME][k] for k in ['n_video_frames', 'fps', 'pitch'])
    period, active = _period_keys(n, half_video_blocks(GAME))
    paths = asset_paths()
    tracking = load_tracking(paths['atd_meta'], paths['atd_raw'], n, fps, pitch, retain_tracks=True)
    crowd = crowd_dead_mask(tracking.track_xy_m, tracking.home_slots, period, pitch, fps) & active
    f = pd.read_csv(ROOT/'data/runs/seventh_trial/diagnostics/v7_features.csv',
                    usecols=['direct_control', 'ball_reliable', 'ball_speed_mps'])
    live_ref = active & (f.direct_control.to_numpy() >= 0) & (f.ball_reliable.to_numpy() == 1) \
        & (np.nan_to_num(f.ball_speed_mps.to_numpy()) > MOVING_BALL_MPS)
    v13 = pd.read_csv(ROOT/'data/runs/thirteenth_trial/predictions/v13_combined.csv')['label'].to_numpy(dtype='<U1')
    result = {'settings': SETTINGS, 'intervals': intervals(crowd, fps),
              'false_dead_on_clear_moving_ball': float(crowd[live_ref].mean()),
              'clear_moving_ball_frames_flagged': int((crowd & live_ref).sum()),
              'share_already_dead_in_v13': float((v13[crowd] == 'D').mean()) if crowd.any() else None,
              'new_dead_frames_vs_v13': int((crowd & (v13 != 'D')).sum())}
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out/'crowd_mask.npy', crowd)
    (args.out/'crowd_bar_ath_check.json').write_text(json.dumps(result, indent=2)+'\n')
    progress(json.dumps(result, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['bench', 'bar-ath'])
    parser.add_argument('--v13-bench', type=Path, default=ROOT/'data/runs/external_bench_v13')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    args.out = (args.out or ROOT/('data/runs/external_bench_crowd' if args.mode == 'bench' else 'data/runs/crowd_bar_ath')).resolve()
    if not args.out.is_relative_to((ROOT/'data').resolve()) or (args.out.exists() and any(args.out.iterdir())):
        raise ValueError('Choose a fresh output directory inside ignored data/')
    args.out.mkdir(parents=True, exist_ok=True)
    progress = lambda message: print(message, flush=True)
    (bench if args.mode == 'bench' else bar_ath)(args, progress)


if __name__ == '__main__':
    main()
