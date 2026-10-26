"""Experiment 3 on BAR-ATH: fill the gated dead-ball blind spots (solution/deadball_bridge.py).

Applies the bridging rule to the frozen candidates' dead ball (V16 gated + calibrated, and V17 which
uses the same dead ball), then runs the answer-free checks and the public sanity score. Bridging only
adds dead frames inside low-coverage stretches. No commit or push.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from .run import ROOT, sha256, write_labels


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'data/runs/nineteenth_trial')
    parser.add_argument('--skip-score', action='store_true')
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if not out.is_relative_to((ROOT/'data').resolve()) or (out.exists() and any(out.iterdir())):
        raise ValueError('Choose a fresh output directory inside ignored data/')
    (out/'predictions').mkdir(parents=True)
    progress = lambda message: print(message, flush=True)
    from games import GAME, GAMES, asset_paths, half_video_blocks
    from starter import load_smart_tagging
    from .data import load_tracking
    from .deadball_bridge import bridge_blind_spots
    from .deadball_formation import calibrated_dead_mask
    from .run_bench10 import GATE_PLAYERS
    from .run_v6 import _period_keys
    from .run_v9 import BASELINES, PUBLIC_WINDOW
    from .run_v12 import spell_stats
    from .run_v14 import overlay
    v7c = json.loads((ROOT/'solution/config_v7.json').read_text())
    n, fps, pitch = (GAMES[GAME][k] for k in ['n_video_frames', 'fps', 'pitch'])
    period, active = _period_keys(n, half_video_blocks(GAME))
    paths = asset_paths()
    tracking = load_tracking(paths['atd_meta'], paths['atd_raw'], n, fps, pitch)
    f = pd.read_csv(ROOT/'data/runs/seventh_trial/diagnostics/v7_features.csv',
                    usecols=['direct_control', 'ball_reliable', 'ball_speed_mps', 'players_observed'])
    live = active & (f.direct_control.to_numpy() >= 0) & (f.ball_reliable.to_numpy() == 1) \
        & (np.nan_to_num(f.ball_speed_mps.to_numpy()) > 2)
    players = f.players_observed.to_numpy()
    full = pd.Series(players >= GATE_PLAYERS).rolling(fps, center=True, min_periods=1).min().to_numpy().astype(bool)
    p16 = np.load(ROOT/'data/runs/sixteenth_trial/v16_p_dead.npy')
    gated = calibrated_dead_mask(np.where(full, p16, 0), period, v7c, live, 0.02, fps)[0]
    read = lambda path: pd.read_csv(path)['label'].to_numpy(dtype='<U1')
    candidates = {'v16_gated_cal': read(ROOT/'data/runs/sixteenth_trial_gated/predictions/v16_gated_cal.csv'),
                  'v17_mixed': read(ROOT/'data/runs/seventeenth_trial/predictions/v17_mixed.csv')}
    # Bridge on each candidate's final dead-ball decision (its D labels), the rule's intended input.
    labels = {**candidates, **{k + '_bridged': overlay(v, bridge_blind_spots(v == 'D', full | ~active, period,
                                                                         tracking.ball_xy_m, pitch, fps), active)
                               for k, v in candidates.items()}}
    tags = [e for e in load_smart_tagging.load() if (e['code'] or '').strip().casefold() == 'set pieces']
    windows = [(max(0, e['start_frame'] - 3*fps), min(n, e['start_frame'] + fps)) for e in tags]
    windows = [(a, b) for a, b in windows if active[a:b].any()]
    public = np.zeros(n, bool); public[PUBLIC_WINDOW[0]:PUBLIC_WINDOW[1]] = True
    blind = active & ~full
    checks = {}
    for name, v in labels.items():
        d = v == 'D'
        checks[name] = {'false_dead': float(d[live].mean()),
                        'false_dead_by_players': {g: float(d[live & sel].mean()) for g, sel in
                                                  (('0-18', players < 19), ('19-20', (players >= 19) & (players < 21)), ('21+', players >= 21))},
                        'dead_share': float(d[active].mean()), 'dead_share_in_blind_spots': float(d[blind].mean()),
                        'dead_share_public_window': float(d[public].mean()),
                        'set_piece_tags_with_D': f"{sum(int(d[a:b].any()) for a, b in windows)}/{len(windows)}",
                        'full_match_spells': spell_stats(v, active, fps)}
        c = checks[name]
        progress(f"  {name}: false dead {c['false_dead']:.3f} by players "
                 f"{ {k: round(x, 3) for k, x in c['false_dead_by_players'].items()} } dead share {c['dead_share']:.3f} "
                 f"(blind spots {c['dead_share_in_blind_spots']:.3f}), tags {c['set_piece_tags_with_D']}, "
                 f"spells/live min {c['full_match_spells']['spells_per_live_minute']:.2f}")
        write_labels(out/'predictions'/(name + '.csv'), v)
    for name in BASELINES:
        (out/'predictions'/(name + '.csv')).write_bytes((ROOT/'evaluation/results'/(name + '.csv')).read_bytes())
    from evaluation.possession_metrics import read_label_csv, validate_submission
    manifest = {'experiment': 'Experiment 3 — bridged dead-ball blind spots', 'blind_spot_share': float(blind.sum() / active.sum()),
                'answer_free_checks': checks, 'models': {},
                'source_sha256': {q.relative_to(ROOT).as_posix(): sha256(q) for q in sorted((ROOT/'solution').glob('*.py'))}}
    for path in sorted((out/'predictions').glob('*.csv')):
        error = validate_submission(read_label_csv(path), n)
        if error:
            raise ValueError(f'{path.stem}: {error}')
        manifest['models'][path.stem] = {'prediction_sha256': sha256(path)}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2, default=float) + '\n', encoding='utf-8')
    if not args.skip_score:
        subprocess.run([sys.executable, str(ROOT/'evaluation/run_evaluation.py'), '--gt', 'public',
                        '--results', str(out/'predictions'), '--out', str(out/'scoring'),
                        '--bootstrap', '1000', '--no-detail'], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
