"""V15: formation dead ball adapted to the target match, plus Smart Tagging set-piece hints.

(1) Self-adaptation: the formation model is retrained with the target match's
certainly-live frames (a player clearly controls a ball moving faster than 2 m/s)
added as live examples. Each half uses examples from the other half only, so false
alarms are measured on frames the model never saw. Standard V7 thresholds (0.7/0.4).
(2) Tag hints (BAR-ATH only; the Metrica samples have no tags): from 15 s before to
0.5 s after an automatic set-piece tag, dead ball may start at 0.5 (continue 0.3).
V15a = V13 + (1); V15b = V13 + (1) + (2), the declared main BAR-ATH candidate.
No answer labels are read before predictions are frozen. No commit or push.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from evaluation import possession_metrics as pm
from .deadball_formation import dead_probability, fit, formation_features
from .run import ROOT, sha256, write_labels
from .run_v14 import MOVING_BALL_MPS, overlay
from .v7_model import dead_mask

TAG_BEFORE_S, TAG_AFTER_S = 15.0, 0.5
TAG_START, TAG_CONTINUE = 0.5, 0.3


def adapted_probability(train_tables, train_labels, table, period, live_ref, stride=5):
    """Per half: Metrica labels plus the other half's certainly-live frames as live examples."""
    p = np.zeros(len(period))
    for half in np.unique(period[period > 0]):
        negatives = np.where(live_ref & (period > 0) & (period != half), 'H', '')
        model = fit([*train_tables, table], [*train_labels, negatives], stride)
        rows = period == half
        p[rows] = dead_probability(model, table[rows])
    return p


def tag_window(events, n, fps):
    window = np.zeros(n, bool)
    for e in events:
        if (e['code'] or '').strip().casefold() == 'set pieces':
            window[max(0, e['start_frame'] - round(TAG_BEFORE_S*fps)):min(n, e['start_frame'] + round(TAG_AFTER_S*fps))] = True
    return window


def bench(args, progress):
    from .ball_masking import bar_ath_gap_lengths, hide_live_ball
    from .external_possession import attach_teams, possession_labels
    from .external_samples import DEFAULT_SAMPLE_ROOT, load_metrica_samples
    from .run_external import GAP_SOURCE, V7_CONFIG, pooled
    from .run_gap_model import LEVELS, paired_difference
    from .v7_features import build_features
    from .v7_sequence import control_evidence

    v7c = json.loads(V7_CONFIG.read_text())
    lengths = bar_ath_gap_lengths(GAP_SOURCE)
    matches = attach_teams(load_metrica_samples(DEFAULT_SAMPLE_ROOT), DEFAULT_SAMPLE_ROOT)
    names = [m.name for m in matches]
    labels = {m.name: possession_labels(m.events, m.period)[0] for m in matches}
    progress('Formation features.')
    tables = {m.name: formation_features(m.player_xy_m, m.home_slots, m.period, m.pitch, m.fps) for m in matches}
    read = lambda path: pd.read_csv(path)['label'].to_numpy(dtype='<U1')
    results = {}
    for level, target in LEVELS.items():
        tag = 'live' + level.zfill(2)
        preds = {}
        for index, m in enumerate(matches):
            ball = m.ball_xy_m
            if target:
                ball = hide_live_ball(m.ball_xy_m, np.isin(labels[m.name], ['H', 'A']), m.period, lengths, target, index)[0]
            masked = type(m)(m.name, m.player_xy_m, m.home_slots, ball, m.period, m.events, m.fps, m.pitch)
            features, _ = build_features(masked, v7c)
            live_ref = ((control_evidence(features, v7c) >= 0) & (features['ball_reliable'] == 1)
                        & (np.nan_to_num(features['ball_speed_mps']) > MOVING_BALL_MPS) & (m.period > 0))
            train = [n for n in names if n != m.name]
            p = adapted_probability([tables[n] for n in train], [labels[n] for n in train], tables[m.name],
                                    m.period, live_ref)
            dead = dead_mask(p, m.period, v7c, m.fps) & (m.period > 0)
            v13 = read(args.v13_bench/'predictions'/tag/m.name/'v13_combined.csv')
            preds[m.name] = {'v15a_adapted_dead': overlay(v13, dead, m.period > 0), 'v13_combined': v13}
            v14_path = args.v14_bench/'predictions'/tag/m.name/'v14_confident_dead.csv'
            if v14_path.exists():
                preds[m.name]['v14_confident_dead'] = read(v14_path)
            (args.out/'predictions'/tag/m.name).mkdir(parents=True, exist_ok=True)
            write_labels(args.out/'predictions'/tag/m.name/'v15a_adapted_dead.csv', preds[m.name]['v15a_adapted_dead'])
            progress(f'  level {level}% {m.name}: false dead on live reference {dead[live_ref].mean():.3f}')
        gt = pooled(labels[n] for n in names)
        keys = [k for k in preds[names[0]] if all(k in preds[n] for n in names)]
        joined = {k: pooled(preds[n][k] for n in names) for k in keys}
        scores = {k: pm.evaluate(pm.label_codes(gt), pm.label_codes(v)) for k, v in joined.items()}
        results[level] = {
            'spell_f1': {k: float(s['spell_f1_mean']) for k, s in scores.items()},
            'd_precision': {k: float(s['per_class']['D']['precision']) for k, s in scores.items()},
            'd_recall': {k: float(s['per_class']['D']['recall']) for k, s in scores.items()},
            'v15a_minus_v13': paired_difference(gt, joined['v15a_adapted_dead'], joined['v13_combined'])}
        if 'v14_confident_dead' in joined:
            results[level]['v15a_minus_v14'] = paired_difference(gt, joined['v15a_adapted_dead'], joined['v14_confident_dead'])
        r = results[level]
        progress(f"level {level}%: " + '  '.join(f'{k} {v:.4f}' for k, v in r['spell_f1'].items())
                 + f"  V15a-V13 {r['v15a_minus_v13']['mean']:+.4f} ({r['v15a_minus_v13']['share_first_better']:.0%})"
                 + f"  D recall {r['d_recall']['v15a_adapted_dead']:.3f} precision {r['d_precision']['v15a_adapted_dead']:.3f}")
    (args.out/'v15_bench_results.json').write_text(json.dumps(
        {'created_utc': datetime.now(timezone.utc).isoformat(),
         'source_sha256': {p.name: sha256(p) for p in sorted((ROOT/'solution').glob('*.py'))},
         'bench_results': results}, indent=2)+'\n')


def bar_ath(args, progress):
    from games import GAME, GAMES, asset_paths, half_video_blocks
    from starter import load_smart_tagging
    from .data import load_tracking
    from .external_possession import attach_teams, possession_labels
    from .external_samples import DEFAULT_SAMPLE_ROOT, load_metrica_samples
    from .run_v6 import _period_keys
    from .run_v9 import BASELINES
    from .run_v12 import spell_stats

    quality = json.loads((ROOT/'solution/config_v7.json').read_text())
    n, fps, pitch = (GAMES[GAME][k] for k in ['n_video_frames', 'fps', 'pitch'])
    period, active = _period_keys(n, half_video_blocks(GAME))
    v13_manifest = json.loads((args.v13_run/'manifest.json').read_text())
    frozen = {}
    for name in ['v13_combined', 'v7_transfer']:
        path = args.v13_run/'predictions'/(name+'.csv')
        if sha256(path) != v13_manifest['models'][name]['prediction_sha256']:
            raise ValueError(f'Frozen prediction changed: {name}')
        frozen[name] = pd.read_csv(path)['label'].to_numpy(dtype='<U1')
    v14_manifest = json.loads((args.v14_run/'manifest.json').read_text())
    v14_path = args.v14_run/'predictions/v14_confident_dead.csv'
    if sha256(v14_path) != v14_manifest['models']['v14_confident_dead']['prediction_sha256']:
        raise ValueError('Frozen V14 prediction changed')
    frozen['v14_confident_dead'] = pd.read_csv(v14_path)['label'].to_numpy(dtype='<U1')

    progress('Metrica formation features and labels.')
    matches = attach_teams(load_metrica_samples(DEFAULT_SAMPLE_ROOT), DEFAULT_SAMPLE_ROOT)
    train_tables = [formation_features(m.player_xy_m, m.home_slots, m.period, m.pitch, m.fps) for m in matches]
    train_labels = [possession_labels(m.events, m.period)[0] for m in matches]
    del matches
    progress('BAR-ATH formation features.')
    paths = asset_paths()
    tracking = load_tracking(paths['atd_meta'], paths['atd_raw'], n, fps, pitch, retain_tracks=True)
    table = formation_features(tracking.track_xy_m, tracking.home_slots, period, pitch, fps)
    f = pd.read_csv(ROOT/'data/runs/seventh_trial/diagnostics/v7_features.csv',
                    usecols=['direct_control', 'ball_reliable', 'ball_speed_mps'])
    live_ref = active & (f.direct_control.to_numpy() >= 0) & (f.ball_reliable.to_numpy() == 1) \
        & (np.nan_to_num(f.ball_speed_mps.to_numpy()) > MOVING_BALL_MPS)
    p = adapted_probability(train_tables, train_labels, table, period, live_ref)
    dead_a = dead_mask(p, period, quality, fps) & active
    tags = load_smart_tagging.load()
    if not load_smart_tagging.check_time_base(tags)['agrees']:
        raise ValueError('Smart Tagging clock mismatch')
    window = tag_window(tags, n, fps) & active
    loose = dict(quality, dead_start_score=TAG_START, dead_continue_score=TAG_CONTINUE)
    dead_b = dead_a | (dead_mask(p, period, loose, fps) & window)
    outputs = {'v15a_adapted_dead': overlay(frozen['v13_combined'], dead_a, active),
               'v15b_adapted_tag_dead': overlay(frozen['v13_combined'], dead_b, active)}
    set_pieces = [e for e in tags if (e['code'] or '').strip().casefold() == 'set pieces']

    def coverage(dead):
        hits = sum(int(dead[max(0, e['start_frame'] - 3*fps):min(n, e['start_frame'] + fps)].any())
                   for e in set_pieces if active[max(0, e['start_frame'] - 3*fps):min(n, e['start_frame'] + fps)].any())
        total = sum(1 for e in set_pieces if active[max(0, e['start_frame'] - 3*fps):min(n, e['start_frame'] + fps)].any())
        return f'{hits}/{total}'

    checks = {}
    for name, labels in {**outputs, **frozen}.items():
        dead = labels == 'D'
        checks[name] = {'set_piece_tags_with_D': coverage(dead),
                        'false_dead_on_clear_moving_ball': float(dead[live_ref].mean()),
                        'dead_share_of_graded': float(dead[active].mean()),
                        'dead_share_outside_tag_windows': float(dead[active & ~window].mean()),
                        'full_match_spells': spell_stats(labels, active, fps)}
        c = checks[name]
        progress(f"  {name}: tags {c['set_piece_tags_with_D']}, false dead {c['false_dead_on_clear_moving_ball']:.3f}, "
                 f"dead share {c['dead_share_of_graded']:.3f}, spells {c['full_match_spells']['spells']}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'predictions').mkdir(exist_ok=True)
    for name, labels in {**outputs, **frozen}.items():
        write_labels(args.out/'predictions'/(name+'.csv'), labels)
    for name in BASELINES:
        (args.out/'predictions'/(name+'.csv')).write_bytes((ROOT/'evaluation/results'/(name+'.csv')).read_bytes())
    np.save(args.out/'v15_p_dead_adapted.npy', p)
    from evaluation.possession_metrics import read_label_csv, validate_submission
    manifest = {'experiment': 'FirstTrial v15 — self-adapted formation dead ball with set-piece tag hints',
                'created_utc': datetime.now(timezone.utc).isoformat(), 'main_candidate': 'v15b_adapted_tag_dead',
                'settings': {'moving_ball_mps': MOVING_BALL_MPS, 'tag_window_s': [TAG_BEFORE_S, TAG_AFTER_S],
                             'tag_thresholds': [TAG_START, TAG_CONTINUE],
                             'standard_thresholds': [quality['dead_start_score'], quality['dead_continue_score']]},
                'answer_free_checks': checks, 'v13_manifest_sha256': sha256(args.v13_run/'manifest.json'),
                'v14_manifest_sha256': sha256(args.v14_run/'manifest.json'),
                'source_sha256': {q.relative_to(ROOT).as_posix(): sha256(q) for q in sorted((ROOT/'solution').glob('*.py'))},
                'models': {}}
    for path in sorted((args.out/'predictions').glob('*.csv')):
        error = validate_submission(read_label_csv(path), n)
        if error:
            raise ValueError(f'{path.stem}: {error}')
        manifest['models'][path.stem] = {'prediction_sha256': sha256(path)}
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    progress('Frozen; scoring on the public sample.')
    if not args.skip_score:
        subprocess.run([sys.executable, str(ROOT/'evaluation/run_evaluation.py'), '--gt', 'public',
                        '--results', str(args.out/'predictions'), '--out', str(args.out/'scoring'),
                        '--bootstrap', str(args.bootstrap), '--no-detail'], cwd=ROOT, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['bench', 'bar-ath'])
    parser.add_argument('--v13-bench', type=Path, default=ROOT/'data/runs/external_bench_v13')
    parser.add_argument('--v14-bench', type=Path, default=ROOT/'data/runs/external_bench_v14')
    parser.add_argument('--v13-run', type=Path, default=ROOT/'data/runs/thirteenth_trial')
    parser.add_argument('--v14-run', type=Path, default=ROOT/'data/runs/fourteenth_trial')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--skip-score', action='store_true')
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args(argv)
    args.out = (args.out or ROOT/('data/runs/external_bench_v15' if args.mode == 'bench' else 'data/runs/fifteenth_trial')).resolve()
    if not args.out.is_relative_to((ROOT/'data').resolve()) or (args.out.exists() and any(args.out.iterdir())):
        raise ValueError('Choose a fresh output directory inside ignored data/')
    progress = lambda message: print(message, flush=True)
    (bench if args.mode == 'bench' else bar_ath)(args, progress)


if __name__ == '__main__':
    main()
