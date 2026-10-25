"""V14: V13 plus confident dead ball from the formation model (V11), calibrated without answers.

On each match, frames where a player clearly controls a ball moving faster than 2 m/s
are treated as certainly live. The formation model's start threshold is raised until
at most 2% of those frames are marked dead (1% and 5% are reported as sensitivity
only). The surviving dead ball is added on top of V13's labels.
  bench   : leave-one-match-out formation model, all four hiding levels, against V13
  bar-ath : saved formation probabilities (trained on all Metrica matches), frozen V13
Outputs stay under data/; no commit or push.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from evaluation import possession_metrics as pm
from .deadball_formation import calibrated_dead_mask
from .run import ROOT, sha256, write_labels

MAIN_RATE = 0.02
RATES = (0.01, 0.02, 0.05)
MOVING_BALL_MPS = 2.0


def overlay(v13, dead, active):
    out = np.asarray(v13).copy()
    out[np.asarray(dead, bool) & np.asarray(active, bool)] = 'D'
    return out


def bench(args, progress):
    from .ball_masking import bar_ath_gap_lengths, hide_live_ball
    from .deadball_formation import dead_probability, fit, formation_features
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
    progress('Formation features and leave-one-match-out models.')
    tables = {m.name: formation_features(m.player_xy_m, m.home_slots, m.period, m.pitch, m.fps) for m in matches}
    p_dead = {}
    for m in matches:
        train = [n for n in names if n != m.name]
        p_dead[m.name] = dead_probability(fit([tables[n] for n in train], [labels[n] for n in train]), tables[m.name])
    read = lambda path: pd.read_csv(path)['label'].to_numpy(dtype='<U1')
    results, thresholds = {}, defaultdict(dict)
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
            v13 = read(args.v13_bench/'predictions'/tag/m.name/'v13_combined.csv')
            preds[m.name] = {'v13_combined': v13}
            for rate in RATES:
                dead, threshold = calibrated_dead_mask(p_dead[m.name], m.period, v7c, live_ref, rate, m.fps)
                thresholds[level][f'{m.name}@{rate}'] = threshold
                name = 'v14_confident_dead' + ('' if rate == MAIN_RATE else f'_{rate:g}')
                preds[m.name][name] = overlay(v13, dead, m.period > 0)
                if rate == MAIN_RATE:
                    (args.out/'predictions'/tag/m.name).mkdir(parents=True, exist_ok=True)
                    write_labels(args.out/'predictions'/tag/m.name/'v14_confident_dead.csv', preds[m.name][name])
        gt = pooled(labels[n] for n in names)
        joined = {k: pooled(preds[n][k] for n in names) for k in preds[names[0]]}
        scores = {k: pm.evaluate(pm.label_codes(gt), pm.label_codes(v)) for k, v in joined.items()}
        results[level] = {
            'spell_f1': {k: float(s['spell_f1_mean']) for k, s in scores.items()},
            'predicted_spells': {k: int(s['n_pred_spells']) for k, s in scores.items()},
            'd_precision': {k: float(s['per_class']['D']['precision']) for k, s in scores.items()},
            'd_recall': {k: float(s['per_class']['D']['recall']) for k, s in scores.items()},
            'v14_minus_v13': paired_difference(gt, joined['v14_confident_dead'], joined['v13_combined'])}
        r = results[level]
        progress(f"level {level}%: V14 {r['spell_f1']['v14_confident_dead']:.4f}  V13 {r['spell_f1']['v13_combined']:.4f}  "
                 f"V14-V13 {r['v14_minus_v13']['mean']:+.4f} ({r['v14_minus_v13']['share_first_better']:.0%})  "
                 f"1%: {r['spell_f1']['v14_confident_dead_0.01']:.4f}  5%: {r['spell_f1']['v14_confident_dead_0.05']:.4f}  "
                 f"D recall {r['d_recall']['v14_confident_dead']:.3f} vs {r['d_recall']['v13_combined']:.3f}, "
                 f"precision {r['d_precision']['v14_confident_dead']:.3f} vs {r['d_precision']['v13_combined']:.3f}")
    (args.out/'v14_bench_results.json').write_text(json.dumps(
        {'created_utc': datetime.now(timezone.utc).isoformat(), 'main_rate': MAIN_RATE, 'rates': RATES,
         'moving_ball_mps': MOVING_BALL_MPS, 'thresholds': thresholds,
         'source_sha256': {p.name: sha256(p) for p in sorted((ROOT/'solution').glob('*.py'))},
         'bench_results': results}, indent=2)+'\n')


def bar_ath(args, progress):
    from games import GAME, GAMES, half_video_blocks
    from starter import load_smart_tagging
    from .run_v6 import _period_keys
    from .run_v9 import BASELINES, PUBLIC_WINDOW
    from .run_v12 import spell_stats

    quality = json.loads((ROOT/'solution/config_v7.json').read_text())
    n, fps = GAMES[GAME]['n_video_frames'], GAMES[GAME]['fps']
    period, active = _period_keys(n, half_video_blocks(GAME))
    v13_manifest = json.loads((args.v13_run/'manifest.json').read_text())
    frozen = {}
    for name in ['v13_combined', 'v12_turnover', 'v7_transfer']:
        path = args.v13_run/'predictions'/(name+'.csv')
        if sha256(path) != v13_manifest['models'][name]['prediction_sha256']:
            raise ValueError(f'Frozen prediction changed: {name}')
        frozen[name] = pd.read_csv(path)['label'].to_numpy(dtype='<U1')
    p_dead = np.load(args.p_dead)
    f = pd.read_csv(ROOT/'data/runs/seventh_trial/diagnostics/v7_features.csv',
                    usecols=['direct_control', 'ball_reliable', 'ball_speed_mps'])
    live_ref = active & (f.direct_control.to_numpy() >= 0) & (f.ball_reliable.to_numpy() == 1) \
        & (np.nan_to_num(f.ball_speed_mps.to_numpy()) > MOVING_BALL_MPS)
    tags = load_smart_tagging.load()
    if not load_smart_tagging.check_time_base(tags)['agrees']:
        raise ValueError('Smart Tagging clock mismatch')
    set_pieces = [e for e in tags if (e['code'] or '').strip().casefold() == 'set pieces']
    v2_dead = pd.read_csv(ROOT/'data/runs/seventh_trial/predictions/v2_shape_motion.csv')['label'].to_numpy(dtype='<U1') == 'D'

    def tag_coverage(dead):
        hits = Counter(); total = Counter()
        for e in set_pieces:
            a, b = max(0, e['start_frame'] - 3*fps), min(n, e['start_frame'] + fps)
            if active[a:b].any():
                kind = e['tags'].get('Type') or 'unknown'
                total[kind] += 1; hits[kind] += int(dead[a:b].any())
        return {'all': f"{sum(hits.values())}/{sum(total.values())}", **{k: f'{hits[k]}/{total[k]}' for k in total}}

    checks, outputs = {}, {}
    for rate in RATES:
        dead, threshold = calibrated_dead_mask(p_dead, period, quality, live_ref, rate, fps)
        labels = overlay(frozen['v13_combined'], dead, active)
        outputs[rate] = labels
        checks[f'v14_at_{rate:g}'] = {'start_threshold': threshold, 'set_piece_tags_with_D': tag_coverage(labels == 'D'),
                                      'false_dead_on_clear_moving_ball': float((labels == 'D')[live_ref].mean()),
                                      'dead_share_of_graded': float((labels == 'D')[active].mean()),
                                      'full_match_spells': spell_stats(labels, active, fps)}
    for name, labels in frozen.items():
        checks[name] = {'set_piece_tags_with_D': tag_coverage(labels == 'D'),
                        'false_dead_on_clear_moving_ball': float((labels == 'D')[live_ref].mean()),
                        'dead_share_of_graded': float((labels == 'D')[active].mean()),
                        'full_match_spells': spell_stats(labels, active, fps)}
    for name, c in checks.items():
        progress(f"  {name}: tags {c['set_piece_tags_with_D']['all']}, false dead {c['false_dead_on_clear_moving_ball']:.3f}, "
                 f"dead share {c['dead_share_of_graded']:.3f}, spells {c['full_match_spells']['spells']}"
                 + (f", threshold {c['start_threshold']:.3f}" if 'start_threshold' in c else ''))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'predictions').mkdir(exist_ok=True)
    write_labels(args.out/'predictions/v14_confident_dead.csv', outputs[MAIN_RATE])
    for name, labels in frozen.items():
        write_labels(args.out/'predictions'/(name+'.csv'), labels)
    for name in BASELINES:
        (args.out/'predictions'/(name+'.csv')).write_bytes((ROOT/'evaluation/results'/(name+'.csv')).read_bytes())
    from evaluation.possession_metrics import read_label_csv, validate_submission
    manifest = {'experiment': 'FirstTrial v14 — V13 plus answer-free calibrated formation dead ball',
                'created_utc': datetime.now(timezone.utc).isoformat(), 'main_rate': MAIN_RATE,
                'moving_ball_mps': MOVING_BALL_MPS, 'p_dead_sha256': sha256(args.p_dead),
                'v13_manifest_sha256': sha256(args.v13_run/'manifest.json'), 'answer_free_checks': checks,
                'only_main_variant_scored_publicly': True,
                'source_sha256': {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted((ROOT/'solution').glob('*.py'))},
                'models': {}}
    for path in sorted((args.out/'predictions').glob('*.csv')):
        error = validate_submission(read_label_csv(path), n)
        if error:
            raise ValueError(f'{path.stem}: {error}')
        manifest['models'][path.stem] = {'prediction_sha256': sha256(path)}
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    progress('Frozen; scoring the main variant with controls on the public sample.')
    if not args.skip_score:
        subprocess.run([sys.executable, str(ROOT/'evaluation/run_evaluation.py'), '--gt', 'public',
                        '--results', str(args.out/'predictions'), '--out', str(args.out/'scoring'),
                        '--bootstrap', str(args.bootstrap), '--no-detail'], cwd=ROOT, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['bench', 'bar-ath'])
    parser.add_argument('--v13-bench', type=Path, default=ROOT/'data/runs/external_bench_v13')
    parser.add_argument('--v13-run', type=Path, default=ROOT/'data/runs/thirteenth_trial')
    parser.add_argument('--p-dead', type=Path, default=ROOT/'data/runs/tenth_trial/v11_p_dead.npy')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--skip-score', action='store_true')
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args(argv)
    args.out = (args.out or ROOT/('data/runs/external_bench_v14' if args.mode == 'bench' else 'data/runs/fourteenth_trial')).resolve()
    if not args.out.is_relative_to((ROOT/'data').resolve()) or (args.out.exists() and any(args.out.iterdir())):
        raise ValueError('Choose a fresh output directory inside ignored data/')
    progress = lambda message: print(message, flush=True)
    (bench if args.mode == 'bench' else bar_ath)(args, progress)


if __name__ == '__main__':
    main()
