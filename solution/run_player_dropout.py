"""Bench with BAR-ATH-like player drop-out: current models versus models retrained with drop-out.

Conditions: (no drop-out, drop-out) x (ball 0% hidden, 40% hidden). Test matches are
predicted leave-one-match-out. "plain" models train as before (no drop-out); "aug"
models add drop-out copies of the training matches (seed 900 + match index; the test
copies use seed 500 + match index). Shared base for every method: V1's stoppage rules
give the D mask and V7 control memory runs on V1 labels (a stated simplification, so
results compare within this run only). V12 switch costs reuse the bench's training-only
choices; V14 uses the 2% answer-free calibration. Outputs stay under data/; no commit or push.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import possession_metrics as pm
from starter import baseline
from .ball_masking import bar_ath_gap_lengths, hide_live_ball
from .deadball_formation import (calibrated_dead_mask, dead_probability, fit as fit_dead, formation_features)
from .external_possession import attach_teams, possession_labels
from .external_samples import DEFAULT_SAMPLE_ROOT, load_metrica_samples
from .features import make_features
from .gap_model import ballfree_features, fit as fit_gap, home_probability as gap_probability
from .player_masking import bar_ath_missing_counts, drop_players
from .predict import predict
from .run import ROOT, DEFAULT_CONFIG, read_config, sha256
from .run_external import GAP_SOURCE, V7_CONFIG, baseline_codes, pooled, tracking_from_match
from .run_gap_model import paired_difference
from .run_turnover_model import length_breakdown
from .run_v13 import combine
from .run_v14 import MOVING_BALL_MPS, overlay
from .turnover_model import decode, fit as fit_turnover, home_probability as turnover_probability, touch_features, training_rows
from .v7_features import build_features
from .v7_labels import period_runs
from .v7_sequence import control_evidence, control_memory

SWITCH_COST = {'Sample_Game_1': 1.0, 'Sample_Game_2': 0.5, 'Sample_Game_3': 0.25}
CONDITIONS = [('clean', 0.0), ('clean', 0.40), ('dropout', 0.0), ('dropout', 0.40)]
TEST_SEED, TRAIN_SEED, BALL_TRAIN_SEED = 500, 900, 100


def subset(table, labels, keep, stride=5):
    rows = np.flatnonzero(np.isin(labels, keep))[::stride]
    return table.iloc[rows].reset_index(drop=True), np.asarray(labels)[rows]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'data/runs/external_bench_dropout')
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if not out.is_relative_to((ROOT/'data').resolve()) or (out.exists() and any(out.iterdir())):
        raise ValueError('Choose a fresh output directory inside ignored data/')
    out.mkdir(parents=True)
    progress = lambda message: print(message, flush=True)
    v7c, base = json.loads(V7_CONFIG.read_text()), read_config(DEFAULT_CONFIG)
    gaps, missing = bar_ath_gap_lengths(GAP_SOURCE), bar_ath_missing_counts(GAP_SOURCE)
    matches = attach_teams(load_metrica_samples(DEFAULT_SAMPLE_ROOT), DEFAULT_SAMPLE_ROOT)
    names = [m.name for m in matches]
    labels = {m.name: possession_labels(m.events, m.period)[0] for m in matches}

    def variant(i, players_seed, ball_target, ball_seed):
        m = matches[i]
        xy = m.player_xy_m if players_seed is None else drop_players(m.player_xy_m, m.period, missing, players_seed)
        ball = m.ball_xy_m if not ball_target else hide_live_ball(
            m.ball_xy_m, np.isin(labels[m.name], ['H', 'A']), m.period, gaps, ball_target, ball_seed)[0]
        return type(m)(m.name, xy, m.home_slots, ball, m.period, m.events, m.fps, m.pitch)

    progress('Training rows (plain and with drop-out).')
    rows = {kind: {} for kind in ('plain', 'aug')}
    for i, name in enumerate(names):
        for kind, seed in (('plain', None), ('aug', TRAIN_SEED + i)):
            parts = {'turnover': [], 'gap': [], 'dead': []}
            for ball_target, ball_seed in ((0.0, 0), (0.25, BALL_TRAIN_SEED + i)):
                m = variant(i, seed, ball_target, ball_seed)
                features, _ = build_features(m, v7c)
                bf = ballfree_features(m.player_xy_m, m.home_slots, m.period, m.pitch, m.fps)
                parts['turnover'].append(training_rows(touch_features(features, m.ball_xy_m, m.period, bf), labels[name]))
                if ball_target == 0.0:  # gap and dead models ignore the ball
                    parts['gap'].append((tuple(subset(t, labels[name], ['H', 'A'])[0] for t in bf),
                                         subset(bf[0], labels[name], ['H', 'A'])[1]))
                    ft = formation_features(m.player_xy_m, m.home_slots, m.period, m.pitch, m.fps)
                    parts['dead'].append(subset(ft, labels[name], ['H', 'A', 'D']))
            rows[kind][name] = parts
        progress(f'  {name} done')

    models = {}
    for kind in ('plain', 'aug'):
        for held in names:
            train = [n for n in names if n != held]
            pool = {k: [p for n in train for p in rows[kind][n][k]] for k in ('turnover', 'gap', 'dead')}
            if kind == 'aug':  # retrained models see both clean and drop-out examples
                pool = {k: pool[k] + [p for n in train for p in rows['plain'][n][k]] for k in pool}
            models[kind, held] = {
                'turnover': fit_turnover(pool['turnover']),
                'gap': fit_gap([t for t, _ in pool['gap']], [l for _, l in pool['gap']], strides=[1]*len(pool['gap'])),
                'dead': fit_dead([t for t, _ in pool['dead']], [l for _, l in pool['dead']], stride=1)}
    progress('Models trained.')

    results, dead_checks = {}, {}
    for players, ball_target in CONDITIONS:
        key = f'{players}_ball{int(ball_target*100)}'
        preds = {n: {} for n in names}
        dead_checks[key] = {}
        for i, name in enumerate(names):
            m = variant(i, TEST_SEED + i if players == 'dropout' else None, ball_target, i)
            tracking = tracking_from_match(m)
            blocks = [(int(m.period[a]), a, b-1) for a, b in period_runs(m.period) if m.period[a] > 0]
            v1 = predict(tracking, make_features(tracking, blocks, base), blocks, base)[0]
            d_mask = v1 == 'D'
            features, _ = build_features(m, v7c)
            bf = ballfree_features(m.player_xy_m, m.home_slots, m.period, m.pitch, m.fps)
            touch = touch_features(features, m.ball_xy_m, m.period, bf)
            formation = formation_features(m.player_xy_m, m.home_slots, m.period, m.pitch, m.fps)
            live_ref = ((control_evidence(features, v7c) >= 0) & (features['ball_reliable'] == 1)
                        & (np.nan_to_num(features['ball_speed_mps']) > MOVING_BALL_MPS) & (m.period > 0))
            preds[name]['smooth50'] = baseline.smooth(baseline_codes(tracking), 50)
            preds[name]['v7_on_v1'] = control_memory(features, m.period, v7c, v1, dead=d_mask)[0]
            for kind in ('plain', 'aug'):
                mod = models[kind, name]
                v12 = decode(turnover_probability(mod['turnover'], touch), m.period, d_mask, SWITCH_COST[name], m.fps)
                v13 = combine(v12, gap_probability(mod['gap'], bf), features, m.period, v7c, m.fps)
                dead, threshold = calibrated_dead_mask(dead_probability(mod['dead'], formation), m.period, v7c,
                                                       live_ref, 0.02, m.fps)
                preds[name][f'v12_{kind}'], preds[name][f'v13_{kind}'] = v12, v13
                preds[name][f'v14_{kind}'] = overlay(v13, dead, m.period > 0)
                graded = labels[name] != 'X'
                dead_checks[key][f'{name}_{kind}'] = {
                    'threshold': threshold, 'false_dead_on_live_ref': float(dead[live_ref].mean()),
                    'dead_share': float(dead[m.period > 0].mean()),
                    'precision_vs_labels': float((labels[name][dead & graded] == 'D').mean()) if (dead & graded).any() else None}
            progress(f'  {key} {name} done')
        gt = pooled(labels[n] for n in names)
        joined = {k: pooled(preds[n][k] for n in names) for k in preds[names[0]]}
        scores = {k: pm.evaluate(pm.label_codes(gt), pm.label_codes(v)) for k, v in joined.items()}
        results[key] = {
            'spell_f1': {k: float(s['spell_f1_mean']) for k, s in scores.items()},
            'predicted_spells': {k: int(s['n_pred_spells']) for k, s in scores.items()},
            'd_precision': {k: float(s['per_class']['D']['precision']) for k, s in scores.items()},
            'd_recall': {k: float(s['per_class']['D']['recall']) for k, s in scores.items()},
            'match_rate_by_length': {k: length_breakdown(gt, joined[k]) for k in ('v13_plain', 'v13_aug')},
            'paired': {f'{a} - {b}': paired_difference(gt, joined[a], joined[b]) for a, b in
                       [('v13_aug', 'v13_plain'), ('v14_aug', 'v14_plain'), ('v14_aug', 'v13_aug'),
                        ('v14_plain', 'v13_plain'), ('v13_plain', 'v7_on_v1')]}}
        r = results[key]
        progress(f'{key}: ' + '  '.join(f'{k} {v:.4f}' for k, v in r['spell_f1'].items()))
        progress('   ' + '  '.join(f"{k}: {v['mean']:+.4f} ({v['share_first_better']:.0%})" for k, v in r['paired'].items()))
    (out/'dropout_results.json').write_text(json.dumps(
        {'created_utc': datetime.now(timezone.utc).isoformat(), 'conditions': CONDITIONS,
         'switch_cost': SWITCH_COST, 'seeds': {'test': TEST_SEED, 'train': TRAIN_SEED},
         'bar_ath_missing_share': {str(k): float(np.mean(missing == k)) for k in range(8)},
         'source_sha256': {p.name: sha256(p) for p in sorted((ROOT/'solution').glob('*.py'))},
         'results': results, 'dead_checks': dead_checks}, indent=2, default=float)+'\n')


if __name__ == '__main__':
    main()
