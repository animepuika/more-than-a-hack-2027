"""Score the V12 turnover-aware possession model on the external bench.

Leave one match out. Each model trains on its matches with the ball as recorded
and with 25% of the live ball hidden (seed 100 + match index, different from the
bench masks). The switch cost is chosen on the two training matches only (train
on one, decode the other at 25% hidden, best mean Spell F1). D comes from the
saved V2 bench predictions. V10 bench predictions are the comparison.
Outputs stay under data/; no commit or push.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import possession_metrics as pm
from .ball_masking import bar_ath_gap_lengths, hide_live_ball
from .external_possession import attach_teams, possession_labels
from .external_samples import DEFAULT_SAMPLE_ROOT, load_metrica_samples
from .gap_model import ballfree_features
from .run import ROOT, sha256, write_labels
from .run_external import GAP_SOURCE, V7_CONFIG, pooled
from .run_gap_model import LEVELS, paired_difference
from .turnover_model import MODEL_SETTINGS, SWITCH_GRID_S, decode, fit, home_probability, touch_features, training_rows
from .v7_features import build_features

TRAIN_MASK = 0.25
TRAIN_SEED = 100
BUCKETS = ((0, 1), (1, 3), (3, 10), (10, np.inf))


def length_breakdown(gt, pred):
    gt, pred = pm.label_codes(gt), pm.label_codes(pred)
    runs = pm.valid_runs(gt)
    pred = pm.fill_abstentions(pred, runs)
    truth, spells = pm.possession_spells(gt, runs), pm.possession_spells(pred, runs)
    hit = pm.matched_gt_flags(pm.spell_overlaps(truth, spells), len(truth), 0.5)
    seconds = np.array([(b - a) / 25 for a, b, _ in truth])
    return {f'{lo:g}-{hi:g}s': float(hit[(seconds >= lo) & (seconds < hi)].mean()) for lo, hi in BUCKETS}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument('--bench-prefix', default='data/runs/external_bench_mid_live')
    parser.add_argument('--v10-bench', type=Path, default=ROOT/'data/runs/external_bench_v10')
    parser.add_argument('--out', type=Path, default=ROOT/'data/runs/external_bench_v12')
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if not out.is_relative_to((ROOT/'data').resolve()) or (out.exists() and any(out.iterdir())):
        raise ValueError('Choose a fresh output directory inside ignored data/')
    progress = lambda message: print(message, flush=True)
    v7c = json.loads(V7_CONFIG.read_text())
    lengths = bar_ath_gap_lengths(GAP_SOURCE)
    matches = attach_teams(load_metrica_samples(args.samples), args.samples)
    names = [m.name for m in matches]
    labels = {m.name: possession_labels(m.events, m.period)[0] for m in matches}
    ballfree = {m.name: ballfree_features(m.player_xy_m, m.home_slots, m.period, m.pitch, m.fps) for m in matches}

    def tables(index, target, seed):
        m = matches[index]
        ball = m.ball_xy_m
        if target:
            live = np.isin(labels[m.name], ['H', 'A'])
            ball = hide_live_ball(m.ball_xy_m, live, m.period, lengths, target, seed)[0]
        masked = type(m)(m.name, m.player_xy_m, m.home_slots, ball, m.period, m.events, m.fps, m.pitch)
        features, _ = build_features(masked, v7c)
        return touch_features(features, ball, m.period, ballfree[m.name])

    progress('Training rows (recorded ball and 25% hidden).')
    rows = {}
    for i, name in enumerate(names):
        rows[name] = [training_rows(tables(i, 0.0, 0), labels[name]),
                      training_rows(tables(i, TRAIN_MASK, TRAIN_SEED + i), labels[name])]
    single = {n: fit(rows[n]) for n in names}
    outer = {n: fit([part for other in names if other != n for part in rows[other]]) for n in names}

    progress('Predicting every match at every hiding level.')
    read = lambda path: pd.read_csv(path)['label'].to_numpy(dtype='<U1')
    p_outer, p_inner, v2 = {}, {}, {}
    for i, name in enumerate(names):
        for level, target in LEVELS.items():
            t = tables(i, target, i)
            p_outer[name, level] = home_probability(outer[name], t)
            v2[name, level] = read(ROOT/(args.bench_prefix + level.zfill(2))/'predictions'/name/'v2_shape_motion.csv')
            if level == '25':
                for other in names:
                    if other != name:
                        p_inner[other, name] = home_probability(single[other], t)
        progress(f'  {name} done')

    chosen, inner_scores = {}, {}
    for held in names:
        train = [n for n in names if n != held]
        scores = {}
        for s in SWITCH_GRID_S:
            values = []
            for model_from, target_match in [(train[0], train[1]), (train[1], train[0])]:
                m = matches[names.index(target_match)]
                pred = decode(p_inner[model_from, target_match], m.period, v2[target_match, '25'] == 'D', s)
                values.append(pm.evaluate(pm.label_codes(labels[target_match]), pm.label_codes(pred))['spell_f1_mean'])
            scores[s] = float(np.mean(values))
        chosen[held] = max(scores, key=scores.get)
        inner_scores[held] = scores
        progress(f'  switch cost for {held}: {chosen[held]} (inner Spell F1 {scores})')

    results = {}
    for level in LEVELS:
        tag = 'live' + level.zfill(2)
        preds = {}
        for name in names:
            m = matches[names.index(name)]
            v12 = decode(p_outer[name, level], m.period, v2[name, level] == 'D', chosen[name])
            (out/'predictions'/tag/name).mkdir(parents=True, exist_ok=True)
            write_labels(out/'predictions'/tag/name/'v12_turnover.csv', v12)
            preds[name] = {'v12_turnover': v12,
                           'v10_gap_model': read(args.v10_bench/'predictions'/tag/name/'v10_gap_model.csv')}
        gt = pooled(labels[n] for n in names)
        joined = {k: pooled(preds[n][k] for n in names) for k in ['v12_turnover', 'v10_gap_model']}
        scores = {k: pm.evaluate(pm.label_codes(gt), pm.label_codes(v)) for k, v in joined.items()}
        results[level] = {
            'spell_f1': {k: float(s['spell_f1_mean']) for k, s in scores.items()},
            'spell_f1_by_iou': {k: {str(i): float(v['f1']) for i, v in s['spell_f1'].items()} for k, s in scores.items()},
            'predicted_spells': {k: int(s['n_pred_spells']) for k, s in scores.items()},
            'true_spells': int(scores['v12_turnover']['n_gt_spells']),
            'match_rate_by_length_iou_0.5': {k: length_breakdown(gt, v) for k, v in joined.items()},
            'v12_minus_v10': paired_difference(gt, joined['v12_turnover'], joined['v10_gap_model'])}
        r = results[level]
        progress(f"level {level}%: V12 {r['spell_f1']['v12_turnover']:.4f}  V10 {r['spell_f1']['v10_gap_model']:.4f}  "
                 f"V12-V10 {r['v12_minus_v10']['mean']:+.4f} ({r['v12_minus_v10']['share_first_better']:.0%})  "
                 f"spells {r['predicted_spells']}  short<1s {r['match_rate_by_length_iou_0.5']['v12_turnover']['0-1s']:.2f}"
                 f" 1-3s {r['match_rate_by_length_iou_0.5']['v12_turnover']['1-3s']:.2f}")
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'model_settings': MODEL_SETTINGS,
              'train_mask': TRAIN_MASK, 'switch_grid_s': SWITCH_GRID_S, 'chosen_switch_cost_s': chosen,
              'inner_scores': {k: {str(s): v for s, v in d.items()} for k, d in inner_scores.items()},
              'source_sha256': {p.name: sha256(p) for p in sorted((ROOT/'solution').glob('*.py'))},
              'bench_results': results}
    (out/'v12_results.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
