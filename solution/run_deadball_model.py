"""Score the V11 ball-free dead-ball model on the external bench, leave one match out.

Reuses the run_external bench folders (controls, same masks) and the V10 bench
predictions, both reproduced exactly here before anything new is scored. V11 is
V10 with its dead mask widened by the formation model (V2 D or formation D);
V11-only replaces V2's dead mask with the formation one (declared sensitivity).
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
from .deadball_formation import MODEL_SETTINGS, dead_probability, fit as fit_dead, formation_features
from .external_possession import attach_teams, possession_labels
from .external_samples import DEFAULT_SAMPLE_ROOT, load_metrica_samples
from .gap_fallback import GAP_FALLBACK_S, gap_fallback_memory
from .gap_model import ballfree_features, fit as fit_gap, home_probability, model_fallback
from .run import ROOT, sha256, write_labels
from .run_external import GAP_SOURCE, V7_CONFIG, pooled
from .run_gap_model import LEVELS, paired_difference
from .v7_features import build_features
from .v7_model import dead_mask, interval_metrics
from .v7_sequence import control_memory


def possession(features, period, config, p_home, v2, dead):
    held = control_memory(features, period, config, v2, dead=dead)[0]
    fallback = model_fallback(p_home, features, period, config, held, GAP_FALLBACK_S, dead=dead)
    return gap_fallback_memory(features, period, config, fallback, GAP_FALLBACK_S, v2, dead=dead)[0]


def dead_scores(labels, pred, period):
    graded = labels != 'X'
    truth = labels == 'D'
    tp, fp, fn = [int((graded & a & b).sum()) for a, b in [(pred, truth), (pred, ~truth), (~pred, truth)]]
    y = np.where(graded, truth.astype(int), -1)
    return {'precision': tp/max(1, tp+fp), 'recall': tp/max(1, tp+fn), 'frame_f1': 2*tp/max(1, 2*tp+fp+fn),
            **interval_metrics(y, pred, period)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument('--bench-prefix', default='data/runs/external_bench_mid_live')
    parser.add_argument('--v10-bench', type=Path, default=ROOT/'data/runs/external_bench_v10')
    parser.add_argument('--out', type=Path, default=ROOT/'data/runs/external_bench_v11')
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
    formation, gap_tables = {}, {}
    for m in matches:
        progress(f'Features: {m.name}')
        formation[m.name] = formation_features(m.player_xy_m, m.home_slots, m.period, m.pitch, m.fps)
        gap_tables[m.name] = ballfree_features(m.player_xy_m, m.home_slots, m.period, m.pitch, m.fps)

    progress('Leave-one-match-out models.')
    p_dead, p_home, new_dead, screen = {}, {}, {}, {}
    for m in matches:
        train = [n for n in names if n != m.name]
        p_dead[m.name] = dead_probability(fit_dead([formation[n] for n in train], [labels[n] for n in train]),
                                          formation[m.name])
        p_home[m.name] = home_probability(fit_gap([gap_tables[n] for n in train], [labels[n] for n in train]),
                                          gap_tables[m.name])
        new_dead[m.name] = dead_mask(p_dead[m.name], m.period, v7c, m.fps) & (m.period > 0)
        screen[m.name] = dead_scores(labels[m.name], new_dead[m.name], m.period)
        progress(f"  {m.name}: D precision {screen[m.name]['precision']:.3f} recall {screen[m.name]['recall']:.3f}")

    results = {}
    for level, target in LEVELS.items():
        tag = 'live' + level.zfill(2)
        bench = ROOT/(args.bench_prefix + level.zfill(2))
        preds, v2_dead_scores = {}, {}
        for index, m in enumerate(matches):
            ball = m.ball_xy_m
            if target:
                live = np.isin(labels[m.name], ['H', 'A'])
                ball = hide_live_ball(m.ball_xy_m, live, m.period, lengths, target, index)[0]
            masked = type(m)(m.name, m.player_xy_m, m.home_slots, ball, m.period, m.events, m.fps, m.pitch)
            features, _ = build_features(masked, v7c)
            read = lambda path: pd.read_csv(path)['label'].to_numpy(dtype='<U1')
            v2 = read(bench/'predictions'/m.name/'v2_shape_motion.csv')
            saved_v10 = read(args.v10_bench/'predictions'/tag/m.name/'v10_gap_model.csv')
            v10 = possession(features, m.period, v7c, p_home[m.name], v2, v2 == 'D')
            if not np.array_equal(v10, saved_v10):
                raise ValueError(f'{m.name} level {level}: V10 not reproduced exactly')
            v2_dead = v2 == 'D'
            v2_dead_scores[m.name] = dead_scores(labels[m.name], v2_dead, m.period)
            preds[m.name] = {'v10_gap_model': v10,
                             'v11_formation_dead': possession(features, m.period, v7c, p_home[m.name], v2,
                                                              v2_dead | new_dead[m.name]),
                             'v11_formation_dead_only': possession(features, m.period, v7c, p_home[m.name], v2,
                                                                   new_dead[m.name])}
            for name in ('v11_formation_dead', 'v11_formation_dead_only'):
                folder = out/'predictions'/tag/m.name
                folder.mkdir(parents=True, exist_ok=True)
                write_labels(folder/(name+'.csv'), preds[m.name][name])
        gt = pooled(labels[n] for n in names)
        joined = {k: pooled(preds[n][k] for n in names) for k in preds[names[0]]}
        scores = {k: pm.evaluate(pm.label_codes(gt), pm.label_codes(v)) for k, v in joined.items()}
        results[level] = {
            'spell_f1': {k: float(s['spell_f1_mean']) for k, s in scores.items()},
            'predicted_spells': {k: int(s['n_pred_spells']) for k, s in scores.items()},
            'd_precision': {k: float(s['per_class']['D']['precision']) for k, s in scores.items()},
            'd_recall': {k: float(s['per_class']['D']['recall']) for k, s in scores.items()},
            'v11_minus_v10': paired_difference(gt, joined['v11_formation_dead'], joined['v10_gap_model']),
            'v11_only_minus_v10': paired_difference(gt, joined['v11_formation_dead_only'], joined['v10_gap_model']),
            'v2_dead_mask_by_match': v2_dead_scores}
        r = results[level]
        progress(f"level {level}%: V11 {r['spell_f1']['v11_formation_dead']:.4f}  V11-only "
                 f"{r['spell_f1']['v11_formation_dead_only']:.4f}  V10 {r['spell_f1']['v10_gap_model']:.4f}  "
                 f"V11-V10 {r['v11_minus_v10']['mean']:+.4f} ({r['v11_minus_v10']['share_first_better']:.0%})  "
                 f"D recall V11 {r['d_recall']['v11_formation_dead']:.3f} vs V10 {r['d_recall']['v10_gap_model']:.3f}")
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'model_settings': MODEL_SETTINGS,
              'dead_mask_settings': {k: v7c[k] for k in ['dead_start_score', 'dead_continue_score',
                                                         'dead_confirmation_s', 'restart_confirmation_s']},
              'source_sha256': {p.name: sha256(p) for p in sorted((ROOT/'solution').glob('*.py'))},
              'formation_dead_leave_one_match_out': screen, 'bench_results': results}
    (out/'v11_results.json').write_text(json.dumps(report, indent=2, default=float)+'\n')


if __name__ == '__main__':
    main()
