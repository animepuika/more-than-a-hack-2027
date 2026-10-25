"""Score the V10 trained gap model on the external bench, leave one match out.

Reuses a completed run_external bench (all four hiding levels, same masks) for the
V2/V7/V9 controls, checks that V7 is reproduced exactly on the re-masked ball,
then decodes long gaps from the held-out model. Settings in gap_model.py are fixed
before scoring. Outputs stay under data/; no commit or push.
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
from .gap_fallback import GAP_FALLBACK_S, gap_fallback_memory
from .gap_model import (MODEL_SETTINGS, ballfree_features, fit, home_probability, model_fallback)
from .run import ROOT, sha256, write_labels
from .run_external import GAP_SOURCE, V7_CONFIG, pooled
from .v7_features import build_features
from .v7_sequence import control_memory

LEVELS = {'0': 0.0, '10': 0.10, '25': 0.25, '40': 0.40}
CONTROLS = ['v9_gap_motion', 'v7_transfer', 'baseline_smooth50']


def paired_difference(gt, first, second, draws=1000, seed=0):
    """Paired spell bootstrap: the same resampled true spells for both predictions."""
    gt = pm.label_codes(gt)
    runs = pm.valid_runs(gt)
    truth = pm.possession_spells(gt, runs)
    n = len(truth)
    rng = np.random.RandomState(seed)
    picks = [rng.randint(0, n, n) for _ in range(draws)]
    scores = []
    for labels in (first, second):
        pred = pm.fill_abstentions(pm.label_codes(labels), runs)
        spells = pm.possession_spells(pred, runs)
        flags = np.vstack([pm.matched_gt_flags(pm.spell_overlaps(truth, spells), n, t) for t in pm.SPELL_IOUS])
        values = []
        for p in picks:
            f1 = []
            for row in flags:
                tp = int(row[p].sum())
                fp, fn = max(0, len(spells)-tp), n-tp
                f1.append(2*tp/(2*tp+fp+fn) if tp else 0.)
            values.append(np.mean(f1))
        scores.append(np.array(values))
    d = scores[0] - scores[1]
    return {'mean': float(d.mean()), 'interval_90': [float(np.percentile(d, 5)), float(np.percentile(d, 95))],
            'share_first_better': float(np.mean(d > 0))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument('--bench-prefix', default='data/runs/external_bench_mid_live',
                        help='Completed run_external folders: prefix + 00/10/25/40')
    parser.add_argument('--out', type=Path, default=ROOT/'data/runs/external_bench_v10')
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
    tables = {}
    for m in matches:
        progress(f'Ball-free features: {m.name}')
        tables[m.name] = ballfree_features(m.player_xy_m, m.home_slots, m.period, m.pitch, m.fps)

    progress('Leave-one-match-out training.')
    probability, screen = {}, {}
    for held in names:
        train = [n for n in names if n != held]
        model = fit([tables[n] for n in train], [labels[n] for n in train])
        p = home_probability(model, tables[held])
        probability[held] = p
        live = np.isin(labels[held], ['H', 'A'])
        truth = labels[held][live] == 'H'
        guess = p[live] > 0.5
        seen = np.isfinite(next(m for m in matches if m.name == held).ball_xy_m).all(axis=1)[live]
        screen[held] = {'live_frames': int(live.sum()), 'accuracy': float(np.mean(guess == truth)),
                        'log_loss': float(-np.mean(np.where(truth, np.log(p[live]), np.log(1-p[live])))),
                        'accuracy_ball_missing': float(np.mean((guess == truth)[~seen])) if (~seen).any() else None,
                        'confident_share_p_over_0.7': float(np.mean(np.abs(p[live]-0.5) > 0.2)),
                        'accuracy_when_confident': float(np.mean((guess == truth)[np.abs(p[live]-0.5) > 0.2]))}
        progress(f"  {held}: accuracy {screen[held]['accuracy']:.3f}, log loss {screen[held]['log_loss']:.3f}")

    results = {}
    for level, target in LEVELS.items():
        bench = ROOT/(args.bench_prefix + level.zfill(2))
        folder = out/'predictions'/f'live{level.zfill(2)}'
        preds = {n: {} for n in names}
        for index, m in enumerate(matches):
            ball = m.ball_xy_m
            if target:
                live = np.isin(labels[m.name], ['H', 'A'])
                ball = hide_live_ball(m.ball_xy_m, live, m.period, lengths, target, index)[0]
            masked = type(m)(m.name, m.player_xy_m, m.home_slots, ball, m.period, m.events, m.fps, m.pitch)
            features, _ = build_features(masked, v7c)
            saved = {k: pd.read_csv(bench/'predictions'/m.name/(k+'.csv'))['label'].to_numpy(dtype='<U1')
                     for k in ['v2_shape_motion', *CONTROLS]}
            v2 = saved['v2_shape_motion']
            v7 = control_memory(features, m.period, v7c, v2, dead=v2 == 'D')[0]
            if not np.array_equal(v7, saved['v7_transfer']):
                raise ValueError(f'{m.name} level {level}: V7 not reproduced; masks or features differ')
            fallback = model_fallback(probability[m.name], features, m.period, v7c, v7, GAP_FALLBACK_S, dead=v2 == 'D')
            v10 = gap_fallback_memory(features, m.period, v7c, fallback, GAP_FALLBACK_S, v2, dead=v2 == 'D')[0]
            (folder/m.name).mkdir(parents=True, exist_ok=True)
            write_labels(folder/m.name/'v10_gap_model.csv', v10)
            preds[m.name] = {'v10_gap_model': v10, **{k: saved[k] for k in CONTROLS}}
        gt = pooled(labels[n] for n in names)
        joined = {k: pooled(preds[n][k] for n in names) for k in ['v10_gap_model', *CONTROLS]}
        scores = {k: pm.evaluate(pm.label_codes(gt), pm.label_codes(v)) for k, v in joined.items()}
        results[level] = {
            'spell_f1': {k: float(s['spell_f1_mean']) for k, s in scores.items()},
            'predicted_spells': {k: int(s['n_pred_spells']) for k, s in scores.items()},
            'v10_minus_v9': paired_difference(gt, joined['v10_gap_model'], joined['v9_gap_motion']),
            'v10_minus_v7': paired_difference(gt, joined['v10_gap_model'], joined['v7_transfer']),
        }
        r = results[level]
        progress(f"level {level}%: V10 {r['spell_f1']['v10_gap_model']:.4f}  V9 {r['spell_f1']['v9_gap_motion']:.4f}  "
                 f"V7 {r['spell_f1']['v7_transfer']:.4f}  V10-V9 {r['v10_minus_v9']['mean']:+.4f} "
                 f"({r['v10_minus_v9']['share_first_better']:.0%} of draws)")
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'model_settings': MODEL_SETTINGS,
              'gap_fallback_s': GAP_FALLBACK_S, 'bench_prefix': args.bench_prefix,
              'source_sha256': {p.name: sha256(p) for p in sorted((ROOT/'solution').glob('*.py'))},
              'leave_one_match_out_screen': screen, 'bench_results': results}
    (out/'v10_results.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
