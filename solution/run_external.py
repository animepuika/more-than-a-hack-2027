"""External bench: score existing methods on event-derived Metrica possession labels.

Each method runs unchanged on the three Metrica sample matches; the official
scorer compares it with labels from solution.external_possession. These labels
are approximate (events, not corrected tracking), and Metrica's ball is almost
always missing during dead ball, so D is easier here than on BAR-ATH. No
BAR-ATH data or answers are read. No command commits or pushes.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import warnings

import numpy as np
import pandas as pd

from starter import baseline
from evaluation import possession_metrics as pm
from .ball_masking import bar_ath_gap_lengths, hide_live_ball
from .context import build_context, proxy_costs
from .data import Tracking
from .external_possession import attach_teams, possession_labels
from .external_samples import DEFAULT_SAMPLE_ROOT, load_metrica_samples
from .features import make_features
from .gap_fallback import GAP_FALLBACK_S, gap_fallback_memory
from .predict import predict
from .run import ROOT, DEFAULT_CONFIG, read_config, sha256, write_labels
from .run_v2 import V2_CONFIG, context_config
from .run_v3 import V3_CONFIG, read_v3_config, _stoppage_variant
from .run_v6 import _sample_fingerprints
from .stoppages import build_stoppage_features
from .v7_features import build_features
from .v7_labels import period_runs
from .v7_sequence import control_memory

V7_CONFIG = ROOT/'solution/config_v7.json'
GAP_SOURCE = ROOT/'data/runs/seventh_trial/diagnostics/v7_features.csv'
BASELINES = (('baseline_raw', 0), ('baseline_smooth25', 25), ('baseline_smooth50', 50),
             ('baseline_smooth125', 125))
DESCRIPTIONS = {
    'baseline_raw': 'Official nearest-player baseline, no smoothing.',
    'baseline_smooth25': 'Official baseline, +/-1 s majority vote.',
    'baseline_smooth50': 'Official baseline, +/-2 s majority vote (published full-match floor).',
    'baseline_smooth125': 'Official baseline, +/-5 s majority vote (best on BAR-ATH public).',
    'v1_gap_stop': 'V1 gap and stoppage sequence decoding.',
    'v2_shape': 'V1 plus team-shape costs.',
    'v2_motion': 'V1 plus motion costs.',
    'v2_shape_motion': 'V1 plus shape and motion costs.',
    'v3_soft': 'V2 shape+motion plus soft stoppage costs, without Smart Tagging (none exists here).',
    'v7_transfer': 'Confirmed control transfers on the V2 shape+motion D mask.',
    'v7_transfer_v3d_overlay': 'Handover arm A: v7_transfer labels with V3 soft D pasted on; ownership unchanged.',
    'v7_transfer_union_redecode': 'Handover arm B: control memory re-decoded with the same V2|V3 D mask as arm A.',
    'v9_gap_motion': 'v7_transfer, but ball gaps of at least 2 s follow V2 motion instead of holding the owner.',
    'v9_gap_motion_union': 'Arm B (V2|V3 D mask) with the same 2 s V2-motion gap fallback.',
    'v9_gap_motion_union_gap1s': 'Sensitivity: v9_gap_motion_union with a 1 s gap threshold.',
    'v9_gap_motion_union_gap5s': 'Sensitivity: v9_gap_motion_union with a 5 s gap threshold.',
}
NOT_RUN = {
    'v3_tag_soft / v8_*': 'Need Smart Tagging, which the Metrica samples do not have.',
    'v7_joint*, v7_dead_*': 'Use the dead-ball model trained on these same three matches; scoring it here would leak.',
}


def tracking_from_match(match, max_player_speed_mps=12.0, walking_speed_mps=1.4):
    """BAR-ATH-style aggregate tracking. Metrica has no speed field, so speed is a
    central difference of positions inside each period."""
    xy = np.asarray(match.player_xy_m, dtype=np.float32)
    ball = np.asarray(match.ball_xy_m, dtype=np.float64)
    home = np.asarray(match.home_slots, bool)
    valid = np.isfinite(xy).all(axis=2)
    distance = np.linalg.norm(xy - ball[:, None], axis=2)
    nearest = []
    for side in (home, ~home):
        d = np.where(valid & side[None], distance, np.inf).min(axis=1)
        nearest.append(np.where(np.isfinite(d), d, np.nan))
    speed = np.full(valid.shape, np.nan, dtype=np.float32)
    for a, b in period_runs(match.period):
        if b - a > 2:
            speed[a+1:b-1] = np.linalg.norm(xy[a+2:b] - xy[a:b-2], axis=2) * match.fps / 2
    good = valid & np.isfinite(speed) & (speed <= max_player_speed_mps)
    count = good.sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # all-missing rows stay NaN
        median = np.nanmedian(np.where(good, speed, np.nan), axis=1)
    walking = np.where(count > 0, (good & (speed <= walking_speed_mps)).sum(axis=1)/np.maximum(count, 1), np.nan)
    return Tracking(float(match.fps), tuple(match.pitch), ball, nearest[0], nearest[1],
                    valid.sum(axis=1).astype(np.int16), count.astype(np.int16), median, walking,
                    xy, home)


def baseline_codes(tracking):
    h, a = tracking.home_distance_m, tracking.away_distance_m
    codes = np.full(len(h), baseline.NOBODY, dtype=np.int8)
    codes[np.isfinite(h) & ~(a < h)] = baseline.HOME
    codes[np.isfinite(a) & ~(h <= a)] = baseline.AWAY
    return codes


def run_methods(match, configs, progress):
    base, v2c, v3c, v7c = configs
    tracking = tracking_from_match(match)
    blocks = [(int(match.period[a]), a, b-1) for a, b in period_runs(match.period) if match.period[a] > 0]
    models = {}
    codes = baseline_codes(tracking)
    for name, window in BASELINES:
        models[name] = baseline.smooth(codes, window)
    basic = make_features(tracking, blocks, base)
    models['v1_gap_stop'] = predict(tracking, basic, blocks, base)[0]
    progress('  team shape and motion context')
    context = build_context(tracking, basic, blocks, v2c)
    shape, _ = proxy_costs(context['shape'], tracking, basic, blocks, v2c, v2c['shape_weight'])
    motion, _ = proxy_costs(context['motion'], tracking, basic, blocks, v2c, v2c['motion_weight'])
    for name, cost in [('v2_shape', shape), ('v2_motion', motion), ('v2_shape_motion', shape+motion)]:
        models[name] = predict(tracking, basic, blocks, base, extra_team_costs=cost)[0]
    clues = build_stoppage_features(tracking, basic, blocks, v3c, ())
    models['v3_soft'] = _stoppage_variant(tracking, basic, blocks, base, shape+motion, clues, v3c,
                                          hard_override=False, use_restart_tags=False,
                                          duration_frames=None)[0]
    progress('  V7 controlled transfers')
    features, _ = build_features(match, v7c)
    v2 = models['v2_shape_motion']
    models['v7_transfer'] = control_memory(features, match.period, v7c, v2, dead=v2 == 'D')[0]
    v3_dead = models['v3_soft'] == 'D'
    overlay = models['v7_transfer'].copy()
    overlay[v3_dead] = 'D'
    models['v7_transfer_v3d_overlay'] = overlay
    union = (v2 == 'D') | v3_dead
    models['v7_transfer_union_redecode'] = control_memory(features, match.period, v7c, v2, dead=union)[0]
    if not np.array_equal(models['v7_transfer_union_redecode'] == 'D', overlay == 'D'):
        raise AssertionError('Handover arms A and B must share one D mask')
    motion = models['v2_motion']
    models['v9_gap_motion'] = gap_fallback_memory(features, match.period, v7c, motion, GAP_FALLBACK_S,
                                                  v2, dead=v2 == 'D')[0]
    for suffix, seconds in [('', GAP_FALLBACK_S), ('_gap1s', 1.0), ('_gap5s', 5.0)]:
        models['v9_gap_motion_union'+suffix] = gap_fallback_memory(features, match.period, v7c, motion,
                                                                  seconds, v2, dead=union)[0]
    return models


def merge_short_spells(labels, max_frames):
    """Label-sensitivity variant: an H/A spell shorter than max_frames between two
    spells of the other team (no D or X in between) is treated as a touch."""
    labels = labels.copy()
    edges = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    runs = [(labels[a], a, b) for a, b in zip(edges[:-1], edges[1:])]
    for (left, _, _), (mid, a, b), (right, _, _) in zip(runs, runs[1:], runs[2:]):
        if mid in 'HA' and left == right and left in 'HA' and left != mid and b - a < max_frames:
            labels[a:b] = left
    return labels


def pooled(arrays):
    """Join matches with one ungraded X frame so no run or spell crosses a match."""
    parts = []
    for values in arrays:
        parts.extend([values, np.array(['X'])])
    return np.concatenate(parts[:-1])


def score(gt, pred, bootstrap):
    r = pm.evaluate(pm.label_codes(gt), pm.label_codes(pred), bootstrap_n=bootstrap)
    keep = ['spell_f1_mean', 'spell_bootstrap_low', 'spell_bootstrap_high', 'spell_f1_at_headline',
            'macro_f1', 'accuracy', 'alive_dead_f1', 'team_acc_both_alive', 'n_gt_spells',
            'n_pred_spells', 'mean_spell_duration_s', 'gt_mean_spell_duration_s']
    out = {k: (float(r[k]) if isinstance(r[k], (float, np.floating)) else r[k]) for k in keep}
    out['spell_f1_by_iou'] = {str(k): float(v['f1']) for k, v in r['spell_f1'].items()}
    out['d_precision'] = float(r['per_class']['D']['precision'])
    out['d_recall'] = float(r['per_class']['D']['recall'])
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument('--out', type=Path, default=ROOT/'data/runs/external_bench')
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--short-spell-s', type=float, default=1.0)
    parser.add_argument('--hide-live-ball', type=float, default=0.0,
                        help='Stress test: hide the ball in at least this share of live frames')
    parser.add_argument('--mask-seed', type=int, default=0)
    parser.add_argument('--gap-source', type=Path, default=GAP_SOURCE,
                        help='Saved BAR-ATH V7 feature diagnostic supplying ball-gap lengths')
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if not out.is_relative_to((ROOT/'data').resolve()):
        raise ValueError('Keep external bench outputs in ignored data/')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Choose a fresh output directory')
    progress = lambda message: print(message, flush=True)
    source_hashes = {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted((ROOT/'solution').glob('*.py'))}
    configs = (read_config(DEFAULT_CONFIG), context_config(V2_CONFIG), read_v3_config(V3_CONFIG),
               json.loads(V7_CONFIG.read_text()))
    (out/'gt').mkdir(parents=True)
    matches = attach_teams(load_metrica_samples(args.samples), args.samples)
    truth, truth_merged, predictions, audits = {}, {}, {}, {}
    masking = {'target_live_share': args.hide_live_ball, 'seed': args.mask_seed, 'matches': {}}
    if args.hide_live_ball:
        lengths = bar_ath_gap_lengths(args.gap_source)
        masking.update(gap_source=str(args.gap_source), gap_source_sha256=sha256(args.gap_source),
                       bar_ath_gaps=len(lengths), bar_ath_median_gap_s=float(np.median(lengths))/25)
    for index, match in enumerate(matches):
        progress(f'{match.name}: {len(match):,} frames')
        labels, audit = possession_labels(match.events, match.period)
        if args.hide_live_ball:
            live = np.isin(labels, ['H', 'A'])
            before = float(np.mean(~np.isfinite(match.ball_xy_m[live]).all(axis=1)))
            match.ball_xy_m, hidden, added = hide_live_ball(match.ball_xy_m, live, match.period, lengths,
                                                            args.hide_live_ball, args.mask_seed+index)
            masking['matches'][match.name] = {'live_missing_before': before,
                                              'live_missing_after': float(hidden[live].mean()),
                                              'graded_missing_after': float(hidden[labels != 'X'].mean()),
                                              'gaps_added': added}
            progress(f"  live ball missing {before:.1%} -> {hidden[live].mean():.1%} ({added} gaps added)")
        truth[match.name] = labels
        truth_merged[match.name] = merge_short_spells(labels, round(args.short_spell_s*match.fps))
        audits[match.name] = audit
        write_labels(out/'gt'/(match.name+'.csv'), labels)
        write_labels(out/'gt'/(match.name+'_short_spells_merged.csv'), truth_merged[match.name])
        predictions[match.name] = run_methods(match, configs, progress)
        folder = out/'predictions'/match.name
        folder.mkdir(parents=True)
        for name, values in predictions[match.name].items():
            if len(values) != len(match) or not set(np.unique(values)) <= set('HADX'):
                raise ValueError(f'{match.name}/{name}: invalid prediction')
            write_labels(folder/(name+'.csv'), values)
    if source_hashes != {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted((ROOT/'solution').glob('*.py'))}:
        raise RuntimeError('Solution source changed during run; rerun frozen code')
    progress('Predictions frozen; scoring.')
    names = list(next(iter(predictions.values())))
    results = {'per_match': {}, 'pooled': {}, 'pooled_short_spells_merged': {}}
    for match_name in truth:
        results['per_match'][match_name] = {name: score(truth[match_name], predictions[match_name][name], 0)
                                            for name in names}
    for key, gt in [('pooled', truth), ('pooled_short_spells_merged', truth_merged)]:
        joined = pooled(gt.values())
        for name in names:
            results[key][name] = score(joined, pooled(p[name] for p in predictions.values()), args.bootstrap)
    rows = []
    for name in names:
        row = {'method': name, **{f'pooled_{k}': results['pooled'][name][k] for k in
               ['spell_f1_mean', 'spell_bootstrap_low', 'spell_bootstrap_high', 'macro_f1',
                'accuracy', 'd_precision', 'd_recall', 'n_pred_spells']},
               'merged_gt_spell_f1_mean': results['pooled_short_spells_merged'][name]['spell_f1_mean']}
        for match_name in truth:
            row[match_name+'_spell_f1'] = results['per_match'][match_name][name]['spell_f1_mean']
        rows.append(row)
    table = pd.DataFrame(rows).sort_values('pooled_spell_f1_mean', ascending=False)
    table.to_csv(out/'scores.csv', index=False, float_format='%.6f')
    (out/'scores.json').write_text(json.dumps(results, indent=2)+'\n')
    manifest = {
        'experiment': 'External bench: event-derived Metrica H/A/D labels, existing methods unchanged',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'method_descriptions': DESCRIPTIONS, 'not_run': NOT_RUN, 'label_audit': audits,
        'short_spell_merge_s': args.short_spell_s, 'bootstrap': args.bootstrap, 'ball_masking': masking,
        'external_input_sha256': _sample_fingerprints(args.samples), 'source_sha256': source_hashes,
        'gt_sha256': {p.name: sha256(p) for p in sorted((out/'gt').glob('*.csv'))},
        'prediction_label_counts': {m: {n: dict(Counter(map(str, v))) for n, v in p.items()}
                                    for m, p in predictions.items()},
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__},
        'limitations': [
            'Labels come from Metrica event logs, not corrected tracking; boundaries are event-timed.',
            'Metrica removes the ball during most dead ball, so D is far easier here than on BAR-ATH.',
            'Metrica loses the ball in only about 3-9% of live frames; BAR-ATH loses it often during live play.',
            'Methods were designed on BAR-ATH; none was tuned on these labels.',
            'The three matches likely share a provider pipeline; they are not BAR-ATH.',
        ],
    }
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(table.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
