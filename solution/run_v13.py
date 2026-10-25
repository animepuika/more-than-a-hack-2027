"""V13: V12 everywhere, V10's gap model inside long ball gaps.

V12 is weaker than V10 at the ends of long gaps on BAR-ATH. V13 keeps V12's labels,
then re-decodes each long gap (>= 2 s, outside D) with V10's ball-free model,
anchored to V12's team before the gap and the first confirmed controller after it.
No new settings. Two modes:
  bench   : leave-one-match-out on the Metrica bench, against saved V12/V10 predictions
  bar-ath : combine the frozen V12 predictions with the saved V10 probabilities
Outputs stay under data/; no commit or push.
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
from .gap_fallback import GAP_FALLBACK_S
from .gap_model import model_fallback
from .run import ROOT, sha256, write_labels


def combine(v12, p_home, features, period, config, fps=25):
    """V12 labels with long gaps re-decoded by the V10 model; D and short gaps unchanged."""
    dead = v12 == 'D'
    fallback = model_fallback(p_home, features, period, config, v12, GAP_FALLBACK_S, dead=dead, fps=fps)
    out = v12.copy()
    use = np.isin(fallback, ['H', 'A']) & ~dead
    out[use] = fallback[use]
    return out


def bench(args, progress):
    from .ball_masking import bar_ath_gap_lengths, hide_live_ball
    from .external_possession import attach_teams, possession_labels
    from .external_samples import DEFAULT_SAMPLE_ROOT, load_metrica_samples
    from .gap_model import ballfree_features, fit, home_probability
    from .run_external import GAP_SOURCE, V7_CONFIG, pooled
    from .run_gap_model import LEVELS, paired_difference
    from .run_turnover_model import length_breakdown
    from .v7_features import build_features

    v7c = json.loads(V7_CONFIG.read_text())
    lengths = bar_ath_gap_lengths(GAP_SOURCE)
    matches = attach_teams(load_metrica_samples(DEFAULT_SAMPLE_ROOT), DEFAULT_SAMPLE_ROOT)
    names = [m.name for m in matches]
    labels = {m.name: possession_labels(m.events, m.period)[0] for m in matches}
    tables = {m.name: ballfree_features(m.player_xy_m, m.home_slots, m.period, m.pitch, m.fps) for m in matches}
    p_home = {}
    for m in matches:
        train = [n for n in names if n != m.name]
        p_home[m.name] = home_probability(fit([tables[n] for n in train], [labels[n] for n in train]), tables[m.name])
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
            v12 = read(args.v12_bench/'predictions'/tag/m.name/'v12_turnover.csv')
            v13 = combine(v12, p_home[m.name], features, m.period, v7c, m.fps)
            (args.out/'predictions'/tag/m.name).mkdir(parents=True, exist_ok=True)
            write_labels(args.out/'predictions'/tag/m.name/'v13_combined.csv', v13)
            preds[m.name] = {'v13_combined': v13, 'v12_turnover': v12,
                             'v10_gap_model': read(args.v10_bench/'predictions'/tag/m.name/'v10_gap_model.csv')}
        gt = pooled(labels[n] for n in names)
        joined = {k: pooled(preds[n][k] for n in names) for k in preds[names[0]]}
        scores = {k: pm.evaluate(pm.label_codes(gt), pm.label_codes(v)) for k, v in joined.items()}
        results[level] = {'spell_f1': {k: float(s['spell_f1_mean']) for k, s in scores.items()},
                          'predicted_spells': {k: int(s['n_pred_spells']) for k, s in scores.items()},
                          'match_rate_by_length_iou_0.5': {k: length_breakdown(gt, v) for k, v in joined.items()},
                          'v13_minus_v12': paired_difference(gt, joined['v13_combined'], joined['v12_turnover'])}
        r = results[level]
        progress(f"level {level}%: V13 {r['spell_f1']['v13_combined']:.4f}  V12 {r['spell_f1']['v12_turnover']:.4f}  "
                 f"V10 {r['spell_f1']['v10_gap_model']:.4f}  V13-V12 {r['v13_minus_v12']['mean']:+.4f} "
                 f"({r['v13_minus_v12']['share_first_better']:.0%})  spells {r['predicted_spells']}")
    (args.out/'v13_bench_results.json').write_text(json.dumps(
        {'created_utc': datetime.now(timezone.utc).isoformat(), 'gap_fallback_s': GAP_FALLBACK_S,
         'source_sha256': {p.name: sha256(p) for p in sorted((ROOT/'solution').glob('*.py'))},
         'bench_results': results}, indent=2)+'\n')


def bar_ath(args, progress):
    from games import GAME, GAMES, asset_paths, half_video_blocks
    from .data import load_tracking
    from .external_samples import ExternalMatch
    from .run_v6 import _period_keys
    from .run_v9 import BASELINES, PUBLIC_WINDOW
    from .run_v10 import TEAM
    from .run_v12 import gap_end_scores, spell_stats
    from .v7_sequence import control_memory
    from .v8_stoppages import build_evidence

    quality = json.loads((ROOT/'solution/config_v7.json').read_text())
    v8config = json.loads((ROOT/'solution/config_v8.json').read_text())
    v12_manifest = json.loads((args.v12_run/'manifest.json').read_text())
    frozen = {}
    for name in ['v12_turnover', 'v10_gap_model', 'v9_gap_motion', 'v7_transfer']:
        path = args.v12_run/'predictions'/(name+'.csv')
        if sha256(path) != v12_manifest['models'][name]['prediction_sha256']:
            raise ValueError(f'Frozen prediction changed: {name}')
        frozen[name] = pd.read_csv(path)['label'].to_numpy(dtype='<U1')
    p_home = np.load(args.v10_run/'v10_p_home.npy')
    n, fps, pitch = (GAMES[GAME][k] for k in ['n_video_frames', 'fps', 'pitch'])
    period, active = _period_keys(n, half_video_blocks(GAME))
    paths = asset_paths()
    tracking = load_tracking(paths['atd_meta'], paths['atd_raw'], n, fps, pitch, retain_tracks=True)
    match = ExternalMatch(GAME, tracking.track_xy_m, tracking.home_slots, tracking.ball_xy_m, period, [], fps, pitch)
    evidence = build_evidence(match, v8config, quality)
    v2 = pd.read_csv(ROOT/'data/runs/seventh_trial/predictions/v2_shape_motion.csv')['label'].to_numpy(dtype='<U1')
    v7, supported, direct = control_memory(evidence, period, quality, v2, dead=v2 == 'D', fps=fps)
    if not np.array_equal(v7, frozen['v7_transfer']):
        raise ValueError('Light evidence did not reproduce V7 control predictions exactly')
    v13 = combine(frozen['v12_turnover'], p_home, evidence, period, quality, fps)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'predictions').mkdir(exist_ok=True)
    write_labels(args.out/'predictions/v13_combined.csv', v13)
    for name, labels in frozen.items():
        write_labels(args.out/'predictions'/(name+'.csv'), labels)
    for name in BASELINES:
        (args.out/'predictions'/(name+'.csv')).write_bytes((ROOT/'evaluation/results'/(name+'.csv')).read_bytes())
    reliable = evidence['ball_reliable'] == 1
    controller = np.where(direct >= 0, TEAM[np.clip(direct, 0, 1)], '')
    clear = active & (direct >= 0) & (v7 == controller)
    public = np.zeros(n, bool); public[PUBLIC_WINDOW[0]:PUBLIC_WINDOW[1]] = True
    checks = {name: {'clear_control_agreement': float(np.mean(labels[clear] == controller[clear])),
                     'gap_ends': gap_end_scores(labels, direct, supported, reliable, v7, period, fps),
                     'full_match_spells': spell_stats(labels, active, fps),
                     'public_window_spells': spell_stats(labels, public, fps)}
              for name, labels in {'v13_combined': v13, **frozen}.items()}
    for name, c in checks.items():
        progress(f"  {name}: clear {c['clear_control_agreement']:.3f}, gap ends {c['gap_ends']['correct']:.3f} "
                 f"(changed {c['gap_ends']['correct_team_changed']:.3f}), spells {c['full_match_spells']['spells']}")
    from evaluation.possession_metrics import read_label_csv, validate_submission
    manifest = {'experiment': 'FirstTrial v13 — V12 with V10 gap model in long ball gaps',
                'created_utc': datetime.now(timezone.utc).isoformat(), 'gap_fallback_s': GAP_FALLBACK_S,
                'v12_manifest_sha256': sha256(args.v12_run/'manifest.json'),
                'v10_p_home_sha256': sha256(args.v10_run/'v10_p_home.npy'),
                'v7_transfer_reproduced_exactly': True, 'answer_free_checks': checks,
                'frames_changed_vs_v12': int((v13 != frozen['v12_turnover']).sum()),
                'source_sha256': {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted((ROOT/'solution').glob('*.py'))},
                'models': {}}
    for path in sorted((args.out/'predictions').glob('*.csv')):
        error = validate_submission(read_label_csv(path), n)
        if error:
            raise ValueError(f'{path.stem}: {error}')
        labels = pd.read_csv(path)['label'].to_numpy(dtype=str)
        manifest['models'][path.stem] = {'prediction_sha256': sha256(path), 'label_counts': dict(Counter(map(str, labels)))}
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    progress(f"Frozen. Frames changed vs V12: {manifest['frames_changed_vs_v12']:,}")
    if not args.skip_score:
        subprocess.run([sys.executable, str(ROOT/'evaluation/run_evaluation.py'), '--gt', 'public',
                        '--results', str(args.out/'predictions'), '--out', str(args.out/'scoring'),
                        '--bootstrap', str(args.bootstrap), '--no-detail'], cwd=ROOT, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['bench', 'bar-ath'])
    parser.add_argument('--v12-bench', type=Path, default=ROOT/'data/runs/external_bench_v12')
    parser.add_argument('--v10-bench', type=Path, default=ROOT/'data/runs/external_bench_v10')
    parser.add_argument('--v12-run', type=Path, default=ROOT/'data/runs/twelfth_trial')
    parser.add_argument('--v10-run', type=Path, default=ROOT/'data/runs/tenth_trial')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--skip-score', action='store_true')
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args(argv)
    args.out = (args.out or ROOT/('data/runs/external_bench_v13' if args.mode == 'bench' else 'data/runs/thirteenth_trial')).resolve()
    if not args.out.is_relative_to((ROOT/'data').resolve()) or (args.out.exists() and any(args.out.iterdir())):
        raise ValueError('Choose a fresh output directory inside ignored data/')
    progress = lambda message: print(message, flush=True)
    (bench if args.mode == 'bench' else bar_ath)(args, progress)


if __name__ == '__main__':
    main()
