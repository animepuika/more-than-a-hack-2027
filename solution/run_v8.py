"""V8: fixed ball-evidence ablations and a V3-stoppage/V7-control combination.

Uses hash-verified V7 controls. No model fitting or public-label parameter search.
Outputs stay local under data/; this command never commits or pushes.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd

from games import GAME, GAMES, asset_paths, half_video_blocks
from starter import load_smart_tagging
from .data import load_tracking
from .external_samples import ExternalMatch, DEFAULT_SAMPLE_ROOT, load_metrica_samples
from .features import runs
from .run import ROOT, sha256, write_labels
from .run_v6 import _period_keys, _sample_fingerprints
from .stoppages import restart_windows
from .v7_labels import build_targets
from .v7_model import interval_metrics
from .v7_sequence import control_memory
from .v8_stoppages import build_evidence, decode_stops

CONFIG_PATH = ROOT/'solution/config_v8.json'
VARIANTS = {
    'v8_boundary': {'stationary': False},
    'v8_stationary': {},
    'v8_restart': {'persistent': True},
    'v8_pressure': {'persistent': True, 'pressure_guard': True},
    'v8_tags': {'persistent': True, 'pressure_guard': True, 'use_tags': True},
}
DESCRIPTIONS = {
    'v7_transfer': 'Unchanged V7 best public-sample control.',
    'v7_joint': 'Unchanged V7 joint team/dead control.',
    'v3_tag_soft': 'Unchanged V3 soft stoppage/tag control.',
    'v8_v3_control': 'V3 dead periods plus V7 confirmed-control memory; no new dead rules.',
    'v8_boundary': 'V7 control plus confirmed observed boundary exits.',
    'v8_stationary': 'Boundary + robust stationary ball and quiet players; nearby taker allowed.',
    'v8_restart': 'Same entries, with up to five seconds of unsupported stop memory and explicit restart.',
    'v8_pressure': 'Persistent rule with a nearby closing-opponent veto on stationary entries.',
    'v8_tags': 'Pressure rule plus automatic set-piece support, still requiring stable-ball/quiet evidence.',
}


def weak_scores(target, pred, period, fps):
    known = target >= 0
    tp = int(((target == 1) & pred).sum())
    fp = int(((target == 0) & pred).sum())
    fn = int(((target == 1) & ~pred).sum())
    return {'known_frames': int(known.sum()), 'dead_precision': tp/max(1, tp+fp),
            'dead_recall': tp/max(1, tp+fn), 'dead_frame_f1': 2*tp/max(1, 2*tp+fp+fn),
            'false_dead_s': fp/fps, 'missed_dead_s': fn/fps,
            **interval_metrics(target, pred, period, fps)}


def external_check(samples, config, quality, progress):
    result = {}
    matches = load_metrica_samples(samples)
    for index, match in enumerate(matches):
        progress(f'Checking fixed tracking rules on {match.name}: {len(match):,} frames.')
        # Events create evaluation targets only, never rule inputs.
        f = build_evidence(match, config, quality)
        target, audit = build_targets(match.events, match.period, match.fps,
            quality['event_boundary_margin_s'], quality['maximum_live_event_bridge_s'])
        result[match.name] = {'target_audit': audit, 'methods': {}}
        for name, arguments in VARIANTS.items():
            if arguments.get('use_tags'):
                continue  # avoid using the same event log for both targets and tag features
            pred, _, _ = decode_stops(f, match.period, match.fps, config, **arguments)
            result[match.name]['methods'][name] = weak_scores(target, pred, match.period, match.fps)
        matches[index] = None
        del f, target, match
        gc.collect()
    return result


def public_review(out, models, evidence, fps):
    """Reporting only, called after frozen prediction files and official scoring."""
    target = pd.read_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv')['label'].to_numpy()
    known = target != 'X'
    result = {'scope': 'Reused public five minutes only; diagnostic, not training data.', 'methods': {}}
    visible = evidence['ball_reliable'] == 1
    for name, labels in models.items():
        pred = labels == 'D'
        slices = {}
        for title, selected in [('all', known), ('reliable_ball', known & visible),
                                ('unreliable_or_missing_ball', known & ~visible)]:
            truth = target == 'D'
            tp = int((selected & pred & truth).sum())
            fp = int((selected & pred & ~truth).sum())
            fn = int((selected & ~pred & truth).sum())
            slices[title] = {'tp_frames': tp, 'fp_frames': fp, 'fn_frames': fn,
                             'precision': tp/max(1, tp+fp), 'recall': tp/max(1, tp+fn)}
        result['methods'][name] = slices
    result['true_dead_intervals'] = []
    for a, b in runs(known & (target == 'D')):
        result['true_dead_intervals'].append({'start_frame': int(a), 'end_frame_exclusive': int(b),
            'duration_s': (b-a)/fps, 'reliable_ball_fraction': float(visible[a:b].mean()),
            'detected_fraction': {name: float((labels[a:b] == 'D').mean()) for name, labels in models.items()}})
    (out/'deadball_review.json').write_text(json.dumps(result, indent=2)+'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run', type=Path, default=ROOT/'data/runs/seventh_trial')
    parser.add_argument('--samples', type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--config', type=Path, default=CONFIG_PATH)
    parser.add_argument('--out', type=Path, default=ROOT/'data/runs/eighth_trial')
    parser.add_argument('--skip-score', action='store_true')
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    quality_path = ROOT/'solution/config_v7.json'
    quality = json.loads(quality_path.read_text())
    out = args.out.resolve()
    if not out.is_relative_to((ROOT/'data').resolve()) or (out.exists() and any(out.iterdir())):
        raise ValueError('Choose a fresh output directory inside ignored data/')
    if args.bootstrap < 0:
        raise ValueError('bootstrap must be nonnegative')
    progress = lambda message: print(message, flush=True)
    source_hashes = {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted((ROOT/'solution').glob('*.py'))}
    previous = json.loads((args.reference_run/'manifest.json').read_text())
    if sha256(quality_path) != previous['config_sha256']:
        raise ValueError('V7 control settings differ from the reference run')
    for name, expected in previous['source_sha256'].items():
        if sha256(ROOT/name) != expected:
            raise ValueError(f'V7 source changed: {name}. Rebuild the reference first.')
    paths = asset_paths(data_dir=str(args.data_dir) if args.data_dir else None)
    for name, expected in previous['bar_ath_input_sha256'].items():
        if sha256(paths[name]) != expected:
            raise ValueError(f'BAR-ATH reference input changed: {name}')
    refs = {}
    for name in ['v7_transfer', 'v7_joint', 'v3_tag_soft', 'v2_shape_motion']:
        path = args.reference_run/'predictions'/(name+'.csv')
        if sha256(path) != previous['models'][name]['prediction_sha256']:
            raise ValueError(f'Reference prediction changed: {name}')
        refs[name] = pd.read_csv(path)['label'].to_numpy(dtype='<U1')
    out.mkdir(parents=True)
    (out/'predictions').mkdir(); (out/'diagnostics').mkdir()
    external = external_check(args.samples, config, quality, progress)
    (out/'external_rule_checks.json').write_text(json.dumps(external, indent=2)+'\n')
    n, fps, pitch = (GAMES[GAME][k] for k in ['n_video_frames', 'fps', 'pitch'])
    blocks = half_video_blocks(GAME)
    period, active = _period_keys(n, blocks)
    progress('Loading BAR-ATH and building observed-ball evidence; public answers are unopened.')
    tracking = load_tracking(paths['atd_meta'], paths['atd_raw'], n, fps, pitch, retain_tracks=True,
        progress=lambda count, total: progress(f'  Tracking: {count:,}/{total:,}'))
    match = ExternalMatch(GAME, tracking.track_xy_m, tracking.home_slots, tracking.ball_xy_m, period, [], fps, pitch)
    f = build_evidence(match, config, quality)
    reproduced, _, _ = control_memory(f, period, quality, refs['v2_shape_motion'],
                                      dead=refs['v2_shape_motion'] == 'D', fps=fps)
    if not np.array_equal(reproduced, refs['v7_transfer']):
        raise ValueError('Light feature calculations did not reproduce V7 control predictions exactly')
    tagging = load_smart_tagging.load(data_dir=str(args.data_dir) if args.data_dir else None)
    clock_check = load_smart_tagging.check_time_base(tagging)
    if not clock_check['agrees']:
        raise ValueError('Smart Tagging clock mismatch')
    tags = restart_windows(tagging, blocks, n, fps, config['tag_padding_s'])
    models = {name: refs[name] for name in ['v7_transfer', 'v7_joint', 'v3_tag_soft']}
    models['v8_v3_control'], _, _ = control_memory(f, period, quality, refs['v3_tag_soft'],
                                                 dead=refs['v3_tag_soft'] == 'D', fps=fps)
    base_dead = refs['v7_transfer'] == 'D'
    for name, arguments in VARIANTS.items():
        options = {k: v for k, v in arguments.items() if k != 'use_tags'}
        dead, reason, diagnostic = decode_stops(f, period, fps, config,
            tag_windows=tags if arguments.get('use_tags') else None, **options)
        models[name], _, _ = control_memory(f, period, quality, refs['v7_transfer'],
                                          dead=(dead | base_dead) & active, fps=fps)
        pd.DataFrame({'frame': np.arange(n), 'new_dead': dead, 'reason': reason, **diagnostic}).to_csv(
            out/'diagnostics'/(name+'.csv'), index=False)
    pd.DataFrame({'frame': np.arange(n), 'period': period, **f}).to_csv(
        out/'diagnostics/ball_evidence.csv', index=False, float_format='%.5f')
    from evaluation.possession_metrics import read_label_csv, validate_submission
    manifest = {'experiment': 'FirstTrial v8 — ball stability, restart memory and reused V3 stoppages',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'fps': fps, 'frames': n, 'game': GAME,
        'report_design': 'Six fixed V8 variants and three hash-verified controls. No fitting or parameter search. Fixed rules checked on three external matches before BAR-ATH scoring.',
        'method_descriptions': DESCRIPTIONS, 'config': config, 'config_sha256': sha256(args.config),
        'source_sha256': source_hashes, 'bar_ath_input_sha256': previous['bar_ath_input_sha256'],
        'reference_manifest_sha256': sha256(args.reference_run/'manifest.json'),
        'external_input_sha256': _sample_fingerprints(args.samples), 'external_rule_checks': external,
        'v7_transfer_reproduced_exactly': True, 'smart_tagging_clock_check': clock_check,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__},
        'limitations': [
            'Public scores reuse only five minutes and 17 possession spells; there is no independent full-match result.',
            'All thresholds are fixed engineering choices, not calibrated probabilities or optimized parameters.',
            'V8 adds no new training. External event-derived targets are approximate evaluation labels; uncertain boundaries are masked.',
            'External checks evaluate fixed standalone dead-ball rules, not official H/A possession Spell F1. Automatic BAR-ATH tags are not used in external checks.',
            'Stationary ball plus quiet players can still be live play. Five-second unsupported continuity can falsely extend a stop.',
            'Restart movement can be a returned or repositioned ball. Two-dimensional tracking cannot guarantee the referee decision.',
            'These methods are offline: confirmation can backdate entries, and V7 ownership can use future observations.',
            'A pressure veto is only a movement clue; it does not identify fouls, cards, fights, injuries or offside.',
            'No broadcast fusion, video model, permanent player identity, referee track or ball height was added.'], 'models': {}}
    for name, labels in models.items():
        path = out/'predictions'/(name+'.csv')
        write_labels(path, labels)
        error = validate_submission(read_label_csv(path), n)
        if error:
            raise ValueError(f'{name}: {error}')
        manifest['models'][name] = {'prediction_sha256': sha256(path), 'official_format_valid': True,
                                   'label_counts': dict(Counter(map(str, labels)))}
        progress(f'Validated {name}: {n:,} rows')
    if source_hashes != {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted((ROOT/'solution').glob('*.py'))}:
        raise RuntimeError('Source changed during run; rerun frozen implementation')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    if not args.skip_score:
        progress('Scoring frozen predictions on the public sample.')
        subprocess.run([sys.executable, str(ROOT/'evaluation/run_evaluation.py'), '--gt', 'public',
            '--results', str(out/'predictions'), '--out', str(out/'scoring'), '--bootstrap', str(args.bootstrap),
            '--no-detail'], cwd=ROOT, check=True)
        public_review(out, models, f, fps)
    from .report import write_report
    summary = write_report(ROOT, out, manifest, models, not args.skip_score)
    progress(json.dumps(summary, indent=2))
    progress(f'Local V8 report: {out / "comparison.html"}')


if __name__ == '__main__':
    main()
