"""V7: repaired external weak labels, quality-aware features, controlled transfers.

Predictions and external selection are frozen before the public scorer runs.
Earlier experiment outputs are never overwritten. No command commits or pushes.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from games import GAME, GAMES, asset_paths, half_video_blocks
from starter import baseline, load_smart_tagging
from .context import build_context, proxy_costs
from .data import load_tracking
from .external_samples import ExternalMatch, DEFAULT_SAMPLE_ROOT, load_metrica_samples
from .features import make_features
from .predict import predict
from .run import ROOT, DEFAULT_CONFIG, read_config, sha256, write_labels, write_diagnostics
from .run_v2 import V2_CONFIG, context_config
from .run_v3 import V3_CONFIG, read_v3_config, _stoppage_variant
from .run_v6 import _period_keys, _sample_fingerprints
from .stoppages import build_stoppage_features, restart_windows
from .v7_labels import build_targets
from .v7_features import build_features, model_columns
from .v7_model import fit_compare, dead_mask
from .v7_sequence import control_memory, joint_decode

CONFIG_PATH = ROOT/'solution/config_v7.json'


def training_data(samples, cache_root, config, progress):
    fingerprint = {'inputs': _sample_fingerprints(samples), 'config': config,
        'source': {name: sha256(ROOT/'solution'/name) for name in
                   ['v7_labels.py', 'v7_features.py', 'context.py', 'deadball_features.py']}}
    key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    cache_root.mkdir(parents=True, exist_ok=True)
    cache = cache_root/(key+'.npz')
    metadata_file = cache_root/(key+'.json')
    if cache.exists() and metadata_file.exists():
        progress('Using hash-matched external feature/target cache.')
        with np.load(cache, allow_pickle=False) as data:
            arrays = [data[k] for k in ['x', 'y', 'groups', 'periods']]
        meta = json.loads(metadata_file.read_text())
        return (*arrays, meta['columns'], meta['label_audit'], fingerprint)
    matrices, labels, groups, periods, audits = [], [], [], [], []
    matches = load_metrica_samples(samples)
    columns = None
    for index, match in enumerate(matches):
        progress(f'Building repaired features/targets for {match.name} ({len(match):,} frames).')
        target, audit = build_targets(match.events, match.period, match.fps,
            config['event_boundary_margin_s'], config['maximum_live_event_bridge_s'])
        features, _ = build_features(match, config)
        matrix = model_columns(features)
        if columns is None:
            columns = list(matrix.columns)
        if list(matrix.columns) != columns:
            raise ValueError('External feature schemas differ')
        matrices.append(matrix.to_numpy(np.float32)); labels.append(target)
        groups.append(np.full(len(match), index, np.int8))
        periods.append((index+1)*10+match.period.astype(np.int16))
        audits.append({'match': match.name, **audit})
        progress(f"  live={audit['weak_live']:,}; dead={audit['weak_dead']:,}; unknown={audit['unknown']:,}; pass/D conflicts=0")
        matches[index] = None
        del features, matrix, match
        gc.collect()
    x, y = np.concatenate(matrices), np.concatenate(labels)
    group, period = np.concatenate(groups), np.concatenate(periods)
    np.savez_compressed(cache, x=x, y=y, groups=group, periods=period)
    metadata_file.write_text(json.dumps({'columns': columns, 'label_audit': audits,
                                        'fingerprint': fingerprint}, indent=2)+'\n')
    return x, y, group, period, columns, audits, fingerprint


def references(tracking, basic, blocks, base, context_config_, stop_config, tagging, progress):
    context = build_context(tracking, basic, blocks, context_config_,
        progress=lambda count, total: progress(f'  Reference context: {count:,}/{total:,}'))
    shape, shape_fit = proxy_costs(context['shape'], tracking, basic, blocks, context_config_,
                                  context_config_['shape_weight'])
    motion, motion_fit = proxy_costs(context['motion'], tracking, basic, blocks, context_config_,
                                    context_config_['motion_weight'])
    models = {}
    for name, cost in [('v2_shape', shape), ('v2_motion', motion), ('v2_shape_motion', shape+motion)]:
        models[name] = predict(tracking, basic, blocks, base, extra_team_costs=cost)
    clues = build_stoppage_features(tracking, basic, blocks, stop_config, tagging)
    v3, reasons, _, _ = _stoppage_variant(tracking, basic, blocks, base, shape+motion,
        clues, stop_config, hard_override=False, use_restart_tags=True, duration_frames=None)
    models['v3_tag_soft'] = (v3, reasons)
    # Positive logit supports Home; weak context affects conditional H/A only.
    context_logit = (shape[:, 1]+motion[:, 1])-(shape[:, 0]+motion[:, 0])
    return models, context_logit, {'shape': shape_fit, 'motion': motion_fit}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--config', type=Path, default=CONFIG_PATH)
    parser.add_argument('--out', type=Path, default=ROOT/'data/runs/seventh_trial')
    parser.add_argument('--skip-score', action='store_true')
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    out = args.out.resolve()
    if not out.is_relative_to((ROOT/'data').resolve()):
        raise ValueError('Keep V7 generated outputs in ignored data/')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Choose a fresh output directory')
    if args.bootstrap < 0:
        raise ValueError('bootstrap must be nonnegative')
    out.mkdir(parents=True, exist_ok=True)
    for name in ['predictions', 'diagnostics']:
        (out/name).mkdir()
    progress = lambda message: print(message, flush=True)
    source_hashes = {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted((ROOT/'solution').glob('*.py'))}
    x, y, groups, periods, columns, audit, training_fingerprint = training_data(
        args.samples, ROOT/'data/cache/v7', config, progress)
    (out/'training_label_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    model, selected, cv, oof = fit_compare(x, y, groups, periods, config, progress)
    (out/'cross_validation.json').write_text(json.dumps(cv, indent=2)+'\n')
    np.savez_compressed(out/'external_oof.npz', target=y, period=periods, group=groups, **oof)
    joblib.dump(model, out/'external_dead_model.joblib')
    del x, y, groups, periods, oof
    gc.collect()
    paths = asset_paths(data_dir=str(args.data_dir) if args.data_dir else None)
    progress('Loading BAR-ATH. Public answer labels remain unopened.')
    n, fps, pitch = (GAMES[GAME][k] for k in ['n_video_frames', 'fps', 'pitch'])
    blocks = half_video_blocks(GAME)
    period, active = _period_keys(n, blocks)
    base = read_config(DEFAULT_CONFIG); v2config = context_config(V2_CONFIG); stopconfig = read_v3_config(V3_CONFIG)
    tracking = load_tracking(paths['atd_meta'], paths['atd_raw'], n, fps, pitch, retain_tracks=True,
        progress=lambda count, total: progress(f'  Tracking: {count:,}/{total:,}'))
    match = ExternalMatch(GAME, tracking.track_xy_m, tracking.home_slots, tracking.ball_xy_m, period, [], fps, pitch)
    progress('Building BAR-ATH quality-aware features.')
    features, _ = build_features(match, config)
    matrix = model_columns(features)
    if list(matrix.columns) != columns:
        raise ValueError('BAR-ATH/external feature schemas differ')
    with threadpool_limits(limits=4):
        dead_score = model.predict_proba(matrix.to_numpy(np.float32))[:, 1]
    learned_dead = dead_mask(dead_score, period, config, fps) & active
    del matrix
    basic = make_features(tracking, blocks, base)
    tagging = load_smart_tagging.load(data_dir=str(args.data_dir) if args.data_dir else None)
    clock_check = load_smart_tagging.check_time_base(tagging)
    if not clock_check['agrees']:
        raise ValueError('Automatic tagging clock check failed')
    progress('Rebuilding unchanged V2/V3 controls.')
    models, context_logit, proxy_fitting = references(tracking, basic, blocks, base, v2config, stopconfig, tagging, progress)
    ref = baseline.smooth(baseline.build(data_dir=str(args.data_dir) if args.data_dir else None), 50)
    models = {'reference_smooth50': (ref, np.full(n, 'official_baseline')), **models}
    descriptions = {
        'reference_smooth50': 'Unchanged official reference.', 'v2_shape': 'Unchanged V2 shape reference.',
        'v2_motion': 'Unchanged V2 motion reference.', 'v2_shape_motion': 'Unchanged V2 shape plus motion reference.',
        'v3_tag_soft': 'Unchanged V3 soft stoppage/tag reference.',
        'v7_transfer': 'Confirmed control transfers; retains V2 shape/motion dead periods.',
        'v7_dead_v2': 'Repaired learned D overlay only, on V2 shape/motion.',
        'v7_dead_motion': 'Missing ablation: repaired learned D overlay on V2 motion-only.',
        'v7_joint': 'Factorized team/dead sequence with control memory and reset after D.',
        'v7_joint_context': 'Joint model plus weak team shape/motion conditional evidence.',
        'v7_joint_tags': 'Joint model plus automatic restart windows corroborated by tracking.',
    }
    v2labels = models['v2_shape_motion'][0]
    transfer, supported, direct = control_memory(features, period, config, v2labels, dead=v2labels == 'D', fps=fps)
    models['v7_transfer'] = (transfer, np.where(supported, 'v7_confirmed_transfer', 'v7_control_memory'))
    for name, parent in [('v7_dead_v2', 'v2_shape_motion'), ('v7_dead_motion', 'v2_motion')]:
        labels = models[parent][0].copy(); labels[learned_dead] = 'D'
        models[name] = (labels, np.where(learned_dead, 'v7_repaired_learned_dead', 'unchanged_parent'))
    for name, context, tags in [('v7_joint', None, None), ('v7_joint_context', context_logit, None),
                                ('v7_joint_tags', None, restart_windows(tagging, blocks, n, fps, 2.0))]:
        labels, _ = joint_decode(features, dead_score, period, config, context, tags)
        models[name] = (labels, np.where(labels == 'D', 'v7_joint_dead', 'v7_joint_control_memory'))
    progress('Writing frozen predictions and diagnostics before scoring.')
    for name, (labels, reasons) in models.items():
        write_labels(out/'predictions'/(name+'.csv'), labels)
        write_diagnostics(out/'diagnostics'/(name+'.csv'), labels, reasons, tracking, basic)
    pd.DataFrame({'frame': np.arange(n), 'period': period, 'dead_score': dead_score,
                  'learned_dead': learned_dead, 'direct_control': direct,
                  'confirmed_control': supported, **features}).to_csv(out/'diagnostics/v7_features.csv', index=False, float_format='%.4f')
    limitations = [
        'BAR-ATH public scores reuse only five minutes and 17 possessions; no full-match winner is established.',
        'External targets remain event-derived weak labels; unknown/conflicting spans and 0.2-second boundaries are masked.',
        'External interval metrics score labeled interiors, not timing inside masked boundaries; they are not official possession Spell F1.',
        'Two model families are selected on the same three external held-out matches; this is model-selection evidence, not a separate final test.',
        'Classifier scores are not calibrated BAR-ATH probabilities. Fixed thresholds and transfer timing are engineering choices, not publicly tuned.',
        'No additional video, broadcast fusion, referee trajectory, player identity or ball height is used.',
        'Control-memory transfer inference is not supervised on external H/A possession answers. D is the externally trained component.',
        'All methods are offline: future observations confirm control boundaries and centered feature windows are used.',
        'V4 finite-distance guard is repaired in source; historical V4 results remain historical and are not rerun here.',
    ]
    manifest = {
        'experiment': 'FirstTrial v7 — repaired supervision and controlled possession changes',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'game': GAME, 'frames': n, 'fps': fps,
        'report_design': 'Two external dead-ball classifiers with repaired labels and whole-match validation. Six fixed V7 ablations; five unchanged controls. Public answers opened only after prediction hashes were recorded.',
        'method_descriptions': descriptions, 'config': config, 'selected_external_model': selected,
        'external_cross_validation': cv, 'training_label_audit': audit, 'training_fingerprint': training_fingerprint,
        'feature_columns': columns, 'proxy_fitting': proxy_fitting, 'smart_tagging_clock_check': clock_check,
        'bar_ath_input_sha256': {k: sha256(paths[k]) for k in ['atd_meta', 'atd_raw', 'smart_tagging']},
        'source_sha256': source_hashes, 'config_sha256': sha256(args.config),
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__},
        'limitations': limitations, 'models': {},
    }
    from evaluation.possession_metrics import read_label_csv, validate_submission
    for name, (labels, reasons) in models.items():
        path = out/'predictions'/(name+'.csv')
        error = validate_submission(read_label_csv(path), n)
        if error:
            raise ValueError(f'{name}: {error}')
        manifest['models'][name] = {'prediction_sha256': sha256(path), 'label_counts': dict(Counter(map(str, labels))),
                                     'reason_counts': dict(Counter(map(str, reasons))), 'official_format_valid': True}
    if source_hashes != {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted((ROOT/'solution').glob('*.py'))}:
        raise RuntimeError('Solution source changed during run; rerun frozen code')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    if not args.skip_score:
        progress('Scoring the frozen predictions on the public five-minute sample.')
        subprocess.run([sys.executable, str(ROOT/'evaluation/run_evaluation.py'), '--gt', 'public',
                        '--results', str(out/'predictions'), '--out', str(out/'scoring'),
                        '--bootstrap', str(args.bootstrap), '--no-detail'], cwd=ROOT, check=True)
    from .report import write_report
    summary = write_report(ROOT, out, manifest, {name: value[0] for name, value in models.items()}, not args.skip_score)
    print(json.dumps(summary, indent=2), flush=True)
    progress(f'V7 comparison: {out / "comparison.html"}')


if __name__ == '__main__':
    main()
