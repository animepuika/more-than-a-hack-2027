"""V12 on BAR-ATH: turnover-aware possession model trained on all three Metrica matches.

Training uses each Metrica match with the ball as recorded and with 25% of the
live ball hidden. The switch cost is fixed before this run at 0.5 s, the median
of the costs chosen on training matches in the bench (1.0, 0.5, 0.25). D comes
from the unchanged V2 shape+motion mask. Predictions are frozen and checked
without answers (gap ends, clear control, spell counts) before the public scorer
runs. Outputs stay under data/; no commit or push.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from games import GAME, GAMES, asset_paths, half_video_blocks
from .ball_masking import bar_ath_gap_lengths, hide_live_ball
from .data import load_tracking
from .external_possession import attach_teams, possession_labels
from .external_samples import DEFAULT_SAMPLE_ROOT, ExternalMatch, load_metrica_samples
from .gap_model import ballfree_features
from .run import ROOT, sha256, write_labels
from .run_external import GAP_SOURCE
from .run_v6 import _period_keys
from .run_v9 import BASELINES, PUBLIC_WINDOW
from .run_v10 import TEAM, gap_end_table
from .turnover_model import MODEL_SETTINGS, decode, fit, home_probability, touch_features, training_rows
from .v7_features import build_features
from .v7_sequence import control_memory
from .v8_stoppages import build_evidence

SWITCH_COST_S = 0.5
TRAIN_MASK, TRAIN_SEED = 0.25, 100


def spell_stats(labels, keep, fps):
    """Spell count, median length and share under 3 s inside the kept frames."""
    labels = np.where(keep, labels, 'X')
    edges = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    lengths = np.array([(b - a) / fps for a, b in zip(edges[:-1], edges[1:]) if labels[a] in ('H', 'A')])
    live_minutes = np.isin(labels, ['H', 'A']).sum() / fps / 60
    return {'spells': int(len(lengths)), 'spells_per_live_minute': float(len(lengths) / max(live_minutes, 1e-9)),
            'median_s': float(np.median(lengths)) if len(lengths) else None,
            'share_under_3s': float(np.mean(lengths < 3)) if len(lengths) else None}


def gap_end_scores(labels, direct, supported, reliable, held, period, fps):
    """Each method's majority label in the last second of a long gap vs the first confirmed controller
    after it. D frames count as half for each team, so a mostly-D second is scored as a guess."""
    p = np.where(labels == 'H', 1.0, np.where(labels == 'A', 0.0, 0.5))
    table = gap_end_table(p, direct, supported, reliable, held, period, fps)
    changed = table.held != table.post
    return {'gaps': len(table), 'correct': float(np.mean(table.model == table.post)),
            'correct_team_changed': float(np.mean(table.model[changed] == table.post[changed])),
            'correct_team_same': float(np.mean(table.model[~changed] == table.post[~changed]))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run', type=Path, default=ROOT/'data/runs/seventh_trial')
    parser.add_argument('--v10-run', type=Path, default=ROOT/'data/runs/tenth_trial')
    parser.add_argument('--samples', type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument('--out', type=Path, default=ROOT/'data/runs/twelfth_trial')
    parser.add_argument('--skip-score', action='store_true')
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if not out.is_relative_to((ROOT/'data').resolve()) or (out.exists() and any(out.iterdir())):
        raise ValueError('Choose a fresh output directory inside ignored data/')
    progress = lambda message: print(message, flush=True)
    source_hashes = {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted((ROOT/'solution').glob('*.py'))}
    quality_path, v8_path = ROOT/'solution/config_v7.json', ROOT/'solution/config_v8.json'
    quality, v8config = json.loads(quality_path.read_text()), json.loads(v8_path.read_text())
    previous = json.loads((args.reference_run/'manifest.json').read_text())
    if sha256(quality_path) != previous['config_sha256']:
        raise ValueError('V7 control settings differ from the reference run')
    for name, expected in previous['source_sha256'].items():
        if sha256(ROOT/name) != expected:
            raise ValueError(f'V7 source changed: {name}. Rebuild the reference first.')
    paths = asset_paths()
    for name, expected in previous['bar_ath_input_sha256'].items():
        if sha256(paths[name]) != expected:
            raise ValueError(f'BAR-ATH reference input changed: {name}')
    refs = {}
    for name in ['v7_transfer', 'v2_shape_motion']:
        path = args.reference_run/'predictions'/(name+'.csv')
        if sha256(path) != previous['models'][name]['prediction_sha256']:
            raise ValueError(f'Reference prediction changed: {name}')
        refs[name] = pd.read_csv(path)['label'].to_numpy(dtype='<U1')
    v10_manifest = json.loads((args.v10_run/'manifest.json').read_text())
    frozen = {}
    for name in ['v10_gap_model', 'v9_gap_motion']:
        path = args.v10_run/'predictions'/(name+'.csv')
        if sha256(path) != v10_manifest['models'][name]['prediction_sha256']:
            raise ValueError(f'Frozen prediction changed: {name}')
        frozen[name] = pd.read_csv(path)['label'].to_numpy(dtype='<U1')

    progress('Training V12 on all three Metrica matches (recorded ball and 25% hidden).')
    lengths = bar_ath_gap_lengths(GAP_SOURCE)
    matches = attach_teams(load_metrica_samples(args.samples), args.samples)
    parts = []
    for i, m in enumerate(matches):
        labels = possession_labels(m.events, m.period)[0]
        ballfree = ballfree_features(m.player_xy_m, m.home_slots, m.period, m.pitch, m.fps)
        for target, seed in ((0.0, 0), (TRAIN_MASK, TRAIN_SEED + i)):
            ball = m.ball_xy_m if not target else hide_live_ball(
                m.ball_xy_m, np.isin(labels, ['H', 'A']), m.period, lengths, target, seed)[0]
            masked = type(m)(m.name, m.player_xy_m, m.home_slots, ball, m.period, m.events, m.fps, m.pitch)
            features, _ = build_features(masked, quality)
            parts.append(training_rows(touch_features(features, ball, m.period, ballfree), labels))
    model = fit(parts)
    del matches, parts

    n, fps, pitch = (GAMES[GAME][k] for k in ['n_video_frames', 'fps', 'pitch'])
    period, active = _period_keys(n, half_video_blocks(GAME))
    progress('Loading BAR-ATH tracking; public answers stay unopened until predictions are frozen.')
    tracking = load_tracking(paths['atd_meta'], paths['atd_raw'], n, fps, pitch, retain_tracks=True)
    match = ExternalMatch(GAME, tracking.track_xy_m, tracking.home_slots, tracking.ball_xy_m, period, [], fps, pitch)
    evidence = build_evidence(match, v8config, quality)
    v2 = refs['v2_shape_motion']
    v7, supported, direct = control_memory(evidence, period, quality, v2, dead=v2 == 'D', fps=fps)
    if not np.array_equal(v7, refs['v7_transfer']):
        raise ValueError('Light evidence did not reproduce V7 control predictions exactly')
    features, _ = build_features(match, quality)
    tables = touch_features(features, tracking.ball_xy_m, period,
                            ballfree_features(tracking.track_xy_m, tracking.home_slots, period, pitch, fps))
    p_home = np.full(n, 0.5)
    p_home[active] = home_probability(model, (tables[0][active], tables[1][active]))
    v12 = decode(p_home, period, (v2 == 'D') & active, SWITCH_COST_S, fps)

    out.mkdir(parents=True)
    (out/'predictions').mkdir()
    write_labels(out/'predictions/v12_turnover.csv', v12)
    for name in ['v10_gap_model', 'v9_gap_motion']:
        shutil.copyfile(args.v10_run/'predictions'/(name+'.csv'), out/'predictions'/(name+'.csv'))
    shutil.copyfile(args.reference_run/'predictions/v7_transfer.csv', out/'predictions/v7_transfer.csv')
    for name in BASELINES:
        shutil.copyfile(ROOT/'evaluation/results'/(name+'.csv'), out/'predictions'/(name+'.csv'))
    np.save(out/'v12_p_home.npy', p_home)

    progress('Answer-free checks.')
    reliable = evidence['ball_reliable'] == 1
    controller = np.where(direct >= 0, TEAM[np.clip(direct, 0, 1)], '')
    clear = active & (direct >= 0) & (v7 == controller)
    window = active & supported & (direct >= 0)
    methods = {'v12_turnover': v12, 'v10_gap_model': frozen['v10_gap_model'], 'v9_gap_motion': frozen['v9_gap_motion'],
               'v7_transfer': v7}
    public = np.zeros(n, bool); public[PUBLIC_WINDOW[0]:PUBLIC_WINDOW[1]] = True
    checks = {name: {
        'clear_control_agreement': float(np.mean(labels[clear] == controller[clear])),
        'transfer_window_agreement': float(np.mean(labels[window] == controller[window])),
        'gap_ends': gap_end_scores(labels, direct, supported, reliable, v7, period, fps),
        'full_match_spells': spell_stats(labels, active, fps),
        'public_window_spells': spell_stats(labels, public, fps)} for name, labels in methods.items()}
    for name, c in checks.items():
        progress(f"  {name}: clear {c['clear_control_agreement']:.3f}, transfer windows {c['transfer_window_agreement']:.3f}, "
                 f"gap ends {c['gap_ends']['correct']:.3f} (changed {c['gap_ends']['correct_team_changed']:.3f}), "
                 f"spells {c['full_match_spells']['spells']} ({c['full_match_spells']['spells_per_live_minute']:.2f}/live min, "
                 f"{c['full_match_spells']['share_under_3s']:.0%} under 3 s)")

    from evaluation.possession_metrics import read_label_csv, validate_submission
    manifest = {
        'experiment': 'FirstTrial v12 — turnover-aware possession model on BAR-ATH',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'game': GAME, 'frames': n, 'fps': fps,
        'model_settings': MODEL_SETTINGS, 'switch_cost_s': SWITCH_COST_S,
        'training': 'All three Metrica matches, recorded ball and 25% hidden live ball (seed 100 + match index)',
        'dead_mask': 'v2_shape_motion D frames (unchanged)', 'answer_free_checks': checks,
        'v7_transfer_reproduced_exactly': True, 'source_sha256': source_hashes,
        'reference_manifest_sha256': sha256(args.reference_run/'manifest.json'),
        'v10_manifest_sha256': sha256(args.v10_run/'manifest.json'),
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__},
        'limitations': [
            'Trained on event-derived Metrica labels; BAR-ATH answers come from corrected tracking.',
            'Answer-free checks use V7 control evidence as a reference; V12 sees similar distance clues.',
            'Public scores reuse 17 inspected spells.',
            'Offline: uses frames after the current one.'],
        'models': {}}
    for path in sorted((out/'predictions').glob('*.csv')):
        error = validate_submission(read_label_csv(path), n)
        if error:
            raise ValueError(f'{path.stem}: {error}')
        labels = pd.read_csv(path)['label'].to_numpy(dtype=str)
        manifest['models'][path.stem] = {'prediction_sha256': sha256(path), 'official_format_valid': True,
                                         'label_counts': dict(Counter(map(str, labels)))}
    if source_hashes != {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted((ROOT/'solution').glob('*.py'))}:
        raise RuntimeError('Source changed during run; rerun frozen implementation')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    if not args.skip_score:
        progress('Scoring frozen predictions on the public sample.')
        subprocess.run([sys.executable, str(ROOT/'evaluation/run_evaluation.py'), '--gt', 'public',
                        '--results', str(out/'predictions'), '--out', str(out/'scoring'),
                        '--bootstrap', str(args.bootstrap), '--no-detail'], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
