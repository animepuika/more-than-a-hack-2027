"""V10 on BAR-ATH: trained ball-free gap model, adapted with BAR-ATH self-labels.

Self-labels are BAR-ATH frames where one team clearly controls a visible ball and
V7 agrees; no answer file is read. Three training sets are compared answer-free,
each half predicted by a model that never saw that half's self-labels:
Metrica only, BAR-ATH self-labels only, and both. The pre-declared choice is the
best long-gap-end accuracy (tie: agreement on V7 transfer windows). The chosen
set is then refitted on everything and its predictions are frozen before scoring.
Outputs stay under data/; no commit or push.
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
from .data import load_tracking
from .external_possession import attach_teams, possession_labels
from .external_samples import DEFAULT_SAMPLE_ROOT, ExternalMatch, load_metrica_samples
from .features import runs
from .gap_fallback import GAP_FALLBACK_S, gap_fallback_memory, long_gap_mask
from .gap_model import MODEL_SETTINGS, ballfree_features, fit, home_probability, model_fallback
from .run import ROOT, sha256, write_labels
from .run_v6 import _period_keys
from .run_v9 import BASELINES, PUBLIC_WINDOW, change_summary
from .v7_sequence import control_memory
from .v8_stoppages import build_evidence

TEAM = np.array(['H', 'A'])
CANDIDATES = ('metrica', 'bar_self', 'metrica_plus_bar_self')


def gap_end_table(p_home, direct, supported, reliable, held, period, fps):
    """Long gaps followed by confirmed control within 3 s: model's last-second call vs controller."""
    gaps = long_gap_mask(reliable, period, round(GAP_FALLBACK_S*fps)) & (period > 0)
    rows = []
    for lo, hi in runs(gaps):
        window = slice(hi, hi + 3*round(fps))
        after = np.flatnonzero((direct[window] >= 0) & supported[window] & reliable[window])
        if lo == 0 or not len(after) or held[lo-1] not in ('H', 'A') or period[lo-1] != period[hi-1]:
            continue
        tail = slice(max(lo, hi - round(fps)), hi)
        rows.append({'half': int(period[lo]), 'held': held[lo-1], 'post': TEAM[direct[hi+after[0]]],
                     'model': 'H' if p_home[tail].mean() > 0.5 else 'A'})
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run', type=Path, default=ROOT/'data/runs/seventh_trial')
    parser.add_argument('--v9-run', type=Path, default=ROOT/'data/runs/ninth_trial')
    parser.add_argument('--samples', type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument('--out', type=Path, default=ROOT/'data/runs/tenth_trial')
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
    v9_manifest = json.loads((args.v9_run/'manifest.json').read_text())
    v9_path = args.v9_run/'predictions/v9_gap_motion.csv'
    if sha256(v9_path) != v9_manifest['models']['v9_gap_motion']['prediction_sha256']:
        raise ValueError('Frozen V9 prediction changed')

    progress('Metrica ball-free features and labels.')
    matches = attach_teams(load_metrica_samples(args.samples), args.samples)
    metrica_tables = [ballfree_features(m.player_xy_m, m.home_slots, m.period, m.pitch, m.fps) for m in matches]
    metrica_labels = [possession_labels(m.events, m.period)[0] for m in matches]
    del matches

    n, fps, pitch = (GAMES[GAME][k] for k in ['n_video_frames', 'fps', 'pitch'])
    blocks = half_video_blocks(GAME)
    period, active = _period_keys(n, blocks)
    progress('Loading BAR-ATH tracking; public answers stay unopened until predictions are frozen.')
    tracking = load_tracking(paths['atd_meta'], paths['atd_raw'], n, fps, pitch, retain_tracks=True)
    match = ExternalMatch(GAME, tracking.track_xy_m, tracking.home_slots, tracking.ball_xy_m, period, [], fps, pitch)
    evidence = build_evidence(match, v8config, quality)
    v2 = refs['v2_shape_motion']
    v7, supported, direct = control_memory(evidence, period, quality, v2, dead=v2 == 'D', fps=fps)
    if not np.array_equal(v7, refs['v7_transfer']):
        raise ValueError('Light evidence did not reproduce V7 control predictions exactly')
    reliable = evidence['ball_reliable'] == 1
    bar_tables = ballfree_features(tracking.track_xy_m, tracking.home_slots, period, pitch, fps)
    controller = np.where(direct >= 0, TEAM[np.clip(direct, 0, 1)], '')
    self_labels = np.where(active & (direct >= 0) & (v7 == controller), controller, '')
    transfer_window = active & supported & (direct >= 0)

    progress('Cross-half comparison of training sets (answer-free).')
    cross = {name: np.full(n, 0.5) for name in CANDIDATES}
    metrica_model = fit(metrica_tables, metrica_labels)
    for half in (1, 2):
        rows = period == half
        other = np.where((period != half) & active, self_labels, '')
        table = (bar_tables[0][rows], bar_tables[1][rows])
        cross['metrica'][rows] = home_probability(metrica_model, table)
        bar_model = fit([bar_tables], [other], strides=[1])
        cross['bar_self'][rows] = home_probability(bar_model, table)
        both = fit([*metrica_tables, bar_tables], [*metrica_labels, other], strides=[5, 5, 5, 1])
        cross['metrica_plus_bar_self'][rows] = home_probability(both, table)
    comparison = {}
    for name, p in cross.items():
        guess = np.where(p > 0.5, 'H', 'A')
        labelled = self_labels != ''
        ends = gap_end_table(p, direct, supported, reliable, v7, period, fps)
        changed = ends.held != ends.post
        comparison[name] = {
            'self_label_agreement_other_half': float(np.mean(guess[labelled] == self_labels[labelled])),
            'transfer_window_agreement': float(np.mean(guess[transfer_window] == controller[transfer_window])),
            'gap_ends': len(ends), 'gap_end_accuracy': float(np.mean(ends.model == ends.post)),
            'gap_end_accuracy_team_changed': float(np.mean(ends.model[changed] == ends.post[changed])),
            'gap_end_accuracy_team_same': float(np.mean(ends.model[~changed] == ends.post[~changed]))}
        progress(f'  {name}: ' + ', '.join(f'{k} {v:.3f}' for k, v in comparison[name].items() if k != 'gap_ends'))
    chosen = max(CANDIDATES, key=lambda k: (round(comparison[k]['gap_end_accuracy'], 6),
                                            comparison[k]['transfer_window_agreement']))
    progress(f'Pre-declared choice: {chosen}')

    labelled_all = np.where(active, self_labels, '')
    if chosen == 'metrica':
        final = metrica_model
    elif chosen == 'bar_self':
        final = fit([bar_tables], [labelled_all], strides=[1])
    else:
        final = fit([*metrica_tables, bar_tables], [*metrica_labels, labelled_all], strides=[5, 5, 5, 1])
    p_home = np.full(n, 0.5)
    p_home[active] = home_probability(final, (bar_tables[0][active], bar_tables[1][active]))
    fallback = model_fallback(p_home, evidence, period, quality, v7, GAP_FALLBACK_S, dead=v2 == 'D', fps=fps)
    v10, long_gap = gap_fallback_memory(evidence, period, quality, fallback, GAP_FALLBACK_S, v2, dead=v2 == 'D', fps=fps)
    v10[~active] = v7[~active]

    out.mkdir(parents=True)
    (out/'predictions').mkdir()
    write_labels(out/'predictions/v10_gap_model.csv', v10)
    shutil.copyfile(v9_path, out/'predictions/v9_gap_motion.csv')
    shutil.copyfile(args.reference_run/'predictions/v7_transfer.csv', out/'predictions/v7_transfer.csv')
    for name in BASELINES:
        shutil.copyfile(ROOT/'evaluation/results'/(name+'.csv'), out/'predictions'/(name+'.csv'))
    np.save(out/'v10_p_home.npy', p_home)
    from evaluation.possession_metrics import read_label_csv, validate_submission
    v9 = pd.read_csv(v9_path)['label'].to_numpy(dtype='<U1')
    manifest = {
        'experiment': 'FirstTrial v10 — trained ball-free gap model with BAR-ATH self-labels',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'game': GAME, 'frames': n, 'fps': fps,
        'model_settings': MODEL_SETTINGS, 'gap_fallback_s': GAP_FALLBACK_S,
        'self_labels': {'definition': 'direct V7 control evidence on a reliable ball, where V7 labels the same team',
                        'frames': int((self_labels != '').sum())},
        'answer_free_comparison': comparison, 'chosen_training_set': chosen,
        'selection_rule': 'highest cross-half long-gap-end accuracy; tie-break transfer-window agreement',
        'v7_transfer_reproduced_exactly': True, 'source_sha256': source_hashes,
        'reference_manifest_sha256': sha256(args.reference_run/'manifest.json'),
        'v9_manifest_sha256': sha256(args.v9_run/'manifest.json'),
        'change_vs_v9_full_match': change_summary(v10, v9, period, long_gap, fps),
        'change_vs_v9_public_window': change_summary(v10, v9, period, long_gap, fps, PUBLIC_WINDOW),
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__},
        'limitations': [
            'Self-labels come from V7 control evidence, not answers; they exist only where the ball is visible.',
            'Cross-half checks reuse one match; gap-end truth is the first confirmed controller after a gap.',
            'Public scores reuse 17 spells and contain little long-gap time.',
            'Offline like V7: the gap decode uses the controller after the gap.'],
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
    progress(json.dumps({k: manifest[k] for k in ['change_vs_v9_full_match', 'change_vs_v9_public_window']}, indent=2))
    if not args.skip_score:
        progress('Scoring frozen predictions on the public sample.')
        subprocess.run([sys.executable, str(ROOT/'evaluation/run_evaluation.py'), '--gt', 'public',
                        '--results', str(out/'predictions'), '--out', str(out/'scoring'),
                        '--bootstrap', str(args.bootstrap), '--no-detail'], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
