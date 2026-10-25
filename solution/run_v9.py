"""V9: V7 control memory that follows V2 motion through long ball gaps, on BAR-ATH.

Selected on the external Metrica bench (solution/run_external.py), not on the
public answers. Reuses hash-verified V7 reference predictions and V8's light
evidence, which must reproduce v7_transfer exactly. Predictions are frozen
before the public scorer runs. Outputs stay under data/; no commit or push.
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
from .external_samples import ExternalMatch
from .gap_fallback import GAP_FALLBACK_S, gap_fallback_memory
from .run import ROOT, sha256, write_labels
from .run_v6 import _period_keys
from .v7_sequence import control_memory
from .v8_stoppages import build_evidence

BASELINES = ['baseline_raw', 'baseline_smooth25', 'baseline_smooth50', 'baseline_smooth125']
PUBLIC_WINDOW = (11185, 18685)  # graded public video frames, as published in DATA.md
DESCRIPTIONS = {
    'v9_gap_motion': 'V7 control memory; ball gaps of at least 2 s follow V2 motion instead of holding the owner.',
    'v7_transfer': 'Unchanged V7 best public-sample control.',
    'v2_motion': 'Unchanged V2 motion control (the V9 gap fallback).',
    **{name: 'Unchanged official baseline shipped in evaluation/results.' for name in BASELINES},
}


def change_summary(new, old, period, long_gap, fps, window=None):
    """Frames and spells changed versus V7, optionally inside one frame window."""
    index = np.arange(len(new))
    keep = period > 0 if window is None else (index >= window[0]) & (index < window[1])
    changed = keep & (new != old)
    return {'frames': int(keep.sum()), 'changed_frames': int(changed.sum()),
            'changed_in_long_gaps': int((changed & long_gap).sum()),
            'long_gap_frames': int((keep & long_gap).sum()), 'long_gap_s': float((keep & long_gap).sum()/fps),
            'v9_spells': _spell_count(new, keep), 'v7_spells': _spell_count(old, keep)}


def _spell_count(labels, keep):
    """Maximal H/A runs inside the kept frames."""
    labels = np.where(keep, labels, 'X')
    starts = np.r_[True, labels[1:] != labels[:-1]]
    return int((starts & np.isin(labels, ['H', 'A'])).sum())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run', type=Path, default=ROOT/'data/runs/seventh_trial')
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--out', type=Path, default=ROOT/'data/runs/ninth_trial')
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
    paths = asset_paths(data_dir=str(args.data_dir) if args.data_dir else None)
    for name, expected in previous['bar_ath_input_sha256'].items():
        if sha256(paths[name]) != expected:
            raise ValueError(f'BAR-ATH reference input changed: {name}')
    refs = {}
    for name in ['v7_transfer', 'v2_shape_motion', 'v2_motion']:
        path = args.reference_run/'predictions'/(name+'.csv')
        if sha256(path) != previous['models'][name]['prediction_sha256']:
            raise ValueError(f'Reference prediction changed: {name}')
        refs[name] = pd.read_csv(path)['label'].to_numpy(dtype='<U1')
    baseline_hashes = {}
    for name in BASELINES:
        local, shipped = ROOT/'data/local_checks/baselines'/(name+'.csv'), ROOT/'evaluation/results'/(name+'.csv')
        if sha256(local) != sha256(shipped):
            raise ValueError(f'Local {name} differs from the shipped baseline')
        baseline_hashes[name] = sha256(shipped)

    n, fps, pitch = (GAMES[GAME][k] for k in ['n_video_frames', 'fps', 'pitch'])
    blocks = half_video_blocks(GAME)
    period, active = _period_keys(n, blocks)
    progress('Loading BAR-ATH tracking; public answers stay unopened until predictions are frozen.')
    tracking = load_tracking(paths['atd_meta'], paths['atd_raw'], n, fps, pitch, retain_tracks=True,
        progress=lambda count, total: progress(f'  Tracking: {count:,}/{total:,}'))
    match = ExternalMatch(GAME, tracking.track_xy_m, tracking.home_slots, tracking.ball_xy_m, period, [], fps, pitch)
    evidence = build_evidence(match, v8config, quality)
    base = refs['v2_shape_motion']
    reproduced, _, _ = control_memory(evidence, period, quality, base, dead=base == 'D', fps=fps)
    if not np.array_equal(reproduced, refs['v7_transfer']):
        raise ValueError('Light evidence did not reproduce V7 control predictions exactly')
    v9, long_gap = gap_fallback_memory(evidence, period, quality, refs['v2_motion'], GAP_FALLBACK_S,
                                       base, dead=base == 'D', fps=fps)
    v9[~active] = refs['v7_transfer'][~active]

    out.mkdir(parents=True)
    (out/'predictions').mkdir()
    write_labels(out/'predictions/v9_gap_motion.csv', v9)
    for name in ['v7_transfer', 'v2_motion']:
        shutil.copyfile(args.reference_run/'predictions'/(name+'.csv'), out/'predictions'/(name+'.csv'))
    for name in BASELINES:
        shutil.copyfile(ROOT/'evaluation/results'/(name+'.csv'), out/'predictions'/(name+'.csv'))
    from evaluation.possession_metrics import read_label_csv, validate_submission
    manifest = {
        'experiment': 'FirstTrial v9 — V7 control memory with a V2-motion fallback in long ball gaps',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'game': GAME, 'frames': n, 'fps': fps,
        'selection': 'Chosen on the external Metrica bench at four live-ball hiding levels before this run; '
                     'the 2 s gap threshold was declared before any scoring.',
        'gap_fallback_s': GAP_FALLBACK_S, 'method_descriptions': DESCRIPTIONS,
        'reference_manifest_sha256': sha256(args.reference_run/'manifest.json'),
        'v7_transfer_reproduced_exactly': True, 'source_sha256': source_hashes,
        'config_sha256': {'config_v7.json': sha256(quality_path), 'config_v8.json': sha256(v8_path)},
        'bar_ath_input_sha256': previous['bar_ath_input_sha256'], 'baseline_sha256': baseline_hashes,
        'change_vs_v7_full_match': change_summary(v9, refs['v7_transfer'], period, long_gap, fps),
        'change_vs_v7_public_window': change_summary(v9, refs['v7_transfer'], period, long_gap, fps,
                                                     PUBLIC_WINDOW),
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__},
        'limitations': [
            'Public scores reuse only five minutes and 17 possession spells; they cannot rank full-match methods.',
            'The public window has short ball gaps; most V9 changes fall outside it.',
            'Selection evidence comes from approximate event-derived Metrica labels with simulated live-ball gaps.',
            'V9 is offline: like V7 it confirms transfers with future observations.',
            'The fallback inherits V2 motion errors; it does not detect turnovers inside a gap by itself.'],
        'models': {}}
    for path in sorted((out/'predictions').glob('*.csv')):
        error = validate_submission(read_label_csv(path), n)
        if error:
            raise ValueError(f'{path.stem}: {error}')
        labels = pd.read_csv(path)['label'].to_numpy(dtype=str)
        manifest['models'][path.stem] = {'prediction_sha256': sha256(path), 'official_format_valid': True,
                                         'label_counts': dict(Counter(map(str, labels)))}
        progress(f'Validated {path.stem}: {n:,} rows')
    if source_hashes != {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted((ROOT/'solution').glob('*.py'))}:
        raise RuntimeError('Source changed during run; rerun frozen implementation')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    progress(json.dumps({k: manifest[k] for k in ['change_vs_v7_full_match', 'change_vs_v7_public_window']},
                        indent=2))
    if not args.skip_score:
        progress('Scoring frozen predictions on the public sample.')
        subprocess.run([sys.executable, str(ROOT/'evaluation/run_evaluation.py'), '--gt', 'public',
                        '--results', str(out/'predictions'), '--out', str(out/'scoring'),
                        '--bootstrap', str(args.bootstrap), '--no-detail'], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
