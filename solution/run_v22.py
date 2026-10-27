"""V22 on BAR-ATH: V17's calmer possession with V20's dead ball, and the anti-split cleanup.

Candidates: v22a = V17 (possession trained on Metrica + SkillCorner, V16 gated dead ball) plus V20's
broadcast-cutaway dead ball; with --cleanup (run_split_tuning.py output, used only if its practice
rules passed), v20_clean and v22a_clean apply the chosen cleanup lengths. Answer-free rules against
V20, fixed before the run: A1 false dead on certainly-live frames at most 4.5%; A2 clear-control
agreement at most 1 point below V20; A3 gap-end accuracy at most 2 points below V20; A4 fewer
possessions per live minute than V20. The public score is computed only for candidates passing
every rule (V20 and the baselines are scored alongside). No commit or push.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from .run import ROOT, sha256, write_labels


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'data/runs/twentysecond_trial')
    parser.add_argument('--cleanup', type=Path, help='split_tuning.json from run_split_tuning')
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if not out.is_relative_to((ROOT/'data').resolve()) or (out.exists() and any(out.iterdir())):
        raise ValueError('Choose a fresh output directory inside ignored data/')
    (out/'predictions').mkdir(parents=True)
    from games import GAME, GAMES, asset_paths, half_video_blocks
    from .camera_cutaway import cutaway_dead, cutaway_mask
    from .run_v6 import _period_keys
    from .run_v9 import BASELINES
    from .run_v10 import TEAM
    from .run_v12 import gap_end_scores, spell_stats
    from .run_v14 import overlay
    from .split_cleanup import tidy
    n, fps = (GAMES[GAME][k] for k in ['n_video_frames', 'fps'])
    period, active = _period_keys(n, half_video_blocks(GAME))
    runs = ROOT/'data/runs'
    read = lambda path: pd.read_csv(path)['label'].to_numpy(dtype='<U1')
    diag = pd.read_csv(runs/'seventh_trial/diagnostics/v7_features.csv',
                       usecols=['direct_control', 'confirmed_control', 'ball_reliable', 'ball_speed_mps', 'speed_median_mps'])
    direct = diag.direct_control.to_numpy()
    supported = diag.confirmed_control.to_numpy().astype(bool)
    reliable = diag.ball_reliable.to_numpy() == 1
    live = active & (direct >= 0) & reliable & (np.nan_to_num(diag.ball_speed_mps.to_numpy()) > 2)
    v7 = read(runs/'seventh_trial/predictions/v7_transfer.csv')
    controller = np.where(direct >= 0, TEAM[np.clip(direct, 0, 1)], '')
    clear = active & (direct >= 0) & (v7 == controller)
    cut, _ = cutaway_mask(asset_paths(source='broadcast')['homography'], period, active)
    extra = cutaway_dead(cut, diag.speed_median_mps.to_numpy(), fps)

    labels = {'v20_cutaway': read(runs/'twentieth_trial/predictions/v20_cutaway.csv'),
              'v17_mixed': read(runs/'seventeenth_trial/predictions/v17_mixed.csv')}
    labels['v22a_calm_cutaway'] = overlay(labels['v17_mixed'], extra, active)
    cleanup = None
    if args.cleanup:
        tuned = json.loads(args.cleanup.read_text())
        if all(tuned['rules'].values()):
            cleanup = tuned['chosen']
            labels['v20_clean'] = tidy(labels['v20_cutaway'], period, fps, **cleanup)
            labels['v22a_clean'] = tidy(labels['v22a_calm_cutaway'], period, fps, **cleanup)

    checks = {}
    for name, v in labels.items():
        d = (v == 'D') & active
        checks[name] = {'false_dead': float(d[live].mean()), 'dead_share': float(d[active].mean()),
                        'clear_control_agreement': float(np.mean(v[clear] == controller[clear])),
                        'gap_ends': gap_end_scores(v, direct, supported, reliable, v7, period, fps),
                        'full_match_spells': spell_stats(v, active, fps)}
        c = checks[name]
        print(f"{name}: false dead {c['false_dead']:.4f}, dead {c['dead_share']:.3f}, clear {c['clear_control_agreement']:.4f}, "
              f"gap ends {c['gap_ends']['correct']:.3f}, spells/live min {c['full_match_spells']['spells_per_live_minute']:.2f}", flush=True)
    ref = checks['v20_cutaway']
    rules = {}
    for name in [k for k in labels if k not in ('v20_cutaway', 'v17_mixed')]:
        c = checks[name]
        rules[name] = {'A1': c['false_dead'] <= 0.045,
                       'A2': c['clear_control_agreement'] >= ref['clear_control_agreement'] - 0.01,
                       'A3': c['gap_ends']['correct'] >= ref['gap_ends']['correct'] - 0.02,
                       'A4': c['full_match_spells']['spells_per_live_minute'] < ref['full_match_spells']['spells_per_live_minute']}
        print(name, rules[name], flush=True)
    passing = [k for k, r in rules.items() if all(r.values())]
    for name in ['v20_cutaway', *passing]:
        write_labels(out/'predictions'/(name + '.csv'), labels[name])
    for name in BASELINES:
        (out/'predictions'/(name + '.csv')).write_bytes((ROOT/'evaluation/results'/(name + '.csv')).read_bytes())
    from evaluation.possession_metrics import read_label_csv, validate_submission
    models = {}
    for path in sorted((out/'predictions').glob('*.csv')):
        error = validate_submission(read_label_csv(path), n)
        if error:
            raise ValueError(f'{path.stem}: {error}')
        models[path.stem] = {'prediction_sha256': sha256(path)}
    report = {'experiment': 'V22 — calm possession with V20 dead ball, and anti-split cleanup', 'cleanup': cleanup,
              'answer_free_checks': checks, 'rules': rules, 'passing': passing, 'models': models,
              'source_sha256': {q.relative_to(ROOT).as_posix(): sha256(q) for q in sorted((ROOT/'solution').glob('*.py'))}}
    (out/'manifest.json').write_text(json.dumps(report, indent=2, default=float) + '\n', encoding='utf-8')
    if passing:
        subprocess.run([sys.executable, str(ROOT/'evaluation/run_evaluation.py'), '--gt', 'public',
                        '--results', str(out/'predictions'), '--out', str(out/'scoring'),
                        '--bootstrap', '1000', '--no-detail'], cwd=ROOT, check=True)
    else:
        print('No candidate passes every rule: public score not computed.')


if __name__ == '__main__':
    main()
