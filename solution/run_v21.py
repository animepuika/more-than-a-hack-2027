"""V21 on BAR-ATH: V20 plus dead ball after line exits and restarts from the edge
(solution/line_restarts.py).

Pass rules, fixed before the run.
Practice matches (run_bench10 --dropout edge, variant v16_gated_cal_lines vs v16_gated_cal, realistic,
all ten matches pooled): P1 Spell F1 at least +0.005, better in at least 80% of bootstrap draws;
P2 at least 80% of added dead frames truly dead; P3 false dead on the live reference at most 4%.
BAR-ATH, answer-free: B1 false dead on certainly-live frames at most 4.5%; B2 dead share in the 15 s
before throw-in tags at least 0.45 and before goal-kick tags at least 0.50; B3 dead share at most 50%;
B4 possessions per live minute at most 7.9. The public score is computed only if every rule passes.
With --timing (solution/run_v21_timing.py output) this runs V21b: the restart timing picked on
practice copies, with P1-P3 taken from its judging copies; the B rules are unchanged.
No commit or push.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from .run import ROOT, sha256, write_labels


def bench_rules(path):
    r = json.loads(Path(path).read_text())
    paired = r['results']['realistic/all']['paired']['v16_gated_cal_lines - v16_gated_cal']
    lines = r['line_restarts']['realistic']
    return {'P1': paired['mean'] >= 0.005 and paired['share_first_better'] >= 0.80,
            'P2': lines['added_truly_dead'] >= 0.80, 'P3': lines['false_dead_on_live_ref'] <= 0.04}, \
        {'spell_f1_change': paired, 'line_restarts': lines,
         'spell_f1': {k: r['results']['realistic/all']['spell_f1'][k] for k in ('v16_gated_cal', 'v16_gated_cal_lines')}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'data/runs/twentyfirst_trial')
    parser.add_argument('--bench', type=Path, default=ROOT/'data/runs/bench10_lines/bench10_results.json')
    parser.add_argument('--timing', type=Path, help='V21b: timing.json from run_v21_timing')
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if not out.is_relative_to((ROOT/'data').resolve()) or (out.exists() and any(out.iterdir())):
        raise ValueError('Choose a fresh output directory inside ignored data/')
    (out/'predictions').mkdir(parents=True)
    from games import GAME, GAMES, asset_paths, half_video_blocks
    from starter import load_smart_tagging
    from .data import load_tracking
    from .line_restarts import line_restart_dead
    from .run_v6 import _period_keys
    from .run_v9 import BASELINES, PUBLIC_WINDOW
    from .run_v12 import spell_stats
    from .run_v14 import overlay
    n, fps, pitch = (GAMES[GAME][k] for k in ['n_video_frames', 'fps', 'pitch'])
    period, active = _period_keys(n, half_video_blocks(GAME))
    paths = asset_paths()
    tracking = load_tracking(paths['atd_meta'], paths['atd_raw'], n, fps, pitch, retain_tracks=True)
    tuned = json.loads(args.timing.read_text()) if args.timing else None
    name_new = 'v21b_lines' if tuned else 'v21_lines'
    lines = line_restart_dead(tracking.ball_xy_m, tracking.track_xy_m, period, pitch, fps,
                              timing=tuned['chosen'] if tuned else None) & active
    v20 = pd.read_csv(ROOT/'data/runs/twentieth_trial/predictions/v20_cutaway.csv')['label'].to_numpy(dtype='<U1')
    labels = {'v20_cutaway': v20, name_new: overlay(v20, lines, active)}
    f = pd.read_csv(ROOT/'data/runs/seventh_trial/diagnostics/v7_features.csv',
                    usecols=['direct_control', 'ball_reliable', 'ball_speed_mps'])
    live = active & (f.direct_control.to_numpy() >= 0) & (f.ball_reliable.to_numpy() == 1) \
        & (np.nan_to_num(f.ball_speed_mps.to_numpy()) > 2)
    tags = [e for e in load_smart_tagging.load() if (e['code'] or '').strip().casefold() == 'set pieces'
            and active[min(n - 1, e['start_frame'])]]
    public = np.zeros(n, bool); public[PUBLIC_WINDOW[0]:PUBLIC_WINDOW[1]] = True
    checks = {}
    for name, v in labels.items():
        d = (v == 'D') & active
        by = defaultdict(list)
        for e in tags:
            by[e['tags'].get('Type')].append(float(d[max(0, e['start_frame'] - 15 * fps):e['start_frame']].mean()))
        spells = spell_stats(v, active, fps)
        checks[name] = {'false_dead': float(d[live].mean()), 'dead_share': float(d[active].mean()),
                        'dead_share_public_window': float(d[public].mean()),
                        'dead_before_tags_15s': {k: float(np.mean(x)) for k, x in by.items()},
                        'spells_per_live_minute': spells['spells_per_live_minute'], 'full_match_spells': spells}
        write_labels(out/'predictions'/(name + '.csv'), v)
        print(name, json.dumps({k: x for k, x in checks[name].items() if k != 'full_match_spells'}, default=float), flush=True)
    c = checks[name_new]
    rules = {'B1': c['false_dead'] <= 0.045,
             'B2': c['dead_before_tags_15s'].get('Throw-in', 0) >= 0.45 and c['dead_before_tags_15s'].get('Goal Kick', 0) >= 0.50,
             'B3': c['dead_share'] <= 0.50, 'B4': c['spells_per_live_minute'] <= 7.9}
    if tuned:
        p_rules, bench = tuned['rules'], {'timing': tuned['chosen'], 'test_copies': tuned['test_copies']}
    else:
        p_rules, bench = bench_rules(args.bench)
    rules.update(p_rules)
    report = {'experiment': 'V21 — V20 plus line-exit and edge-restart dead ball', 'line_dead_frames': int(lines.sum()),
              'bench': bench, 'answer_free_checks': checks, 'rules': {k: bool(x) for k, x in rules.items()},
              'all_pass': bool(all(rules.values())), 'models': {},
              'source_sha256': {q.relative_to(ROOT).as_posix(): sha256(q) for q in sorted((ROOT/'solution').glob('*.py'))}}
    print('rules:', report['rules'], flush=True)
    for name in BASELINES:
        (out/'predictions'/(name + '.csv')).write_bytes((ROOT/'evaluation/results'/(name + '.csv')).read_bytes())
    from evaluation.possession_metrics import read_label_csv, validate_submission
    for path in sorted((out/'predictions').glob('*.csv')):
        error = validate_submission(read_label_csv(path), n)
        if error:
            raise ValueError(f'{path.stem}: {error}')
        report['models'][path.stem] = {'prediction_sha256': sha256(path)}
    (out/'manifest.json').write_text(json.dumps(report, indent=2, default=float) + '\n', encoding='utf-8')
    if report['all_pass']:
        subprocess.run([sys.executable, str(ROOT/'evaluation/run_evaluation.py'), '--gt', 'public',
                        '--results', str(out/'predictions'), '--out', str(out/'scoring'),
                        '--bootstrap', '1000', '--no-detail'], cwd=ROOT, check=True)
    else:
        print('Not all rules pass: public score not computed.')


if __name__ == '__main__':
    main()
