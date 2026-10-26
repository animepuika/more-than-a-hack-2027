"""V20 on BAR-ATH: V16 gated + calibrated plus dead ball while the broadcast cuts away
(solution/camera_cutaway.py: edges trimmed, the main tracking decides the restart).

Pass rules, fixed before the run:
A. SkillCorner (cutaway = no player detected, trimmed the same way; its tracking stops during
   cutaways, so the restart rule cannot run there): trimmed cutaways at least 90% dead ball and
   covering at least 50% of dead ball, all 20 matches pooled.
B. BAR-ATH false dead ball on certainly-live frames at most 4.0%.
C. BAR-ATH dead-ball share between 30% and 50% of graded frames.
D. Set-piece tags with dead ball from 3 s before to 1 s after: at least 44 of 58 (V16 gated).
E. Possessions per live minute at most 7.9 (V16 gated 6.33 plus 25%).
The public score is computed only if A-E pass, as a sanity check; a drop of more than 0.05 below
V16 gated is flagged. No practice match has both a broadcast and a main tracking, so V20 has no
practice-match score. No commit or push.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from .run import ROOT, sha256, write_labels


def skillcorner_check(root):
    from .camera_cutaway import trim
    from .skillcorner_open import load_skillcorner_match
    totals = np.zeros(3)
    for folder in sorted(p for p in Path(root).iterdir() if p.is_dir() and (p / f'{p.name}_match.json').exists()):
        m = load_skillcorner_match(folder)
        keep = m.labels != 'X'
        cut = trim((np.isfinite(m.match.player_xy_m[..., 0]).sum(axis=1) == 0) & keep, 25)
        dead = (m.labels == 'D') & keep
        totals += [(cut & dead).sum(), cut.sum(), dead.sum()]
    both, n_cut, n_dead = totals
    precision, coverage = both / n_cut, both / n_dead
    return {'trimmed_cutaway_is_dead': float(precision), 'dead_in_trimmed_cutaway': float(coverage),
            'pass_A': bool(precision >= 0.90 and coverage >= 0.50)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'data/runs/twentieth_trial')
    parser.add_argument('--skillcorner', type=Path, default=ROOT/'data/external/skillcorner')
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if not out.is_relative_to((ROOT/'data').resolve()) or (out.exists() and any(out.iterdir())):
        raise ValueError('Choose a fresh output directory inside ignored data/')
    (out/'predictions').mkdir(parents=True)
    from games import GAME, GAMES, half_video_blocks
    from starter import load_smart_tagging
    from .camera_cutaway import cutaway_dead, cutaway_mask
    from .run_v6 import _period_keys
    from .run_v9 import BASELINES, PUBLIC_WINDOW
    from .run_v12 import spell_stats
    from .run_v14 import overlay
    n, fps = (GAMES[GAME][k] for k in ['n_video_frames', 'fps'])
    period, active = _period_keys(n, half_video_blocks(GAME))

    report = {'experiment': 'V20 — V16 gated plus broadcast-cutaway dead ball'}
    report['skillcorner'] = skillcorner_check(args.skillcorner)
    print('SkillCorner:', report['skillcorner'], flush=True)

    f = pd.read_csv(ROOT/'data/runs/seventh_trial/diagnostics/v7_features.csv',
                    usecols=['direct_control', 'ball_reliable', 'ball_speed_mps', 'speed_median_mps'])
    live = active & (f.direct_control.to_numpy() >= 0) & (f.ball_reliable.to_numpy() == 1) \
        & (np.nan_to_num(f.ball_speed_mps.to_numpy()) > 2)
    cut, seen = cutaway_mask(ROOT/'data/broadcast/BAR-ATH_broadcast_homography.csv', period, active)
    extra = cutaway_dead(cut, f.speed_median_mps.to_numpy(), fps)
    v16 = pd.read_csv(ROOT/'data/runs/sixteenth_trial_gated/predictions/v16_gated_cal.csv')['label'].to_numpy(dtype='<U1')
    labels = {'v16_gated_cal': v16, 'v20_cutaway': overlay(v16, extra, active)}
    tags = [e for e in load_smart_tagging.load() if (e['code'] or '').strip().casefold() == 'set pieces']
    windows = [(max(0, e['start_frame'] - 3 * fps), min(n, e['start_frame'] + fps)) for e in tags]
    windows = [(a, b) for a, b in windows if active[a:b].any()]
    public = np.zeros(n, bool); public[PUBLIC_WINDOW[0]:PUBLIC_WINDOW[1]] = True
    checks = {}
    for name, v in labels.items():
        d = v == 'D'
        spells = spell_stats(v, active, fps)
        tag_hits = sum(int(d[a:b].any()) for a, b in windows)
        checks[name] = {'false_dead': float(d[live].mean()), 'dead_share': float(d[active].mean()),
                        'dead_share_public_window': float(d[public].mean()),
                        'set_piece_tags_with_D': tag_hits, 'set_piece_tags': len(windows),
                        'spells_per_live_minute': spells['spells_per_live_minute'], 'full_match_spells': spells}
        write_labels(out/'predictions'/(name + '.csv'), v)
        print(name, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in checks[name].items() if k != 'full_match_spells'}, flush=True)
    c = checks['v20_cutaway']
    rules = {'A': report['skillcorner']['pass_A'], 'B': c['false_dead'] <= 0.040,
             'C': 0.30 <= c['dead_share'] <= 0.50, 'D': c['set_piece_tags_with_D'] >= 44,
             'E': c['spells_per_live_minute'] <= 7.9}
    report.update({'extra_dead_frames': int((extra & active).sum()), 'cutaway_share': float(cut[seen].mean()),
                   'answer_free_checks': checks, 'rules': rules, 'all_pass': bool(all(rules.values())), 'models': {},
                   'source_sha256': {q.relative_to(ROOT).as_posix(): sha256(q) for q in sorted((ROOT/'solution').glob('*.py'))}})
    print('rules:', rules, flush=True)
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
