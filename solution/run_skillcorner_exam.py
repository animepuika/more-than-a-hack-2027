"""Held-out exam on SkillCorner A-League matches: nothing from SkillCorner is used for training.

Models train on the 10 bench matches (Metrica + DFL) exactly as in run_bench10, then predict each
SkillCorner match as recorded (real broadcast drop-outs). Scores are pooled over the matches two ways:
all graded frames, and only frames where at least VISIBLE_MIN players are detected (others set to X),
because the broadcast cuts away during stoppages and leaves no players visible (an artifact absent
from BAR-ATH's tactical camera). Outputs stay under data/; no commit or push.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import possession_metrics as pm
from starter import baseline
from .run import ROOT, sha256
from .run_bench10 import (CALIBRATION, GATE_PLAYERS, SWITCH_COST, bundle_job, features_dict, get_bundle,
                          load_sources)

VISIBLE_MIN = 10
SC_SWITCH_COST = 0.5
METHODS = ['smooth50', 'v7', 'v13', 'v14', 'v14_gated', 'v16', 'v16_cal', 'v16_gated_cal']


def spell_rate(labels, keep, fps=25):
    labels = np.where(keep, labels, 'X')
    starts = np.r_[True, labels[1:] != labels[:-1]] & np.isin(labels, ['H', 'A'])
    return float(starts.sum() / max(np.isin(labels, ['H', 'A']).sum() / fps / 60, 1e-9))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'data/runs/skillcorner_exam')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if not out.is_relative_to((ROOT/'data').resolve()) or (out.exists() and any(out.iterdir())):
        raise ValueError('Choose a fresh output directory inside ignored data/')
    out.mkdir(parents=True)
    progress = lambda message: print(message, flush=True)
    from .deadball_formation import calibrated_dead_mask, dead_probability, fit as fit_dead
    from .gap_model import fit as fit_gap, home_probability as gap_probability
    from .run_external import V7_CONFIG, baseline_codes, pooled, tracking_from_match
    from .run_gap_model import paired_difference
    from .run_player_dropout import subset
    from .run_v13 import combine
    from .run_v14 import MOVING_BALL_MPS, overlay
    from .turnover_model import decode, fit as fit_turnover, home_probability as turnover_probability, touch_features, training_rows
    from .v7_model import dead_mask
    from .v7_sequence import control_evidence, control_memory
    v7c = json.loads(V7_CONFIG.read_text())

    train = load_sources(('metrica', 'dfl'))
    exam = load_sources(('skillcorner',))
    jobs = [(s, i, v) for i, s in enumerate(train) for v in ('recorded', 'train_realistic')]
    jobs += [(s, i, 'ball25') for i, s in enumerate(train) if s['kind'] == 'metrica']
    jobs += [(s, i, 'recorded') for i, s in enumerate(exam)]
    progress(f'Feature bundles: {len(jobs)} jobs (training ones should be cached).')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name, version in pool.map(bundle_job, jobs):
            progress(f'  {name} {version}')

    dead_rows, metrica_dead, poss_rows = [], [], {}
    for i, s in enumerate(train):
        for version in ('recorded', 'train_realistic'):
            _, b = get_bundle(s, i, version)
            dead_rows.append(subset(b['formation'], s['labels'], ['H', 'A', 'D']))
            if version == 'recorded' and s['kind'] == 'metrica':
                metrica_dead.append(dead_rows[-1])
        if s['kind'] == 'metrica':
            turn = []
            for version in ('recorded', 'ball25'):
                m, b = get_bundle(s, i, version)
                turn.append(training_rows(touch_features(features_dict(b), m.ball_xy_m, m.period, (b['bf_home'], b['bf_away'])), s['labels']))
            _, b = get_bundle(s, i, 'recorded')
            poss_rows[s['name']] = (turn, (b['bf_home'], b['bf_away']), s['labels'])
    v16_model = fit_dead([t for t, _ in dead_rows], [l for _, l in dead_rows], stride=1)
    v14_model = fit_dead([t for t, _ in metrica_dead], [l for _, l in metrica_dead], stride=1)
    turnover_model = fit_turnover([p for v in poss_rows.values() for p in v[0]])
    gap_model = fit_gap([v[1] for v in poss_rows.values()], [v[2] for v in poss_rows.values()])
    progress('Models trained on the 10 bench matches.')

    preds, visible, checks = {}, {}, {}
    for i, s in enumerate(exam):
        m, b = get_bundle(s, i, 'recorded')
        feats = features_dict(b)
        bf = (b['bf_home'], b['bf_away'])
        active = m.period > 0
        d_v1 = b['v1'] == 'D'
        tracking = tracking_from_match(m)
        v12 = decode(turnover_probability(turnover_model, touch_features(feats, m.ball_xy_m, m.period, bf)),
                     m.period, d_v1, SC_SWITCH_COST, m.fps)
        v13 = combine(v12, gap_probability(gap_model, bf), feats, m.period, v7c, m.fps)
        live_ref = ((control_evidence(feats, v7c) >= 0) & (feats['ball_reliable'] == 1)
                    & (np.nan_to_num(feats['ball_speed_mps']) > MOVING_BALL_MPS) & active)
        full = pd.Series(b['players'] >= GATE_PLAYERS).rolling(round(m.fps), center=True, min_periods=1).min().to_numpy().astype(bool)
        p14 = dead_probability(v14_model, b['formation'])
        p16 = dead_probability(v16_model, b['formation'])
        masks = {'v14': calibrated_dead_mask(p14, m.period, v7c, live_ref, CALIBRATION, m.fps)[0],
                 'v14_gated': calibrated_dead_mask(np.where(full, p14, 0), m.period, v7c, live_ref, CALIBRATION, m.fps)[0],
                 'v16': dead_mask(p16, m.period, v7c, m.fps) & active,
                 'v16_cal': calibrated_dead_mask(p16, m.period, v7c, live_ref, CALIBRATION, m.fps)[0],
                 'v16_gated_cal': calibrated_dead_mask(np.where(full, p16, 0), m.period, v7c, live_ref, CALIBRATION, m.fps)[0]}
        preds[s['name']] = {'smooth50': baseline.smooth(baseline_codes(tracking), 50),
                            'v7': control_memory(feats, m.period, v7c, b['v1'], dead=d_v1)[0],
                            'v13': v13, **{k: overlay(v13, mask, active) for k, mask in masks.items()}}
        visible[s['name']] = b['players'] >= VISIBLE_MIN
        graded = s['labels'] != 'X'
        checks[s['name']] = {'true_spells_per_live_min': spell_rate(s['labels'], graded),
                             'players_visible_median_live': float(np.median(b['players'][np.isin(s['labels'], ['H', 'A'])])),
                             **{k: {'spells_per_live_min': spell_rate(v, graded),
                                    'dead_recall': float((v == 'D')[graded & (s['labels'] == 'D')].mean()),
                                    'dead_precision': float((s['labels'][(v == 'D') & graded] == 'D').mean()) if ((v == 'D') & graded).any() else None}
                                for k, v in preds[s['name']].items()}}
        progress(f"  {s['name']}: true {checks[s['name']]['true_spells_per_live_min']:.2f} spells/min; "
                 + '  '.join(f"{k} {checks[s['name']][k]['spells_per_live_min']:.2f}" for k in METHODS))

    results = {}
    for scope in ('all_frames', 'visible_frames'):
        truth = [np.where(visible[s['name']] | (scope == 'all_frames'), s['labels'], 'X') for s in exam]
        gt = pooled(truth)
        joined = {k: pooled(preds[s['name']][k] for s in exam) for k in METHODS}
        scores = {k: float(pm.evaluate(pm.label_codes(gt), pm.label_codes(v))['spell_f1_mean']) for k, v in joined.items()}
        paired = {f'{k} - v13': paired_difference(gt, joined[k], joined['v13']) for k in METHODS if k != 'v13'}
        paired['v16_gated_cal - v14'] = paired_difference(gt, joined['v16_gated_cal'], joined['v14'])
        results[scope] = {'spell_f1': scores, 'paired': paired}
        progress(f'{scope}: ' + '  '.join(f'{k} {v:.4f}' for k, v in scores.items()))
        progress('   ' + '  '.join(f"{k}: {v['mean']:+.4f} ({v['share_first_better']:.0%})" for k, v in paired.items()))
    (out/'skillcorner_exam.json').write_text(json.dumps(
        {'created_utc': datetime.now(timezone.utc).isoformat(), 'visible_min': VISIBLE_MIN,
         'exam_matches': [s['name'] for s in exam], 'results': results, 'checks': checks,
         'skillcorner_sha256': (ROOT/'data/external/skillcorner-sha256.txt').read_text().splitlines(),
         'source_sha256': {q.name: sha256(q) for q in sorted((ROOT/'solution').glob('*.py'))}}, indent=2, default=float)+'\n')


if __name__ == '__main__':
    main()
