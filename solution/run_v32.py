"""V32: a sequence model for possession, trained on all practice games (practice only).

Fixed 1 November 2026, before any feature export, training or scoring.

Why. On the 10 practice games, 75% of missed possessions last under 5 s, and the current model
misses most of them even with complete tracking. The current touch model is trained on the
3 Metrica games only and looks at short fixed windows. V32 replaces that one step with a neural
network that reads the whole sequence and is trained on the other 9 games (Metrica and DFL).

Setup (leave one game out, 10 folds; the target game is never used for training or for the
input scaling):
- Inputs per frame: the same touch and ball-free features the current turnover model uses
  (turnover_model.touch_features, both team orientations), no new information.
- Training copies of each training game: as recorded plus the realistic 'train_edge' copy.
  Test copy: the realistic 'test_edge' copy of the held-out game, as in V27/V31.
- Targets: Metrica provider labels; DFL controlled labels (touches under 1 s relabelled), as in V31.
- Model (no search): linear 128 + GELU, 2-layer bidirectional GRU (96 per direction), one
  output P(home has the ball). 40 s random windows, batch 16, AdamW lr 1e-3 weight decay 1e-4,
  4,000 steps, seed 3201 + fold, last weights. Home/away swap augmentation (50%); at test the
  two orientations are averaged; prediction uses the same 40 s windows with 50% overlap, keeping
  each window's middle half. Loss is binary cross-entropy on frames with a true H/A label,
  weighted x3 inside true possessions shorter than 5 s.
- Decoding: the existing two-state decoder with the frozen V27 dead-ball mask, then the
  existing 1 s flip cleanup. Switch cost for each target game is picked from {0.25, 0.5, 1.0} s
  by the best pooled Spell F1 on the OTHER nine games' out-of-fold predictions.
- Baseline: the saved V27 reference predictions (pooled controlled Spell F1 0.5258446186742693).

Pass rules (all required):
  P1 pooled controlled Spell F1 gain >= +0.03 over the baseline;
  P2 better in at least 8 of 10 games and >= 95% of whole-match bootstrap draws positive;
  P3 Metrica and DFL each gain >= 0, and the raw-target (original DFL labels) gain >= +0.02;
  P4 identical dead-ball masks, baseline reproduced, target game excluded from training.
If it passes, applying it to BAR-ATH needs its own checks first. No BAR-ATH answers, no
submission change, no commit or push.

Stages: `export` and `score` run with the project Python; training runs with the GPU environment
(solution/seq_model.py).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .run import ROOT

OUT = ROOT/'data/runs/v32_sequence'
BASELINE = 0.5258446186742693
PRIOR = ROOT/'data/runs/twentyseventh_trial/predictions'
GRID = (0.25, 0.5, 1.0)


def export():
    from evaluation.possession_metrics import read_label_csv
    from .run_bench10 import features_dict, get_bundle, load_sources
    from .run_v16 import controlled_possession
    from .turnover_model import touch_features
    (OUT/'features').mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(load_sources()):
        name, original = source['name'], source['match']
        raw = np.asarray(source['labels'])
        controlled = controlled_possession(raw, original.period, original.fps) if source['kind'] == 'dfl' else raw
        for version in ('recorded', 'train_edge', 'test_edge'):
            path = OUT/'features'/f'{name}_{version}.npz'
            if path.exists():
                continue
            m, b = get_bundle(source, index, version)
            tables = touch_features(features_dict(b), m.ball_xy_m, m.period, (b['bf_home'], b['bf_away']))
            extra = {}
            if version == 'test_edge':
                base = read_label_csv(PRIOR/f'realistic_{name}_reference.csv')['label'].to_numpy(dtype='<U1')
                extra = {'baseline': base, 'dead': base == 'D'}
            np.savez(path, x_home=tables[0].to_numpy(np.float16), x_away=tables[1].to_numpy(np.float16),
                     columns=np.array(tables[0].columns, dtype=str), period=m.period, fps=m.fps,
                     controlled=np.asarray(controlled, '<U1'), raw=raw.astype('<U1'), kind=source['kind'], **extra)
            print('exported', name, version, tables[0].shape, flush=True)


def score():
    from .run_v23 import cluster_difference, counts_score, spell_counts
    from .run_v27 import ownership_cleanup
    from .turnover_model import decode
    games = sorted(p.name[:-len('_test_edge.npz')] for p in (OUT/'features').glob('*_test_edge.npz'))
    data, preds = {}, {}
    for g in games:
        d = np.load(OUT/'features'/f'{g}_test_edge.npz')
        p = np.load(OUT/'predictions'/f'{g}.npz')
        if p['train_games'].tolist().count(g):
            raise ValueError(f'{g} was in its own training set')
        data[g] = d
        fps = int(d['fps'])
        preds[g] = {c: ownership_cleanup(decode(p['p_home'], d['period'], d['dead'], c, fps), d['period'], fps)
                    for c in GRID}
        for c in GRID:
            if not np.array_equal(preds[g][c] == 'D', d['dead']):
                raise ValueError('Dead mask changed')
    counts = {g: {t: {c: spell_counts(data[g][t], preds[g][c]) for c in GRID} for t in ('controlled', 'raw')}
              for g in games}
    base = {g: {t: spell_counts(data[g][t], data[g]['baseline']) for t in ('controlled', 'raw')} for g in games}
    chosen = {}
    for g in games:  # switch cost picked on the other nine games only
        others = [o for o in games if o != g]
        chosen[g] = max(GRID, key=lambda c: counts_score(sum(counts[o]['controlled'][c] for o in others)))
    rows, groups = [], {}
    for g in games:
        kind = str(data[g]['kind'])
        rows.append({'name': g, 'kind': kind, 'switch_cost': chosen[g],
                     **{f'{t}_{arm}': float(counts_score(c)) for t in ('controlled', 'raw')
                        for arm, c in (('baseline', base[g][t]), ('v32', counts[g][t][chosen[g]]))}})
    for t in ('controlled', 'raw'):
        for kind in ('all', 'metrica', 'dfl'):
            sel = [g for g in games if kind == 'all' or str(data[g]['kind']) == kind]
            b_ = float(counts_score(sum(base[g][t] for g in sel)))
            v_ = float(counts_score(sum(counts[g][t][chosen[g]] for g in sel)))
            groups[f'{t}/{kind}'] = {'baseline': b_, 'v32': v_, 'gain': v_ - b_}
    fixed = {c: float(counts_score(sum(counts[g]['controlled'][c] for g in games))) for c in GRID}
    boot = cluster_difference([counts[g]['controlled'][chosen[g]] for g in games], [base[g]['controlled'] for g in games])
    wins = sum(r['controlled_v32'] > r['controlled_baseline'] for r in rows)
    rules = {'P1': groups['controlled/all']['gain'] >= 0.03,
             'P2': wins >= 8 and boot['share_first_better'] >= 0.95,
             'P3': groups['controlled/metrica']['gain'] >= 0 and groups['controlled/dfl']['gain'] >= 0
                   and groups['raw/all']['gain'] >= 0.02,
             'P4': abs(groups['controlled/all']['baseline'] - BASELINE) < 1e-12}
    report = {'experiment': 'V32 sequence possession model (practice only)', 'groups': groups, 'wins': int(wins),
              'bootstrap': boot, 'fixed_switch_cost_diagnostic': fixed, 'rules': rules, 'pass': all(rules.values()),
              'per_match': rows, 'public_answers_read': False, 'submission_changed': False}
    (ROOT/'solution/results/first_trial_v32.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('groups', 'wins', 'bootstrap', 'fixed_switch_cost_diagnostic', 'rules')},
                     indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['export', 'score'])
    {'export': export, 'score': score}[parser.parse_args().stage]()
