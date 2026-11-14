"""V36: the V34 sequence model trained on practice copies with a broad range of ball-gap lengths.

Fixed 13 November 2026, before any V36 copy, training or score. One attempt; if it fails, the
submission stays V33 and the sequence model is not pursued further.

Why. V34 passed every practice rule (+0.072) but V35 failed an answer-free check on BAR-ATH: on
frames where one player clearly controls a moving ball, its team agreed with that player 85.5% of
the time against 90.6% for V33. V34's practice copies hid the ball only in SkillCorner's short
gaps (median 1.5 s); real tracking can lose the ball far longer. The idea for this fix came from
that BAR-ATH check, and is recorded as such. Nothing in it is measured on BAR-ATH.

Gap lengths, generic and fixed: log-uniform between 0.5 s and 60 s (5,000 draws, seed 3600).
Practice copies (no players dropped; ball hidden in the same share of live and of dead frames):
  'recorded'      as provided (reused from V34)
  'train_broad25' 25% hidden, seed 3700+i;  'train_broad50' 50% hidden, seed 3800+i
  'test_broad25'  25% hidden, seed 3900+i
Model, windows, steps, seeds, loss and decoding exactly as V34 (solution/seq_model.py), trained on
the other nine games' 'recorded' + 'train_broad25' + 'train_broad50' copies. Switch cost picked
from {0.25, 0.5, 1.0} s on the other nine games. Baseline arm and dead-ball mask exactly as V34
(the submission's possession steps, unchanged).

Practice rules (all required): primary condition 'test_broad25'.
  P1 pooled controlled Spell F1 gain >= +0.03;  P2 >= 8/10 wins and >= 95% bootstrap draws positive;
  P3 Metrica and DFL each >= 0, raw-target gain >= +0.02, and 'recorded' gain >= 0.
If they pass: final model on all 10 games (seed 3211 as V35), V35's integration unchanged, then the
BAR-ATH checks of V35, with C5 required: clear-control agreement >= 0.906 (V33). C3 (clearly live
frames called dead) must not exceed V33's 6.0% by more than 0.5 points. Only if all pass does it
replace V33. The public five minutes are scored once afterwards, for information only.

    .venv/Scripts/python.exe -m solution.run_v36 export
    data/tools/media_env/Scripts/python.exe solution/seq_model.py --out data/runs/v36_sequence \
        --train recorded train_broad25 train_broad50 --test test_broad25 recorded
    .venv/Scripts/python.exe -m solution.run_v36 score
"""
import argparse
import json
import shutil

import numpy as np

from .run import ROOT
from .run_v34 import baseline_models

OUT = ROOT/'data/runs/v36_sequence'
V34 = ROOT/'data/runs/v34_sequence'
COPIES = {'train_broad25': (0.25, 3700), 'train_broad50': (0.50, 3800), 'test_broad25': (0.25, 3900)}
CONDITIONS = ('test_broad25', 'recorded')
GRID = (0.25, 0.5, 1.0)


def broad_lengths(fps=25):
    rng = np.random.RandomState(3600)
    return np.round(np.exp(rng.uniform(np.log(0.5 * fps), np.log(60 * fps), 5000))).astype(int)


def export():
    from dataclasses import replace
    from .deadball_formation import fixed_dead_mask
    from .deadball_v16 import hide_ball_realistic
    from .external_settings import load as external, practice_sources
    from .gap_model import ballfree_features, home_probability as gp
    from .run_v13 import combine
    from .run_v16 import controlled_possession
    from .run_v27 import ownership_cleanup
    from .turnover_model import decode, home_probability as tp, touch_features
    from .v7_features import build_features
    settings = external()
    quality = json.loads((ROOT/'solution/config_v7.json').read_text())
    (OUT/'features').mkdir(parents=True, exist_ok=True)
    games = practice_sources()
    for name, _, _ in games:  # the as-recorded copies are identical to V34's
        if not (OUT/'features'/f'{name}_recorded.npz').exists():
            shutil.copy(V34/'features'/f'{name}_recorded.npz', OUT/'features'/f'{name}_recorded.npz')
    lengths = broad_lengths()
    models = baseline_models([g for g in games if g[0].startswith('Sample_Game')],
                             np.asarray(settings['ball_gap_lengths_frames']), quality)
    print('baseline models trained', flush=True)
    for i, (name, m, raw) in enumerate(games):
        kind = 'metrica' if name.startswith('Sample_Game') else 'dfl'
        controlled = controlled_possession(raw, m.period, m.fps) if kind == 'dfl' else raw
        dead = fixed_dead_mask(np.load(ROOT/'data/runs/external_threshold'/f'{name}.npy'), m.period, quality,
                               settings['dead_start_threshold'], m.fps)
        model = models[name if kind == 'metrica' else 'all']
        for version, (share, seed) in COPIES.items():
            path = OUT/'features'/f'{name}_{version}.npz'
            if path.exists():
                continue
            c = replace(m, ball_xy_m=hide_ball_realistic(m.ball_xy_m, raw, m.period, lengths, seed + i, share, share))
            feats, _ = build_features(c, quality)
            bf = ballfree_features(c.player_xy_m, c.home_slots, c.period, c.pitch, c.fps)
            tables = touch_features(feats, c.ball_xy_m, c.period, bf)
            extra = {}
            if version in CONDITIONS:
                active = c.period > 0
                pturn, pgap = np.full(len(raw), 0.5), np.full(len(raw), 0.5)
                pturn[active] = tp(model['turn'], (tables[0][active], tables[1][active]))
                pgap[active] = gp(model['gap'], (bf[0][active], bf[1][active]))
                base = ownership_cleanup(combine(decode(pturn, c.period, dead, .5, c.fps), pgap, feats, c.period,
                                                 quality, c.fps), c.period, c.fps)
                if not np.array_equal(base == 'D', dead):
                    raise ValueError('Baseline changed the dead mask')
                extra = {'baseline': base, 'dead': dead}
            np.savez(path, x_home=tables[0].to_numpy(np.float16), x_away=tables[1].to_numpy(np.float16),
                     columns=np.array(tables[0].columns, dtype=str), period=c.period, fps=c.fps,
                     controlled=np.asarray(controlled, '<U1'), raw=np.asarray(raw, '<U1'), kind=kind, **extra)
            print('exported', name, version, flush=True)


def score():
    from .run_v23 import cluster_difference, counts_score, spell_counts
    from .run_v27 import ownership_cleanup
    from .turnover_model import decode
    games = sorted(p.name[:-len('_test_broad25.npz')] for p in (OUT/'features').glob('*_test_broad25.npz'))
    report = {'experiment': 'V36 sequence model with broad generic ball gaps (practice only)', 'conditions': {}}
    for cond in CONDITIONS:
        data, counts, base = {}, {}, {}
        for g in games:
            d = np.load(OUT/'features'/f'{g}_{cond}.npz')
            p = np.load(OUT/'predictions'/f'{g}_{cond}.npz')
            if g in p['train_games'].tolist():
                raise ValueError(f'{g} was in its own training set')
            fps = int(d['fps'])
            preds = {c: ownership_cleanup(decode(p['p_home'], d['period'], d['dead'], c, fps), d['period'], fps) for c in GRID}
            if any(not np.array_equal(v == 'D', d['dead']) for v in preds.values()):
                raise ValueError('Dead masks differ')
            data[g] = d
            counts[g] = {t: {c: spell_counts(d[t], preds[c]) for c in GRID} for t in ('controlled', 'raw')}
            base[g] = {t: spell_counts(d[t], d['baseline']) for t in ('controlled', 'raw')}
        chosen = {g: max(GRID, key=lambda c: counts_score(sum(counts[o]['controlled'][c] for o in games if o != g)))
                  for g in games}
        groups = {}
        for t in ('controlled', 'raw'):
            for kind in ('all', 'metrica', 'dfl'):
                sel = [g for g in games if kind == 'all' or str(data[g]['kind']) == kind]
                b_ = float(counts_score(sum(base[g][t] for g in sel)))
                v_ = float(counts_score(sum(counts[g][t][chosen[g]] for g in sel)))
                groups[f'{t}/{kind}'] = {'baseline': b_, 'v36': v_, 'gain': v_ - b_}
        per = [{'name': g, 'switch_cost': chosen[g], 'baseline': float(counts_score(base[g]['controlled'])),
                'v36': float(counts_score(counts[g]['controlled'][chosen[g]]))} for g in games]
        report['conditions'][cond] = {
            'groups': groups, 'per_match': per, 'wins': int(sum(r['v36'] > r['baseline'] for r in per)),
            'bootstrap': cluster_difference([counts[g]['controlled'][chosen[g]] for g in games],
                                            [base[g]['controlled'] for g in games]),
            'fixed_switch_cost_diagnostic': {c: float(counts_score(sum(counts[g]['controlled'][c] for g in games)))
                                             for c in GRID}}
    p, r = report['conditions']['test_broad25'], report['conditions']['recorded']
    report['practice_rules'] = {
        'P1': p['groups']['controlled/all']['gain'] >= 0.03,
        'P2': p['wins'] >= 8 and p['bootstrap']['share_first_better'] >= 0.95,
        'P3': p['groups']['controlled/metrica']['gain'] >= 0 and p['groups']['controlled/dfl']['gain'] >= 0
              and p['groups']['raw/all']['gain'] >= 0.02 and r['groups']['controlled/all']['gain'] >= 0}
    report['practice_pass'] = all(report['practice_rules'].values())
    report.update(public_or_bar_ath_data_used=False, submission_changed=False)
    (ROOT/'solution/results/first_trial_v36.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({c: {k: v for k, v in report['conditions'][c].items() if k != 'per_match'} for c in CONDITIONS}
                     | {'rules': report['practice_rules']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['export', 'score'])
    {'export': export, 'score': score}[parser.parse_args().stage]()
