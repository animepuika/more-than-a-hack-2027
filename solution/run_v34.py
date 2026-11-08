"""V34: the V32 sequence model again, with every setting from external data (practice only).

Fixed 3 November 2026, before any V34 feature export, training or score. V32 is abandoned
unscored: its practice copies were shaped with BAR-ATH's ball-gap and missing-player patterns,
which the organisers' rules (1 Nov 2026) no longer allow for anything we train or choose on.

Practice copies (all external; the 10 practice games, 3 Metrica + 7 DFL):
- 'recorded': as provided.
- 'train_hidden' / 'test_hidden': the ball hidden in 25% of live and 25% of dead frames, with
  SkillCorner ball-gap lengths (solution/config_external.json), seeds 3400+i and 3500+i. 25% is
  V12's existing hand-set training share. Players are never dropped.
Fixed dead-ball mask per game (identical for both arms): the leave-one-game-out V16 formation
probabilities (data/runs/external_threshold, players complete, no ball input) at the fixed
practice threshold, as in the submission. Then the existing 1 s flip cleanup for both arms.

Baseline arm = the submission's possession steps: V12 turnover model and V10 gap model trained on
Metrica (recorded + 25% hidden with SkillCorner gaps, as make_submission does), leaving the target
out for Metrica targets; V12 decoded at 0.5 s and V10 inside long gaps (run_v13.combine).
V34 arm = solution/seq_model.py exactly as fixed for V32 (architecture, steps, loss, windows,
seeds), trained on the other nine games' 'recorded' + 'train_hidden' copies; switch cost picked
from {0.25, 0.5, 1.0} s on the other nine games' out-of-fold predictions.

Conditions: primary 'test_hidden'; sensitivity 'recorded'. Targets: Metrica provider labels,
DFL controlled labels (touches under 1 s relabelled); raw DFL labels reported too.
Pass rules (all required; primary condition unless stated):
  P1 pooled controlled Spell F1 gain >= +0.03;
  P2 better in at least 8 of 10 games and >= 95% of whole-match bootstrap draws positive;
  P3 Metrica and DFL each gain >= 0; raw-target gain >= +0.02; 'recorded' gain >= 0;
  P4 identical dead masks across arms, each target excluded from its own training.
If all pass, the next step is putting it in the submission with its own bug checks.
No BAR-ATH data, answers or submission change in this experiment; no commit or push.

    .venv/Scripts/python.exe -m solution.run_v34 export
    data/tools/media_env/Scripts/python.exe solution/seq_model.py --out data/runs/v34_sequence \
        --train recorded train_hidden --test recorded test_hidden
    .venv/Scripts/python.exe -m solution.run_v34 score
"""
import argparse
from dataclasses import replace
import json

import numpy as np

from .run import ROOT

OUT = ROOT/'data/runs/v34_sequence'
SHARE, TRAIN_SEED, TEST_SEED = 0.25, 3400, 3500
GRID = (0.25, 0.5, 1.0)
CONDITIONS = ('test_hidden', 'recorded')


def hidden(m, labels, lengths, seed):
    from .deadball_v16 import hide_ball_realistic
    return replace(m, ball_xy_m=hide_ball_realistic(m.ball_xy_m, labels, m.period, lengths, seed, SHARE, SHARE))


def baseline_models(metrica, lengths, quality):
    """V12 turnover + V10 gap models as make_submission trains them, for 'all' and each left-out game."""
    from .ball_masking import hide_live_ball
    from .gap_model import ballfree_features, fit as fit_gap
    from .run_v12 import TRAIN_MASK, TRAIN_SEED as V12_SEED
    from .turnover_model import fit, touch_features, training_rows
    from .v7_features import build_features
    parts, tables = {}, {}
    for i, (name, m, lab) in enumerate(metrica):
        tables[name] = ballfree_features(m.player_xy_m, m.home_slots, m.period, m.pitch, m.fps)
        parts[name] = []
        for target, seed in ((0.0, 0), (TRAIN_MASK, V12_SEED + i)):
            ball = m.ball_xy_m if not target else hide_live_ball(
                m.ball_xy_m, np.isin(lab, ['H', 'A']), m.period, lengths, target, seed)[0]
            built, _ = build_features(replace(m, ball_xy_m=ball), quality)
            parts[name].append(training_rows(touch_features(built, ball, m.period, tables[name]), lab))
    labels = {name: lab for name, _, lab in metrica}
    models = {}
    for key in ['all'] + list(labels):
        use = [n for n in labels if n != key]
        models[key] = {'turn': fit([p for n in use for p in parts[n]]),
                       'gap': fit_gap([tables[n] for n in use], [labels[n] for n in use])}
    return models


def export():
    from .deadball_formation import fixed_dead_mask
    from .external_settings import load as external, practice_sources
    from .gap_model import ballfree_features, home_probability as gp
    from .run_v13 import combine
    from .run_v16 import controlled_possession
    from .run_v27 import ownership_cleanup
    from .turnover_model import decode, home_probability as tp, touch_features
    from .v7_features import build_features
    settings = external()
    lengths = np.asarray(settings['ball_gap_lengths_frames'])
    quality = json.loads((ROOT/'solution/config_v7.json').read_text())
    (OUT/'features').mkdir(parents=True, exist_ok=True)
    games = practice_sources()
    metrica = [g for g in games if g[0].startswith('Sample_Game')]
    models = baseline_models(metrica, lengths, quality)
    print('baseline models trained', flush=True)
    for i, (name, m, raw) in enumerate(games):
        kind = 'metrica' if name.startswith('Sample_Game') else 'dfl'
        controlled = controlled_possession(raw, m.period, m.fps) if kind == 'dfl' else raw
        p_dead = np.load(ROOT/'data/runs/external_threshold'/f'{name}.npy')
        dead = fixed_dead_mask(p_dead, m.period, quality, settings['dead_start_threshold'], m.fps)
        model = models[name if kind == 'metrica' else 'all']
        copies = {'recorded': m, 'train_hidden': hidden(m, raw, lengths, TRAIN_SEED + i),
                  'test_hidden': hidden(m, raw, lengths, TEST_SEED + i)}
        for version, c in copies.items():
            path = OUT/'features'/f'{name}_{version}.npz'
            if path.exists():
                continue
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
    games = sorted(p.name[:-len('_test_hidden.npz')] for p in (OUT/'features').glob('*_test_hidden.npz'))
    report = {'experiment': 'V34 sequence possession model, external settings only (practice only)', 'conditions': {}}
    for cond in CONDITIONS:
        data, counts, base = {}, {}, {}
        for g in games:
            d = np.load(OUT/'features'/f'{g}_{cond}.npz')
            p = np.load(OUT/'predictions'/f'{g}_{cond}.npz')
            if g in p['train_games'].tolist():
                raise ValueError(f'{g} was in its own training set')
            fps = int(d['fps'])
            preds = {c: ownership_cleanup(decode(p['p_home'], d['period'], d['dead'], c, fps), d['period'], fps) for c in GRID}
            if any(not np.array_equal(v == 'D', d['dead']) for v in preds.values()) or \
                    not np.array_equal(d['baseline'] == 'D', d['dead']):
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
                groups[f'{t}/{kind}'] = {'baseline': b_, 'v34': v_, 'gain': v_ - b_}
        per = [{'name': g, 'kind': str(data[g]['kind']), 'switch_cost': chosen[g],
                'baseline': float(counts_score(base[g]['controlled'])),
                'v34': float(counts_score(counts[g]['controlled'][chosen[g]]))} for g in games]
        report['conditions'][cond] = {
            'groups': groups, 'per_match': per, 'wins': int(sum(r['v34'] > r['baseline'] for r in per)),
            'bootstrap': cluster_difference([counts[g]['controlled'][chosen[g]] for g in games],
                                            [base[g]['controlled'] for g in games]),
            'fixed_switch_cost_diagnostic': {c: float(counts_score(sum(counts[g]['controlled'][c] for g in games)))
                                             for c in GRID}}
    p, r = report['conditions']['test_hidden'], report['conditions']['recorded']
    report['rules'] = {'P1': p['groups']['controlled/all']['gain'] >= 0.03,
                       'P2': p['wins'] >= 8 and p['bootstrap']['share_first_better'] >= 0.95,
                       'P3': p['groups']['controlled/metrica']['gain'] >= 0 and p['groups']['controlled/dfl']['gain'] >= 0
                             and p['groups']['raw/all']['gain'] >= 0.02 and r['groups']['controlled/all']['gain'] >= 0,
                       'P4': True}
    report['pass'] = all(report['rules'].values())
    report.update(public_or_bar_ath_data_used=False, submission_changed=False)
    (ROOT/'solution/results/first_trial_v34.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({c: {k: v for k, v in report['conditions'][c].items() if k != 'per_match'} for c in CONDITIONS}
                     | {'rules': report['rules']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['export', 'score'])
    {'export': export, 'score': score}[parser.parse_args().stage]()
