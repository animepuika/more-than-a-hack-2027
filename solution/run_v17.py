"""V17: possession models trained on Metrica + SkillCorner (calm, phase-level possession labels).

Leave one match out over 3 Metrica + 5 SkillCorner matches. Turnover (V12) and gap (V10) models train
on every other match; each match also has a copy with 25% of its live ball hidden. The switch cost is
chosen from SWITCH_GRID_S using only the other held-out matches of the same source. Reference: V13
possession (Metrica-only models). D comes from V1's stoppage rules on both, so only possession differs.
BAR-ATH: final models on all eight matches, switch cost = best mean over the SkillCorner folds (the label
definition closest to BAR-ATH's), V2 D mask as in V12/V13, then the frozen V16 gated + calibrated dead
ball on top. Answer-free checks, frozen predictions, public sanity score. No commit or push.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from evaluation import possession_metrics as pm
from .run import ROOT, sha256, write_labels
from .run_bench10 import SWITCH_COST, bundle_job, features_dict, get_bundle, load_sources

SWITCH_GRID_S = (0.25, 0.5, 1.0, 2.0, 4.0)
VISIBLE_MIN = 10


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'data/runs/seventeenth_trial')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--skip-score', action='store_true')
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if not out.is_relative_to((ROOT/'data').resolve()) or (out.exists() and any(out.iterdir())):
        raise ValueError('Choose a fresh output directory inside ignored data/')
    out.mkdir(parents=True)
    progress = lambda message: print(message, flush=True)
    from .gap_model import fit as fit_gap, home_probability as gap_probability
    from .run_external import V7_CONFIG, pooled
    from .run_gap_model import paired_difference
    from .run_v13 import combine
    from .turnover_model import decode, fit as fit_turnover, home_probability as turnover_probability, touch_features, training_rows
    v7c = json.loads(V7_CONFIG.read_text())

    sources = load_sources(('metrica', 'skillcorner'))
    names = [s['name'] for s in sources]
    kind = {s['name']: s['kind'] for s in sources}
    jobs = [(s, i, v) for i, s in enumerate(sources) for v in ('recorded', 'ball25')]
    progress(f'Feature bundles: {len(jobs)} jobs.')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name, version in pool.map(bundle_job, jobs):
            progress(f'  {name} {version}')

    rows, tables = {}, {}
    for i, s in enumerate(sources):
        turn = []
        for version in ('recorded', 'ball25'):
            m, b = get_bundle(s, i, version)
            turn.append(training_rows(touch_features(features_dict(b), m.ball_xy_m, m.period, (b['bf_home'], b['bf_away'])), s['labels']))
        m, b = get_bundle(s, i, 'recorded')
        rows[s['name']] = {'turnover': turn, 'gap': ((b['bf_home'], b['bf_away']), s['labels'])}
        tables[s['name']] = (m, b)

    def models_for(train):
        return {'turnover': fit_turnover([p for n in train for p in rows[n]['turnover']]),
                'gap': fit_gap([rows[n]['gap'][0] for n in train], [rows[n]['gap'][1] for n in train])}

    def possession(models, name, cost):
        m, b = tables[name]
        feats, bf = features_dict(b), (b['bf_home'], b['bf_away'])
        v12 = decode(turnover_probability(models['turnover'], touch_features(feats, m.ball_xy_m, m.period, bf)),
                     m.period, b['v1'] == 'D', cost, m.fps)
        return combine(v12, gap_probability(models['gap'], bf), feats, m.period, v7c, m.fps)

    metrica = [n for n in names if kind[n] == 'metrica']
    progress('Leave-one-match-out models.')
    mixed = {h: models_for([n for n in names if n != h]) for h in names}
    reference = {h: models_for([n for n in metrica if n != h]) for h in names}
    # Mixed-model predictions for every switch cost, so costs can be chosen on the other folds.
    grid = {h: {c: possession(mixed[h], h, c) for c in SWITCH_GRID_S} for h in names}
    f1 = lambda gt, pred: float(pm.evaluate(pm.label_codes(gt), pm.label_codes(pred))['spell_f1_mean'])
    fold_scores = {h: {c: f1(sources[names.index(h)]['labels'], grid[h][c]) for c in SWITCH_GRID_S} for h in names}
    chosen, preds = {}, {}
    for h in names:
        others = [n for n in names if n != h and kind[n] == kind[h]]
        chosen[h] = max(SWITCH_GRID_S, key=lambda c: np.mean([fold_scores[n][c] for n in others]))
        ref_cost = SWITCH_COST.get(h, 0.5)
        preds[h] = {'v13_reference': possession(reference[h], h, ref_cost), 'v17_mixed': grid[h][chosen[h]]}
        progress(f'  {h}: chosen switch cost {chosen[h]}')

    results = {}
    for group, members in (('skillcorner', [n for n in names if kind[n] == 'skillcorner']), ('metrica', metrica)):
        scopes = [('all_frames', False)] + ([('visible_frames', True)] if group == 'skillcorner' else [])
        for scope, visible_only in scopes:
            truth = []
            for n in members:
                labels = sources[names.index(n)]['labels']
                if visible_only:
                    labels = np.where(tables[n][1]['players'] >= VISIBLE_MIN, labels, 'X')
                truth.append(labels)
            gt = pooled(truth)
            joined = {k: pooled(preds[n][k] for n in members) for k in ('v13_reference', 'v17_mixed')}
            scores = {k: f1(gt, v) for k, v in joined.items()}
            diff = paired_difference(gt, joined['v17_mixed'], joined['v13_reference'])
            results[f'{group}/{scope}'] = {'spell_f1': scores, 'v17_minus_v13': diff}
            progress(f"{group}/{scope}: V17 {scores['v17_mixed']:.4f}  V13 {scores['v13_reference']:.4f}  "
                     f"diff {diff['mean']:+.4f} ({diff['share_first_better']:.0%})")
    sc = [n for n in names if kind[n] == 'skillcorner']
    final_cost = max(SWITCH_GRID_S, key=lambda c: np.mean([fold_scores[n][c] for n in sc]))
    (out/'v17_bench_results.json').write_text(json.dumps(
        {'created_utc': datetime.now(timezone.utc).isoformat(), 'switch_grid_s': SWITCH_GRID_S,
         'chosen_switch_cost': chosen, 'fold_scores': fold_scores, 'bar_ath_switch_cost': final_cost,
         'results': results, 'source_sha256': {q.name: sha256(q) for q in sorted((ROOT/'solution').glob('*.py'))}},
        indent=2, default=float)+'\n')

    progress(f'BAR-ATH with models on all eight matches, switch cost {final_cost}.')
    from games import GAME, GAMES, asset_paths, half_video_blocks
    from .data import load_tracking
    from .deadball_formation import calibrated_dead_mask
    from .external_samples import ExternalMatch
    from .gap_model import ballfree_features
    from .run_v6 import _period_keys
    from .run_v9 import BASELINES, PUBLIC_WINDOW
    from .run_v10 import TEAM
    from .run_v12 import gap_end_scores, spell_stats
    from .run_v14 import overlay
    from .v7_features import build_features
    from .v7_sequence import control_memory
    from .v8_stoppages import build_evidence
    final = models_for(names)
    n, fps, pitch = (GAMES[GAME][k] for k in ['n_video_frames', 'fps', 'pitch'])
    period, active = _period_keys(n, half_video_blocks(GAME))
    paths = asset_paths()
    tracking = load_tracking(paths['atd_meta'], paths['atd_raw'], n, fps, pitch, retain_tracks=True)
    match = ExternalMatch(GAME, tracking.track_xy_m, tracking.home_slots, tracking.ball_xy_m, period, [], fps, pitch)
    features, _ = build_features(match, v7c)
    bf = ballfree_features(tracking.track_xy_m, tracking.home_slots, period, pitch, fps)
    v2 = pd.read_csv(ROOT/'data/runs/seventh_trial/predictions/v2_shape_motion.csv')['label'].to_numpy(dtype='<U1')
    p_turn = np.full(n, 0.5); p_gap = np.full(n, 0.5)
    touch = touch_features(features, tracking.ball_xy_m, period, bf)
    p_turn[active] = turnover_probability(final['turnover'], (touch[0][active], touch[1][active]))
    p_gap[active] = gap_probability(final['gap'], (bf[0][active], bf[1][active]))
    v12 = decode(p_turn, period, (v2 == 'D') & active, final_cost, fps)
    evidence = build_evidence(match, json.loads((ROOT/'solution/config_v8.json').read_text()), v7c)
    v17_poss = combine(v12, p_gap, evidence, period, v7c, fps)
    # Same dead ball as the current candidate: V16 probabilities, 21+ player gate, 2% calibration.
    f = pd.read_csv(ROOT/'data/runs/seventh_trial/diagnostics/v7_features.csv',
                    usecols=['direct_control', 'ball_reliable', 'ball_speed_mps', 'players_observed'])
    live_ref = active & (f.direct_control.to_numpy() >= 0) & (f.ball_reliable.to_numpy() == 1) \
        & (np.nan_to_num(f.ball_speed_mps.to_numpy()) > 2)
    full = pd.Series(f.players_observed.to_numpy() >= 21).rolling(fps, center=True, min_periods=1).min().to_numpy().astype(bool)
    p16 = np.load(ROOT/'data/runs/sixteenth_trial/v16_p_dead.npy')
    dead16 = calibrated_dead_mask(np.where(full, p16, 0), period, v7c, live_ref, 0.02, fps)[0]
    v17 = overlay(v17_poss, dead16, active)
    current = pd.read_csv(ROOT/'data/runs/sixteenth_trial_gated/predictions/v16_gated_cal.csv')['label'].to_numpy(dtype='<U1')
    v7 = pd.read_csv(ROOT/'data/runs/seventh_trial/predictions/v7_transfer.csv')['label'].to_numpy(dtype='<U1')
    v7_check, supported, direct = control_memory(evidence, period, v7c, v2, dead=v2 == 'D', fps=fps)
    if not np.array_equal(v7_check, v7):
        raise ValueError('V7 not reproduced exactly')
    controller = np.where(direct >= 0, TEAM[np.clip(direct, 0, 1)], '')
    clear = active & (direct >= 0) & (v7 == controller)
    reliable = evidence['ball_reliable'] == 1
    public = np.zeros(n, bool); public[PUBLIC_WINDOW[0]:PUBLIC_WINDOW[1]] = True
    checks = {name: {'clear_control_agreement': float(np.mean(labels[clear] == controller[clear])),
                     'gap_ends': gap_end_scores(labels, direct, supported, reliable, v7, period, fps),
                     'full_match_spells': spell_stats(labels, active, fps),
                     'public_window_spells': spell_stats(labels, public, fps),
                     'dead_share': float((labels == 'D')[active].mean())}
              for name, labels in {'v17_mixed': v17, 'v16_gated_cal': current}.items()}
    for name, c in checks.items():
        progress(f"  {name}: clear {c['clear_control_agreement']:.3f}, gap ends {c['gap_ends']['correct']:.3f} "
                 f"(changed {c['gap_ends']['correct_team_changed']:.3f}), spells/live min "
                 f"{c['full_match_spells']['spells_per_live_minute']:.2f}, dead share {c['dead_share']:.3f}")
    (out/'predictions').mkdir()
    write_labels(out/'predictions/v17_mixed.csv', v17)
    for name in ['v16_gated_cal', 'v14_confident_dead', 'v13_combined', 'v7_transfer']:
        shutil.copyfile(ROOT/'data/runs/sixteenth_trial_gated/predictions'/(name+'.csv'), out/'predictions'/(name+'.csv'))
    for name in BASELINES:
        shutil.copyfile(ROOT/'evaluation/results'/(name+'.csv'), out/'predictions'/(name+'.csv'))
    from evaluation.possession_metrics import read_label_csv, validate_submission
    manifest = {'experiment': 'FirstTrial v17 — possession trained on Metrica + SkillCorner, V16 gated dead ball',
                'created_utc': datetime.now(timezone.utc).isoformat(), 'switch_cost': final_cost,
                'answer_free_checks': checks,
                'skillcorner_sha256': (ROOT/'data/external/skillcorner-sha256.txt').read_text().splitlines(),
                'source_sha256': {q.relative_to(ROOT).as_posix(): sha256(q) for q in sorted((ROOT/'solution').glob('*.py'))},
                'models': {}}
    for path in sorted((out/'predictions').glob('*.csv')):
        error = validate_submission(read_label_csv(path), n)
        if error:
            raise ValueError(f'{path.stem}: {error}')
        manifest['models'][path.stem] = {'prediction_sha256': sha256(path)}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2, default=float)+'\n', encoding='utf-8')
    progress('Frozen; scoring on the public sample.')
    if not args.skip_score:
        subprocess.run([sys.executable, str(ROOT/'evaluation/run_evaluation.py'), '--gt', 'public',
                        '--results', str(out/'predictions'), '--out', str(out/'scoring'),
                        '--bootstrap', '1000', '--no-detail'], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
