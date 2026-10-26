"""Experiments 4 and 5: more SkillCorner matches (V17+) and a high-ball input (V18).

Test folds are the eight V17 folds (3 Metrica + the first 5 SkillCorner matches), each with the switch
cost V17 chose for it, so only the training data or the input set changes:
  v17_plus  V17's possession models trained on Metrica + every other SkillCorner match (20 in total)
  v18       v17_plus with P(ball above 2 m) from the high-ball detector as an extra input
The high-ball detector is itself trained leave-one-match-out on SkillCorner heights, so no match's own
heights feed its inputs. BAR-ATH: models on all matches, V16 gated + calibrated dead ball, answer-free
checks, frozen predictions, public sanity score. No commit or push.
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
from .aerial_model import aerial_features, aerial_probability, fit as fit_aerial, training_rows as aerial_rows
from .run import ROOT, sha256, write_labels
from .run_bench10 import bundle_job, features_dict, get_bundle, load_sources

V17_FOLDS = ROOT/'data/runs/seventeenth_trial/v17_bench_results.json'
VISIBLE_MIN = 10


def with_aerial(tables, p_aerial):
    return tuple(t.assign(p_aerial=np.asarray(p_aerial, np.float32)) for t in tables)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'data/runs/eighteenth_trial')
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
    from .skillcorner_open import load_skillcorner_matches
    from .turnover_model import decode, fit as fit_turnover, home_probability as turnover_probability, touch_features, training_rows
    v7c = json.loads(V7_CONFIG.read_text())
    folds = json.loads(V17_FOLDS.read_text())
    chosen = folds['chosen_switch_cost']

    sources = load_sources(('metrica', 'skillcorner'))
    names = [s['name'] for s in sources]
    kind = {s['name']: s['kind'] for s in sources}
    heights = {f'SC_{m.match_id}': m.ball_z for m in load_skillcorner_matches()}
    tests = list(chosen)
    jobs = [(s, i, v) for i, s in enumerate(sources) for v in ('recorded', 'ball25')]
    progress(f'Feature bundles: {len(jobs)} jobs ({len(names)} matches).')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name, version in pool.map(bundle_job, jobs):
            progress(f'  {name} {version}')

    progress('High-ball detector, leave one SkillCorner match out.')
    aerial_tables = {}
    for i, s in enumerate(sources):
        for version in ('recorded', 'ball25'):
            m, _ = get_bundle(s, i, version)
            aerial_tables[s['name'], version] = aerial_features(m.ball_xy_m, m.player_xy_m, m.period, m.fps)
    sc = [n for n in names if kind[n] == 'skillcorner']
    parts = {n: aerial_rows(aerial_tables[n, 'recorded'], heights[n]) for n in sc}
    aerial_models = {n: fit_aerial([parts[o] for o in sc if o != n]) for n in sc}
    aerial_all = fit_aerial([parts[o] for o in sc])
    detector = {}
    for n in sc:
        x, y = parts[n]
        p = aerial_probability(aerial_models[n], x)
        tp, fp, fn = int(((p > 0.5) & (y == 1)).sum()), int(((p > 0.5) & (y == 0)).sum()), int(((p <= 0.5) & (y == 1)).sum())
        order = np.argsort(p); ranks = np.empty(len(p)); ranks[order] = np.arange(len(p))
        pos = y == 1
        auc = (ranks[pos].mean() - (pos.sum() - 1) / 2) / max((~pos).sum(), 1) if pos.any() and (~pos).any() else None
        detector[n] = {'aerial_share': float(y.mean()), 'precision': tp / max(tp + fp, 1), 'recall': tp / max(tp + fn, 1), 'auc': auc}
    progress('  detector on held-out matches: ' + '  '.join(
        f"{n} AUC {v['auc']:.2f} P{v['precision']:.2f}/R{v['recall']:.2f}" for n, v in detector.items()))
    p_aerial = {(n, v): aerial_probability(aerial_models[n] if n in aerial_models else aerial_all, aerial_tables[n, v])
                for n in names for v in ('recorded', 'ball25')}

    rows = {}
    for i, s in enumerate(sources):
        plain, extra = [], []
        for version in ('recorded', 'ball25'):
            m, b = get_bundle(s, i, version)
            touch = touch_features(features_dict(b), m.ball_xy_m, m.period, (b['bf_home'], b['bf_away']))
            plain.append(training_rows(touch, s['labels']))
            extra.append(training_rows(with_aerial(touch, p_aerial[s['name'], version]), s['labels']))
        m, b = get_bundle(s, i, 'recorded')
        rows[s['name']] = {'plain': plain, 'aerial': extra, 'gap': ((b['bf_home'], b['bf_away']), s['labels'])}

    def models_for(train, variant):
        return {'turnover': fit_turnover([p for n in train for p in rows[n][variant]]),
                'gap': fit_gap([rows[n]['gap'][0] for n in train], [rows[n]['gap'][1] for n in train])}

    def possession(models, i, name, cost, variant):
        m, b = get_bundle(sources[i], i, 'recorded')
        feats, bf = features_dict(b), (b['bf_home'], b['bf_away'])
        touch = touch_features(feats, m.ball_xy_m, m.period, bf)
        if variant == 'aerial':
            touch = with_aerial(touch, p_aerial[name, 'recorded'])
        v12 = decode(turnover_probability(models['turnover'], touch), m.period, b['v1'] == 'D', cost, m.fps)
        return combine(v12, gap_probability(models['gap'], bf), feats, m.period, v7c, m.fps), b['players']

    progress('Leave-one-match-out possession models on the eight V17 folds.')
    preds, visible = {}, {}
    for h in tests:
        i = names.index(h)
        train = [n for n in names if n != h]
        preds[h] = {}
        for variant, key in (('plain', 'v17_plus'), ('aerial', 'v18')):
            preds[h][key], players = possession(models_for(train, variant), i, h, chosen[h], variant)
        visible[h] = players >= VISIBLE_MIN
        progress(f'  {h} done')
    f1 = lambda gt, pred: float(pm.evaluate(pm.label_codes(gt), pm.label_codes(pred))['spell_f1_mean'])
    reference = {k: v['spell_f1'] for k, v in folds['results'].items()}
    results = {}
    for group, members in (('skillcorner', [h for h in tests if kind[h] == 'skillcorner']),
                           ('metrica', [h for h in tests if kind[h] == 'metrica'])):
        for scope in (['all_frames', 'visible_frames'] if group == 'skillcorner' else ['all_frames']):
            truth = [np.where(visible[h] | (scope == 'all_frames'), sources[names.index(h)]['labels'], 'X') for h in members]
            gt = pooled(truth)
            joined = {k: pooled(preds[h][k] for h in members) for k in ('v17_plus', 'v18')}
            scores = {k: f1(gt, v) for k, v in joined.items()}
            diff = paired_difference(gt, joined['v18'], joined['v17_plus'])
            ref = reference.get(f'{group}/{scope}', {})
            results[f'{group}/{scope}'] = {'spell_f1': scores, 'v17_reference': ref, 'v18_minus_v17_plus': diff}
            progress(f"{group}/{scope}: V18 {scores['v18']:.4f}  V17+ {scores['v17_plus']:.4f}  "
                     f"(V17 {ref.get('v17_mixed', float('nan')):.4f}, V13 {ref.get('v13_reference', float('nan')):.4f})  "
                     f"V18-V17+ {diff['mean']:+.4f} ({diff['share_first_better']:.0%})")
    (out/'v18_bench_results.json').write_text(json.dumps(
        {'created_utc': datetime.now(timezone.utc).isoformat(), 'sources': names, 'detector': detector,
         'results': results, 'source_sha256': {q.name: sha256(q) for q in sorted((ROOT/'solution').glob('*.py'))}},
        indent=2, default=float)+'\n')

    progress('BAR-ATH with models on all matches (switch cost 0.5, V16 gated + calibrated dead ball).')
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
    n, fps, pitch = (GAMES[GAME][k] for k in ['n_video_frames', 'fps', 'pitch'])
    period, active = _period_keys(n, half_video_blocks(GAME))
    paths = asset_paths()
    tracking = load_tracking(paths['atd_meta'], paths['atd_raw'], n, fps, pitch, retain_tracks=True)
    match = ExternalMatch(GAME, tracking.track_xy_m, tracking.home_slots, tracking.ball_xy_m, period, [], fps, pitch)
    features, _ = build_features(match, v7c)
    bf = ballfree_features(tracking.track_xy_m, tracking.home_slots, period, pitch, fps)
    touch = touch_features(features, tracking.ball_xy_m, period, bf)
    p_air = aerial_probability(aerial_all, aerial_features(tracking.ball_xy_m, tracking.track_xy_m, period, fps))
    v2 = pd.read_csv(ROOT/'data/runs/seventh_trial/predictions/v2_shape_motion.csv')['label'].to_numpy(dtype='<U1')
    evidence = build_evidence(match, json.loads((ROOT/'solution/config_v8.json').read_text()), v7c)
    f = pd.read_csv(ROOT/'data/runs/seventh_trial/diagnostics/v7_features.csv',
                    usecols=['direct_control', 'ball_reliable', 'ball_speed_mps', 'players_observed'])
    live_ref = active & (f.direct_control.to_numpy() >= 0) & (f.ball_reliable.to_numpy() == 1) \
        & (np.nan_to_num(f.ball_speed_mps.to_numpy()) > 2)
    full = pd.Series(f.players_observed.to_numpy() >= 21).rolling(fps, center=True, min_periods=1).min().to_numpy().astype(bool)
    dead16 = calibrated_dead_mask(np.where(full, np.load(ROOT/'data/runs/sixteenth_trial/v16_p_dead.npy'), 0),
                                  period, v7c, live_ref, 0.02, fps)[0]
    labels_out = {}
    for variant, key in (('plain', 'v17_plus'), ('aerial', 'v18')):
        final = models_for(names, variant)
        t = with_aerial(touch, p_air) if variant == 'aerial' else touch
        p_turn = np.full(n, 0.5); p_gap = np.full(n, 0.5)
        p_turn[active] = turnover_probability(final['turnover'], (t[0][active], t[1][active]))
        p_gap[active] = gap_probability(final['gap'], (bf[0][active], bf[1][active]))
        poss = combine(decode(p_turn, period, (v2 == 'D') & active, 0.5, fps), p_gap, evidence, period, v7c, fps)
        labels_out[key] = overlay(poss, dead16, active)
    v7 = pd.read_csv(ROOT/'data/runs/seventh_trial/predictions/v7_transfer.csv')['label'].to_numpy(dtype='<U1')
    v7_check, supported, direct = control_memory(evidence, period, v7c, v2, dead=v2 == 'D', fps=fps)
    if not np.array_equal(v7_check, v7):
        raise ValueError('V7 not reproduced exactly')
    controller = np.where(direct >= 0, TEAM[np.clip(direct, 0, 1)], '')
    clear = active & (direct >= 0) & (v7 == controller)
    reliable = evidence['ball_reliable'] == 1
    public = np.zeros(n, bool); public[PUBLIC_WINDOW[0]:PUBLIC_WINDOW[1]] = True
    v17 = pd.read_csv(ROOT/'data/runs/seventeenth_trial/predictions/v17_mixed.csv')['label'].to_numpy(dtype='<U1')
    checks = {name: {'clear_control_agreement': float(np.mean(labels[clear] == controller[clear])),
                     'gap_ends': gap_end_scores(labels, direct, supported, reliable, v7, period, fps),
                     'full_match_spells': spell_stats(labels, active, fps),
                     'public_window_spells': spell_stats(labels, public, fps)}
              for name, labels in {**labels_out, 'v17_mixed': v17}.items()}
    for name, c in checks.items():
        progress(f"  {name}: clear {c['clear_control_agreement']:.3f}, gap ends {c['gap_ends']['correct']:.3f} "
                 f"(changed {c['gap_ends']['correct_team_changed']:.3f}), spells/live min {c['full_match_spells']['spells_per_live_minute']:.2f}")
    (out/'predictions').mkdir()
    for key, labels in labels_out.items():
        write_labels(out/'predictions'/(key+'.csv'), labels)
    for name in ['v17_mixed']:
        shutil.copyfile(ROOT/'data/runs/seventeenth_trial/predictions'/(name+'.csv'), out/'predictions'/(name+'.csv'))
    for name in ['v16_gated_cal', 'v14_confident_dead', 'v13_combined', 'v7_transfer']:
        shutil.copyfile(ROOT/'data/runs/sixteenth_trial_gated/predictions'/(name+'.csv'), out/'predictions'/(name+'.csv'))
    for name in BASELINES:
        shutil.copyfile(ROOT/'evaluation/results'/(name+'.csv'), out/'predictions'/(name+'.csv'))
    np.save(out/'bar_ath_p_aerial.npy', p_air)
    from evaluation.possession_metrics import read_label_csv, validate_submission
    manifest = {'experiment': 'FirstTrial v18 — 20 SkillCorner matches and a high-ball input',
                'created_utc': datetime.now(timezone.utc).isoformat(), 'answer_free_checks': checks,
                'bar_ath_aerial_share_p_over_half': float((p_air[active] > 0.5).mean()),
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
