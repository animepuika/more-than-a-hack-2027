"""Cached, parallel multi-source bench: V13, V14 and V16 dead-ball variants with coverage gating.

Sources: Metrica (3) and DFL (7). Expensive per-match features are cached on disk
(solution/feature_cache.py) and computed by several worker processes at once.
Variants added to V13's labels:
  v14            formation model trained on Metrica only, answer-free 2% calibration (as V14)
  v14_gated      the same, trusted only where 21+ players are tracked for a full second
  v16            formation model trained on the other nine matches, standard thresholds
  v16_cal        V16 with the 2% calibration
  v16_gated_cal  V16 trusted only where 21+ players are tracked, with the 2% calibration
  v16_gated_cal_lines  v16_gated_cal plus dead ball after line exits and restarts from the edge
                 (solution/line_restarts.py)
Conditions: clean (as recorded) and realistic (BAR-ATH-like player drop-out, ball hidden in 30% of
live and 35% of dead frames). Leave one match out. Outputs stay under data/; no commit or push.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import possession_metrics as pm
from .feature_cache import cached
from .run import ROOT, DEFAULT_CONFIG, read_config, sha256

MODULES = ['deadball_formation.py', 'v7_features.py', 'gap_model.py', 'features.py', 'predict.py', 'run_external.py',
           'data.py', 'context.py', 'deadball_features.py', 'v7_labels.py']
BUNDLE_VERSION = 'b1'  # bump when compute_bundle below changes
GATE_PLAYERS = 21
CALIBRATION = 0.02
DFL_SWITCH_COST = 0.5
SWITCH_COST = {'Sample_Game_1': 1.0, 'Sample_Game_2': 0.5, 'Sample_Game_3': 0.25}
SEEDS = {'train_realistic': 900, 'test_realistic': 500, 'ball25': 100, 'train_edge': 900, 'test_edge': 500}
VARIANTS = ['v13', 'v14', 'v14_gated', 'v16', 'v16_cal', 'v16_gated_cal', 'v16_gated_cal_bridged', 'v16_gated_cal_lines']


def load_sources(kinds=('metrica', 'dfl')):
    from .dfl_open import load_dfl_matches
    from .external_possession import attach_teams, possession_labels
    from .external_samples import DEFAULT_SAMPLE_ROOT, load_metrica_samples
    sources = []
    if 'metrica' in kinds:
        for m in attach_teams(load_metrica_samples(DEFAULT_SAMPLE_ROOT), DEFAULT_SAMPLE_ROOT):
            sources.append({'name': m.name, 'kind': 'metrica', 'match': m, 'labels': possession_labels(m.events, m.period)[0]})
    if 'dfl' in kinds:
        sources += [{'name': d.match.name, 'kind': 'dfl', 'match': d.match, 'labels': d.labels} for d in load_dfl_matches()]
    if 'skillcorner' in kinds:
        from .skillcorner_open import load_skillcorner_matches
        sources += [{'name': s.match.name, 'kind': 'skillcorner', 'match': s.match, 'labels': s.labels}
                    for s in load_skillcorner_matches()]
    return sources


def make_version(source, index, version):
    """Deterministic copy of a source match: recorded, realistic (train/test seeds) or 25% ball hidden."""
    from .ball_masking import bar_ath_gap_lengths, hide_live_ball
    from .deadball_v16 import hide_ball_realistic
    from .player_masking import bar_ath_missing_counts, drop_players
    from .run_external import GAP_SOURCE
    m = source['match']
    if version == 'recorded':
        return m
    gaps = bar_ath_gap_lengths(GAP_SOURCE)
    seed = SEEDS[version] + index
    if version == 'ball25':
        ball = hide_live_ball(m.ball_xy_m, np.isin(source['labels'], ['H', 'A']), m.period, gaps, 0.25, seed)[0]
        return type(m)(m.name, m.player_xy_m, m.home_slots, ball, m.period, m.events, m.fps, m.pitch)
    # '_edge' versions drop players with BAR-ATH's measured edge weights instead of uniformly.
    xy = drop_players(m.player_xy_m, m.period, bar_ath_missing_counts(GAP_SOURCE), seed,
                      pitch=m.pitch if version.endswith('_edge') else None)
    ball = hide_ball_realistic(m.ball_xy_m, source['labels'], m.period, gaps, seed)
    return type(m)(m.name, xy, m.home_slots, ball, m.period, m.events, m.fps, m.pitch)


def compute_bundle(m):
    from .deadball_formation import formation_features
    from .features import make_features
    from .gap_model import ballfree_features
    from .predict import predict
    from .run_external import V7_CONFIG, tracking_from_match
    from .v7_features import build_features
    from .v7_labels import period_runs
    v7c = json.loads(V7_CONFIG.read_text())
    base = read_config(DEFAULT_CONFIG)
    tracking = tracking_from_match(m)
    blocks = [(int(m.period[a]), a, b-1) for a, b in period_runs(m.period) if m.period[a] > 0]
    features, _ = build_features(m, v7c)
    home, away = ballfree_features(m.player_xy_m, m.home_slots, m.period, m.pitch, m.fps)
    return {'formation': formation_features(m.player_xy_m, m.home_slots, m.period, m.pitch, m.fps),
            'v7': pd.DataFrame(features), 'bf_home': home, 'bf_away': away,
            'v1': predict(tracking, make_features(tracking, blocks, base), blocks, base)[0],
            'players': np.isfinite(m.player_xy_m).all(axis=2).sum(axis=1)}


def bundle_job(job):
    source, index, version = job
    m = make_version(source, index, version)
    cached('bundle', m, MODULES, compute_bundle, extra=BUNDLE_VERSION)
    return source['name'], version


def get_bundle(source, index, version):
    m = make_version(source, index, version)
    return m, cached('bundle', m, MODULES, compute_bundle, extra=BUNDLE_VERSION)


def features_dict(bundle):
    return {c: bundle['v7'][c].to_numpy() for c in bundle['v7'].columns}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'data/runs/bench10_gated')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--dropout', choices=['random', 'edge'], default='random',
                        help='How realistic copies lose players: uniformly or with BAR-ATH edge weights')
    parser.add_argument('--bar-ath', action='store_true', help='Also run answer-free BAR-ATH dead-ball checks')
    parser.add_argument('--save-tuning', action='store_true',
                        help='Also predict on the training-seed copies and save V13 and V16 gated labels for timing tuning')
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if not out.is_relative_to((ROOT/'data').resolve()) or (out.exists() and any(out.iterdir())):
        raise ValueError('Choose a fresh output directory inside ignored data/')
    out.mkdir(parents=True)
    progress = lambda message: print(message, flush=True)
    from .deadball_formation import calibrated_dead_mask, dead_probability, fit as fit_dead
    from .gap_model import fit as fit_gap, home_probability as gap_probability
    from .run_external import V7_CONFIG, pooled
    from .run_gap_model import paired_difference
    from .run_player_dropout import subset
    from .run_v13 import combine
    from .run_v14 import MOVING_BALL_MPS, overlay
    from .turnover_model import decode, fit as fit_turnover, home_probability as turnover_probability, touch_features, training_rows
    from .v7_model import dead_mask
    from .v7_sequence import control_evidence
    from .deadball_bridge import bridge_blind_spots
    from .line_restarts import line_restart_dead
    v7c = json.loads(V7_CONFIG.read_text())
    train_v, test_v = (('train_edge', 'test_edge') if args.dropout == 'edge'
                       else ('train_realistic', 'test_realistic'))

    sources = load_sources()
    names = [s['name'] for s in sources]
    jobs = [(s, i, v) for i, s in enumerate(sources) for v in ('recorded', train_v, test_v)]
    jobs += [(s, i, 'ball25') for i, s in enumerate(sources) if s['kind'] == 'metrica']
    progress(f'Feature bundles: {len(jobs)} jobs on {args.workers} workers (cached ones are instant).')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name, version in pool.map(bundle_job, jobs):
            progress(f'  {name} {version}')

    progress('Training.')
    dead_rows, metrica_dead, poss_rows = {}, {}, {}
    for i, s in enumerate(sources):
        parts = []
        for version in ('recorded', train_v):
            _, b = get_bundle(s, i, version)
            parts.append(subset(b['formation'], s['labels'], ['H', 'A', 'D']))
            if version == 'recorded' and s['kind'] == 'metrica':
                metrica_dead[s['name']] = parts[-1]
        dead_rows[s['name']] = parts
        if s['kind'] == 'metrica':
            turn = []
            for version in ('recorded', 'ball25'):
                m, b = get_bundle(s, i, version)
                bf = (b['bf_home'], b['bf_away'])
                turn.append(training_rows(touch_features(features_dict(b), m.ball_xy_m, m.period, bf), s['labels']))
            _, b = get_bundle(s, i, 'recorded')
            poss_rows[s['name']] = {'turnover': turn, 'gap': ((b['bf_home'], b['bf_away']), s['labels'])}
    fit_rows = lambda rows: fit_dead([t for t, _ in rows], [l for _, l in rows], stride=1)
    v16_models = {h: fit_rows([p for n in names if n != h for p in dead_rows[n]]) for h in names}
    metrica_names = list(poss_rows)
    v14_models = {h: fit_rows([metrica_dead[n] for n in metrica_names if n != h]) for h in names}
    def poss_models(train):
        return {'turnover': fit_turnover([p for n in train for p in poss_rows[n]['turnover']]),
                'gap': fit_gap([poss_rows[n]['gap'][0] for n in train], [poss_rows[n]['gap'][1] for n in train])}
    poss_all = poss_models(metrica_names)
    poss = {n: (poss_models([x for x in metrica_names if x != n]) if n in metrica_names else poss_all) for n in names}
    progress('Models trained.')

    preds, checks = {c: {} for c in ('clean', 'realistic')}, {}
    line_counts = {c: np.zeros(4) for c in ('clean', 'realistic', 'tune')}
    if args.save_tuning:
        preds['tune'] = {}
    conditions = (('clean', 'recorded'), ('realistic', test_v)) + ((('tune', train_v),) if args.save_tuning else ())
    for i, s in enumerate(sources):
        for condition, version in conditions:
            m, b = get_bundle(s, i, version)
            feats = features_dict(b)
            bf = (b['bf_home'], b['bf_away'])
            active = m.period > 0
            d_v1 = b['v1'] == 'D'
            models = poss[s['name']]
            cost = SWITCH_COST.get(s['name'], DFL_SWITCH_COST)
            v12 = decode(turnover_probability(models['turnover'], touch_features(feats, m.ball_xy_m, m.period, bf)),
                         m.period, d_v1, cost, m.fps)
            v13 = combine(v12, gap_probability(models['gap'], bf), feats, m.period, v7c, m.fps)
            live_ref = ((control_evidence(feats, v7c) >= 0) & (feats['ball_reliable'] == 1)
                        & (np.nan_to_num(feats['ball_speed_mps']) > MOVING_BALL_MPS) & active)
            full = pd.Series(b['players'] >= GATE_PLAYERS).rolling(round(m.fps), center=True, min_periods=1).min().to_numpy().astype(bool)
            p14 = dead_probability(v14_models[s['name']], b['formation'])
            p16 = dead_probability(v16_models[s['name']], b['formation'])
            masks = {'v14': calibrated_dead_mask(p14, m.period, v7c, live_ref, CALIBRATION, m.fps)[0],
                     'v14_gated': calibrated_dead_mask(np.where(full, p14, 0), m.period, v7c, live_ref, CALIBRATION, m.fps)[0],
                     'v16': dead_mask(p16, m.period, v7c, m.fps) & active,
                     'v16_cal': calibrated_dead_mask(p16, m.period, v7c, live_ref, CALIBRATION, m.fps)[0],
                     'v16_gated_cal': calibrated_dead_mask(np.where(full, p16, 0), m.period, v7c, live_ref, CALIBRATION, m.fps)[0]}
            # Bridge on the candidate's final dead-ball decision (V13's D plus the gated formation D).
            masks['v16_gated_cal_bridged'] = bridge_blind_spots(masks['v16_gated_cal'] | (v13 == 'D'), full | ~active,
                                                                m.period, m.ball_xy_m, m.pitch, m.fps)
            masks['v16_gated_cal_lines'] = masks['v16_gated_cal'] | line_restart_dead(m.ball_xy_m, m.player_xy_m, m.period,
                                                                                      m.pitch, m.fps)
            preds[condition][s['name']] = {'v13': v13, **{k: overlay(v13, mask, active) for k, mask in masks.items()}}
            graded = s['labels'] != 'X'
            added = (preds[condition][s['name']]['v16_gated_cal_lines'] == 'D') & (preds[condition][s['name']]['v16_gated_cal'] != 'D') & graded
            line_counts[condition] += [added.sum(), (added & (s['labels'] == 'D')).sum(),
                                       ((preds[condition][s['name']]['v16_gated_cal_lines'] == 'D') & live_ref).sum(), live_ref.sum()]
            checks[f"{condition}/{s['name']}"] = {k: {
                'false_dead_on_live_ref': float((v == 'D')[live_ref].mean()) if live_ref.any() else None,
                'dead_precision': float((s['labels'][(v == 'D') & graded] == 'D').mean()) if ((v == 'D') & graded).any() else None,
                'dead_recall': float((v == 'D')[graded & (s['labels'] == 'D')].mean()),
                'false_dead_by_players': {g: float((v == 'D')[live_ref & sel].mean()) if (live_ref & sel).any() else None
                                          for g, sel in (('0-18', b['players'] < 19), ('19-20', (b['players'] >= 19) & (b['players'] < 21)),
                                                         ('21+', b['players'] >= 21))}}
                for k, v in preds[condition][s['name']].items()}
            progress(f"  {condition} {s['name']}: " + '  '.join(
                f"{k} R{c['dead_recall']:.2f}/P{(c['dead_precision'] or 0):.2f}" for k, c in checks[f"{condition}/{s['name']}"].items()))

    results = {}
    for condition in ('clean', 'realistic'):
        for group, members in (('metrica', metrica_names), ('dfl', [n for n in names if n not in metrica_names]), ('all', names)):
            gt = pooled(sources[names.index(n)]['labels'] for n in members)
            joined = {k: pooled(preds[condition][n][k] for n in members) for k in VARIANTS}
            scores = {k: float(pm.evaluate(pm.label_codes(gt), pm.label_codes(v))['spell_f1_mean']) for k, v in joined.items()}
            paired = {f'{k} - v13': paired_difference(gt, joined[k], joined['v13']) for k in VARIANTS[1:]}
            paired.update({f'{k} - v14': paired_difference(gt, joined[k], joined['v14']) for k in ('v14_gated', 'v16_gated_cal')})
            paired['v16_gated_cal_bridged - v16_gated_cal'] = paired_difference(gt, joined['v16_gated_cal_bridged'], joined['v16_gated_cal'])
            paired['v16_gated_cal_lines - v16_gated_cal'] = paired_difference(gt, joined['v16_gated_cal_lines'], joined['v16_gated_cal'])
            results[f'{condition}/{group}'] = {'spell_f1': scores, 'paired': paired}
            progress(f'{condition}/{group}: ' + '  '.join(f'{k} {v:.4f}' for k, v in scores.items()))
            progress('   ' + '  '.join(f"{k}: {v['mean']:+.4f} ({v['share_first_better']:.0%})" for k, v in paired.items()))
    if args.save_tuning:
        np.savez_compressed(out/'tuning_labels.npz', **{f'{c}|{n}|{k}': preds[c][n][k] for c in preds for n in preds[c]
                                                        for k in ('v13', 'v16_gated_cal')})
    lines = {c: {'added_frames': int(v[0]), 'added_truly_dead': float(v[1] / max(v[0], 1)),
                 'false_dead_on_live_ref': float(v[2] / max(v[3], 1))} for c, v in line_counts.items()}
    progress(f'line restarts: {lines}')
    (out/'bench10_results.json').write_text(json.dumps(
        {'created_utc': datetime.now(timezone.utc).isoformat(), 'gate_players': GATE_PLAYERS, 'calibration': CALIBRATION,
         'dropout': args.dropout, 'results': results, 'checks': checks, 'line_restarts': lines,
         'source_sha256': {q.name: sha256(q) for q in sorted((ROOT/'solution').glob('*.py'))}}, indent=2, default=float)+'\n')

    if args.bar_ath:
        bar_ath_check(out, fit_rows([p for n in names for p in dead_rows[n]]), progress)


def bar_ath_check(out, model, progress):
    """Answer-free BAR-ATH dead-ball behaviour of a formation model, by tracking coverage."""
    from games import GAME, GAMES, asset_paths, half_video_blocks
    from .data import load_tracking
    from .deadball_formation import calibrated_dead_mask, dead_probability, formation_features
    from .run_external import V7_CONFIG
    from .run_v6 import _period_keys
    from .v7_model import dead_mask
    v7c = json.loads(V7_CONFIG.read_text())
    n, fps, pitch = (GAMES[GAME][k] for k in ['n_video_frames', 'fps', 'pitch'])
    period, active = _period_keys(n, half_video_blocks(GAME))
    paths = asset_paths()
    tracking = load_tracking(paths['atd_meta'], paths['atd_raw'], n, fps, pitch, retain_tracks=True)
    table = formation_features(tracking.track_xy_m, tracking.home_slots, period, pitch, fps)
    p = np.zeros(n)
    p[active] = dead_probability(model, table[active])
    np.save(out/'bar_ath_p_dead.npy', p)
    f = pd.read_csv(ROOT/'data/runs/seventh_trial/diagnostics/v7_features.csv',
                    usecols=['direct_control', 'ball_reliable', 'ball_speed_mps', 'players_observed'])
    live = (active & (f.direct_control.to_numpy() >= 0) & (f.ball_reliable.to_numpy() == 1)
            & (np.nan_to_num(f.ball_speed_mps.to_numpy()) > 2))
    players = f.players_observed.to_numpy()
    full = pd.Series(players >= GATE_PLAYERS).rolling(fps, center=True, min_periods=1).min().to_numpy().astype(bool)
    masks = {'standard': dead_mask(p, period, v7c, fps) & active,
             'calibrated': calibrated_dead_mask(p, period, v7c, live, CALIBRATION, fps)[0],
             'gated_calibrated': calibrated_dead_mask(np.where(full, p, 0), period, v7c, live, CALIBRATION, fps)[0]}
    result = {}
    for name, d in masks.items():
        result[name] = {'false_dead': float(d[live].mean()), 'dead_share': float(d[active].mean()),
                        'false_dead_by_players': {g: float(d[live & sel].mean()) for g, sel in
                                                  (('0-18', players < 19), ('19-20', (players >= 19) & (players < 21)),
                                                   ('21+', players >= 21))}}
        progress(f"  BAR-ATH {name}: false dead {result[name]['false_dead']:.3f} by players "
                 f"{ {k: round(v, 3) for k, v in result[name]['false_dead_by_players'].items()} } dead share {result[name]['dead_share']:.3f}")
    (out/'bar_ath_check.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
