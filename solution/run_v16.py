"""V16: dead ball learned from DFL + Metrica from player formations, robust to data loss.

Ball clues are switched off (USE_BALL_CLUES). Both external sources carry a ball artifact at
stoppages: Metrica's ball vanishes, and DFL's ball is frozen with exactly zero speed in every
dead frame (about 1-2% of live frames). Neither happens on BAR-ATH, where the ball keeps moving
during stoppages, so a model trained on those clues learns a shortcut that does not transfer.
Sources: 3 Metrica matches (event-derived labels) and 7 DFL matches (tracking-based
BallStatus/BallPossession).
Each dead-ball model trains on the other nine matches, each as recorded and as a realistic copy
(BAR-ATH-like player drop-out; ball hidden in 30% of live and 35% of dead frames). Thresholds are
the V7 defaults. Possession comes from V13 (V12 turnovers + V10 gaps, Metrica-trained; V1 stoppage
rules as base). V16 = V13 with V16 dead ball added. Then BAR-ATH: model on all ten matches,
answer-free checks, frozen predictions, public sanity score. No commit or push.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from evaluation import possession_metrics as pm
from .ball_masking import bar_ath_gap_lengths, hide_live_ball
from .deadball_formation import dead_probability, fit as fit_dead
from .deadball_v16 import dead_features, hide_ball_realistic
from .dfl_open import load_dfl_matches
from .external_possession import attach_teams, possession_labels
from .external_samples import DEFAULT_SAMPLE_ROOT, load_metrica_samples
from .features import make_features
from .gap_model import ballfree_features, fit as fit_gap, home_probability as gap_probability
from .player_masking import bar_ath_missing_counts, drop_players
from .predict import predict
from .run import ROOT, DEFAULT_CONFIG, read_config, sha256, write_labels
from .run_external import GAP_SOURCE, V7_CONFIG, pooled, tracking_from_match
from .run_gap_model import paired_difference
from .run_player_dropout import SWITCH_COST, subset
from .run_v13 import combine
from .run_v14 import MOVING_BALL_MPS, overlay
from .turnover_model import decode, fit as fit_turnover, home_probability as turnover_probability, touch_features, training_rows
from .v7_features import build_features
from .v7_labels import period_runs
from .v7_model import dead_mask
from .v7_sequence import control_evidence

CONTROL_MIN_S = 1.0   # CHALLENGE.md: a touch without control is not a change of possession
USE_BALL_CLUES = False  # see module docstring: external ball artifacts at stoppages
DFL_SWITCH_COST = 0.5
TRAIN_SEED, TEST_SEED = 900, 500


def controlled_possession(labels, period, fps=25, min_s=CONTROL_MIN_S):
    """Relabel brief opposing touches inside a possession (or just before a stoppage) as the old team."""
    out = np.asarray(labels).copy()
    for a, b in period_runs(period):
        local = out[a:b]
        edges = np.r_[0, np.flatnonzero(local[1:] != local[:-1]) + 1, len(local)]
        runs = [(local[s], s, e) for s, e in zip(edges[:-1], edges[1:])]
        for (prev, _, _), (mid, s, e), (nxt, _, _) in zip(runs, runs[1:], runs[2:]):
            if (mid in 'HA' and prev in 'HA' and prev != mid and nxt in (prev, 'D') and (e - s) < min_s * fps):
                local[s:e] = prev
        out[a:b] = local
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'data/runs/sixteenth_trial')
    parser.add_argument('--v13-run', type=Path, default=ROOT/'data/runs/thirteenth_trial')
    parser.add_argument('--skip-score', action='store_true')
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if not out.is_relative_to((ROOT/'data').resolve()) or (out.exists() and any(out.iterdir())):
        raise ValueError('Choose a fresh output directory inside ignored data/')
    out.mkdir(parents=True)
    progress = lambda message: print(message, flush=True)
    v7c, base = json.loads(V7_CONFIG.read_text()), read_config(DEFAULT_CONFIG)
    gaps, missing = bar_ath_gap_lengths(GAP_SOURCE), bar_ath_missing_counts(GAP_SOURCE)

    metrica = attach_teams(load_metrica_samples(DEFAULT_SAMPLE_ROOT), DEFAULT_SAMPLE_ROOT)
    sources = [{'name': m.name, 'kind': 'metrica', 'match': m, 'labels': possession_labels(m.events, m.period)[0]}
               for m in metrica]
    sources += [{'name': d.match.name, 'kind': 'dfl', 'match': d.match, 'labels': d.labels} for d in load_dfl_matches()]
    names = [s['name'] for s in sources]

    def realistic(i, seed):
        s = sources[i]; m = s['match']
        xy = drop_players(m.player_xy_m, m.period, missing, seed)
        ball = hide_ball_realistic(m.ball_xy_m, s['labels'], m.period, gaps, seed)
        return type(m)(m.name, xy, m.home_slots, ball, m.period, m.events, m.fps, m.pitch)

    progress('Dead-ball training rows (recorded + realistic copies).')
    dead_rows = {}
    for i, s in enumerate(sources):
        parts = []
        for version in (s['match'], realistic(i, TRAIN_SEED + i)):
            table = dead_features(version, blank_ball=s['kind'] == 'metrica', ball=USE_BALL_CLUES)
            parts.append(subset(table, s['labels'], ['H', 'A', 'D']))
        dead_rows[s['name']] = parts
        progress(f"  {s['name']}")
    dead_models = {}
    for held in names:
        pool = [p for n in names if n != held for p in dead_rows[n]]
        dead_models[held] = fit_dead([t for t, _ in pool], [l for _, l in pool], stride=1)
    final_dead = fit_dead([t for n in names for t, _ in dead_rows[n]], [l for n in names for _, l in dead_rows[n]], stride=1)
    columns = list(dead_rows[names[0]][0][0].columns)
    progress('Dead-ball models trained.')

    progress('Possession models (Metrica, as in V12/V10).')
    poss_rows = {}
    for i, m in enumerate(metrica):
        labels = sources[i]['labels']
        bf = ballfree_features(m.player_xy_m, m.home_slots, m.period, m.pitch, m.fps)
        turn = []
        for target, seed in ((0.0, 0), (0.25, 100 + i)):
            ball = m.ball_xy_m if not target else hide_live_ball(m.ball_xy_m, np.isin(labels, ['H', 'A']), m.period, gaps, target, seed)[0]
            version = type(m)(m.name, m.player_xy_m, m.home_slots, ball, m.period, m.events, m.fps, m.pitch)
            features, _ = build_features(version, v7c)
            turn.append(training_rows(touch_features(features, ball, m.period, bf), labels))
        poss_rows[m.name] = {'turnover': turn, 'gap': (bf, labels)}
    def poss_models(train):
        return {'turnover': fit_turnover([p for n in train for p in poss_rows[n]['turnover']]),
                'gap': fit_gap([poss_rows[n]['gap'][0] for n in train], [poss_rows[n]['gap'][1] for n in train])}
    metrica_names = [m.name for m in metrica]
    poss = {n: poss_models([x for x in metrica_names if x != n]) for n in metrica_names}
    poss_all = poss_models(metrica_names)

    results, checks, preds = {}, {}, {c: {} for c in ('realistic', 'clean')}
    for i, s in enumerate(sources):
        for condition in ('realistic', 'clean'):
            m = realistic(i, TEST_SEED + i) if condition == 'realistic' else s['match']
            tracking = tracking_from_match(m)
            blocks = [(int(m.period[a]), a, b-1) for a, b in period_runs(m.period) if m.period[a] > 0]
            v1 = predict(tracking, make_features(tracking, blocks, base), blocks, base)[0]
            features, _ = build_features(m, v7c)
            bf = ballfree_features(m.player_xy_m, m.home_slots, m.period, m.pitch, m.fps)
            models = poss.get(s['name'], poss_all)
            cost = SWITCH_COST.get(s['name'], DFL_SWITCH_COST)
            v12 = decode(turnover_probability(models['turnover'], touch_features(features, m.ball_xy_m, m.period, bf)),
                         m.period, v1 == 'D', cost, m.fps)
            v13 = combine(v12, gap_probability(models['gap'], bf), features, m.period, v7c, m.fps)
            p = dead_probability(dead_models[s['name']], dead_features(m, ball=USE_BALL_CLUES)[columns])
            active = m.period > 0
            dead = dead_mask(p, m.period, v7c, m.fps) & active
            v16 = overlay(v13, dead, active)
            live_ref = ((control_evidence(features, v7c) >= 0) & (features['ball_reliable'] == 1)
                        & (np.nan_to_num(features['ball_speed_mps']) > MOVING_BALL_MPS) & active)
            graded = s['labels'] != 'X'
            checks[f"{condition}/{s['name']}"] = {
                'false_dead_on_live_ref': float(dead[live_ref].mean()) if live_ref.any() else None,
                'dead_share_pred': float(dead[active].mean()), 'dead_share_true': float((s['labels'][graded] == 'D').mean()),
                'dead_precision': float((s['labels'][dead & graded] == 'D').mean()) if (dead & graded).any() else None,
                'dead_recall': float(dead[graded & (s['labels'] == 'D')].mean())}
            preds[condition][s['name']] = {'v13': v13, 'v16': v16}
            c = checks[f"{condition}/{s['name']}"]
            progress(f"  {condition} {s['name']}: D precision {c['dead_precision']:.3f} recall {c['dead_recall']:.3f} "
                     f"pred share {c['dead_share_pred']:.3f} (true {c['dead_share_true']:.3f}) false-on-live {c['false_dead_on_live_ref']}")
    for condition in ('realistic', 'clean'):
        for group, members in (('metrica', metrica_names), ('dfl', [n for n in names if n not in metrica_names]),
                               ('all', names)):
            for relabel in (False, True):
                if relabel and group == 'metrica':
                    continue
                truth = [controlled_possession(sources[names.index(n)]['labels'], sources[names.index(n)]['match'].period)
                         if relabel and n not in metrica_names else sources[names.index(n)]['labels'] for n in members]
                gt = pooled(truth)
                joined = {k: pooled(preds[condition][n][k] for n in members) for k in ('v13', 'v16')}
                scores = {k: float(pm.evaluate(pm.label_codes(gt), pm.label_codes(v))['spell_f1_mean']) for k, v in joined.items()}
                key = f"{condition}/{group}" + ('/controlled' if relabel else '')
                results[key] = {'spell_f1': scores, 'v16_minus_v13': paired_difference(gt, joined['v16'], joined['v13'])}
                r = results[key]
                progress(f"{key}: V16 {scores['v16']:.4f}  V13 {scores['v13']:.4f}  diff {r['v16_minus_v13']['mean']:+.4f} "
                         f"({r['v16_minus_v13']['share_first_better']:.0%})")
    (out/'v16_bench_results.json').write_text(json.dumps(
        {'created_utc': datetime.now(timezone.utc).isoformat(), 'results': results, 'checks': checks,
         'source_sha256': {q.name: sha256(q) for q in sorted((ROOT/'solution').glob('*.py'))}}, indent=2, default=float)+'\n')

    progress('BAR-ATH with the model trained on all ten matches.')
    from games import GAME, GAMES, asset_paths, half_video_blocks
    from starter import load_smart_tagging
    from .data import load_tracking
    from .external_samples import ExternalMatch
    from .run_v6 import _period_keys
    from .run_v9 import BASELINES, PUBLIC_WINDOW
    from .run_v12 import spell_stats
    n, fps, pitch = (GAMES[GAME][k] for k in ['n_video_frames', 'fps', 'pitch'])
    period, active = _period_keys(n, half_video_blocks(GAME))
    v13_manifest = json.loads((args.v13_run/'manifest.json').read_text())
    frozen = {}
    for name in ['v13_combined', 'v7_transfer']:
        path = args.v13_run/'predictions'/(name+'.csv')
        if sha256(path) != v13_manifest['models'][name]['prediction_sha256']:
            raise ValueError(f'Frozen prediction changed: {name}')
        frozen[name] = pd.read_csv(path)['label'].to_numpy(dtype='<U1')
    v14_path = ROOT/'data/runs/fourteenth_trial/predictions/v14_confident_dead.csv'
    frozen['v14_confident_dead'] = pd.read_csv(v14_path)['label'].to_numpy(dtype='<U1')
    paths = asset_paths()
    tracking = load_tracking(paths['atd_meta'], paths['atd_raw'], n, fps, pitch, retain_tracks=True)
    match = ExternalMatch(GAME, tracking.track_xy_m, tracking.home_slots, tracking.ball_xy_m, period, [], fps, pitch)
    p = np.zeros(n)
    table = dead_features(match, ball=USE_BALL_CLUES)[columns]
    p[active] = dead_probability(final_dead, table[active])
    dead = dead_mask(p, period, v7c, fps) & active
    v16 = overlay(frozen['v13_combined'], dead, active)
    f = pd.read_csv(ROOT/'data/runs/seventh_trial/diagnostics/v7_features.csv',
                    usecols=['direct_control', 'ball_reliable', 'ball_speed_mps', 'players_observed', 'ball_seen'])
    live_ref = active & (f.direct_control.to_numpy() >= 0) & (f.ball_reliable.to_numpy() == 1) \
        & (np.nan_to_num(f.ball_speed_mps.to_numpy()) > MOVING_BALL_MPS)
    players = f.players_observed.to_numpy()
    tags = load_smart_tagging.load()
    set_pieces = [e for e in tags if (e['code'] or '').strip().casefold() == 'set pieces']
    public = np.zeros(n, bool); public[PUBLIC_WINDOW[0]:PUBLIC_WINDOW[1]] = True

    def summary(labels):
        d = labels == 'D'
        windows = [(max(0, e['start_frame'] - 3*fps), min(n, e['start_frame'] + fps)) for e in set_pieces]
        windows = [(a, b) for a, b in windows if active[a:b].any()]
        return {'set_piece_tags_with_D': f"{sum(int(d[a:b].any()) for a, b in windows)}/{len(windows)}",
                'false_dead_on_clear_moving_ball': float(d[live_ref].mean()),
                'dead_share_full': float(d[active].mean()), 'dead_share_public_window': float(d[public].mean()),
                'dead_share_by_players': {f'{lo}-{hi-1}': float(d[active & (players >= lo) & (players < hi)].mean())
                                          for lo, hi in [(0, 19), (19, 21), (21, 99)]},
                'false_dead_by_players': {f'{lo}-{hi-1}': float(d[live_ref & (players >= lo) & (players < hi)].mean())
                                          for lo, hi in [(0, 19), (19, 21), (21, 99)]},
                'full_match_spells': spell_stats(labels, active, fps)}
    bar = {name: summary(labels) for name, labels in {'v16_dead_model': v16, **frozen}.items()}
    for name, c in bar.items():
        progress(f"  {name}: tags {c['set_piece_tags_with_D']}, false dead {c['false_dead_on_clear_moving_ball']:.3f}, "
                 f"dead share full {c['dead_share_full']:.3f} public {c['dead_share_public_window']:.3f}, "
                 f"by players {c['dead_share_by_players']}")
    (out/'predictions').mkdir()
    write_labels(out/'predictions/v16_dead_model.csv', v16)
    for name, labels in frozen.items():
        write_labels(out/'predictions'/(name+'.csv'), labels)
    for name in BASELINES:
        (out/'predictions'/(name+'.csv')).write_bytes((ROOT/'evaluation/results'/(name+'.csv')).read_bytes())
    np.save(out/'v16_p_dead.npy', p)
    from evaluation.possession_metrics import read_label_csv, validate_submission
    manifest = {'experiment': 'FirstTrial v16 — dead ball from DFL + Metrica, robust to data loss',
                'created_utc': datetime.now(timezone.utc).isoformat(), 'answer_free_checks': bar,
                'dfl_files_sha256': (ROOT/'data/external/dfl-open-sha256.txt').read_text().splitlines(),
                'source_sha256': {q.relative_to(ROOT).as_posix(): sha256(q) for q in sorted((ROOT/'solution').glob('*.py'))},
                'models': {}}
    for path in sorted((out/'predictions').glob('*.csv')):
        error = validate_submission(read_label_csv(path), n)
        if error:
            raise ValueError(f'{path.stem}: {error}')
        manifest['models'][path.stem] = {'prediction_sha256': sha256(path)}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    progress('Frozen; scoring on the public sample.')
    if not args.skip_score:
        subprocess.run([sys.executable, str(ROOT/'evaluation/run_evaluation.py'), '--gt', 'public',
                        '--results', str(out/'predictions'), '--out', str(out/'scoring'),
                        '--bootstrap', str(args.bootstrap), '--no-detail'], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
