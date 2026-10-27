"""Rebuild the submission (V20) from the provided data and the public practice data, in one command.

    python -m solution.make_submission --out data/submission

Stages, each with the same functions and settings as the experiment that introduced it:
  1. V2 shape+motion labels, V7 features and control evidence on BAR-ATH (run_v7.py); the feature
     diagnostic is written with four decimals and read back, exactly as later experiments read it.
  2. V12 turnover possession model, trained on the three Metrica matches (run_v12.py).
  3. V10 ball-free gap model, trained on the three Metrica matches (run_v10.py).
  4. V13: V12 with V10 inside long ball gaps (run_v13.py).
  5. V16 dead-ball model on Metrica + DFL, trusted only where 21+ players are tracked, with the 2%
     answer-free calibration (run_v16.py; the gate as in run_v19.py).
  6. V20: V16 gated plus dead ball while the broadcast cuts away (run_v20.py).
Inputs: the BAR-ATH bundle (games.py / MTAH_DATA_DIR), the Metrica sample data and the DFL open data
(see solution/README.md). The public answers are never read. With --check, every stage is compared
with the frozen experiment outputs under data/runs.
"""
import argparse
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import sklearn

from .run import ROOT, DEFAULT_CONFIG, read_config, sha256, write_labels

FEATURE_COLUMNS = ['ball_seen', 'ball_reliable', 'ball_speed_mps', 'players_observed', 'speed_median_mps']
GATE_PLAYERS, CALIBRATION, MOVING_BALL_MPS = 21, 0.02, 2.0
FROZEN = {  # experiment outputs this build must reproduce, for --check
    'v2_shape_motion': 'seventh_trial/predictions/v2_shape_motion.csv',
    'v7_transfer': 'seventh_trial/predictions/v7_transfer.csv',
    'v12_turnover': 'twelfth_trial/predictions/v12_turnover.csv',
    'v13_combined': 'thirteenth_trial/predictions/v13_combined.csv',
    'v16_gated_cal': 'sixteenth_trial_gated/predictions/v16_gated_cal.csv',
    'v20_cutaway': 'twentieth_trial/predictions/v20_cutaway.csv',
}
FROZEN_ARRAYS = {'v12_p_home': 'twelfth_trial/v12_p_home.npy', 'v10_p_home': 'tenth_trial/v10_p_home.npy',
                 'v16_p_dead': 'sixteenth_trial/v16_p_dead.npy'}


def stage_v7(match, tracking, blocks, period, quality, out, progress):
    from .context import build_context, proxy_costs
    from .features import make_features
    from .predict import predict
    from .run_v2 import V2_CONFIG, context_config
    from .v7_features import build_features
    from .v7_sequence import control_memory
    base, v2config = read_config(DEFAULT_CONFIG), context_config(V2_CONFIG)
    features, _ = build_features(match, quality)
    basic = make_features(tracking, blocks, base)
    progress('  V2 shape and motion context.')
    context = build_context(tracking, basic, blocks, v2config)
    shape, _ = proxy_costs(context['shape'], tracking, basic, blocks, v2config, v2config['shape_weight'])
    motion, _ = proxy_costs(context['motion'], tracking, basic, blocks, v2config, v2config['motion_weight'])
    v2 = predict(tracking, basic, blocks, base, extra_team_costs=shape + motion)[0]
    v7, supported, direct = control_memory(features, period, quality, v2, dead=v2 == 'D', fps=match.fps)
    path = out/'v7_features.csv'
    pd.DataFrame({'frame': np.arange(len(period)), 'period': period, 'direct_control': direct,
                  **{c: features[c] for c in FEATURE_COLUMNS}}).to_csv(path, index=False, float_format='%.4f')
    return {'v2_shape_motion': v2, 'v7_transfer': v7}, features, path


def metrica_sources():
    from .external_possession import attach_teams, possession_labels
    from .external_samples import DEFAULT_SAMPLE_ROOT, load_metrica_samples
    matches = attach_teams(load_metrica_samples(DEFAULT_SAMPLE_ROOT), DEFAULT_SAMPLE_ROOT)
    return matches, [possession_labels(m.events, m.period)[0] for m in matches]


def stage_possession(match, features, v2, period, active, quality, lengths, progress):
    from .ball_masking import hide_live_ball
    from .gap_model import ballfree_features, fit as fit_gap, home_probability as gap_probability
    from .run_v12 import SWITCH_COST_S, TRAIN_MASK, TRAIN_SEED
    from .turnover_model import decode, fit, home_probability, touch_features, training_rows
    from .v7_features import build_features
    matches, labels = metrica_sources()
    progress('  V12 turnover model on Metrica.')
    parts, tables = [], []
    for i, (m, lab) in enumerate(zip(matches, labels)):
        ballfree = ballfree_features(m.player_xy_m, m.home_slots, m.period, m.pitch, m.fps)
        tables.append(ballfree)
        for target, seed in ((0.0, 0), (TRAIN_MASK, TRAIN_SEED + i)):
            ball = m.ball_xy_m if not target else hide_live_ball(
                m.ball_xy_m, np.isin(lab, ['H', 'A']), m.period, lengths, target, seed)[0]
            masked = type(m)(m.name, m.player_xy_m, m.home_slots, ball, m.period, m.events, m.fps, m.pitch)
            built, _ = build_features(masked, quality)
            parts.append(training_rows(touch_features(built, ball, m.period, ballfree), lab))
    turnover = fit(parts)
    progress('  V10 gap model on Metrica.')
    gap = fit_gap(tables, labels)
    del matches, parts, tables
    n, fps = len(period), match.fps
    bar = ballfree_features(match.player_xy_m, match.home_slots, period, match.pitch, fps)
    touch = touch_features(features, match.ball_xy_m, period, bar)
    p12, p10 = np.full(n, 0.5), np.full(n, 0.5)
    p12[active] = home_probability(turnover, (touch[0][active], touch[1][active]))
    p10[active] = gap_probability(gap, (bar[0][active], bar[1][active]))
    v12 = decode(p12, period, (v2 == 'D') & active, SWITCH_COST_S, fps)
    return v12, p12, p10


def stage_v16(match, period, active, lengths, missing, progress):
    from .deadball_formation import dead_probability, fit as fit_dead
    from .deadball_v16 import dead_features, hide_ball_realistic
    from .dfl_open import load_dfl_matches
    from .player_masking import drop_players
    from .run_player_dropout import subset
    from .run_v16 import TRAIN_SEED, USE_BALL_CLUES
    matches, labels = metrica_sources()
    sources = [{'kind': 'metrica', 'match': m, 'labels': lab} for m, lab in zip(matches, labels)]
    sources += [{'kind': 'dfl', 'match': d.match, 'labels': d.labels} for d in load_dfl_matches()]
    rows = []
    for i, s in enumerate(sources):
        m = s['match']
        copy = type(m)(m.name, drop_players(m.player_xy_m, m.period, missing, TRAIN_SEED + i), m.home_slots,
                       hide_ball_realistic(m.ball_xy_m, s['labels'], m.period, lengths, TRAIN_SEED + i),
                       m.period, m.events, m.fps, m.pitch)
        for version in (m, copy):
            rows.append(subset(dead_features(version, blank_ball=s['kind'] == 'metrica', ball=USE_BALL_CLUES),
                               s['labels'], ['H', 'A', 'D']))
        progress(f'  dead-ball rows: {m.name}')
        sources[i] = None
    model = fit_dead([t for t, _ in rows], [l for _, l in rows], stride=1)
    columns = list(rows[0][0].columns)
    del rows
    p = np.zeros(len(period))
    p[active] = dead_probability(model, dead_features(match, ball=USE_BALL_CLUES)[columns][active])
    return p


def compare(name, labels, root):
    path = root/FROZEN[name]
    if not path.exists():
        return {'frozen_file': None}
    old = pd.read_csv(path)['label'].to_numpy(dtype='<U1')
    return {'frames_different': int((old != labels).sum())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'data/submission')
    parser.add_argument('--name', default='Letinsh', help='Team name; the file is <name>.csv')
    parser.add_argument('--check', action='store_true', help='Compare every stage with the frozen runs in data/runs')
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if not out.is_relative_to((ROOT/'data').resolve()) or (out.exists() and any(out.iterdir())):
        raise ValueError('Choose a fresh output directory inside ignored data/')
    out.mkdir(parents=True)
    progress = lambda message: print(message, flush=True)
    from games import GAME, GAMES, asset_paths, half_video_blocks
    from .ball_masking import bar_ath_gap_lengths
    from .camera_cutaway import cutaway_dead, cutaway_mask
    from .data import load_tracking
    from .deadball_formation import calibrated_dead_mask
    from .external_samples import ExternalMatch
    from .player_masking import bar_ath_missing_counts
    from .run_v6 import _period_keys
    from .run_v13 import combine
    from .run_v14 import overlay
    from .v8_stoppages import build_evidence
    quality = json.loads((ROOT/'solution/config_v7.json').read_text())
    v8config = json.loads((ROOT/'solution/config_v8.json').read_text())
    n, fps, pitch = (GAMES[GAME][k] for k in ['n_video_frames', 'fps', 'pitch'])
    blocks = half_video_blocks(GAME)
    period, active = _period_keys(n, blocks)
    paths = asset_paths()
    progress('Loading BAR-ATH tracking.')
    tracking = load_tracking(paths['atd_meta'], paths['atd_raw'], n, fps, pitch, retain_tracks=True)
    match = ExternalMatch(GAME, tracking.track_xy_m, tracking.home_slots, tracking.ball_xy_m, period, [], fps, pitch)

    progress('Stage 1: V2 and V7 features.')
    labels, features, features_csv = stage_v7(match, tracking, blocks, period, quality, out, progress)
    diag = pd.read_csv(features_csv)
    lengths, missing = bar_ath_gap_lengths(features_csv), bar_ath_missing_counts(features_csv)

    progress('Stages 2-4: V12, V10 and V13 possession.')
    labels['v12_turnover'], p12, p10 = stage_possession(match, features, labels['v2_shape_motion'], period, active,
                                                        quality, lengths, progress)
    evidence = build_evidence(match, v8config, quality)
    labels['v13_combined'] = combine(labels['v12_turnover'], p10, evidence, period, quality, fps)

    progress('Stage 5: V16 dead ball, gated and calibrated.')
    p16 = stage_v16(match, period, active, lengths, missing, progress)
    live = active & (diag.direct_control.to_numpy() >= 0) & (diag.ball_reliable.to_numpy() == 1) \
        & (np.nan_to_num(diag.ball_speed_mps.to_numpy()) > MOVING_BALL_MPS)
    full = pd.Series(diag.players_observed.to_numpy() >= GATE_PLAYERS).rolling(fps, center=True, min_periods=1).min() \
        .to_numpy().astype(bool)
    gated = calibrated_dead_mask(np.where(full, p16, 0), period, quality, live, CALIBRATION, fps)[0]
    labels['v16_gated_cal'] = overlay(labels['v13_combined'], gated, active)

    progress('Stage 6: V20 broadcast cutaways.')
    cut, _ = cutaway_mask(asset_paths(source='broadcast')['homography'], period, active)
    extra = cutaway_dead(cut, diag.speed_median_mps.to_numpy(), fps)
    labels['v20_cutaway'] = overlay(labels['v16_gated_cal'], extra, active)

    final = out/f'{args.name}.csv'
    write_labels(final, labels['v20_cutaway'])
    from evaluation.possession_metrics import read_label_csv, validate_submission
    error = validate_submission(read_label_csv(final), n)
    if error:
        raise ValueError(f'Submission invalid: {error}')
    manifest = {'submission': final.name, 'sha256': sha256(final), 'method': 'V20',
                'label_counts': {k: int(v) for k, v in zip(*np.unique(labels['v20_cutaway'], return_counts=True))},
                'inputs_sha256': {k: sha256(paths[k]) for k in ['atd_meta', 'atd_raw']} |
                                 {'broadcast_homography': sha256(asset_paths(source='broadcast')['homography'])},
                'source_sha256': {q.relative_to(ROOT).as_posix(): sha256(q) for q in sorted((ROOT/'solution').glob('*.py'))},
                'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                                'sklearn': sklearn.__version__}}
    if args.check:
        runs = ROOT/'data/runs'
        checks = {name: compare(name, value, runs) for name, value in labels.items() if name in FROZEN}
        arrays = {'v12_p_home': p12, 'v10_p_home': p10, 'v16_p_dead': p16}
        for name, value in arrays.items():
            path = runs/FROZEN_ARRAYS[name]
            checks[name] = {'max_abs_difference': float(np.max(np.abs(np.load(path) - value)))} if path.exists() else {}
        frozen_final = runs/FROZEN['v20_cutaway']
        checks['final_file_identical'] = frozen_final.exists() and sha256(frozen_final) == manifest['sha256']
        manifest['checks'] = checks
        progress(json.dumps(checks, indent=2))
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    progress(f'Wrote {final} ({manifest["sha256"][:12]}).')


if __name__ == '__main__':
    main()
