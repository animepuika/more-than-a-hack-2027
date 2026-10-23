"""Quality-aware tracking features. All temporal operations reset per period.

V6's feature generator is retained for historical reproducibility. V7 uses this
separate implementation with finite fragments, bounded ball age and coverage.
"""
from types import SimpleNamespace
import warnings
import numpy as np
import pandas as pd

from .context import fragment_motion
from .deadball_features import _team_geometry, _row_percentile, _nanmin_rows
from .features import true_run_lengths
from .v7_labels import period_runs


def clean_positions(xy, home, pitch, duplicate_m=0.5):
    xy = np.asarray(xy, dtype=np.float32).copy()
    valid = (np.isfinite(xy).all(axis=2) & (xy[:, :, 0] >= -3) &
             (xy[:, :, 0] <= pitch[0]+3) & (xy[:, :, 1] >= -3) &
             (xy[:, :, 1] <= pitch[1]+3))
    xy[~valid] = np.nan
    for side in (home, ~home):
        slots = np.flatnonzero(side)
        for i, left in enumerate(slots):
            for right in slots[i+1:]:
                delta = xy[:, left] - xy[:, right]
                duplicate = np.sum(delta*delta, axis=1) < duplicate_m**2
                xy[duplicate, right] = np.nan
    return xy


def _rolling(values, window, operation='mean', center=False):
    series = pd.Series(np.asarray(values, dtype=float))
    rolled = series.rolling(window, min_periods=max(1, window//2), center=center)
    return getattr(rolled, operation)().to_numpy(dtype=np.float32)


def _local_features(xy, ball, home, fps, pitch, config):
    n = len(ball)
    seen = np.isfinite(ball).all(axis=1)
    reliable = seen.copy()
    ball_step = np.full(n, np.nan)
    ball_step[1:] = np.linalg.norm(np.diff(ball, axis=0), axis=1)*fps
    bad_edge = np.isfinite(ball_step) & (ball_step > config['maximum_ball_speed_mps'])
    # Reject both endpoints of implausible observed jumps; retain raw visibility.
    reliable[bad_edge] = False
    reliable[np.maximum(np.flatnonzero(bad_edge)-1, 0)] = False
    clean_ball = ball.copy(); clean_ball[~reliable] = np.nan
    motion_config = {'velocity_window_frames': 5,
                     'maximum_track_speed_mps': config['maximum_player_speed_mps'],
                     'maximum_acceleration_mps2': config['maximum_acceleration_mps2']}
    velocity, speed, accel = fragment_motion(xy, [(1, 0, n-1)], fps, motion_config)
    count = np.isfinite(xy).all(axis=2)
    speed_valid = np.isfinite(speed)
    speed_count = speed_valid.sum(axis=1)
    features = {
        'ball_seen': seen.astype(float), 'ball_reliable': reliable.astype(float),
        'ball_gap_age_s': true_run_lengths(~reliable)/fps,
        'ball_gap_remaining_s': true_run_lengths((~reliable)[::-1])[::-1]/fps,
        'players_observed': count.sum(axis=1), 'speed_tracks': speed_count,
        'speed_median_mps': _row_percentile(speed, 50),
        'speed_q75_mps': _row_percentile(speed, 75),
        'walking_fraction': np.divide(((speed <= 1.4) & speed_valid).sum(axis=1), speed_count,
                                     out=np.full(n, np.nan), where=speed_count > 0),
        'acceleration_q90_mps2': _row_percentile(np.abs(accel), 90),
        'braking_fraction': np.divide((accel < -3).sum(axis=1), np.isfinite(accel).sum(axis=1),
                                     out=np.full(n, np.nan), where=np.isfinite(accel).sum(axis=1) > 0),
    }
    ball_velocity = np.full_like(clean_ball, np.nan)
    ball_velocity[1:] = np.diff(clean_ball, axis=0)*fps
    ball_speed = np.linalg.norm(ball_velocity, axis=1)
    x, y = clean_ball.T
    margin = np.maximum.reduce([-x, x-pitch[0], -y, y-pitch[1]])
    features.update(ball_speed_mps=ball_speed, outside_margin_m=margin,
                    ball_near_boundary=(np.abs(margin) <= 1).astype(float))
    nearest = []
    for label, side in [('home', home), ('away', ~home)]:
        distance = np.linalg.norm(xy[:, side] - clean_ball[:, None], axis=2)
        near = _nanmin_rows(distance)
        nearest.append(near)
        features[label+'_nearest_ball_m'] = near
        features[label+'_players'] = count[:, side].sum(axis=1)
        features[label+'_speed_median_mps'] = _row_percentile(speed[:, side], 50)
        # Relative movement with the closest player requires a reliable fragment.
        slots = np.flatnonzero(side)
        index = np.argmin(np.where(np.isfinite(distance), distance, np.inf), axis=1)
        nearest_velocity = velocity[np.arange(n), slots[index]]
        rel_speed = np.linalg.norm(ball_velocity-nearest_velocity, axis=1)
        rel_speed[~np.isfinite(near)] = np.nan
        features[label+'_ball_relative_speed_mps'] = rel_speed
    features['nearest_ball_m'] = np.fmin(*nearest)
    # Geometric context is masked when the observation does not support it.
    geometry = _team_geometry(xy, home, pitch, max(1, round(fps)))
    enough = np.minimum(features['home_players'], features['away_players']) >= config['minimum_team_tracks']
    for name in ['home_hull_area_m2', 'away_hull_area_m2', 'team_centroid_separation_m',
                 'home_voronoi_share', 'opponent_pairs_within_2m']:
        values = geometry[name].copy(); values[~enough] = np.nan
        features[name] = values
    # Old ball location is only retained for a short, declared age.
    last_seen = np.maximum.accumulate(np.where(reliable, np.arange(n), -1))
    age = (np.arange(n)-last_seen)/fps
    known = (last_seen >= 0) & (age <= config['maximum_last_ball_age_s'])
    last_ball = np.full_like(ball, np.nan)
    last_ball[known] = clean_ball[last_seen[known]]
    for label, side in [('home', home), ('away', ~home)]:
        dist = np.linalg.norm(xy[:, side]-last_ball[:, None], axis=2)
        features[label+'_near_last_ball'] = np.where(known, (dist <= 3).sum(axis=1), np.nan)
    for seconds in [0.5, 2.0, 5.0]:
        window = max(1, round(seconds*fps))
        for name in ['speed_median_mps', 'walking_fraction', 'ball_reliable', 'ball_speed_mps']:
            features[f'{name}_mean_{seconds:g}s'] = _rolling(features[name], window, center=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            positions = pd.DataFrame(clean_ball).rolling(window, min_periods=max(2, window//2), center=True)
            features[f'ball_span_{seconds:g}s_m'] = np.linalg.norm((positions.max()-positions.min()).to_numpy(), axis=1)
    return {name: np.asarray(values, dtype=np.float32) for name, values in features.items()}


def build_features(match, config):
    if match.fps != 25:
        raise ValueError('V7 external/source geometry currently requires 25 fps')
    home = np.asarray(match.home_slots, bool)
    if not home.any() or home.all():
        raise ValueError('Both teams need track slots')
    output = {}
    clean_xy = np.full_like(match.player_xy_m, np.nan, dtype=np.float32)
    for a, b in period_runs(match.period):
        if match.period[a] <= 0:
            continue
        local = clean_positions(match.player_xy_m[a:b], home, match.pitch, config['duplicate_distance_m'])
        clean_xy[a:b] = local
        features = _local_features(local, match.ball_xy_m[a:b], home, match.fps, match.pitch, config)
        for name, values in features.items():
            if name not in output:
                output[name] = np.full(len(match.ball_xy_m), np.nan, dtype=np.float32)
            output[name][a:b] = values
    if not output:
        raise ValueError('No active periods')
    return output, clean_xy


def model_columns(features):
    # Symmetric transforms prevent Home/Away or pitch-end shortcuts for D.
    result = dict(features)
    for suffix in ['nearest_ball_m', 'players', 'speed_median_mps', 'ball_relative_speed_mps',
                   'hull_area_m2', 'near_last_ball']:
        h, a = result.pop('home_'+suffix), result.pop('away_'+suffix)
        result['teams_'+suffix+'_min'] = np.fmin(h, a)
        result['teams_'+suffix+'_max'] = np.fmax(h, a)
    result['territory_imbalance'] = np.abs(result.pop('home_voronoi_share')-0.5)
    return pd.DataFrame(result, dtype=np.float32)
