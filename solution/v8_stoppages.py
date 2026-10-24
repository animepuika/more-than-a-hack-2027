"""V8: observed ball stability, boundary evidence, and separate restart logic.

Fixed engineering hypotheses, not calibrated probabilities or referee decisions.
No event answers are read here. Offline confirmation may backdate a stop.
"""
import math
import warnings

import numpy as np
import pandas as pd

from .context import fragment_motion
from .features import runs, true_run_lengths
from .v7_features import clean_positions
from .v7_labels import period_runs


def _nanmin(values):
    result = np.min(np.where(np.isfinite(values), values, np.inf), axis=1)
    result[~np.isfinite(result)] = np.nan
    return result


def build_evidence(match, config, quality_config):
    """Build light tracking features, resetting every contiguous active period.

    Control fields reproduce V7's calculations, avoiding a geometry recompute.
    Small visibility holes are tolerated for stability, never for travel speed.
    """
    fps = match.fps
    window = max(2, math.ceil(config['stationary_window_s'] * fps))
    needed = max(2, math.ceil(window * config['stationary_coverage']))
    lag = max(1, round(config['travel_window_s'] * fps))
    output = {}
    home = np.asarray(match.home_slots, bool)
    motion = {'velocity_window_frames': 5,
              'maximum_track_speed_mps': quality_config['maximum_player_speed_mps'],
              'maximum_acceleration_mps2': quality_config['maximum_acceleration_mps2']}
    for first, stop in period_runs(match.period):
        if match.period[first] <= 0:
            continue
        ball = match.ball_xy_m[first:stop].copy()
        xy = clean_positions(match.player_xy_m[first:stop], home, match.pitch,
                             quality_config['duplicate_distance_m'])
        n = len(ball)
        reliable = np.isfinite(ball).all(axis=1)
        step = np.r_[np.nan, np.linalg.norm(np.diff(ball, axis=0), axis=1) * fps]
        bad = np.isfinite(step) & (step > quality_config['maximum_ball_speed_mps'])
        reliable[bad] = False
        reliable[np.maximum(np.flatnonzero(bad)-1, 0)] = False
        ball[~reliable] = np.nan
        velocity, speed, _ = fragment_motion(xy, [(1, 0, n-1)], fps, motion)
        ball_velocity = np.full_like(ball, np.nan)
        ball_velocity[1:] = np.diff(ball, axis=0) * fps
        counts = np.isfinite(xy).all(axis=2)
        speed_tracks = np.isfinite(speed).sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            median = np.nanpercentile(speed, 50, axis=1)
        walking = np.divide(((speed <= 1.4) & np.isfinite(speed)).sum(axis=1), speed_tracks,
                            out=np.full(n, np.nan), where=speed_tracks > 0)
        f = {'ball_reliable': reliable.astype(float), 'speed_tracks': speed_tracks,
             'speed_median_mps': median, 'walking_fraction': walking}
        near_sides, closing_sides = [], []
        for label, side in [('home', home), ('away', ~home)]:
            offset = ball[:, None] - xy[:, side]
            distance = np.linalg.norm(offset, axis=2)
            near = _nanmin(distance)
            slots = np.flatnonzero(side)
            index = np.argmin(np.where(np.isfinite(distance), distance, np.inf), axis=1)
            rel = np.linalg.norm(ball_velocity - velocity[np.arange(n), slots[index]], axis=1)
            rel[~np.isfinite(near)] = np.nan
            f[label+'_nearest_ball_m'] = near
            f[label+'_players'] = counts[:, side].sum(axis=1)
            f[label+'_ball_relative_speed_mps'] = rel
            near_sides.append(near <= config['pressure_radius_m'])
            direction = np.divide(offset, distance[..., None], out=np.zeros_like(offset),
                                  where=distance[..., None] > 0.05)
            closing = np.sum(velocity[:, side] * direction, axis=2)
            closing_sides.append(((distance <= config['pressure_radius_m']) &
                                  (closing >= config['pressure_closing_speed_mps'])).any(axis=1))
        pressure = near_sides[0] & near_sides[1] & (closing_sides[0] | closing_sides[1])
        f['pressure_fraction'] = pd.Series(pressure).rolling(window, min_periods=window).mean().to_numpy()
        f['nearest_ball_m'] = np.fmin(f['home_nearest_ball_m'], f['away_nearest_ball_m'])
        x, y = ball.T
        margin = np.maximum.reduce([-x, x-match.pitch[0], -y, y-match.pitch[1]])
        f['outside_margin_m'] = margin
        quiet = ((speed_tracks >= config['minimum_speed_tracks']) &
                 (median <= config['quiet_median_speed_mps']) &
                 (walking >= config['quiet_walking_fraction']))
        f['quiet_fraction'] = pd.Series(quiet).rolling(window, min_periods=window).mean().to_numpy()
        positions = pd.DataFrame(ball).rolling(window, min_periods=needed)
        q = config['stationary_quantile']
        span = positions.quantile(1-q) - positions.quantile(q)
        f['robust_ball_span_m'] = np.linalg.norm(span.to_numpy(), axis=1)
        coverage = pd.Series(reliable).rolling(window, min_periods=window).mean().to_numpy()
        max_gap = pd.Series(true_run_lengths(~reliable)/fps).rolling(window, min_periods=window).max().to_numpy()
        f['stability_covered'] = ((coverage >= config['stationary_coverage']) &
                                  (max_gap <= config['stationary_max_gap_s']) & reliable).astype(float)
        # Net displacement/path length rejects jitter and back-and-forth placement.
        travel = np.full(n, np.nan)
        path = pd.Series(np.r_[np.nan, np.linalg.norm(np.diff(ball, axis=0), axis=1)]).rolling(
            lag, min_periods=lag).sum().to_numpy()
        if n > lag:
            travel[lag:] = np.linalg.norm(ball[lag:] - ball[:-lag], axis=1)
        uninterrupted = true_run_lengths(reliable) >= lag+1
        travel[~uninterrupted] = np.nan
        efficiency = np.divide(travel, path, out=np.zeros(n), where=path > 0.05)
        travel_speed = travel * fps / lag
        active_players = ((speed_tracks >= config['minimum_speed_tracks']) &
                          (median >= config['restart_player_speed_mps']))
        restart = (reliable & (margin < -config['ball_radius_m']) &
                   (efficiency >= config['restart_path_efficiency']) &
                   (((travel_speed >= config['restart_speed_mps']) & active_players) |
                    ((travel_speed >= config['restart_fast_speed_mps']) &
                     (f['nearest_ball_m'] >= config['restart_ball_separation_m']))))
        f.update(travel_speed_mps=travel_speed, path_efficiency=efficiency,
                 restart_candidate=restart.astype(float))
        for name, values in f.items():
            if name not in output:
                output[name] = np.full(len(match), np.nan, np.float32)
            output[name][first:stop] = np.asarray(values, np.float32)
    return output


def _confirmed(mask, minimum, lookback=0):
    output = np.zeros(len(mask), bool)
    for a, b in runs(mask):
        if b-a >= minimum:
            output[max(0, a-lookback):b] = True
    return output


def decode_stops(features, period, fps, config, *, stationary=True, persistent=False,
                 pressure_guard=False, tag_windows=None):
    """No missingness-only entry. Remember supported stops for a bounded time.

    'Persistent' changes continuation only. Restart confirmation backdates the
    release to its first supporting frame, then applies a short re-entry cooldown.
    A return to the pitch or movement alone does not automatically restart play.
    """
    result = np.zeros(len(period), bool)
    reasons = np.full(len(period), '', dtype='<U40')
    diagnostics = {k: np.zeros(len(period), bool) for k in
                   ['outside_entry', 'stationary_entry', 'tag_entry', 'restart', 'pressure_veto']}
    window = max(2, math.ceil(config['stationary_window_s']*fps))
    hold = math.ceil(config['persistent_hold_s' if persistent else 'brief_hold_s']*fps)
    for a, b in period_runs(period):
        if period[a] <= 0:
            continue
        f = {k: v[a:b] for k, v in features.items()}
        outside = _confirmed((f['ball_reliable'] == 1) &
            (f['outside_margin_m'] > config['ball_radius_m']+config['boundary_uncertainty_m']),
            math.ceil(config['outside_confirmation_s']*fps))
        covered = (f['stability_covered'] == 1) & (f['outside_margin_m'] <= config['ball_radius_m'])
        pressure = np.nan_to_num(f['pressure_fraction']) >= config['pressure_window_fraction']
        still = covered & (f['robust_ball_span_m'] <= config['stationary_span_m']) & (
            f['quiet_fraction'] >= config['quiet_window_fraction'])
        if pressure_guard:
            still &= ~pressure
        still = _confirmed(still, math.ceil(config['stationary_confirmation_s']*fps), window-1)
        tag = np.zeros(b-a, bool)
        if stationary and tag_windows is not None:
            supported = (tag_windows[a:b] & covered &
                (f['robust_ball_span_m'] <= config['tag_stationary_span_m']) &
                (f['quiet_fraction'] >= config['tag_quiet_window_fraction']))
            if pressure_guard:
                supported &= ~pressure
            tag = _confirmed(supported, math.ceil(config['tag_confirmation_s']*fps), window-1)
        if not stationary:
            still[:] = False
        # Never backfill across earlier credible motion or an observed pressure duel.
        if pressure_guard:
            still[pressure] = False
            tag[pressure] = False
        restart = _confirmed(f['restart_candidate'] == 1,
                             math.ceil(config['restart_confirmation_s']*fps))
        for key, value in [('outside_entry', outside), ('stationary_entry', still),
                           ('tag_entry', tag), ('restart', restart), ('pressure_veto', pressure)]:
            diagnostics[key][a:b] = value
        last_support, stopped, cooldown_until = -hold-1, False, -1
        for i in range(b-a):
            if outside[i]:
                stopped, last_support = True, i
                result[a+i] = True
                reasons[a+i] = 'v8_observed_outside'
            elif restart[i]:
                stopped = False
                cooldown_until = i + math.ceil(config['restart_cooldown_s']*fps)
            elif i <= cooldown_until:
                continue
            elif still[i] or tag[i]:
                stopped, last_support = True, i
                result[a+i] = True
                reasons[a+i] = 'v8_tag_supported_still' if tag[i] else 'v8_stable_ball_and_quiet'
            elif stopped and i-last_support <= hold:
                result[a+i] = True
                reasons[a+i] = 'v8_waiting_for_restart'
            else:
                stopped = False
    return result, reasons, diagnostics
