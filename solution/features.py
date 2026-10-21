"""Build gap and stoppage clues independently inside each published half."""

import math

import numpy as np


def true_run_lengths(mask):
    """Consecutive true observations, starting at one; false resets to zero."""
    mask = np.asarray(mask, dtype=bool)
    index = np.arange(len(mask))
    last_false = np.maximum.accumulate(np.where(~mask, index, -1))
    return np.where(mask, index - last_false, 0)


def runs(mask):
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return zip(edges[::2], edges[1::2])


def make_features(tracking, blocks, config):
    n = len(tracking)
    seen = np.isfinite(tracking.ball_xy_m).all(axis=1)
    reliable = seen.copy()
    speed = np.full(n, np.nan)
    gap_s = np.zeros(n)
    outside_confirmed = np.zeros(n, dtype=bool)
    slow_gap_confirmed = np.zeros(n, dtype=bool)
    stationary_confirmed = np.zeros(n, dtype=bool)
    active = np.zeros(n, dtype=bool)
    length, width = tracking.pitch
    margin = config["outside_margin_m"]
    outside = seen & (
        (tracking.ball_xy_m[:, 0] < -margin) |
        (tracking.ball_xy_m[:, 0] > length + margin) |
        (tracking.ball_xy_m[:, 1] < -margin) |
        (tracking.ball_xy_m[:, 1] > width + margin)
    )
    nearest_distance = np.fmin(tracking.home_distance_m, tracking.away_distance_m)
    previous_stop = -1
    for _, start, last in blocks:
        stop = last + 1
        if start < 0 or stop > n or start >= stop or start <= previous_stop:
            raise ValueError("Half blocks must be ordered, disjoint, and inside the recording")
        previous_stop = last
        active[start:stop] = True
        sl = slice(start, stop)
        local_seen = seen[sl]
        gap_s[sl] = true_run_lengths(~local_seen) / tracking.fps
        local_speed = np.full(stop - start, np.nan)
        both = local_seen[1:] & local_seen[:-1]
        delta = np.linalg.norm(np.diff(tracking.ball_xy_m[sl], axis=0), axis=1) * tracking.fps
        local_speed[1:] = np.where(both, delta, np.nan)
        speed[sl] = local_speed
        reliable[sl] &= ~np.isfinite(local_speed) | (local_speed <= config["max_ball_speed_mps"])
        enough_speed = tracking.speed_players[sl] >= config["minimum_players"]
        slow = enough_speed & (
            tracking.median_player_speed_mps[sl] <= config["slow_median_speed_mps"]
        ) & (tracking.walking_fraction[sl] >= config["slow_fraction"])
        slow_long = true_run_lengths(slow) >= math.ceil(config["slow_confirm_s"] * tracking.fps)
        outside_confirmed[sl] = true_run_lengths(outside[sl] & reliable[sl]) >= math.ceil(
            config["outside_confirm_s"] * tracking.fps)
        slow_gap_confirmed[sl] = slow_long & (
            gap_s[sl] >= config["missing_stop_after_s"])
        stationary = reliable[sl] & np.isfinite(local_speed) & (
            local_speed <= config["stationary_ball_speed_mps"])
        stationary_long = true_run_lengths(stationary) >= math.ceil(
            config["stationary_confirm_s"] * tracking.fps)
        stationary_confirmed[sl] = stationary_long & slow_long & (
            nearest_distance[sl] >= config["stationary_min_player_distance_m"])

    # Low collective speed on its own is deliberately never a stoppage cue.
    enough_players = tracking.observed_players >= config["minimum_players"]
    control_usable = reliable & enough_players & ~outside
    return {
        "active": active,
        "ball_seen": seen,
        "ball_reliable": reliable,
        "ball_speed_mps": speed,
        "gap_s": gap_s,
        "control_usable": control_usable,
        "outside_stop": outside_confirmed,
        "slow_gap_stop": slow_gap_confirmed,
        "stationary_stop": stationary_confirmed,
    }
