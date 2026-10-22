"""V3 stoppage hypotheses. No public answer files are read here.

These rules use collective speed, persistence, ball position stability, and
optional automatic restart windows. They do not identify fouls/cards/subs.
"""

import math

import numpy as np
import pandas as pd

from .features import runs, true_run_lengths


def restart_windows(events, blocks, n, fps, padding_s):
    """Candidate windows only, never confirmed dead-ball labels or team owners."""
    candidate = np.zeros(n, dtype=bool)
    padding = math.ceil(padding_s * fps)
    for event in events:
        if (event["code"] or "").strip().casefold() != "set pieces":
            continue
        start, stop = event["start_frame"] - padding, event["end_frame"] + padding
        for _, first, last in blocks:
            a, b = max(first, start), min(last + 1, stop)
            if a < b:
                candidate[a:b] = True
    return candidate


def build_stoppage_features(tracking, features, blocks, config, events=()):
    n, fps = len(tracking), tracking.fps
    result = {name: np.zeros(n, dtype=bool) for name in
              ("quiet", "stationary", "missing_quiet", "stationary_quiet", "live_restart",
               "missing_seed", "stationary_seed", "tag_seed", "outside_seed")}
    result["tag_window"] = restart_windows(events, blocks, n, fps, config["restart_window_padding_s"])
    ball_span = np.full(n, np.nan)
    ball_travel_speed = np.full(n, np.nan)
    ball = tracking.ball_xy_m.copy()
    ball[~features["ball_reliable"]] = np.nan
    window = max(2, math.ceil(config["ball_cluster_window_s"] * fps))
    periods = max(2, math.ceil(window * config["ball_cluster_coverage"]))
    lag = max(1, math.ceil(config["ball_travel_window_s"] * fps))
    for _, start, last in blocks:
        sl = slice(start, last + 1)
        local_ball = pd.DataFrame(ball[sl])
        extent = local_ball.rolling(window, min_periods=periods).max() - local_ball.rolling(
            window, min_periods=periods).min()
        span = np.linalg.norm(extent.to_numpy(), axis=1)
        ball_span[sl] = span
        travel = np.full(last + 1 - start, np.nan)
        if len(travel) > lag:
            # Require an uninterrupted reliable window: do not differentiate across occlusions.
            uninterrupted = true_run_lengths(features["ball_reliable"][sl]) >= lag + 1
            difference = np.linalg.norm(ball[start + lag:last + 1] - ball[start:last + 1 - lag], axis=1)
            travel[lag:] = difference * fps / lag
            travel[~uninterrupted] = np.nan
        ball_travel_speed[sl] = travel
        covered = tracking.speed_players[sl] >= config["minimum_speed_tracks"]
        quiet = covered & (tracking.median_player_speed_mps[sl] <= config["quiet_median_speed_mps"])
        quiet &= tracking.walking_fraction[sl] >= config["quiet_walking_fraction"]
        stationary = features["ball_reliable"][sl] & (span <= config["ball_cluster_diameter_m"])
        moving_ball = features["control_usable"][sl] & (travel >= config["restart_ball_speed_mps"])
        moving_ball &= span > config["ball_cluster_diameter_m"]
        moving_players = covered & (tracking.median_player_speed_mps[sl] >= config["restart_player_speed_mps"])
        moving_players &= tracking.walking_fraction[sl] <= config["restart_walking_fraction"]
        live_candidate = moving_ball | (~features["ball_seen"][sl] & moving_players)
        live = true_run_lengths(live_candidate) >= math.ceil(config["restart_confirmation_s"] * fps)
        # Confirmed live play resets the quiet/stationary timers. Otherwise a long
        # dead-ball episode could immediately re-trigger D after play resumes.
        quiet_for_confirmation = quiet & ~live
        quiet_long = true_run_lengths(quiet_for_confirmation) >= math.ceil(
            config["quiet_confirmation_s"] * fps)
        missing_quiet = quiet_for_confirmation & ~features["ball_seen"][sl]
        stationary_quiet = quiet_for_confirmation & stationary & ~moving_ball
        result["quiet"][sl] = quiet
        result["stationary"][sl] = stationary
        result["missing_quiet"][sl] = missing_quiet
        result["stationary_quiet"][sl] = stationary_quiet
        result["live_restart"][sl] = live
        result["missing_seed"][sl] = quiet_long & (true_run_lengths(missing_quiet) >= math.ceil(
            config["missing_confirmation_s"] * fps))
        result["stationary_seed"][sl] = quiet_long & (true_run_lengths(stationary_quiet) >= math.ceil(
            config["stationary_confirmation_s"] * fps))
        supported_tag = result["tag_window"][sl] & (missing_quiet | stationary_quiet)
        result["tag_seed"][sl] = quiet_long & (true_run_lengths(supported_tag) >= math.ceil(
            config["restart_hint_confirmation_s"] * fps))
        result["outside_seed"][sl] = features["outside_stop"][sl]
    result["ball_span_m"] = ball_span
    result["ball_travel_speed_mps"] = ball_travel_speed
    return result


def enhanced_dead_mask(clues, blocks, fps, config, use_restart_tags=False, retrofill=True):
    """Persistent evidence + short bounded continuity; reset independently per half.

    Retrofill marks the beginning of a qualifying quiet/still episode after
    later observations confirm it. Disabling it does not make the full model causal.
    """
    mask = np.zeros(len(clues["quiet"]), dtype=bool)
    reasons = np.full(len(mask), "", dtype="<U40")
    hold = math.ceil(config["maximum_dead_hold_s"] * fps)
    for _, start, last in blocks:
        base = np.zeros(last + 1 - start, dtype=bool)
        base_reason = np.full(len(base), "", dtype="<U40")
        for eligible_name, seed_name, reason in (
            ("missing_quiet", "missing_seed", "v3_missing_and_quiet"),
            ("stationary_quiet", "stationary_seed", "v3_stationary_and_quiet"),
        ):
            live = clues["live_restart"][start:last + 1]
            eligible = clues[eligible_name][start:last + 1] & ~live
            seed = clues[seed_name][start:last + 1].copy() & ~live
            if use_restart_tags:
                seed |= clues["tag_seed"][start:last + 1] & eligible
            for a, b in runs(eligible):
                confirmed = np.flatnonzero(seed[a:b])
                if not len(confirmed):
                    continue
                begin = a if retrofill else a + int(confirmed[0])
                base[begin:b] = True
                base_reason[begin:b] = reason
                if use_restart_tags and clues["tag_seed"][start + begin:start + b].any():
                    base_reason[begin:b] = "v3_restart_tag_with_tracking"
        outside = clues["outside_seed"][start:last + 1]
        base[outside] = True
        base_reason[outside] = "v3_ball_outside_pitch"
        latched, support_frame = False, -hold - 1
        for i in range(len(base)):
            frame = start + i
            if clues["live_restart"][frame] and not outside[i]:
                latched = False
                continue
            if base[i]:
                latched, support_frame = True, i
                mask[frame] = True
                reasons[frame] = base_reason[i]
            elif latched and i - support_frame <= hold:
                mask[frame] = True
                reasons[frame] = "v3_bounded_dead_continuity"
            else:
                latched = False
    return mask, reasons
