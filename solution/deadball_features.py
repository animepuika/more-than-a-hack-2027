"""Frame-level ball, movement, and team-shape features for the V6 study.

Features use tracking only. Event logs are used to make weak training targets,
never as inference features. The territory measure is a coarse nearest-player
Voronoi proxy, not Spearman's time-to-intercept pitch-control model.
"""

import warnings

import numpy as np
from scipy.spatial import ConvexHull, QhullError


def _row_percentile(values, percentile):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(values, percentile, axis=1).astype(np.float32)


def _nanmin_rows(values):
    good = np.isfinite(values)
    result = np.min(np.where(good, values, np.inf), axis=1)
    result[~good.any(axis=1)] = np.nan
    return result.astype(np.float32)


def _interpolate_sampled(sample_frames, sampled_values, n_frames):
    output = np.full(n_frames, np.nan, dtype=np.float32)
    valid = np.isfinite(sampled_values)
    if valid.sum() == 1:
        output[:] = sampled_values[valid][0]
    elif valid.sum() > 1:
        frame_axis = np.arange(n_frames, dtype=np.int32)
        output[:] = np.interp(frame_axis, sample_frames[valid], sampled_values[valid])
    return output


def _sample_frames(n_frames, stride):
    frames = np.arange(0, n_frames, stride, dtype=np.int32)
    if not len(frames) or frames[-1] != n_frames - 1:
        frames = np.r_[frames, np.int32(n_frames - 1)]
    return frames


def _team_geometry(xy, team_slots, pitch, stride):
    n_frames = len(xy)
    ids = np.flatnonzero(team_slots)
    other_ids = np.flatnonzero(~team_slots)
    features = {}
    centers = []
    valid_counts = []
    for label, slots in (("home", ids), ("away", other_ids)):
        team = xy[:, slots, :]
        valid = np.isfinite(team).all(axis=2)
        count = valid.sum(axis=1)
        center = np.divide(np.nansum(team, axis=1), count[:, None],
                           out=np.full((n_frames, 2), np.nan, dtype=np.float32),
                           where=count[:, None] > 0)
        centers.append(center)
        valid_counts.append(count)
        x_values = np.where(valid, team[:, :, 0], np.nan)
        y_values = np.where(valid, team[:, :, 1], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            width = np.nanmax(x_values, axis=1) - np.nanmin(x_values, axis=1)
            height = np.nanmax(y_values, axis=1) - np.nanmin(y_values, axis=1)
            radial = np.linalg.norm(team - center[:, None, :], axis=2)
            radial_q90 = np.nanpercentile(radial, 90, axis=1)
        width[count == 0] = np.nan
        height[count == 0] = np.nan
        radial_q90[count == 0] = np.nan
        features[label + "_players"] = count.astype(np.float32)
        features[label + "_centroid_x_m"] = center[:, 0]
        features[label + "_centroid_y_m"] = center[:, 1]
        features[label + "_bbox_area_m2"] = (width * height).astype(np.float32)
        features[label + "_radial_spread_q90_m"] = radial_q90.astype(np.float32)

        sample_frames = _sample_frames(n_frames, stride)
        hull_area = np.full(len(sample_frames), np.nan, dtype=np.float32)
        bbox_ratio = np.full(len(sample_frames), np.nan, dtype=np.float32)
        length, breadth = pitch
        for index, frame in enumerate(sample_frames):
            points = team[frame]
            points = points[np.isfinite(points).all(axis=1)]
            if len(points) < 3:
                continue
            span = np.ptp(points, axis=0)
            bbox_ratio[index] = float(span[0] / max(length, 1e-6) + span[1] / max(breadth, 1e-6))
            if len(np.unique(points, axis=0)) < 3:
                hull_area[index] = 0.0
                continue
            try:
                hull_area[index] = float(ConvexHull(points).volume)
            except QhullError:
                hull_area[index] = 0.0
        features[label + "_hull_area_m2"] = _interpolate_sampled(sample_frames, hull_area, n_frames)
        features[label + "_shape_span_ratio"] = _interpolate_sampled(sample_frames, bbox_ratio, n_frames)

    home, away = centers
    features["team_centroid_separation_m"] = np.linalg.norm(home - away, axis=1).astype(np.float32)
    features["team_centroid_x_gap_m"] = np.abs(home[:, 0] - away[:, 0]).astype(np.float32)
    features["team_player_count_min"] = np.minimum(valid_counts[0], valid_counts[1]).astype(np.float32)
    features["team_player_count_gap"] = np.abs(valid_counts[0] - valid_counts[1]).astype(np.float32)
    for label, center in (("home", home), ("away", away)):
        lag = max(1, int(round(stride)))
        displacement = np.full(n_frames, np.nan, dtype=np.float32)
        if n_frames > lag:
            valid = np.isfinite(center[lag:]).all(axis=1) & np.isfinite(center[:-lag]).all(axis=1)
            local = np.linalg.norm(center[lag:] - center[:-lag], axis=1) * 25.0 / lag
            displacement[lag:][valid] = local[valid]
        features[label + "_centroid_speed_mps"] = displacement

    # Coarse nearest-player Voronoi control share, calculated at one sample per
    # second and interpolated. It is a spatial ownership baseline, not a true
    # interception-probability model.
    grid_x = np.linspace(2.5, pitch[0] - 2.5, 13, dtype=np.float32)
    grid_y = np.linspace(2.5, pitch[1] - 2.5, 9, dtype=np.float32)
    gx, gy = np.meshgrid(grid_x, grid_y)
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    vor_frames = _sample_frames(n_frames, 25)
    vor_share = np.full(len(vor_frames), np.nan, dtype=np.float32)
    for index, frame in enumerate(vor_frames):
        players = xy[frame]
        good = np.isfinite(players).all(axis=1)
        home_good = good & team_slots
        away_good = good & ~team_slots
        if home_good.sum() < 6 or away_good.sum() < 6:
            continue
        distance = np.linalg.norm(players[good, None, :] - grid[None, :, :], axis=2)
        nearest = np.argmin(distance, axis=0)
        valid_slots = np.flatnonzero(good)
        vor_share[index] = float(team_slots[valid_slots[nearest]].mean())
    features["home_voronoi_share"] = _interpolate_sampled(vor_frames, vor_share, n_frames)

    # Opponent convergence is sampled with the hull geometry cadence.
    duel_frames = sample_frames
    close_pairs = np.full(len(duel_frames), np.nan, dtype=np.float32)
    nearest_opponent = np.full(len(duel_frames), np.nan, dtype=np.float32)
    for index, frame in enumerate(duel_frames):
        home_xy, away_xy = xy[frame, ids], xy[frame, other_ids]
        home_xy = home_xy[np.isfinite(home_xy).all(axis=1)]
        away_xy = away_xy[np.isfinite(away_xy).all(axis=1)]
        if len(home_xy) and len(away_xy):
            pair_distance = np.linalg.norm(home_xy[:, None, :] - away_xy[None, :, :], axis=2)
            close_pairs[index] = float((pair_distance <= 2.0).sum())
            nearest_opponent[index] = float(pair_distance.min())
    features["opponent_pairs_within_2m"] = _interpolate_sampled(duel_frames, close_pairs, n_frames)
    features["nearest_opponent_distance_m"] = _interpolate_sampled(duel_frames, nearest_opponent, n_frames)
    return features


def build_deadball_features(match, player_speed_cap_mps=12.0, ball_radius_m=0.11):
    """Create tracking-only dead-ball, press, movement, and formation features."""
    xy = np.asarray(match.player_xy_m, dtype=np.float32)
    ball = np.asarray(match.ball_xy_m, dtype=np.float32)
    n_frames, n_players = xy.shape[:2]
    fps = float(match.fps)
    if fps <= 0 or ball.shape != (n_frames, 2):
        raise ValueError("Tracking arrays must align and frame rate must be positive")
    if match.home_slots is None or len(match.home_slots) != n_players:
        raise ValueError("Team slot map must match the player tracking array")
    features = {}

    # A centered two-frame difference reduces coordinate jitter versus a single
    # frame jump while preserving the 25 fps timing. The 1.4 m/s and 3.5 m/s
    # bands retain the earlier walking and sprint proxy thresholds.
    velocity = np.full_like(xy, np.nan)
    if n_frames >= 3:
        valid = np.isfinite(xy[2:]).all(axis=2) & np.isfinite(xy[:-2]).all(axis=2)
        centered = (xy[2:] - xy[:-2]) * (fps / 2.0)
        velocity[1:-1] = np.where(valid[:, :, None], centered, np.nan)
    if n_frames >= 2:
        edge_valid = np.isfinite(xy[1]).all(axis=1) & np.isfinite(xy[0]).all(axis=1)
        velocity[0] = np.where(edge_valid[:, None], (xy[1] - xy[0]) * fps, np.nan)
        edge_valid = np.isfinite(xy[-1]).all(axis=1) & np.isfinite(xy[-2]).all(axis=1)
        velocity[-1] = np.where(edge_valid[:, None], (xy[-1] - xy[-2]) * fps, np.nan)
    speed = np.linalg.norm(velocity, axis=2)
    speed[(speed > player_speed_cap_mps) | ~np.isfinite(speed)] = np.nan
    valid_speed = np.isfinite(speed)
    speed_count = valid_speed.sum(axis=1)
    acceleration = np.full_like(speed, np.nan)
    if n_frames >= 3:
        acceleration[1:-1] = np.linalg.norm(velocity[2:] - velocity[:-2], axis=2) * fps / 2.0
    signed_speed_change = np.full_like(speed, np.nan)
    if n_frames >= 2:
        signed_speed_change[1:] = (speed[1:] - speed[:-1]) * fps
    features["players_observed"] = np.isfinite(xy).all(axis=2).sum(axis=1).astype(np.float32)
    features["speed_tracks"] = np.isfinite(speed).sum(axis=1).astype(np.float32)
    features["speed_median_mps"] = _row_percentile(speed, 50)
    features["speed_q75_mps"] = _row_percentile(speed, 75)
    features["speed_q90_mps"] = _row_percentile(speed, 90)
    features["speed_max_mps"] = _row_percentile(speed, 100)
    features["walking_fraction"] = np.divide(
        ((speed <= 1.4) & valid_speed).sum(axis=1), speed_count,
        out=np.full(n_frames, np.nan, dtype=np.float32), where=speed_count > 0)
    features["burst_fraction"] = np.divide(
        ((speed >= 3.5) & valid_speed).sum(axis=1), speed_count,
        out=np.full(n_frames, np.nan, dtype=np.float32), where=speed_count > 0)
    features["acceleration_q90_mps2"] = _row_percentile(acceleration, 90)
    braking_valid = np.isfinite(signed_speed_change)
    braking_count = braking_valid.sum(axis=1)
    features["hard_braking_fraction"] = np.divide(
        ((signed_speed_change <= -4.0) & braking_valid).sum(axis=1), braking_count,
        out=np.full(n_frames, np.nan, dtype=np.float32), where=braking_count > 0)

    seen = np.isfinite(ball).all(axis=1)
    features["ball_seen"] = seen.astype(np.float32)
    features["ball_gap_age_s"] = _true_run_length(~seen).astype(np.float32) / fps
    pitch_length, pitch_width = match.pitch
    x, y = ball[:, 0], ball[:, 1]
    outside_margin = np.maximum.reduce([-x, x - pitch_length, -y, y - pitch_width])
    features["ball_outside_margin_m"] = np.where(seen, outside_margin, np.nan).astype(np.float32)
    features["ball_entirely_outside"] = (seen & (outside_margin > ball_radius_m)).astype(np.float32)
    edge_dist = np.minimum.reduce([np.abs(x), np.abs(pitch_length - x), np.abs(y), np.abs(pitch_width - y)])
    features["ball_boundary_distance_m"] = np.where(seen, edge_dist, np.nan).astype(np.float32)
    features["ball_near_boundary_1m"] = (seen & (edge_dist <= 1.0)).astype(np.float32)
    features["ball_x_normalized"] = np.where(seen, x / pitch_length, np.nan).astype(np.float32)
    features["ball_y_normalized"] = np.where(seen, y / pitch_width, np.nan).astype(np.float32)
    zone_position = np.where(seen, np.clip(x / pitch_length, 0.0, 0.9999), 0.0)
    x_zone = np.floor(zone_position * 3).astype(np.int8)
    features["ball_left_third"] = (seen & (x_zone == 0)).astype(np.float32)
    features["ball_middle_third"] = (seen & (x_zone == 1)).astype(np.float32)
    features["ball_right_third"] = (seen & (x_zone == 2)).astype(np.float32)
    ball_velocity = np.full((n_frames, 2), np.nan, dtype=np.float32)
    if n_frames >= 2:
        pair_valid = seen[1:] & seen[:-1]
        ball_velocity[1:] = np.where(pair_valid[:, None], (ball[1:] - ball[:-1]) * fps, np.nan)
    ball_speed = np.linalg.norm(ball_velocity, axis=1)
    ball_speed[(ball_speed > 45.0) | ~np.isfinite(ball_speed)] = np.nan
    features["ball_speed_mps"] = ball_speed.astype(np.float32)
    features["ball_deceleration_mps2"] = np.r_[np.nan, (ball_speed[:-1] - ball_speed[1:]) * fps].astype(np.float32)
    outward = np.zeros(n_frames, dtype=bool)
    outward |= (x <= 1.5) & (ball_velocity[:, 0] < -0.5)
    outward |= (x >= pitch_length - 1.5) & (ball_velocity[:, 0] > 0.5)
    outward |= (y <= 1.5) & (ball_velocity[:, 1] < -0.5)
    outward |= (y >= pitch_width - 1.5) & (ball_velocity[:, 1] > 0.5)
    features["ball_moving_outward_near_line"] = (seen & outward).astype(np.float32)

    # Short path summaries are available only when every point in the window
    # is observed; they never bridge a missing-ball gap with a straight line.
    missing_prefix = np.r_[0, np.cumsum(~seen)]
    step_distance = np.linalg.norm(np.diff(ball, axis=0), axis=1)
    valid_step = seen[1:] & seen[:-1]
    step_prefix = np.r_[0.0, np.cumsum(np.where(valid_step, step_distance, 0.0))]
    for seconds in (0.4, 1.0, 2.0):
        lag = max(1, int(round(seconds * fps)))
        displacement = np.full(n_frames, np.nan, dtype=np.float32)
        path_length = np.full(n_frames, np.nan, dtype=np.float32)
        if n_frames > lag:
            ends = np.arange(lag, n_frames)
            complete = (missing_prefix[ends + 1] - missing_prefix[ends - lag] == 0)
            displacement[ends[complete]] = np.linalg.norm(
                ball[ends[complete]] - ball[ends[complete] - lag], axis=1)
            # Use a prefix sum indexed by frame; it contains one element more
            # than the edge vector so interval endpoints stay unambiguous.
            edge_counts = np.r_[0, np.cumsum(valid_step.astype(np.int32))]
            complete_edges = (edge_counts[ends] - edge_counts[ends - lag] == lag)
            ok = complete & complete_edges
            path_length[ends[ok]] = step_prefix[ends[ok]] - step_prefix[ends[ok] - lag]
        features["ball_net_displacement_{:.1f}s_m".format(seconds)] = displacement
        features["ball_path_length_{:.1f}s_m".format(seconds)] = path_length
        features["ball_path_efficiency_{:.1f}s".format(seconds)] = np.divide(
            displacement, path_length, out=np.full(n_frames, np.nan, dtype=np.float32),
            where=np.isfinite(path_length) & (path_length > 1e-3))

    # Last known ball position supports measured convergence and pressure while
    # the ball is occluded. Age and visibility remain explicit model inputs.
    frame_ids = np.arange(n_frames)
    last_seen = np.maximum.accumulate(np.where(seen, frame_ids, -1))
    last_ball = np.full_like(ball, np.nan)
    known = last_seen >= 0
    last_ball[known] = ball[last_seen[known]]
    player_to_last_ball = np.linalg.norm(xy - last_ball[:, None, :], axis=2)
    home = np.asarray(match.home_slots, dtype=bool)
    home_dist = _nanmin_rows(player_to_last_ball[:, home])
    away_dist = _nanmin_rows(player_to_last_ball[:, ~home])
    features["home_nearest_ball_m"] = home_dist
    features["away_nearest_ball_m"] = away_dist
    features["nearest_team_ball_distance_gap_m"] = np.abs(home_dist - away_dist).astype(np.float32)
    within_three_m = np.isfinite(player_to_last_ball) & (player_to_last_ball <= 3.0)
    features["home_players_within_1m_last_ball"] = (
        np.isfinite(player_to_last_ball[:, home]) & (player_to_last_ball[:, home] <= 1.0)).sum(axis=1).astype(np.float32)
    features["away_players_within_1m_last_ball"] = (
        np.isfinite(player_to_last_ball[:, ~home]) & (player_to_last_ball[:, ~home] <= 1.0)).sum(axis=1).astype(np.float32)
    features["home_players_within_1_5m_last_ball"] = (
        np.isfinite(player_to_last_ball[:, home]) & (player_to_last_ball[:, home] <= 1.5)).sum(axis=1).astype(np.float32)
    features["away_players_within_1_5m_last_ball"] = (
        np.isfinite(player_to_last_ball[:, ~home]) & (player_to_last_ball[:, ~home] <= 1.5)).sum(axis=1).astype(np.float32)
    features["home_players_within_3m_last_ball"] = within_three_m[:, home].sum(axis=1).astype(np.float32)
    features["away_players_within_3m_last_ball"] = within_three_m[:, ~home].sum(axis=1).astype(np.float32)
    lag = min(max(1, int(round(0.4 * fps))), max(1, n_frames - 1))
    if n_frames > lag:
        closer = (player_to_last_ball[:-lag] - player_to_last_ball[lag:]) >= 0.5
        converging = closer & (player_to_last_ball[lag:] <= 12.0) & np.isfinite(player_to_last_ball[lag:])
        convergence = np.zeros((n_frames, n_players), dtype=bool)
        convergence[lag:] = converging
    else:
        convergence = np.zeros((n_frames, n_players), dtype=bool)
    features["home_converging_players"] = convergence[:, home].sum(axis=1).astype(np.float32)
    features["away_converging_players"] = convergence[:, ~home].sum(axis=1).astype(np.float32)
    features["all_converging_players"] = convergence.sum(axis=1).astype(np.float32)

    geometry = _team_geometry(xy, home, match.pitch, stride=max(1, int(round(fps))))
    features.update(geometry)
    # Team-relative movement and pressing signals. They describe movement
    # patterns but cannot prove a foul, goal, or possession change on their own.
    features["team_speed_median_gap_mps"] = np.abs(
        _team_speed_median(speed, home) - _team_speed_median(speed, ~home)).astype(np.float32)
    features["home_team_speed_median_mps"] = _team_speed_median(speed, home)
    features["away_team_speed_median_mps"] = _team_speed_median(speed, ~home)
    return features


def _team_speed_median(speed, team_mask):
    if not team_mask.any():
        return np.full(len(speed), np.nan, dtype=np.float32)
    return _row_percentile(speed[:, team_mask], 50)


def _true_run_length(mask):
    mask = np.asarray(mask, dtype=bool)
    index = np.arange(len(mask))
    last_false = np.maximum.accumulate(np.where(~mask, index, -1))
    return np.where(mask, index - last_false, 0)


FEATURE_GROUPS = {
    "ball": ["ball_seen", "ball_gap_age_s", "ball_outside_margin_m", "ball_entirely_outside",
             "ball_boundary_distance_m", "ball_near_boundary_1m", "ball_speed_mps",
             "ball_deceleration_mps2", "ball_moving_outward_near_line", "ball_x_normalized",
             "ball_y_normalized", "ball_left_third", "ball_middle_third", "ball_right_third",
             "ball_net_displacement_0.4s_m", "ball_path_length_0.4s_m", "ball_path_efficiency_0.4s",
             "ball_net_displacement_1.0s_m", "ball_path_length_1.0s_m", "ball_path_efficiency_1.0s",
             "ball_net_displacement_2.0s_m", "ball_path_length_2.0s_m", "ball_path_efficiency_2.0s"],
    "movement": ["players_observed", "speed_tracks", "speed_median_mps", "speed_q75_mps",
                 "speed_q90_mps", "speed_max_mps", "walking_fraction", "burst_fraction",
                 "acceleration_q90_mps2", "hard_braking_fraction", "team_speed_median_gap_mps",
                 "home_team_speed_median_mps", "away_team_speed_median_mps"],
    "spatial": ["home_players", "away_players", "home_bbox_area_m2", "away_bbox_area_m2",
                "home_hull_area_m2", "away_hull_area_m2", "home_radial_spread_q90_m",
                "away_radial_spread_q90_m", "team_centroid_separation_m", "team_centroid_x_gap_m",
                "home_centroid_speed_mps", "away_centroid_speed_mps", "home_voronoi_share",
                "opponent_pairs_within_2m", "nearest_opponent_distance_m"],
    "convergence": ["home_nearest_ball_m", "away_nearest_ball_m", "nearest_team_ball_distance_gap_m",
                    "home_players_within_3m_last_ball", "away_players_within_3m_last_ball",
                    "home_players_within_1m_last_ball", "away_players_within_1m_last_ball",
                    "home_players_within_1_5m_last_ball", "away_players_within_1_5m_last_ball",
                    "home_converging_players", "away_converging_players", "all_converging_players"],
}


def feature_columns(group_names=("ball", "movement", "spatial", "convergence")):
    return [column for group in group_names for column in FEATURE_GROUPS[group]]
