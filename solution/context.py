"""Collective geometry and short track-fragment motion, without answer labels.

No named-player roles or permanent identity assumptions are used. Prototype
fitting uses only strong visible-ball proximity as noisy proxy labels.
"""

import numpy as np
import pandas as pd

from .features import runs

SHAPE_NAMES = ("log_hull_area", "width_m", "length_m", "centroid_x_m", "centroid_y_m", "spread_m")
MOTION_NAMES = ("speed_mps", "acceleration_mps2", "burst_fraction", "braking_fraction", "closing_fraction")


def hull_area(points):
    """Monotone-chain convex hull and shoelace area in square metres."""
    points = sorted(set(map(tuple, np.asarray(points, dtype=float))))
    if len(points) < 3:
        return 0.0
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    ring = np.asarray(lower[:-1] + upper[:-1])
    return float(abs(np.sum(ring[:, 0] * np.roll(ring[:, 1], -1) -
                            ring[:, 1] * np.roll(ring[:, 0], -1))) / 2)


def unique_positions(points, minimum_distance):
    """Remove very close duplicate detections before estimating team geometry."""
    kept = []
    for point in points:
        if not kept or np.min(np.linalg.norm(np.asarray(kept) - point, axis=1)) >= minimum_distance:
            kept.append(point)
    return np.asarray(kept).reshape(-1, 2)


def estimate_ball(tracking, features, blocks, config):
    """Interpolate short, internally bounded gaps only; this is an estimate."""
    result = tracking.ball_xy_m.copy()
    result[~features["ball_reliable"]] = np.nan
    estimated = np.zeros(len(tracking), dtype=bool)
    stopped = features["outside_stop"] | features["slow_gap_stop"] | features["stationary_stop"]
    for _, start, last in blocks:
        for a, b in runs(~np.isfinite(result[start:last + 1]).all(axis=1)):
            a, b = start + a, start + b
            if a == start or b == last + 1 or (b - a) / tracking.fps > config["maximum_ball_interpolation_s"]:
                continue
            if not features["control_usable"][a - 1] or not features["control_usable"][b] or stopped[a:b].any():
                continue
            left, right = result[a - 1], result[b]
            travel_speed = np.linalg.norm(right - left) * tracking.fps / (b - a + 1)
            if travel_speed > config["maximum_interpolation_speed_mps"]:
                continue
            result[a:b] = np.linspace(left, right, b - a + 2)[1:-1]
            estimated[a:b] = True
    return result, estimated


def fragment_motion(xy, blocks, fps, config):
    """Trailing-smoothed velocity; gaps, half boundaries, and jumps reset it."""
    velocity = np.full_like(xy, np.nan, dtype=np.float32)
    speed = np.full(xy.shape[:2], np.nan, dtype=np.float32)
    acceleration = np.full_like(speed, np.nan)
    window = config["velocity_window_frames"]
    for _, start, last in blocks:
        sl = slice(start, last + 1)
        local = xy[sl]
        raw = np.full_like(local, np.nan)
        raw[1:] = np.diff(local, axis=0) * fps
        raw_speed = np.linalg.norm(raw, axis=2)
        raw[raw_speed > config["maximum_track_speed_mps"]] = np.nan
        for axis in range(2):
            velocity[sl, :, axis] = pd.DataFrame(raw[:, :, axis]).rolling(
                window, min_periods=window).mean().to_numpy(dtype=np.float32)
        speed[sl] = np.linalg.norm(velocity[sl], axis=2)
        local_accel = np.full_like(speed[sl], np.nan)
        local_accel[1:] = np.diff(speed[sl], axis=0) * fps
        local_accel[np.abs(local_accel) > config["maximum_acceleration_mps2"]] = np.nan
        acceleration[sl] = local_accel
    return velocity, speed, acceleration


def build_context(tracking, features, blocks, config, progress=None):
    if tracking.track_xy_m is None or tracking.home_slots is None:
        raise ValueError("Load tracking with retain_tracks=True for v2")
    n = len(tracking)
    shape = np.full((n, 12), np.nan, dtype=np.float32)
    counts = np.zeros((n, 2), dtype=np.int16)
    xy = tracking.track_xy_m.copy()
    length, width = tracking.pitch
    margin = config["position_margin_m"]
    bad = (~np.isfinite(xy).all(axis=2) | (xy[:, :, 0] < -margin) |
           (xy[:, :, 0] > length + margin) | (xy[:, :, 1] < -margin) | (xy[:, :, 1] > width + margin))
    xy[bad] = np.nan
    sides = (tracking.home_slots, ~tracking.home_slots)
    for i in np.flatnonzero(features["active"]):
        for side, slots in enumerate(sides):
            points = xy[i, slots]
            points = points[np.isfinite(points).all(axis=1)]
            points = unique_positions(points, config["duplicate_distance_m"])
            counts[i, side] = len(points)
            if len(points) < config["minimum_team_tracks"]:
                continue
            center = points.mean(axis=0)
            # Robust spans reduce the influence of a single goalkeeper or outlier.
            span = np.quantile(points, 0.9, axis=0) - np.quantile(points, 0.1, axis=0)
            shape[i, side * 6:(side + 1) * 6] = [
                np.log1p(hull_area(points)), span[1], span[0], center[0], center[1],
                np.sqrt(np.mean(np.sum((points - center) ** 2, axis=1)))]
        if progress is not None and (i + 1) % 50000 == 0:
            progress(i + 1, n)

    if progress is not None:
        progress(n, n)
    shape[counts.min(axis=1) < config["minimum_team_tracks"]] = np.nan
    velocity, speed, acceleration = fragment_motion(xy, blocks, tracking.fps, config)
    ball, estimated = estimate_ball(tracking, features, blocks, config)
    towards = ball[:, None, :] - xy
    distance = np.linalg.norm(towards, axis=2)
    closing = np.divide(np.sum(velocity * towards, axis=2), distance,
                        out=np.full_like(speed, np.nan), where=distance > 0.1)
    motion = np.full((n, 10), np.nan, dtype=np.float32)
    converging = np.zeros((n, 2), dtype=np.int16)
    motion_counts = np.zeros((n, 2), dtype=np.int16)
    for side, slots in enumerate(sides):
        valid_speed = np.isfinite(speed[:, slots])
        valid_accel = np.isfinite(acceleration[:, slots])
        number = valid_speed.sum(axis=1)
        motion_counts[:, side] = number
        accel_number = valid_accel.sum(axis=1)
        usable = number >= config["minimum_motion_tracks"]
        usable_accel = accel_number >= config["minimum_motion_tracks"]
        motion[usable, side * 5] = pd.DataFrame(speed[:, slots]).median(axis=1).to_numpy()[usable]
        motion[usable_accel, side * 5 + 1] = pd.DataFrame(acceleration[:, slots]).median(axis=1).to_numpy()[usable_accel]
        for offset, condition in ((2, acceleration[:, slots] >= config["burst_acceleration_mps2"]),
                                  (3, acceleration[:, slots] <= -config["burst_acceleration_mps2"])):
            fraction = condition.sum(axis=1) / np.maximum(accel_number, 1)
            motion[usable_accel, side * 5 + offset] = fraction[usable_accel]
        near = distance[:, slots] <= config["convergence_radius_m"]
        converging[:, side] = ((closing[:, slots] >= config["convergence_closing_speed_mps"]) & near).sum(axis=1)
        known_ball = np.isfinite(ball).all(axis=1)
        mask = usable & known_ball
        motion[mask, side * 5 + 4] = converging[mask, side] / number[mask]
    # A common target is a weak clue only when at least three tracks from both sides close in.
    both_converge = (converging.sum(axis=1) >= 3) & (converging.min(axis=1) >= 1)
    motion[~both_converge, 4] = np.nan
    motion[~both_converge, 9] = np.nan
    motion[~features["active"]] = np.nan
    motion[motion_counts.min(axis=1) < config["minimum_motion_tracks"]] = np.nan
    return {"shape": shape, "motion": motion, "team_track_counts": counts,
            "ball_interpolated": estimated, "converging_track_counts": converging,
            "both_teams_converging": both_converge, "motion_track_counts": motion_counts,
            "shape_names": [side + "_" + name for side in ("home", "away") for name in SHAPE_NAMES],
            "motion_names": [side + "_" + name for side in ("home", "away") for name in MOTION_NAMES]}


def proxy_costs(values, tracking, features, blocks, config, weight):
    """Half-specific robust prototypes, fitted to noisy tracking-only anchors.

    Subsample anchors to one frame per second. The public answer labels never
    enter this function. Each feature must have anchors from BOTH teams.
    """
    costs = np.zeros((len(tracking), 2))
    metadata = []
    distances = np.column_stack([tracking.home_distance_m, tracking.away_distance_m])
    closest = np.argmin(np.nan_to_num(distances, nan=np.inf), axis=1)
    strong = features["control_usable"] & np.isfinite(distances).all(axis=1)
    strong &= np.min(distances, axis=1) <= config["anchor_radius_m"]
    strong &= np.max(distances, axis=1) >= config["anchor_opponent_distance_m"]
    stopped = features["outside_stop"] | features["slow_gap_stop"] | features["stationary_stop"]
    for half, start, last in blocks:
        sampled = np.arange(start, last + 1, max(1, round(tracking.fps)))
        anchors = [sampled[strong[sampled] & (closest[sampled] == side)] for side in (0, 1)]
        minimum = config["minimum_anchors_per_team"]
        keep = np.array([all(np.isfinite(values[rows, column]).sum() >= minimum for rows in anchors)
                         for column in range(values.shape[1])])
        info = {"half": half, "anchor_frames_home": len(anchors[0]), "anchor_frames_away": len(anchors[1]),
                "usable_features": int(keep.sum()), "weight": weight}
        metadata.append(info)
        if not keep.any():
            continue
        data = values[start:last + 1, keep]
        anchor_data = [values[rows][:, keep] for rows in anchors]
        centers = np.array([np.nanmedian(part, axis=0) for part in anchor_data])
        pooled = np.vstack(anchor_data)
        scale = np.nanmedian(np.abs(pooled - np.nanmedian(pooled, axis=0)), axis=0) * 1.4826
        scale = np.maximum(scale, 0.1)
        valid = np.isfinite(data)
        errors = np.nan_to_num(np.clip(((data[:, None, :] - centers) / scale) ** 2, 0, 9))
        scores = errors.sum(axis=2) / np.maximum(valid.sum(axis=1, keepdims=True), 1)
        scores -= scores.min(axis=1, keepdims=True)
        scores = np.minimum(scores, 1.0) * weight
        allowed = ~strong[start:last + 1] & ~stopped[start:last + 1]
        # Require at least half the retained features; low coverage contributes no evidence.
        allowed &= valid.sum(axis=1) >= max(1, int(np.ceil(keep.sum() / 2)))
        costs[start:last + 1][allowed] = scores[allowed]
    return costs, metadata


def tagging_costs(events, team_names, features, config):
    """Soft team priors from automatic phases; never copy intervals as answers.

    Set pieces and goals have uncertain boundaries and are deliberately excluded.
    Defending phases name the defending team, so their possession prior is inverted.
    """
    n = len(features["active"])
    votes = np.zeros((n, 2))
    mapping = {name.strip().casefold(): side for side, name in enumerate(team_names)}
    attacking = {"build up", "progression", "attacking transition", "long ball",
                 "creating chances", "ball in final third", "ball in the box"}
    defensive = {"defending in defensive third", "defending in middle third",
                 "defending in attacking third", "defensive transition"}
    used = 0
    for event in events:
        code = (event["code"] or "").strip().casefold()
        side = mapping.get((event["tags"].get("Team") or "").strip().casefold())
        if side is None or code not in attacking | defensive:
            continue
        if code in defensive:
            side = 1 - side
        start, stop = max(0, event["start_frame"]), min(n, event["end_frame"])
        if start >= stop:
            continue
        votes[start:stop, side] += 1
        used += 1
    total = votes.sum(axis=1)
    prior = np.divide(votes, total[:, None], out=np.zeros_like(votes), where=total[:, None] > 0)
    costs = (prior.max(axis=1, keepdims=True) - prior) * config["tag_weight"]
    costs[~features["active"]] = 0
    return costs, {"automatic_events_used": used, "frames_with_team_prior": int((total > 0).sum()),
                   "set_pieces_goals_and_unknown_teams_used": False}
