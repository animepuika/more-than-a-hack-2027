"""Missing-ball abstention policy based on the official X-fill rule.

The policy keeps the V2/V3 possession estimate during fast, multi-player movement,
and writes X during calmer missing-ball frames. The official spell scorer fills X
from the previous prediction, so this is a deliberate "hold the last owner" rule.
"""

import numpy as np


def chaotic_movement_mask(motion, motion_track_counts, config):
    """Flag a possible scramble from track acceleration/braking, without claiming an event.

    The two team acceleration and braking fractions are converted back to approximate
    reacting-track counts. A frame is chaotic when at least three tracks react across
    both teams, one team has at least three reacting tracks while moving quickly, or
    both team medians show sprint-like speed. This is a movement proxy, not a tackle
    or turnover detector.
    """
    motion = np.asarray(motion, dtype=float)
    counts = np.asarray(motion_track_counts, dtype=float)
    if motion.ndim != 2 or motion.shape[1] != 10:
        raise ValueError("Expected frame-by-10 motion features")
    if counts.shape != (len(motion), 2):
        raise ValueError("Expected frame-by-two team motion track counts")
    if config["minimum_reacting_players_each_team"] < 1:
        raise ValueError("Each-team reacting-track threshold must be positive")
    if config["minimum_total_reacting_players"] < 2:
        raise ValueError("Combined reacting-track threshold must be at least two")
    if config["minimum_one_team_reacting_players"] < 1:
        raise ValueError("One-team reacting-track threshold must be positive")

    # motion columns: home speed/accel/burst/brake/closing, then away equivalents.
    h_speed, a_speed = motion[:, 0], motion[:, 5]
    h_fraction = np.nan_to_num(motion[:, 2], nan=0.0) + np.nan_to_num(motion[:, 3], nan=0.0)
    a_fraction = np.nan_to_num(motion[:, 7], nan=0.0) + np.nan_to_num(motion[:, 8], nan=0.0)
    h_reacting = np.rint(h_fraction * counts[:, 0])
    a_reacting = np.rint(a_fraction * counts[:, 1])
    adequate = np.isfinite(h_speed) & np.isfinite(a_speed)
    adequate &= counts.min(axis=1) >= config["minimum_motion_tracks_per_team"]

    two_sided = (h_reacting >= config["minimum_reacting_players_each_team"])
    two_sided &= a_reacting >= config["minimum_reacting_players_each_team"]
    two_sided &= h_reacting + a_reacting >= config["minimum_total_reacting_players"]
    one_sided = np.maximum(h_reacting, a_reacting) >= config["minimum_one_team_reacting_players"]
    one_sided &= np.maximum(h_speed, a_speed) >= config["one_team_speed_threshold_mps"]
    both_fast = np.minimum(h_speed, a_speed) >= config["both_teams_speed_threshold_mps"]
    return adequate & (two_sided | one_sided | both_fast)


def apply_missing_ball_policy(labels, ball_seen, active, chaos, mode):
    """Copy labels and abstain on eligible missing-ball frames.

    ``mode='all'`` abstains on every missing-ball H/A frame. ``mode='calm'`` only
    abstains when the movement proxy does not flag a possible scramble. D labels,
    visible-ball frames, and frames outside the scored halves are preserved.
    """
    labels = np.asarray(labels, dtype="<U1")
    ball_seen = np.asarray(ball_seen, dtype=bool)
    active = np.asarray(active, dtype=bool)
    chaos = np.asarray(chaos, dtype=bool)
    if any(array.shape != labels.shape for array in (ball_seen, active, chaos)):
        raise ValueError("All policy inputs must have the same frame shape")
    if mode not in ("all", "calm"):
        raise ValueError("mode must be 'all' or 'calm'")
    eligible = active & ~ball_seen & np.isin(labels, ("H", "A"))
    mask = eligible if mode == "all" else eligible & ~chaos
    output = labels.copy()
    output[mask] = "X"
    return output, mask
