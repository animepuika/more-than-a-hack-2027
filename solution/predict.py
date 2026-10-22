"""Three-state sequence inference plus a separate, optional X output policy.

This is an offline rules-based experiment, not a trained or calibrated model.
No label/answer file is imported or read by this module.
"""

import numpy as np

from .features import runs

LABELS = np.array(["H", "A", "D"])


def decode_sequence(costs, team_switch_cost, dead_switch_cost):
    """Viterbi minimum-cost path for H/A/D; neutral gaps retain surrounding evidence."""
    costs = np.asarray(costs, dtype=float)
    if costs.ndim != 2 or costs.shape[1] != 3 or not np.isfinite(costs).all():
        raise ValueError("Expected finite frame-by-three costs")
    if team_switch_cost < 0 or dead_switch_cost < 0:
        raise ValueError("Transition costs cannot be negative")
    if not len(costs):
        return np.empty(0, dtype=np.int8)
    transitions = np.array([
        [0.0, team_switch_cost, dead_switch_cost],
        [team_switch_cost, 0.0, dead_switch_cost],
        [dead_switch_cost, dead_switch_cost, 0.0],
    ])
    back = np.zeros((len(costs), 3), dtype=np.int8)
    value = costs[0].copy()
    columns = np.arange(3)
    for i in range(1, len(costs)):
        alternatives = value[:, None] + transitions
        parent = np.argmin(alternatives, axis=0)
        back[i] = parent
        value = alternatives[parent, columns] + costs[i]
        value -= value.min()
    states = np.empty(len(costs), dtype=np.int8)
    states[-1] = int(np.argmin(value))
    for i in range(len(states) - 1, 0, -1):
        states[i - 1] = back[i, states[i]]
    return states


def evidence_costs(tracking, features, config, use_stoppages):
    """Relative costs, not probabilities. Thresholds are declared before scoring."""
    n = len(tracking)
    costs = np.zeros((n, 3))
    costs[:, 2] = 0.2  # A weak live-play prior, not an assumption that missing = dead.
    distances = np.column_stack([tracking.home_distance_m, tracking.away_distance_m])
    nearest = np.fmin(distances[:, 0], distances[:, 1])
    usable = features["control_usable"] & np.isfinite(distances).all(axis=1)
    usable &= nearest <= config["control_radius_m"]
    usable &= np.abs(distances[:, 0] - distances[:, 1]) >= config["control_margin_m"]
    strength = np.clip((config["control_radius_m"] - nearest) /
                       config["control_radius_m"], 0.0, 1.0)
    strength = np.nan_to_num(strength)
    team_costs = np.clip((distances - nearest[:, None]) /
                         config["distance_scale_m"], 0.0, 4.0)
    costs[usable, :2] = team_costs[usable] * strength[usable, None]
    costs[usable, 2] = 0.2 + 3.0 * strength[usable]

    if use_stoppages:
        slow = features["slow_gap_stop"] | features["stationary_stop"]
        costs[slow, :2] = 1.2
        costs[slow, 2] = 0.0
        outside = features["outside_stop"]
        costs[outside, :2] = 4.0
        costs[outside, 2] = 0.0
    return costs


def stable_gap_mask(labels, features, blocks, fps, max_gap_s):
    """Abstain only inside short gaps bounded by observed, same-team predictions."""
    mask = np.zeros(len(labels), dtype=bool)
    stop_cue = features["outside_stop"] | features["slow_gap_stop"] | features["stationary_stop"]
    for _, start, last in blocks:
        stop = last + 1
        for a, b in runs(~features["ball_seen"][start:stop]):
            a, b = a + start, b + start
            if a == start or b == stop or (b - a) / fps > max_gap_s:
                continue
            if not features["control_usable"][a - 1] or not features["control_usable"][b]:
                continue
            owner = labels[a - 1]
            if owner not in ("H", "A") or labels[b] != owner:
                continue
            if np.all(labels[a:b] == owner) and not stop_cue[a:b].any():
                mask[a:b] = True
    return mask


def predict(tracking, features, blocks, config, use_stoppages=True, abstain=False,
            extra_team_costs=None):
    costs = evidence_costs(tracking, features, config, use_stoppages)
    if extra_team_costs is not None:
        extra_team_costs = np.asarray(extra_team_costs, dtype=float)
        if (extra_team_costs.shape != (len(tracking), 2) or
                not np.isfinite(extra_team_costs).all() or (extra_team_costs < 0).any()):
            raise ValueError("Extra team costs must be finite nonnegative frame-by-two values")
        costs[:, :2] += extra_team_costs
    labels = np.full(len(tracking), "X", dtype="<U1")
    reasons = np.full(len(tracking), "outside_scored_halves", dtype="<U40")
    for _, start, last in blocks:
        stop = last + 1
        states = decode_sequence(costs[start:stop], config["team_switch_cost"], config["dead_switch_cost"])
        labels[start:stop] = LABELS[states]
        reasons[start:stop] = "ball_visible_ambiguous"
    active = features["active"]
    missing = active & ~features["ball_seen"]
    reasons[missing] = "missing_ball_sequence_estimate"
    reasons[active & features["ball_seen"] & ~features["ball_reliable"]] = "ball_jump_sequence_estimate"
    distance = np.fmin(tracking.home_distance_m, tracking.away_distance_m)
    margin = np.abs(tracking.home_distance_m - tracking.away_distance_m)
    control = active & features["control_usable"] & (distance <= config["control_radius_m"])
    control &= margin >= config["control_margin_m"]
    reasons[control] = "ball_control_evidence"
    nearest_label = np.where(tracking.home_distance_m <= tracking.away_distance_m, "H", "A")
    reasons[control & (labels != nearest_label)] = "sequence_overrode_nearest_team"
    if extra_team_costs is not None:
        reasons[active & extra_team_costs.any(axis=1) & (labels != "D")] = "additional_team_context"
    reasons[active & (labels == "D")] = "dead_ball_sequence_state"
    if use_stoppages:
        reasons[active & (labels == "D") & features["slow_gap_stop"]] = "dead_ball_slow_missing_gap"
        reasons[active & (labels == "D") & features["stationary_stop"]] = "dead_ball_stationary_and_slow"
        reasons[active & (labels == "D") & features["outside_stop"]] = "dead_ball_outside_pitch"

    # Do not silently invent a team for an entire half without control or stop evidence.
    for _, start, last in blocks:
        sl = slice(start, last + 1)
        stop_evidence = (features["outside_stop"][sl] | features["slow_gap_stop"][sl] |
                         features["stationary_stop"][sl]) if use_stoppages else np.zeros(last + 1 - start, bool)
        extra_evidence = extra_team_costs is not None and extra_team_costs[sl].any()
        if not control[sl].any() and not stop_evidence.any() and not extra_evidence:
            labels[sl] = "X"
            reasons[sl] = "insufficient_evidence_in_half"
    if abstain:
        gaps = stable_gap_mask(labels, features, blocks, tracking.fps, config["max_abstention_gap_s"])
        labels[gaps] = "X"
        reasons[gaps] = "short_stable_gap_abstention"
    return labels, reasons
