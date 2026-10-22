"""Minimum-duration sequence decoding with partial segments allowed at edges."""

import numpy as np


def decode_minimum_duration(costs, team_switch_cost, dead_switch_cost, minimum_frames):
    """Minimum-cost H/A/D path; interior segments must satisfy their duration.

    Each state expands into age states. Only mature states can switch. The
    first and last segments may be truncated by the half boundary.
    """
    costs = np.asarray(costs, dtype=float)
    if costs.ndim != 2 or costs.shape[1] != 3 or not np.isfinite(costs).all():
        raise ValueError("Expected finite frame-by-three costs")
    if team_switch_cost < 0 or dead_switch_cost < 0:
        raise ValueError("Switch costs cannot be negative")
    durations = np.asarray(minimum_frames)
    if durations.shape != (3,) or not np.isfinite(durations).all() or (durations < 1).any() or not np.equal(durations, durations.astype(int)).all():
        raise ValueError("Three positive integer minimum durations are required")
    durations = durations.astype(int)
    if not len(costs):
        return np.empty(0, dtype=np.int8)
    starts = np.r_[0, np.cumsum(durations)[:-1]]
    mature = starts + durations - 1
    owners = np.repeat(np.arange(3), durations)
    count = len(owners)
    # Each state has sparse predecessors: age advancement or a mature state switch.
    predecessors = np.full((count, 4), -1, dtype=int)
    penalties = np.full((count, 4), np.inf)
    transition = np.array([[0, team_switch_cost, dead_switch_cost],
                           [team_switch_cost, 0, dead_switch_cost],
                           [dead_switch_cost, dead_switch_cost, 0]], dtype=float)
    for side in range(3):
        for age in range(durations[side]):
            state = starts[side] + age
            incoming = []
            if age > 0:
                incoming.append((state - 1, 0.0))
            if age == durations[side] - 1:
                incoming.append((state, 0.0))
            if age == 0:
                incoming += [(int(mature[other]), transition[other, side]) for other in range(3) if other != side]
            for j, (parent, penalty) in enumerate(incoming):
                predecessors[state, j], penalties[state, j] = parent, penalty
    safe_parents = np.maximum(predecessors, 0)
    back = np.zeros((len(costs), count), dtype=np.int16)
    value = np.full(count, np.inf)
    value[mature] = costs[0]
    columns = np.arange(count)
    for frame in range(1, len(costs)):
        options = value[safe_parents] + penalties
        chosen = np.argmin(options, axis=1)
        back[frame] = predecessors[columns, chosen]
        value = options[columns, chosen] + costs[frame, owners]
        value -= value.min()
    state = int(np.argmin(value))
    result = np.empty(len(costs), dtype=np.int8)
    result[-1] = owners[state]
    for frame in range(len(costs) - 1, 0, -1):
        state = int(back[frame, state])
        result[frame - 1] = owners[state]
    return result
