"""A Spearman-inspired, acceleration-aware player reach prior.

This is deliberately a small adaptation for possession inference, not a full
Potential Pitch Control Field implementation. The ball target must be observed
or safely interpolated; this module does not invent a hidden ball coordinate.
"""

import numpy as np


def time_to_intercept(position, velocity, target, reaction_time_s=0.7,
                      max_acceleration_mps2=7.0, max_speed_mps=5.0):
    """Approximate arrival time using reaction, projected velocity, and accel limit.

    Arrays broadcast over any leading dimensions and end in XY coordinates.
    During reaction time the player continues at the measured velocity. After
    that, acceleration points toward the target until max speed is reached.
    """
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    target = np.asarray(target, dtype=float)
    if position.shape[-1:] != (2,) or velocity.shape[-1:] != (2,) or target.shape[-1:] != (2,):
        raise ValueError("Position, velocity, and target must end in two XY coordinates")
    if reaction_time_s < 0 or max_acceleration_mps2 <= 0 or max_speed_mps <= 0:
        raise ValueError("Reaction time must be nonnegative; acceleration and speed must be positive")

    valid = np.isfinite(position).all(axis=-1) & np.isfinite(velocity).all(axis=-1)
    valid &= np.isfinite(target).all(axis=-1)
    reaction_position = position + velocity * reaction_time_s
    delta = target - reaction_position
    distance = np.linalg.norm(delta, axis=-1)
    direction = np.divide(delta, distance[..., None], out=np.zeros_like(delta),
                          where=distance[..., None] > 1e-9)
    speed_toward = np.sum(velocity * direction, axis=-1)
    speed_toward = np.clip(speed_toward, -max_speed_mps, max_speed_mps)
    accel = max_acceleration_mps2

    time_to_cap = np.maximum((max_speed_mps - speed_toward) / accel, 0.0)
    distance_to_cap = speed_toward * time_to_cap + 0.5 * accel * time_to_cap ** 2
    distance_to_cap = np.maximum(distance_to_cap, 0.0)
    accelerated_distance = np.minimum(distance, distance_to_cap)
    discriminant = np.maximum(speed_toward ** 2 + 2.0 * accel * accelerated_distance, 0.0)
    accelerated_time = (-speed_toward + np.sqrt(discriminant)) / accel
    movement_time = np.where(
        distance <= distance_to_cap,
        accelerated_time,
        time_to_cap + (distance - distance_to_cap) / max_speed_mps,
    )
    result = reaction_time_s + movement_time
    result = np.where(distance <= 1e-9, reaction_time_s, result)
    return np.where(valid, result, np.inf)


def intercept_probability(position, velocity, target, horizon_s=1.0,
                          reaction_time_s=0.7, max_acceleration_mps2=7.0,
                          max_speed_mps=5.0, uncertainty_s=0.45):
    """Logistic-CDF chance a player can reach a target by a short time horizon."""
    if horizon_s < 0 or uncertainty_s <= 0:
        raise ValueError("Horizon must be nonnegative and uncertainty must be positive")
    tau = time_to_intercept(position, velocity, target, reaction_time_s,
                            max_acceleration_mps2, max_speed_mps)
    z = (np.pi / (np.sqrt(3.0) * uncertainty_s)) * (horizon_s - tau)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def team_reach_probabilities(xy, velocity, targets, home_slots, config):
    """Independent-arrival approximation to each team's chance to reach a target.

    Returns a frame-by-two probability array ordered Home, Away and a boolean
    mask for frames with a finite target plus at least one tracked player per side.
    The team probabilities are normalized against each other for use as a weak
    possession cost, not presented as calibrated match probabilities.
    """
    xy = np.asarray(xy, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    targets = np.asarray(targets, dtype=float)
    home_slots = np.asarray(home_slots, dtype=bool)
    if xy.ndim != 3 or xy.shape[-1] != 2 or velocity.shape != xy.shape:
        raise ValueError("Player positions and velocities must have shape frames x slots x 2")
    if targets.shape != (xy.shape[0], 2) or home_slots.shape != (xy.shape[1],):
        raise ValueError("Targets or team mapping do not match tracking dimensions")

    n = xy.shape[0]
    reach = np.zeros((n, 2), dtype=float)
    supported = np.isfinite(targets).all(axis=1)
    for side, slots in enumerate((home_slots, ~home_slots)):
        if not slots.any():
            supported[:] = False
            continue
        positions = xy[:, slots, :]
        speeds = velocity[:, slots, :].copy()
        speeds[~np.isfinite(speeds)] = 0.0
        valid = np.isfinite(positions).all(axis=2)
        player_target = targets[:, None, :]
        probability = intercept_probability(
            positions, speeds, player_target,
            horizon_s=config["control_horizon_s"],
            reaction_time_s=config["reaction_time_s"],
            max_acceleration_mps2=config["max_acceleration_mps2"],
            max_speed_mps=config["max_speed_mps"],
            uncertainty_s=config["arrival_uncertainty_s"],
        )
        probability = np.where(valid, probability, 0.0)
        reach[:, side] = 1.0 - np.prod(1.0 - probability, axis=1)
        supported &= valid.any(axis=1)

    total = reach.sum(axis=1)
    supported &= total > 1e-12
    normalized = np.divide(reach, total[:, None], out=np.full_like(reach, 0.5),
                           where=total[:, None] > 1e-12)
    normalized[~supported] = 0.5
    return normalized, supported


def probability_costs(probabilities, eligible, weight=0.35, maximum_cost=2.0):
    """Convert weak Home/Away reach probabilities into relative nonnegative costs."""
    probabilities = np.asarray(probabilities, dtype=float)
    eligible = np.asarray(eligible, dtype=bool)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2 or eligible.shape != (len(probabilities),):
        raise ValueError("Expected frame-by-two probabilities and one eligibility flag per frame")
    if weight < 0 or maximum_cost < 0:
        raise ValueError("Cost weight and maximum cost must be nonnegative")
    safe = np.clip(probabilities, 1e-6, 1.0)
    cost = -np.log(safe)
    cost -= cost.min(axis=1, keepdims=True)
    cost = np.clip(cost, 0.0, maximum_cost) * weight
    cost[~eligible] = 0.0
    return cost
