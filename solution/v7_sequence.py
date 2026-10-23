"""Controlled-transfer memory and joint live/dead/team sequence evidence.

The model can confirm with future observations, then backdate to the start of
supported control. It never reads answer labels. Scores are uncalibrated.
"""
import numpy as np
from scipy.special import expit

from .v7_labels import period_runs
from .predict import decode_sequence


def control_evidence(features, config):
    h, a = features['home_nearest_ball_m'], features['away_nearest_ball_m']
    usable = (np.isfinite(h) & np.isfinite(a) & (features['ball_reliable'] == 1) &
              (features['home_players'] >= config['minimum_team_tracks']) &
              (features['away_players'] >= config['minimum_team_tracks']) &
              (np.fmin(h, a) <= config['control_radius_m']) &
              (np.abs(h-a) >= config['control_margin_m']))
    owner = np.where(h <= a, 0, 1).astype(np.int8)
    relative = np.where(owner == 0, features['home_ball_relative_speed_mps'],
                        features['away_ball_relative_speed_mps'])
    # Fast ball passage is weaker evidence than sustained close movement.
    usable &= ~np.isfinite(relative) | (relative <= 12.0)
    owner[~usable] = -1
    return owner


def control_memory(features, period, config, initial_labels=None, dead=None, fps=25):
    direct = control_evidence(features, config)
    labels = np.full(len(direct), 'X', dtype='<U1')
    supported = np.zeros(len(direct), bool)
    needed = max(1, round(config['transfer_confirmation_s']*fps))
    hole_limit = round(config['transfer_maximum_hole_s']*fps)
    dead = np.zeros(len(direct), bool) if dead is None else np.asarray(dead, bool)
    for a, b in period_runs(period):
        if period[a] <= 0:
            continue
        owner, candidate, start, count, holes = -1, -1, a, 0, 0
        segment_start = a
        for t in range(a, b):
            if dead[t]:
                labels[t] = 'D'
                owner, candidate, count, holes, segment_start = -1, -1, 0, 0, t+1
                continue
            obs = int(direct[t])
            if obs == owner and owner >= 0:
                candidate, count, holes = -1, 0, 0
            elif obs >= 0:
                if candidate != obs:
                    candidate, start, count = obs, t, 0
                count += 1; holes = 0
                if count >= needed:
                    old = owner
                    owner = candidate
                    labels[start:t+1] = 'H' if owner == 0 else 'A'
                    supported[start:t+1] = True
                    # Leading uncertainty uses the first confirmed control in this live episode.
                    if old < 0:
                        labels[segment_start:start] = 'H' if owner == 0 else 'A'
                    candidate, count = -1, 0
            else:
                holes += 1
                if holes > hole_limit:
                    candidate, count = -1, 0
            if owner >= 0:
                labels[t] = 'H' if owner == 0 else 'A'
            elif initial_labels is not None and initial_labels[t] in ('H', 'A'):
                labels[t] = initial_labels[t]
        # If an episode never acquires control, leave its fallback/X explicit.
    return labels, supported, direct


def joint_decode(features, dead_score, period, config, context_logit=None, restart_tags=None):
    """Factorized H/A-given-live and D evidence, decoded together per half.

    Unlike a D overlay, restart ownership is initialized again after supported
    stoppages. The H/A conditional distribution always sums to one.
    """
    h, a = features['home_nearest_ball_m'], features['away_nearest_ball_m']
    valid = np.isfinite(h) & np.isfinite(a) & (features['ball_reliable'] == 1)
    logit = np.where(valid, np.clip((a-h)/1.5, -4, 4), 0.)
    if context_logit is not None:
        weak = control_evidence(features, config) < 0
        logit[weak] += config['joint_context_weight']*np.asarray(context_logit)[weak]
    p_home = expit(logit)
    d = np.clip(np.nan_to_num(dead_score, nan=0.2), 0.01, 0.99)
    # Tags can only support a stop when motion also agrees; they never label an interval alone.
    if restart_tags is not None:
        quiet = features['speed_median_mps_mean_2s'] < 1.5
        low_ball_motion = (features['ball_span_2s_m'] < 1.5) | (features['ball_reliable'] == 0)
        support = restart_tags & quiet & low_ball_motion
        d[support] = np.maximum(d[support], 0.65)
    costs = -np.log(np.column_stack([(1-d)*p_home, (1-d)*(1-p_home), d]))
    labels = np.full(len(d), 'X', dtype='<U1')
    for first, stop in period_runs(period):
        if period[first] <= 0:
            continue
        states = decode_sequence(costs[first:stop], config['joint_team_switch_cost'],
                                 config['joint_dead_switch_cost'])
        labels[first:stop] = np.array(['H', 'A', 'D'])[states]
    # Confirm transfers inside the jointly inferred live episodes, resetting at D.
    memory, _, _ = control_memory(features, period, config, initial_labels=labels, dead=labels == 'D')
    return memory, d
