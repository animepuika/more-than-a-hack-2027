"""V9 candidate: V7 control memory that stops holding the owner in long ball gaps.

Inside a missing/unreliable ball run of at least ``gap_min_s``, the label comes
from a fallback sequence (V2 motion) instead of the last confirmed owner, and
that team becomes the owner when the ball returns. Short gaps, D resets and
confirmation/backdating behave exactly as in v7_sequence.control_memory.
"""
import numpy as np

from .v7_labels import period_runs
from .v7_sequence import control_evidence

GAP_FALLBACK_S = 2.0  # declared before scoring; >99% of BAR-ATH missing frames are in longer gaps


def long_gap_mask(reliable, period, min_frames):
    reliable, period = np.asarray(reliable, bool), np.asarray(period)
    mask = np.zeros(len(reliable), bool)
    for a, b in period_runs(period):
        if period[a] <= 0:
            continue
        padded = np.r_[False, ~reliable[a:b], False]
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        for lo, hi in zip(edges[::2], edges[1::2]):
            if hi - lo >= min_frames:
                mask[a+lo:a+hi] = True
    return mask


def gap_fallback_memory(features, period, config, fallback, gap_min_s, initial_labels=None,
                        dead=None, fps=25):
    direct = control_evidence(features, config)
    long_gap = long_gap_mask(features['ball_reliable'] == 1, period, max(1, round(gap_min_s*fps)))
    fallback = np.asarray(fallback)
    labels = np.full(len(direct), 'X', dtype='<U1')
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
            if long_gap[t] and fallback[t] in ('H', 'A'):
                # No ball to confirm anything: follow the fallback team and own it afterwards.
                if owner < 0:
                    labels[segment_start:t] = fallback[t]
                owner = 0 if fallback[t] == 'H' else 1
                candidate, count, holes = -1, 0, 0
                labels[t] = fallback[t]
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
    return labels, long_gap
