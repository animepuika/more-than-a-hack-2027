"""Stress test: hide the ball during live play with BAR-ATH-like gap lengths.

Gap lengths come from BAR-ATH tracking alone (ball visibility in the graded
halves, no answers). Gaps start on randomly chosen live frames of an external
match, may run into following dead ball, and never cross a period boundary.
Labels choose where play is live; methods only ever see the masked ball.
"""
import numpy as np
import pandas as pd

from .v7_labels import period_runs


def gap_lengths(seen, period):
    """Missing-ball run lengths in frames, per active period."""
    seen, period = np.asarray(seen, bool), np.asarray(period)
    lengths = []
    for a, b in period_runs(period):
        if period[a] <= 0:
            continue
        padded = np.r_[False, ~seen[a:b], False]
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        lengths.extend(edges[1::2] - edges[::2])
    return np.asarray(lengths, dtype=int)


def bar_ath_gap_lengths(features_csv):
    """Read BAR-ATH ball visibility from a saved V7 feature diagnostic."""
    table = pd.read_csv(features_csv, usecols=['period', 'ball_seen'])
    return gap_lengths(table['ball_seen'].to_numpy() == 1, table['period'].to_numpy())


def hide_live_ball(ball_xy, live, period, lengths, target, seed):
    """Return (masked ball, hidden mask, gaps added) with at least `target` of live frames hidden."""
    ball = np.asarray(ball_xy, dtype=float)
    live, period = np.asarray(live, bool), np.asarray(period)
    if not 0 <= target < 1 or not len(lengths) or (target > 0 and not live.any()):
        raise ValueError('Need a target in [0, 1), gap lengths, and live frames')
    hidden = ~np.isfinite(ball).all(axis=1)
    period_end = np.zeros(len(period), dtype=int)
    for a, b in period_runs(period):
        period_end[a:b] = b
    rng = np.random.RandomState(seed)
    candidates = np.flatnonzero(live)
    needed = int(np.ceil(target * live.sum()))
    count, added = int((hidden & live).sum()), 0
    while count < needed:
        start = int(rng.choice(candidates))
        stop = min(start + int(rng.choice(lengths)), period_end[start])
        new = ~hidden[start:stop]
        count += int((new & live[start:stop]).sum())
        hidden[start:stop] = True
        added += 1
    masked = ball.copy()
    masked[hidden] = np.nan
    return masked, hidden, added
