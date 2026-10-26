"""Experiment 3: fill the gated dead-ball blind spots from the evidence on both sides.

Where too few players are tracked, the coverage gate switches the formation model off. A blind
spot is a run of such frames inside a period. Fixed rule, declared before testing:
  dead on both sides                        -> dead throughout
  live on both sides                        -> unchanged
  mixed (one side dead, the other live)     -> dead only if, within BALL_LOOKBACK_S before the blind
                                               spot, the ball was last seen at or beyond a line
                                               (within OUT_MARGIN_M of it); otherwise unchanged
A blind spot touching a period edge uses the one side it has.
"""
import numpy as np

from .features import runs
from .v7_labels import period_runs

BALL_LOOKBACK_S = 2.0
OUT_MARGIN_M = 1.0


def ball_went_out(ball_xy, start, pitch, fps):
    """Was the last seen ball position within BALL_LOOKBACK_S before `start` at or beyond a line?"""
    length, width = pitch
    window = ball_xy[max(0, start - round(BALL_LOOKBACK_S * fps)):start]
    seen = np.flatnonzero(np.isfinite(window).all(axis=1))
    if not len(seen):
        return False
    x, y = window[seen[-1]]
    edge = min(x, length - x, y, width - y)
    return edge <= OUT_MARGIN_M


def bridge_blind_spots(dead, covered, period, ball_xy, pitch, fps=25):
    """Return a copy of `dead` with uncovered stretches filled by the rule above."""
    out = np.asarray(dead, bool).copy()
    covered = np.asarray(covered, bool)
    for a, b in period_runs(period):
        if period[a] <= 0:
            continue
        for lo, hi in runs(~covered[a:b]):
            lo, hi = a + lo, a + hi
            before = out[lo - 1] if lo > a else None
            after = out[hi] if hi < b else None
            sides = [s for s in (before, after) if s is not None]
            if not sides:
                continue
            if all(sides):
                out[lo:hi] = True
            elif any(sides) and ball_went_out(ball_xy, lo, pitch, fps):
                out[lo:hi] = True
    return out
