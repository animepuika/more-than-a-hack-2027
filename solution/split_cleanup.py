"""Anti-split cleanup: fewer false possession changes and dead-ball fragments.

Spell F1 punishes a possession split in two, and our predictions change possession more often than
BAR-ATH's public answers (6.5 vs 4.2 per live minute). Three rules, applied in this order inside
each half; their lengths are picked on practice copies (solution/run_split_tuning.py):
- join dead ball: a live stretch shorter than dead_gap_s between two dead stretches becomes dead;
- drop short dead ball: a dead stretch shorter than short_dead_s between two stretches of the same
  team becomes that team;
- drop short flips: a possession shorter than flip_s between two stretches of the other team, or
  between the other team and a stoppage, goes back to the team that had the ball (the challenge:
  a touch without control is not a change of possession).
"""
import numpy as np

from .run_v16 import controlled_possession
from .v7_labels import period_runs


def _runs(labels):
    edges = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    return [(labels[s], s, e) for s, e in zip(edges[:-1], edges[1:])]


def tidy(labels, period, fps, flip_s=0.0, short_dead_s=0.0, dead_gap_s=0.0):
    out = np.asarray(labels).copy()
    for a, b in period_runs(period):
        if period[a] <= 0:
            continue
        local = out[a:b]
        if dead_gap_s:
            for (prev, _, _), (mid, s, e), (nxt, _, _) in zip(_runs(local), _runs(local)[1:], _runs(local)[2:]):
                if prev == 'D' and nxt == 'D' and mid in 'HA' and (e - s) < dead_gap_s * fps:
                    local[s:e] = 'D'
        if short_dead_s:
            for (prev, _, _), (mid, s, e), (nxt, _, _) in zip(_runs(local), _runs(local)[1:], _runs(local)[2:]):
                if mid == 'D' and prev in 'HA' and prev == nxt and (e - s) < short_dead_s * fps:
                    local[s:e] = prev
        out[a:b] = local
    if flip_s:
        out = controlled_possession(out, period, fps, flip_s)
    return out
