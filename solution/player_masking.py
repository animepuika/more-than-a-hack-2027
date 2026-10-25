"""Stress test: remove players from external matches the way BAR-ATH's tracking loses them.

BAR-ATH's per-frame count of missing players (22 minus tracked, from tracking only, no
answers) is replayed onto an external match from a random starting point, so both the
share of frames and the duration of drop-outs follow the real match. Which players vanish
is random (a stated simplification: real drop-outs may favour players far from the action).
A vanished player stays hidden until the missing count falls again.
"""
import numpy as np
import pandas as pd

from .v7_labels import period_runs

FULL_TEAM = 22


def bar_ath_missing_counts(features_csv):
    """Missing players per graded BAR-ATH frame, from a saved V7 feature diagnostic."""
    table = pd.read_csv(features_csv, usecols=['period', 'players_observed'])
    observed = table['players_observed'].to_numpy()[table['period'].to_numpy() > 0]
    return np.clip(FULL_TEAM - observed, 0, FULL_TEAM).astype(int)


def drop_players(player_xy, period, missing, seed):
    """Return a copy of player_xy with players hidden to follow the `missing` count sequence."""
    xy = np.asarray(player_xy, dtype=np.float32).copy()
    missing = np.asarray(missing, int)
    rng = np.random.RandomState(seed)
    position = rng.randint(len(missing))
    valid = np.isfinite(xy).all(axis=2)
    for a, b in period_runs(period):
        if period[a] <= 0:
            continue
        dropped = []
        for t in range(a, b):
            target = missing[position % len(missing)]
            position += 1
            visible = np.flatnonzero(valid[t])
            dropped = [s for s in dropped if valid[t, s]]
            while len(dropped) > target:
                dropped.pop(rng.randint(len(dropped)))
            candidates = np.setdiff1d(visible, dropped)
            extra = min(target - len(dropped), len(candidates))
            if extra > 0:
                dropped.extend(rng.choice(candidates, extra, replace=False).tolist())
            if dropped:
                xy[t, dropped] = np.nan
    return xy
