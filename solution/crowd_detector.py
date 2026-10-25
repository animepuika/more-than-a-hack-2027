"""Mixed-crowd stoppage detector: scuffles, arguments and players gathering around an injury.

A general rule, fixed before any testing and never tuned on a specific match moment:
at least CROWD_MIN_PLAYERS players within CROWD_RADIUS_M of one player, at least
CROWD_MIN_PER_TEAM from each team, all slower than CROWD_MAX_SPEED_MPS, lasting at
least CROWD_MIN_S. Such a crowd means play has been stopped, so the frames are dead.
Only player positions are used.
"""
import numpy as np

from .features import runs
from .v7_features import clean_positions
from .v7_labels import period_runs

CROWD_RADIUS_M = 5.0
CROWD_MIN_PLAYERS = 6
CROWD_MIN_PER_TEAM = 2
CROWD_MAX_SPEED_MPS = 1.5
CROWD_MIN_S = 4.0
CHUNK = 5000


def crowd_frames(xy, home, speed):
    """Per frame: is there a slow, mixed crowd around some player?"""
    n = len(xy)
    found = np.zeros(n, bool)
    for start in range(0, n, CHUNK):
        pts, v = xy[start:start+CHUNK], speed[start:start+CHUNK]
        slow = np.isfinite(pts).all(axis=2) & np.isfinite(v) & (v < CROWD_MAX_SPEED_MPS)
        d = np.linalg.norm(pts[:, :, None] - pts[:, None], axis=3)
        near = (d <= CROWD_RADIUS_M) & slow[:, :, None] & slow[:, None, :]
        home_count = (near & home[None, None, :]).sum(axis=2)
        away_count = (near & ~home[None, None, :]).sum(axis=2)
        ok = ((home_count + away_count >= CROWD_MIN_PLAYERS) & (home_count >= CROWD_MIN_PER_TEAM)
              & (away_count >= CROWD_MIN_PER_TEAM) & slow)
        found[start:start+CHUNK] = ok.any(axis=1)
    return found


def crowd_dead_mask(player_xy, home, period, pitch, fps):
    """Dead-ball frames from sustained mixed crowds, per active period."""
    home = np.asarray(home, bool)
    mask = np.zeros(len(period), bool)
    minimum = round(CROWD_MIN_S * fps)
    for a, b in period_runs(period):
        if period[a] <= 0:
            continue
        xy = clean_positions(player_xy[a:b], home, pitch)
        speed = np.full(xy.shape[:2], np.nan, dtype=np.float32)
        if b - a > 2:
            speed[1:-1] = np.linalg.norm(xy[2:] - xy[:-2], axis=2) * fps / 2
        speed[speed > 12] = np.nan
        found = crowd_frames(xy, home, speed)
        for lo, hi in runs(found):
            if hi - lo >= minimum:
                mask[a+lo:a+hi] = True
    return mask
