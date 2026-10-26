"""Dead ball from the broadcast camera leaving the wide pitch view (V20).

TV directors cut to close-ups and replays when play stops. A tactical frame is a cutaway when its
broadcast frame (constant offset per half, measured from player positions) has no valid
homography. Replays overrun the restart and cuts can start before the stoppage, so each cutaway
loses its first TRIM_START_S and last TRIM_END_S; what remains must last MIN_S. The main tracking
keeps running during cutaways, so it decides the restart: the dead ball ends at the first frame
where the players' median speed, averaged over RESTART_WINDOW_S, exceeds RESTART_MPS. All settings
were fixed before V20 was run.
"""
import numpy as np
import pandas as pd

from .features import runs

OFFSETS = {1: 4014, 2: 26920}
TRIM_START_S, TRIM_END_S, MIN_S = 3.0, 6.0, 2.0
RESTART_MPS, RESTART_WINDOW_S = 2.0, 1.0


def cutaway_mask(homography_csv, period, active, offsets=OFFSETS):
    """Tactical frames whose broadcast frame exists and shows no valid pitch view; also returns
    the frames that have a broadcast frame at all."""
    h = pd.read_csv(homography_csv, usecols=['frame', 'valid'])
    valid = np.zeros(int(h.frame.max()) + 1, bool)
    valid[h.frame.to_numpy()] = h.valid.to_numpy() == 1
    broadcast = np.arange(len(period)) - np.where(period == 2, offsets[2], offsets[1])
    seen = np.asarray(active, bool) & (broadcast >= 0) & (broadcast < len(valid))
    return seen & ~valid[np.clip(broadcast, 0, len(valid) - 1)], seen


def trim(mask, fps, start_s=TRIM_START_S, end_s=TRIM_END_S, min_s=MIN_S):
    out = np.zeros(len(mask), bool)
    for a, b in runs(mask):
        a2, b2 = a + int(round(start_s * fps)), b - int(round(end_s * fps))
        if b2 - a2 >= min_s * fps:
            out[a2:b2] = True
    return out


def stop_at_restart(mask, player_speed, fps, mps=RESTART_MPS, window_s=RESTART_WINDOW_S):
    """Cut each stretch at the first frame where the players' rolling mean speed exceeds mps."""
    speed = pd.Series(player_speed).rolling(int(window_s * fps), min_periods=1).mean().to_numpy()
    out = mask.copy()
    for a, b in runs(mask):
        fast = np.flatnonzero(np.nan_to_num(speed[a:b]) > mps)
        if len(fast):
            out[a + fast[0]:b] = False
    return out


def cutaway_dead(cut, player_speed, fps):
    return stop_at_restart(trim(cut, fps), player_speed, fps)
