"""Dead ball after the ball leaves over a touchline or goal line (throw-ins, goal kicks, corners).

Practice data treats the ball unnaturally after it goes out (Metrica's ball vanishes, DFL's freezes),
so only the moment of crossing is used, never what the ball does afterwards. Rules from general
football knowledge, all settings fixed before testing:
- ball out: the ball is seen at least OUT_M beyond a touchline or goal line in EXIT_FRAMES
  consecutive frames;
- waiting taker: while the ball is not seen inside the pitch, a player who was on the pitch in the
  last ON_PITCH_LOOKBACK_S stands (slower than TAKER_STILL_MPS) at least TAKER_OUT_M beyond a
  touchline for TAKER_HOLD_S (substitutes warming up were never on the pitch);
- restart from the edge: the ball is seen within EDGE_TOUCHLINE_M of a touchline (or beyond it), or
  within CORNER_M of a corner flag, and within ENTRY_WITHIN_S it is seen INSIDE_M inside the pitch
  moving faster than KICK_MPS; the time before it was dead for as long as the players were calm
  (1 s mean of their median speed under CALM_MPS, at most MAX_S back), provided that spell holds
  STILL_NEED_S under STILL_TEAM_MPS (players rarely stand that still in live play);
- restart: dead ball lasts until the ball is seen INSIDE_M inside the pitch moving faster than
  KICK_MPS, or the players calm down (1 s mean of their median speed under CALM_MPS) and then speed
  up again (over RESUME_MPS), or MAX_S passes. Stretches shorter than MIN_S are dropped, so a ball
  that comes straight back adds nothing.
"""
import warnings

import numpy as np
import pandas as pd

from .features import runs
from .v7_labels import period_runs

OUT_M, EXIT_FRAMES = 0.5, 2
TAKER_OUT_M, TAKER_STILL_MPS, TAKER_HOLD_S, ON_PITCH_LOOKBACK_S, ON_PITCH_M = 0.3, 1.0, 2.0, 10.0, 2.0
INSIDE_M, KICK_MPS = 1.0, 5.0
CALM_MPS, RESUME_MPS = 1.5, 2.0
EDGE_TOUCHLINE_M, CORNER_M, ENTRY_WITHIN_S = 0.3, 1.5, 1.0
STILL_TEAM_MPS, STILL_NEED_S = 1.0, 2.0
MIN_S, MAX_S = 3.0, 30.0
# Restart timing (V21 values). V21b picks these on practice copies: end when the ball is seen back
# inside at any speed or only when it is kicked/thrown in, the time limit, a cut off the end of every
# stretch, and the calm limit used when an edge restart looks backwards.
TIMING = {'end_on_inside': False, 'max_s': MAX_S, 'end_trim_s': 0.0, 'calm_back_mps': CALM_MPS}


def outside_m(xy, pitch):
    """Distance beyond the nearest line (positive outside the pitch, negative inside)."""
    length, width = pitch
    x, y = xy[..., 0], xy[..., 1]
    return np.maximum.reduce([-x, x - length, -y, y - width])


def _speed(xy, fps):
    """Central-difference speed (m/s) along axis 0; NaN where a neighbour is missing."""
    speed = np.full(xy.shape[:-1], np.nan)
    if len(xy) > 2:
        speed[1:-1] = np.linalg.norm(xy[2:] - xy[:-2], axis=-1) * fps / 2
    return speed


def ball_exits(ball, pitch):
    out = np.nan_to_num(outside_m(ball, pitch), nan=-np.inf) >= OUT_M
    return [a for a, b in runs(out) if b - a >= EXIT_FRAMES]


def taker_starts(players, ball_inside, pitch, fps):
    width = pitch[1]
    y = players[..., 1]
    speed = _speed(players, fps)
    beyond = np.nan_to_num(np.maximum(-y, y - width), nan=-np.inf) >= TAKER_OUT_M
    on_pitch = np.nan_to_num(-outside_m(players, pitch), nan=-np.inf) >= ON_PITCH_M
    recently_on = pd.DataFrame(on_pitch).rolling(round(ON_PITCH_LOOKBACK_S * fps), min_periods=1).max().to_numpy() > 0
    waiting = beyond & (np.nan_to_num(speed, nan=np.inf) < TAKER_STILL_MPS) & recently_on & ~ball_inside[:, None]
    hold = round(TAKER_HOLD_S * fps)
    starts = set()
    for s in range(waiting.shape[1]):
        starts.update(a for a, b in runs(waiting[:, s]) if b - a >= hold)
    return sorted(starts)


def restart_end(start, ball_kicked_in, team_speed, n, fps, max_s=MAX_S):
    stop = min(n, start + round(max_s * fps))
    kicked = np.flatnonzero(ball_kicked_in[start + 1:stop])
    end = start + 1 + kicked[0] if len(kicked) else stop
    calm = np.flatnonzero(np.nan_to_num(team_speed[start:end], nan=np.inf) < CALM_MPS)
    if len(calm):
        resumed = np.flatnonzero(np.nan_to_num(team_speed[start + calm[0]:end]) > RESUME_MPS)
        if len(resumed):
            end = start + calm[0] + resumed[0]
    return end


def edge_restarts(ball, kicked_in, team_speed, pitch, fps, calm_mps=CALM_MPS, max_s=MAX_S):
    """(start, end) of calm spells that end with the ball put into play from a touchline or corner."""
    length, width = pitch
    x, y = ball[:, 0], ball[:, 1]
    with np.errstate(invalid='ignore'):
        touchline = np.minimum(y, width - y) <= EDGE_TOUCHLINE_M
        corner = np.min([np.hypot(x - cx, y - cy) for cx in (0, length) for cy in (0, width)], axis=0) <= CORNER_M
    at_edge = np.isfinite(ball).all(axis=1) & (touchline | corner)
    within = round(ENTRY_WITHIN_S * fps)
    calm = np.nan_to_num(team_speed, nan=np.inf) < calm_mps
    still = np.nan_to_num(team_speed, nan=np.inf) < STILL_TEAM_MPS
    spells = []
    for a, b in runs(at_edge):
        t = b - 1  # last frame at the edge
        if not kicked_in[t + 1:t + 1 + within].any():
            continue
        start = t
        while start > 0 and calm[start - 1] and t - start < max_s * fps:
            start -= 1
        if still[start:t].sum() >= STILL_NEED_S * fps:
            spells.append((start, t))
    return spells


def line_restart_dead(ball_xy, player_xy, period, pitch, fps, triggers=('exit', 'taker', 'edge'), timing=None):
    """Dead-ball mask from line exits, waiting throw-in takers and restarts from the edge."""
    timing = {**TIMING, **(timing or {})}
    trim = round(timing['end_trim_s'] * fps)
    ball = np.asarray(ball_xy, float)
    players = np.asarray(player_xy, float)
    dead = np.zeros(len(ball), bool)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        team = np.nanmedian(_speed(players, fps), axis=1)
    team = pd.Series(team).rolling(round(fps), min_periods=1).mean().to_numpy()
    for a, b in period_runs(period):
        if period[a] <= 0:
            continue
        local = ball[a:b]
        inside = np.nan_to_num(-outside_m(local, pitch), nan=-np.inf) >= INSIDE_M
        kicked_in = inside & (np.nan_to_num(_speed(local, fps)) > KICK_MPS)
        back_in = inside if timing['end_on_inside'] else kicked_in
        starts = set(ball_exits(local, pitch)) if 'exit' in triggers else set()
        if 'taker' in triggers:
            starts |= set(taker_starts(players[a:b], inside, pitch, fps))
        starts = sorted(starts)
        for s in starts:
            e = restart_end(s, back_in, team[a:b], b - a, fps, timing['max_s']) - trim
            if e - s >= MIN_S * fps:
                dead[a + s:a + e] = True
        edges = edge_restarts(local, kicked_in, team[a:b], pitch, fps, timing['calm_back_mps'], timing['max_s'])             if 'edge' in triggers else []
        for s, e in edges:
            e -= trim
            if e - s >= MIN_S * fps:
                dead[a + s:a + e] = True
    return dead
