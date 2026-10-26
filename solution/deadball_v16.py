"""V16 dead-ball clues: player formations plus ball clues, built to survive BAR-ATH-like data loss.

Ball clues (NaN wherever the ball is not reliably seen):
- how long the ball has been, and will stay, nearly still (offline);
- distance to restart spots: nearest corner flag, touchline, goal line, six-yard box, centre spot;
- how far outside the pitch it is;
- a sudden kick from standstill (speed jump after a still spell).
Realistic hiding hides the ball in a share of live AND dead frames with BAR-ATH gap lengths,
because on BAR-ATH the ball is often visible during stoppages and often missing in play.
"""
import numpy as np
import pandas as pd

from .ball_masking import hide_live_ball
from .deadball_formation import formation_features
from .v7_labels import period_runs

STILL_MPS = 1.0
AGE_CAP = 500
LIVE_HIDE, DEAD_HIDE = 0.30, 0.35   # declared from BAR-ATH: about 41% of graded frames lack the ball
BALL_COLUMNS = ('ball_still_age', 'ball_still_next', 'ball_speed', 'ball_speed_mean_1s', 'ball_corner_m',
                'ball_touchline_m', 'ball_goalline_m', 'ball_six_yard_m', 'ball_centre_m', 'ball_outside_m',
                'ball_kick_from_still')


def _age(flag, forward=True):
    """Frames since (until) the flag was last (next) set; the period edge counts as set."""
    seq = flag if forward else flag[::-1]
    index = np.arange(len(seq))
    last = np.maximum.accumulate(np.where(seq, index, -1))
    age = np.minimum(index - last, AGE_CAP).astype(np.float32)
    return age if forward else age[::-1]


def ball_clues(ball_xy, period, pitch, fps):
    length, width = pitch
    ball = np.asarray(ball_xy, dtype=float)
    n = len(ball)
    out = {c: np.full(n, np.nan, dtype=np.float32) for c in BALL_COLUMNS}
    for a, b in period_runs(period):
        if period[a] <= 0:
            continue
        local = ball[a:b]
        seen = np.isfinite(local).all(axis=1)
        speed = np.full(b - a, np.nan)
        if b - a > 2:
            speed[1:-1] = np.linalg.norm(local[2:] - local[:-2], axis=1) * fps / 2
        speed[speed > 45] = np.nan
        moving = np.nan_to_num(speed, nan=0) > STILL_MPS
        # Frames since the ball last moved / until it next moves; only meaningful while seen.
        still_age, still_next = _age(moving, True), _age(moving, False)
        x, y = local[:, 0], local[:, 1]
        corner = np.min([np.hypot(x - cx, y - cy) for cx in (0, length) for cy in (0, width)], axis=0)
        six = np.minimum(np.hypot(np.maximum(0, x - 5.5), np.maximum(0, np.abs(y - width/2) - 9.16)),
                         np.hypot(np.maximum(0, length - 5.5 - x), np.maximum(0, np.abs(y - width/2) - 9.16)))
        outside = np.maximum.reduce([-x, x - length, -y, y - width])
        rolled = pd.Series(speed).rolling(round(fps), min_periods=1, center=True).mean().to_numpy()
        before = pd.Series(np.nan_to_num(speed, nan=0)).shift(round(fps)).rolling(round(fps), min_periods=1).max().to_numpy()
        kick = (np.nan_to_num(speed, nan=0) > 8) & (before < STILL_MPS)
        values = {'ball_still_age': still_age, 'ball_still_next': still_next, 'ball_speed': speed,
                  'ball_speed_mean_1s': rolled, 'ball_corner_m': corner, 'ball_touchline_m': np.minimum(y, width - y),
                  'ball_goalline_m': np.minimum(x, length - x), 'ball_six_yard_m': six,
                  'ball_centre_m': np.hypot(x - length/2, y - width/2), 'ball_outside_m': outside,
                  'ball_kick_from_still': kick.astype(float)}
        for name, v in values.items():
            v = np.asarray(v, dtype=np.float32)
            v[~seen] = np.nan
            out[name][a:b] = v
    return pd.DataFrame(out)


def dead_features(match, blank_ball=False, ball=True):
    """Formation table plus ball clues; ball clues are blanked (NaN) when `blank_ball`,
    and left out entirely when not `ball` (all-NaN columns cannot be trained on)."""
    table = formation_features(match.player_xy_m, match.home_slots, match.period, match.pitch, match.fps)
    if not ball:
        return table
    clues = ball_clues(match.ball_xy_m, match.period, match.pitch, match.fps)
    if blank_ball:
        clues[:] = np.nan
    return pd.concat([table, clues], axis=1)


def hide_ball_realistic(ball_xy, labels, period, gap_lengths, seed, live_share=LIVE_HIDE, dead_share=DEAD_HIDE):
    """Hide the ball in `live_share` of live and `dead_share` of dead frames (BAR-ATH gap lengths)."""
    labels = np.asarray(labels)
    ball = hide_live_ball(ball_xy, np.isin(labels, ['H', 'A']), period, gap_lengths, live_share, seed)[0]
    if (labels == 'D').any():
        ball = hide_live_ball(ball, labels == 'D', period, gap_lengths, dead_share, seed + 7919)[0]
    return ball
