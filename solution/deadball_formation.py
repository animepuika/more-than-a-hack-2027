"""V11 candidate: a ball-free dead-ball model from restart formations and stopped play.

Only player positions are used, so the model works while the ball is invisible
and cannot learn Metrica's "no ball means dead" shortcut. Each restart is
described by the shape it forces on the players:
- kick-off: both teams fully in their own half, players around the centre spot;
- corner: a player at a corner flag and a crowded penalty area;
- goal kick: the attacking side out of the penalty area while defenders stay in;
- throw-in: a player on or beyond a touchline;
- free kick: a still cluster of teammates (the wall);
- any stoppage: players slow down, walk and stand still.
Scores become D frames with the V7 hysteresis settings (config_v7.json).
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .v7_features import clean_positions
from .v7_labels import period_runs

MODEL_SETTINGS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                      l2_regularization=1.0, random_state=2027)
TRAIN_STRIDE = 5
WINDOWS_S = (1.0, 3.0, 6.0)
BOX_DEPTH_M, BOX_HALF_WIDTH_M = 16.5, 20.16
WALL_RADIUS_M, STILL_MPS = 2.5, 1.0


def _formation(xy, home, speed, pitch):
    length, width = pitch
    n = len(xy)
    valid = np.isfinite(xy).all(axis=2)
    x, y = xy[:, :, 0], xy[:, :, 1]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        gap = np.nanmean(x[:, home]) - np.nanmean(x[:, ~home])
    home_left = gap < 0  # each side defends its own half on average
    out = {'players': valid.sum(axis=1).astype(float)}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        out['speed_median'] = np.nanmedian(speed, axis=1)
        out['speed_q90'] = np.nanpercentile(speed, 90, axis=1)
        moving = np.isfinite(speed)
        count = np.maximum(moving.sum(axis=1), 1)
        out['share_still'] = ((speed < 0.5) & moving).sum(axis=1) / count
        out['share_walking'] = ((speed < 1.4) & moving).sum(axis=1) / count
        # Kick-off: everyone in their own half; players around the centre spot.
        own = []
        for side, defends_left in ((home, home_left), (~home, not home_left)):
            in_own = (x[:, side] <= length/2 + 0.5) if defends_left else (x[:, side] >= length/2 - 0.5)
            own.append((in_own & valid[:, side]).sum(axis=1) / np.maximum(valid[:, side].sum(axis=1), 1))
        out['own_half_share_min'] = np.minimum(*own)
        centre = np.hypot(x - length/2, y - width/2)
        out['near_centre_spot'] = (centre < 9.15).sum(axis=1).astype(float)
        # Corner: someone at a flag and a crowded box.
        corners = [(0, 0), (0, width), (length, 0), (length, width)]
        flag = np.min([np.hypot(x - cx, y - cy) for cx, cy in corners], axis=0)
        out['corner_flag_min_m'] = np.nanmin(np.where(valid, flag, np.inf), axis=1)
        in_box_y = np.abs(y - width/2) <= BOX_HALF_WIDTH_M
        ends = []
        for left_end in (True, False):
            in_box = in_box_y & ((x <= BOX_DEPTH_M) if left_end else (x >= length - BOX_DEPTH_M)) & valid
            defending = home if left_end == home_left else ~home
            ends.append(((in_box & defending).sum(axis=1), (in_box & ~defending).sum(axis=1)))
        left_busier = ends[0][0] + ends[0][1] >= ends[1][0] + ends[1][1]
        busy = [np.where(left_busier, ends[0][k], ends[1][k]) for k in (0, 1)]
        quiet = [np.where(left_busier, ends[1][k], ends[0][k]) for k in (0, 1)]
        out['busy_box_defenders'], out['busy_box_attackers'] = busy
        out['quiet_box_defenders'], out['quiet_box_attackers'] = quiet
        # Throw-in and out of play: someone on or beyond a line.
        touch = np.minimum(y, width - y)
        out['touchline_min_m'] = np.nanmin(np.where(valid, touch, np.inf), axis=1)
        outside = (x < -0.5) | (x > length + 0.5) | (y < -0.5) | (y > width + 0.5)
        out['players_outside'] = (outside & valid).sum(axis=1).astype(float)
        # Free-kick wall: the largest still cluster of teammates.
        wall = np.zeros(n)
        still = (speed < STILL_MPS) & valid
        for side in (home, ~home):
            pts, s = xy[:, side], still[:, side]
            d = np.linalg.norm(pts[:, :, None] - pts[:, None], axis=3)
            close = (d < WALL_RADIUS_M) & s[:, :, None] & s[:, None, :]
            wall = np.maximum(wall, close.sum(axis=2).max(axis=1))
        out['still_cluster_max'] = wall
        # Overall shape: a stoppage often scatters or compresses everyone.
        out['x_spread'] = np.nanpercentile(x, 90, axis=1) - np.nanpercentile(x, 10, axis=1)
        out['y_spread'] = np.nanpercentile(y, 90, axis=1) - np.nanpercentile(y, 10, axis=1)
    for name, values in out.items():
        out[name] = np.where(np.isfinite(values), values, np.nan).astype(np.float32)
    return out


def formation_features(player_xy, home, period, pitch, fps):
    home = np.asarray(home, bool)
    n = len(player_xy)
    columns = {}
    for a, b in period_runs(period):
        if period[a] <= 0:
            continue
        xy = clean_positions(player_xy[a:b], home, pitch)
        speed = np.full(xy.shape[:2], np.nan, dtype=np.float32)
        if b - a > 2:
            speed[1:-1] = np.linalg.norm(xy[2:] - xy[:-2], axis=2) * fps / 2
        speed[speed > 12] = np.nan
        frame = pd.DataFrame(_formation(xy, home, speed, pitch))
        extra = {}
        for seconds in WINDOWS_S:
            window = max(1, round(seconds * fps))
            rolled = frame.rolling(window, min_periods=max(1, window // 2), center=True)
            means = rolled.mean()
            extra.update({f'{c}_mean_{seconds:g}s': means[c].to_numpy() for c in frame})
        lag = round(2 * fps)
        for c in ['speed_median', 'share_still', 'share_walking']:
            values = frame[c].to_numpy()
            extra[c+'_change_next'] = np.r_[values[lag:], np.full(lag, np.nan)] - values
            extra[c+'_change_prev'] = values - np.r_[np.full(lag, np.nan), values[:-lag]]
        frame = pd.concat([frame, pd.DataFrame(extra)], axis=1)
        for c in frame:
            if c not in columns:
                columns[c] = np.full(n, np.nan, dtype=np.float32)
            columns[c][a:b] = frame[c].to_numpy(dtype=np.float32)
    return pd.DataFrame(columns)


def fit(tables, labels, stride=TRAIN_STRIDE):
    """Dead (1) vs live (0) on labelled frames; X frames are skipped."""
    xs, ys = [], []
    for table, label in zip(tables, labels):
        rows = np.flatnonzero(np.isin(label, ['H', 'A', 'D']))[::stride]
        xs.append(table.iloc[rows]); ys.append((np.asarray(label)[rows] == 'D').astype(int))
    return HistGradientBoostingClassifier(**MODEL_SETTINGS).fit(pd.concat(xs, ignore_index=True), np.concatenate(ys))


def dead_probability(model, table):
    return model.predict_proba(table)[:, 1]


def calibrated_dead_mask(p_dead, period, config, live_reference, max_false_rate, fps=25, gap=0.3):
    """Raise the start threshold until at most `max_false_rate` of certainly-live frames are D.

    `live_reference` marks frames where a player clearly controls a moving ball; no labels
    are used. The continue threshold stays `gap` below the start threshold, as in V7.
    Returns (mask, start_threshold).
    """
    from .v7_model import dead_mask
    period = np.asarray(period)
    live_reference = np.asarray(live_reference, bool)

    def mask(threshold):
        settings = {**config, 'dead_start_score': threshold, 'dead_continue_score': max(0.05, threshold - gap)}
        return dead_mask(p_dead, period, settings, fps) & (period > 0)

    def rate(m):
        return float(m[live_reference].mean()) if live_reference.any() else 0.0

    low, high = 0.5, 0.995
    if rate(mask(low)) <= max_false_rate:
        return mask(low), low
    for _ in range(10):  # false alarms fall as the threshold rises
        middle = (low + high) / 2
        if rate(mask(middle)) <= max_false_rate:
            high = middle
        else:
            low = middle
    return mask(high), high
