"""High-ball detector: estimates whether the ball is in the air from x/y tracking only.

BAR-ATH has no ball height. SkillCorner's open data has one, so a model is trained there to
recognise "ball above AERIAL_HEIGHT_M" from clues that exist on BAR-ATH: how steady the ball's
speed is, how straight its path is, how far it travels, how close the nearest players are, and
tracking gaps around it. Its probability can then feed the possession model as one more input.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .v7_labels import period_runs

AERIAL_HEIGHT_M = 2.0
MODEL_SETTINGS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                      l2_regularization=1.0, random_state=2027)
WINDOWS = (5, 12, 25)   # frames: 0.2, 0.5 and 1 s at 25 fps


def aerial_features(ball_xy, player_xy, period, fps):
    ball = np.asarray(ball_xy, dtype=float)
    n = len(ball)
    columns = {}
    for a, b in period_runs(period):
        if period[a] <= 0:
            continue
        local = pd.DataFrame(ball[a:b], columns=['x', 'y'])
        seen = local.notna().all(axis=1).to_numpy()
        step = np.full(b - a, np.nan)
        if b - a > 2:
            step[1:-1] = np.linalg.norm(ball[a+2:b] - ball[a:b-2], axis=1) * fps / 2
        speed = pd.Series(step)
        frame = {'speed': step, 'accel': speed.diff().to_numpy() * fps}
        players = player_xy[a:b]
        d = np.linalg.norm(players - ball[a:b, None], axis=2)
        with np.errstate(all='ignore'):
            nearest = np.nanmin(np.where(np.isfinite(d), d, np.inf), axis=1)
        nearest[~np.isfinite(nearest)] = np.nan
        frame['nearest_player_m'] = nearest
        frame['players_within_3m'] = (d < 3).sum(axis=1).astype(float)
        for w in WINDOWS:
            roll = speed.rolling(2 * w + 1, center=True, min_periods=max(3, w))
            frame[f'speed_mean_{w}'] = roll.mean().to_numpy()
            frame[f'speed_std_{w}'] = roll.std().to_numpy()
            ahead, behind = local.shift(-w), local.shift(w)
            chord = np.hypot(ahead.x - behind.x, ahead.y - behind.y).to_numpy()
            path = pd.Series(np.r_[0, np.hypot(np.diff(local.x), np.diff(local.y))]).rolling(2 * w, center=True, min_periods=w).sum().to_numpy()
            frame[f'travel_{w}'] = chord
            frame[f'straightness_{w}'] = np.divide(chord, path, out=np.full(b - a, np.nan), where=path > 0.2)
            frame[f'nearest_min_{w}'] = pd.Series(nearest).rolling(2 * w + 1, center=True, min_periods=1).min().to_numpy()
        frame['missing_before'] = pd.Series(~seen).rolling(25, min_periods=1).sum().to_numpy()
        frame['missing_after'] = pd.Series(~seen[::-1]).rolling(25, min_periods=1).sum().to_numpy()[::-1]
        for name, values in frame.items():
            columns.setdefault(name, np.full(n, np.nan, dtype=np.float32))[a:b] = np.asarray(values, dtype=np.float32)
    return pd.DataFrame(columns)


def training_rows(table, ball_z, stride=5):
    """Frames with a detected ball height: y = 1 above AERIAL_HEIGHT_M."""
    rows = np.flatnonzero(np.isfinite(ball_z))[::stride]
    return table.iloc[rows], (ball_z[rows] > AERIAL_HEIGHT_M).astype(int)


def fit(parts):
    return HistGradientBoostingClassifier(**MODEL_SETTINGS).fit(
        pd.concat([x for x, _ in parts], ignore_index=True), np.concatenate([y for _, y in parts]))


def aerial_probability(model, table):
    return model.predict_proba(table)[:, 1]
