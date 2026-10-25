"""V12 candidate: a trained possession model that can follow quick turnovers.

Per frame it estimates P(home has the ball) from touch clues (which team is
close to the ball now, a moment before and after, who touched it last and who
touches it next, sudden changes of ball direction) plus V10's ball-free team
shape. A two-state decode then turns those probabilities into possession spells,
with D taken from an existing dead mask. The switch cost decides how readily
short possessions appear; it is chosen on the training matches only.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .predict import decode_sequence
from .v7_labels import period_runs

MODEL_SETTINGS = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                      l2_regularization=1.0, random_state=2027)
TRAIN_STRIDE = 5
SHIFTS = (-25, -10, 10, 25)
CLOSE_M = 1.0
AGE_CAP = 250
SWITCH_GRID_S = (0.25, 0.5, 1.0, 2.0, 4.0)


def _shift(values, lag, a, b):
    out = np.full(b - a, np.nan, dtype=np.float32)
    local = values[a:b]
    if lag > 0 and lag < len(local):
        out[:-lag] = local[lag:]
    elif lag < 0 and -lag < len(local):
        out[-lag:] = local[:lag]
    return out


def _age(close, forward):
    """Frames since (or until) the last (next) close frame, capped; NaN if none in the period."""
    seq = close if forward else close[::-1]
    index = np.arange(len(seq))
    last = np.maximum.accumulate(np.where(seq, index, -1))
    age = np.where(last >= 0, np.minimum(index - last, AGE_CAP), np.nan).astype(np.float32)
    return age if forward else age[::-1]


def touch_features(features, ball_xy, period, ballfree):
    """Return (us=home, us=away) tables; `features` come from v7_features.build_features."""
    n = len(period)
    tables = [{}, {}]
    shared = ['ball_speed_mps', 'ball_reliable', 'ball_gap_age_s', 'ball_gap_remaining_s', 'ball_span_0.5s_m',
              'ball_span_2s_m', 'ball_speed_mps_mean_0.5s', 'nearest_ball_m']
    paired = ['nearest_ball_m', 'ball_relative_speed_mps', 'near_last_ball']
    ball = np.asarray(ball_xy, dtype=float)
    turn = np.full(n, np.nan, dtype=np.float32)
    columns = [{}, {}]
    for a, b in period_runs(period):
        if period[a] <= 0:
            continue
        local = ball[a:b]
        if b - a > 6:
            v_in, v_out = local[3:-3] - local[:-6], local[6:] - local[3:-3]
            norm = np.linalg.norm(v_in, axis=1) * np.linalg.norm(v_out, axis=1)
            cos = np.sum(v_in * v_out, axis=1) / np.where(norm > 1e-6, norm, np.nan)
            turn[a+3:b-3] = np.arccos(np.clip(cos, -1, 1))
        for us, (mine, theirs) in enumerate((('home', 'away'), ('away', 'home'))):
            col = columns[us]
            for name in paired:
                for role, side in (('us', mine), ('them', theirs)):
                    col.setdefault(f'{role}_{name}', np.full(n, np.nan, np.float32))[a:b] = features[f'{side}_{name}'][a:b]
            for role, side in (('us', mine), ('them', theirs)):
                near = features[f'{side}_nearest_ball_m']
                for lag in SHIFTS:
                    col.setdefault(f'{role}_near_shift_{lag}', np.full(n, np.nan, np.float32))[a:b] = _shift(near, lag, a, b)
                close = np.nan_to_num(near[a:b], nan=np.inf) <= CLOSE_M
                col.setdefault(f'{role}_close_age', np.full(n, np.nan, np.float32))[a:b] = _age(close, True)
                col.setdefault(f'{role}_close_next', np.full(n, np.nan, np.float32))[a:b] = _age(close, False)
    for us in (0, 1):
        frame = {**{k: np.asarray(features[k], np.float32) for k in shared}, 'ball_turn_rad': turn, **columns[us]}
        tables[us] = pd.concat([pd.DataFrame(frame), ballfree[us].reset_index(drop=True)], axis=1)
    return tables[0], tables[1]


def training_rows(tables, labels, stride=TRAIN_STRIDE):
    """Live labelled frames in both orientations: y = 1 when "us" has the ball."""
    labels = np.asarray(labels)
    rows = np.flatnonzero(np.isin(labels, ['H', 'A']))[::stride]
    x = pd.concat([tables[0].iloc[rows], tables[1].iloc[rows]], ignore_index=True)
    return x, np.r_[labels[rows] == 'H', labels[rows] == 'A'].astype(int)


def fit(parts):
    """Fit on a list of (x, y) parts from training_rows."""
    return HistGradientBoostingClassifier(**MODEL_SETTINGS).fit(
        pd.concat([x for x, _ in parts], ignore_index=True), np.concatenate([y for _, y in parts]))


def home_probability(model, tables):
    p_home = model.predict_proba(tables[0])[:, 1]
    p_away = model.predict_proba(tables[1])[:, 1]
    return np.clip((p_home + 1 - p_away) / 2, 1e-3, 1 - 1e-3)


def decode(p_home, period, dead, switch_cost_s, fps=25):
    """H/A sequence from per-frame probabilities; D wherever `dead` is set; X outside periods."""
    dead = np.asarray(dead, bool)
    labels = np.full(len(p_home), 'X', dtype='<U1')
    costs = np.zeros((len(p_home), 3))
    costs[:, 0], costs[:, 1] = -np.log(p_home) / fps, -np.log(1 - p_home) / fps
    costs[:, 2] = 1e6
    costs[dead] = [1e6, 1e6, 0.0]
    for a, b in period_runs(period):
        if period[a] <= 0:
            continue
        states = decode_sequence(costs[a:b], switch_cost_s, 0.0)
        labels[a:b] = np.array(['H', 'A', 'D'])[states]
    return labels
