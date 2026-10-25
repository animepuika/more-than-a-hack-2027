"""V10 candidate: a trained, ball-free model of which team has the ball in long gaps.

Features use player positions only, so they exist when the ball is missing. Each
period is turned so that the home team attacks +x, and every row is built twice
(home as "us", then away as "us") so the model cannot learn a side preference.
Labels come from solution.external_possession on other matches only.
Inside a long ball gap, a two-state sequence decode places at most a few team
switches where the model is confident, anchored to the owners on both sides.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import runs
from .gap_fallback import long_gap_mask
from .predict import decode_sequence
from .v7_features import clean_positions
from .v7_labels import period_runs
from .v7_sequence import control_evidence

MODEL_SETTINGS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                      l2_regularization=1.0, random_state=2027)
TRAIN_STRIDE = 5
SMOOTH_WINDOWS_S = (1.0, 3.0)
TREND_S = 2.0
SWITCH_COST_S = 1.0     # a switch must be worth about one second of confident evidence
ANCHOR_COST_S = 2.0     # pull towards the known owners at both ends of a gap


def _quantile(values, q):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanpercentile(values, q, axis=1)


def _team_frame(xy, forward, side, speed, length):
    """Per-frame summaries of one team, in its own attacking direction."""
    team = xy[:, side]
    f = forward * (team[:, :, 0] - length / 2)
    y = team[:, :, 1]
    v = speed[:, side]
    count = np.isfinite(f).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        out = {'players': count.astype(float),
               'forward_mean': np.nanmean(f, axis=1), 'forward_q10': _quantile(f, 10),
               'forward_q90': _quantile(f, 90), 'width': _quantile(y, 90) - _quantile(y, 10),
               'centre_y_abs': np.abs(np.nanmean(y, axis=1) - 34.0),
               'speed_median': np.nanmedian(v, axis=1), 'speed_q90': _quantile(v, 90),
               'in_attacking_half': (f > 0).sum(axis=1).astype(float)}
    out['length'] = out['forward_q90'] - out['forward_q10']
    return out


def ballfree_features(player_xy, home, period, pitch, fps):
    """Return (us=home table, us=away table) of identical columns, all frames."""
    home = np.asarray(home, bool)
    n = len(player_xy)
    tables = [{}, {}]
    for a, b in period_runs(period):
        if period[a] <= 0:
            continue
        xy = clean_positions(player_xy[a:b], home, pitch)
        step = np.full(xy.shape[:2], np.nan, dtype=np.float32)
        if b - a > 2:
            step[1:-1] = np.linalg.norm(xy[2:] - xy[:-2], axis=2) * fps / 2
        step[step > 12] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            gap = np.nanmean(xy[:, home, 0]) - np.nanmean(xy[:, ~home, 0])
        home_forward = 1.0 if gap < 0 else -1.0  # each side defends its own half on average
        sides = [_team_frame(xy, home_forward, home, step, pitch[0]),
                 _team_frame(xy, -home_forward, ~home, step, pitch[0])]
        for us in (0, 1):
            mine, theirs = sides[us], sides[1-us]
            local = {}
            for name in mine:
                local['us_'+name], local['them_'+name] = mine[name], theirs[name]
            # Opponent depth is measured in our direction: their deepest line is our target.
            local['line_gap'] = mine['forward_q90'] + theirs['forward_q90']
            local['block_gap'] = mine['forward_mean'] + theirs['forward_mean']
            frame = pd.DataFrame(local)
            extra = {}
            for seconds in SMOOTH_WINDOWS_S:
                window = max(1, round(seconds * fps))
                rolled = frame.rolling(window, min_periods=max(1, window // 2), center=True).mean()
                extra.update({f'{c}_mean_{seconds:g}s': rolled[c].to_numpy() for c in frame})
            lag = max(1, round(TREND_S * fps))
            for c in ['us_forward_mean', 'them_forward_mean', 'block_gap', 'us_length', 'them_length']:
                values = frame[c].to_numpy()
                ahead = np.r_[values[lag:], np.full(lag, np.nan)]
                behind = np.r_[np.full(lag, np.nan), values[:-lag]]
                extra[c+'_trend_next'] = ahead - values
                extra[c+'_trend_prev'] = values - behind
            frame = pd.concat([frame, pd.DataFrame(extra)], axis=1)
            for c in frame:
                if c not in tables[us]:
                    tables[us][c] = np.full(n, np.nan, dtype=np.float32)
                tables[us][c][a:b] = frame[c].to_numpy(dtype=np.float32)
    return pd.DataFrame(tables[0]), pd.DataFrame(tables[1])


def training_rows(tables, labels, stride=TRAIN_STRIDE):
    """Live labelled frames, both orientations: y = 1 when "us" has the ball."""
    labels = np.asarray(labels)
    rows = np.flatnonzero(np.isin(labels, ['H', 'A']))[::stride]
    x = pd.concat([tables[0].iloc[rows], tables[1].iloc[rows]], ignore_index=True)
    y = np.r_[labels[rows] == 'H', labels[rows] == 'A'].astype(int)
    return x, y


def fit(tables_list, labels_list, strides=None):
    strides = strides or [TRAIN_STRIDE] * len(tables_list)
    parts = [training_rows(t, l, s) for t, l, s in zip(tables_list, labels_list, strides)]
    x = pd.concat([p[0] for p in parts], ignore_index=True)
    y = np.concatenate([p[1] for p in parts])
    return HistGradientBoostingClassifier(**MODEL_SETTINGS).fit(x, y)


def home_probability(model, tables):
    """Symmetric P(home has the ball): average of both orientations."""
    p_home = model.predict_proba(tables[0])[:, 1]
    p_away = model.predict_proba(tables[1])[:, 1]
    return np.clip((p_home + 1 - p_away) / 2, 1e-3, 1 - 1e-3)


def model_fallback(p_home, features, period, config, held, gap_min_s, dead=None, fps=25):
    """Fallback labels for long gaps from a two-state decode of the model probabilities."""
    direct = control_evidence(features, config)
    held = np.asarray(held)
    dead = np.zeros(len(direct), bool) if dead is None else np.asarray(dead, bool)
    needed = max(1, round(config['transfer_confirmation_s']*fps))
    gaps = long_gap_mask(features['ball_reliable'] == 1, period, max(1, round(gap_min_s*fps)))
    fallback = np.full(len(direct), '', dtype='<U1')
    for a, b in period_runs(period):
        if period[a] <= 0:
            continue
        for lo, hi in runs(gaps[a:b] & ~dead[a:b]):
            lo, hi = a+lo, a+hi
            before = held[lo-1] if lo > a and held[lo-1] in ('H', 'A') else ''
            stops = np.flatnonzero(dead[hi:b])
            values = direct[hi:hi+stops[0] if len(stops) else b]
            after = ''
            if len(values):
                edges = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, len(values)]
                for s, e in zip(edges[:-1], edges[1:]):
                    if values[s] >= 0 and e - s >= needed:
                        after = 'HA'[values[s]]
                        break
            p = p_home[lo:hi]
            costs = np.zeros((hi-lo, 3))
            costs[:, 0], costs[:, 1] = -np.log(p) / fps, -np.log(1-p) / fps
            costs[:, 2] = 1e6  # no D inside this decode; D comes from the dead mask
            if before:
                costs[0, 1 if before == 'H' else 0] += ANCHOR_COST_S
            if after:
                costs[-1, 1 if after == 'H' else 0] += ANCHOR_COST_S
            states = decode_sequence(costs, SWITCH_COST_S, 1e6)
            fallback[lo:hi] = np.array(['H', 'A', 'D'])[states]
    return fallback
