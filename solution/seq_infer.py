"""Run the trained V34 sequence model (solution/seq_model.py) with numpy only.

The submission ships the weights trained on the 10 practice games (solution/seq_weights.npz) and
runs them here, so rebuilding the file needs neither PyTorch nor a GPU, and does not depend on GPU
training being bit-reproducible. The maths matches torch.nn.GRU / GELU / Linear exactly; windows
and the home/away averaging are the same as seq_model.predict.
"""
from pathlib import Path

import numpy as np
from scipy.special import erf

WEIGHTS = Path(__file__).with_name('seq_weights.npz')


def load(path=WEIGHTS):
    d = np.load(path)
    w = {k[3:]: d[k].astype(np.float64) for k in d.files if k.startswith('w__')}
    return {'w': w, 'mean': d['mean'].astype(np.float64), 'std': d['std'].astype(np.float64),
            'columns': [str(c) for c in d['columns']], 'window': int(d['window'])}


def _prepare(x1, x2, mean, std):
    x = np.concatenate([x1, x2], axis=-1).astype(np.float32).astype(np.float64)
    m, s = np.r_[mean, mean], np.r_[std, std]
    missing = np.isnan(x)
    scaled = np.clip(np.nan_to_num((x - m) / s, nan=0.0), -5, 5)
    return np.concatenate([scaled, missing.astype(np.float64)], axis=-1)


def _gru_direction(x, w, layer, reverse):
    """x: (batch, time, in). Returns (batch, time, hidden)."""
    suffix = f'_l{layer}' + ('_reverse' if reverse else '')
    w_ih, w_hh = w['gru.weight_ih' + suffix], w['gru.weight_hh' + suffix]
    b_ih, b_hh = w['gru.bias_ih' + suffix], w['gru.bias_hh' + suffix]
    hidden = w_hh.shape[1]
    gates_x = x @ w_ih.T + b_ih
    h = np.zeros((x.shape[0], hidden))
    out = np.empty((x.shape[0], x.shape[1], hidden))
    steps = range(x.shape[1] - 1, -1, -1) if reverse else range(x.shape[1])
    for t in steps:
        gx, gh = gates_x[:, t], h @ w_hh.T + b_hh
        r = 1 / (1 + np.exp(-(gx[:, :hidden] + gh[:, :hidden])))
        z = 1 / (1 + np.exp(-(gx[:, hidden:2 * hidden] + gh[:, hidden:2 * hidden])))
        n = np.tanh(gx[:, 2 * hidden:] + r * gh[:, 2 * hidden:])
        h = (1 - z) * n + z * h
        out[:, t] = h
    return out


def forward(model, x):
    """Probability that the first-listed team has the ball, per frame. x: (batch, time, features)."""
    w = model['w']
    a = x @ w['inp.0.weight'].T + w['inp.0.bias']
    a = 0.5 * a * (1 + erf(a / np.sqrt(2)))
    for layer in (0, 1):
        a = np.concatenate([_gru_direction(a, w, layer, False), _gru_direction(a, w, layer, True)], axis=-1)
    logit = (a @ w['out.weight'].T + w['out.bias'])[..., 0]
    return 1 / (1 + np.exp(-logit))


def _periods(period):
    edges = np.r_[0, np.flatnonzero(np.diff(period)) + 1, len(period)]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:]) if period[a] > 0]


def home_probability(model, x_home, x_away, period):
    """As seq_model.predict: 50%-overlapping windows, middle halves kept, both orientations averaged."""
    window = model['window']
    half, quarter = window // 2, window // 4
    p = np.full(len(period), 0.5)
    for a, b in _periods(np.asarray(period)):
        starts = sorted({*range(a, max(a, b - window) + 1, half), max(a, b - window)})
        spans = [(s, min(s + window, b)) for s in starts]
        by_length = {}
        for s, e in spans:
            by_length.setdefault(e - s, []).append((s, e))
        for group in by_length.values():  # equal-length windows run together as one batch
            one = forward(model, np.stack([_prepare(x_home[s:e], x_away[s:e], model['mean'], model['std']) for s, e in group]))
            two = forward(model, np.stack([_prepare(x_away[s:e], x_home[s:e], model['mean'], model['std']) for s, e in group]))
            local = (one + 1 - two) / 2
            for (s, e), values in zip(group, local):
                lo = 0 if s == a else quarter
                hi = e - s if e == b else e - s - quarter
                values_kept = values[lo:hi]
                p[s + lo:s + hi] = values_kept
    return np.clip(p, 1e-3, 1 - 1e-3)
