"""Disk cache for expensive per-match features.

A cache entry is keyed by the feature name, a fingerprint of the input arrays (positions, ball,
periods, team slots) and the source hashes of the modules that compute the feature, so a code
change or different input never reuses a stale result. Values are dicts of arrays or DataFrames,
stored as compressed .npz (no pickle).
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_ROOT = Path(__file__).resolve().parents[1] / 'data/cache/features'
SOLUTION = Path(__file__).resolve().parent


def fingerprint(match):
    digest = hashlib.sha256()
    for array in (match.player_xy_m, match.ball_xy_m, match.period, match.home_slots):
        a = np.ascontiguousarray(array)
        digest.update(str(a.dtype).encode() + str(a.shape).encode())
        digest.update(a.tobytes())
    digest.update(repr((match.fps, tuple(match.pitch))).encode())
    return digest.hexdigest()[:20]


def source_hash(modules):
    digest = hashlib.sha256()
    for name in sorted(modules):
        digest.update((SOLUTION / name).read_bytes())
    return digest.hexdigest()[:12]


def _path(name, match, modules, extra=''):
    return CACHE_ROOT / f'{name}_{match.name}_{fingerprint(match)}_{source_hash(modules)}{extra}.npz'


def cached(name, match, modules, compute, extra=''):
    """Return compute(match) from the cache when present; values: dict of ndarray / DataFrame."""
    path = _path(name, match, modules, extra)
    if path.exists():
        return _load(path)
    value = compute(match)
    path.parent.mkdir(parents=True, exist_ok=True)
    _save(path, value)
    return value


def _save(path, value):
    arrays = {}
    for key, item in value.items():
        if isinstance(item, pd.DataFrame):
            arrays[f'df__{key}__values'] = item.to_numpy(dtype=np.float32)
            arrays[f'df__{key}__columns'] = np.array(item.columns, dtype=str)
        else:
            arrays[f'arr__{key}'] = np.asarray(item)
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def _load(path):
    out = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            if key.startswith('arr__'):
                out[key[5:]] = data[key]
            elif key.endswith('__values'):
                name = key[4:-8]
                out[name] = pd.DataFrame(data[key], columns=[str(c) for c in data[f'df__{name}__columns']])
    return out
