"""Load SkillCorner's open A-League 2024/25 matches (github.com/SkillCorner/opendata, MIT, (c) SkillCorner).

Broadcast tracking at 10 fps. Players and the ball carry an `is_detected` flag; undetected
positions are SkillCorner's estimates and are treated as missing here, which gives real camera
drop-outs (typically about 14 of 22 players and 60% of ball frames are detected).
Labels: phases of play cover the ball in play, each with the team in possession, so frames in a
phase take that team and frames outside every phase are dead ball (D). Frames outside the match
periods are X. Everything is linearly resampled to 25 fps (positions only between two observed
frames; labels from the nearest 10 fps frame) and cached as compressed arrays.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .external_samples import ExternalMatch

DEFAULT_SC_ROOT = Path(__file__).resolve().parents[1] / 'data/external/skillcorner'
DEFAULT_CACHE = Path(__file__).resolve().parents[1] / 'data/cache/skillcorner'
SOURCE_FPS, TARGET_FPS = 10, 25


@dataclass
class SkillCornerMatch:
    match: ExternalMatch
    labels: np.ndarray
    match_id: str
    ball_z: np.ndarray = None   # detected ball height (m), NaN when undetected; never a model input for BAR-ATH


def _resample(values, n_out):
    """Linear 10 -> 25 fps along axis 0; NaN unless both neighbouring source frames exist."""
    n_in = len(values)
    t = np.arange(n_out) * SOURCE_FPS / TARGET_FPS
    lo = np.minimum(np.floor(t).astype(int), n_in - 1)
    hi = np.minimum(lo + 1, n_in - 1)
    w = (t - lo).reshape((-1,) + (1,) * (values.ndim - 1))
    return values[lo] * (1 - w) + values[hi] * w


def _parse(folder, match_id):
    info = json.loads((folder / f'{match_id}_match.json').read_text(encoding='utf-8'))
    home_id = info['home_team']['id']
    team_of = {p['id']: p['team_id'] for p in info['players']}
    slots = sorted(team_of)
    index = {pid: k for k, pid in enumerate(slots)}
    frames = [json.loads(line) for line in (folder / f'{match_id}_tracking_extrapolated.jsonl').open(encoding='utf-8')]
    first = min(f['frame'] for f in frames)
    n = max(f['frame'] for f in frames) - first + 1
    xy = np.full((n, len(slots), 2), np.nan, np.float32)
    ball = np.full((n, 2), np.nan)
    height = np.full(n, np.nan)
    period = np.zeros(n, np.int8)
    for f in frames:
        t = f['frame'] - first
        period[t] = f['period'] or 0
        b = f['ball_data']
        if b and b.get('is_detected') and b.get('x') is not None:
            ball[t] = (b['x'], b['y'])
            height[t] = b['z'] if b.get('z') is not None else np.nan
        for p in f['player_data']:
            if p.get('is_detected') and p['player_id'] in index:
                xy[t, index[p['player_id']]] = (p['x'], p['y'])
    labels = np.full(n, 'X', dtype='<U1')
    labels[period > 0] = 'D'
    phases = pd.read_csv(folder / f'{match_id}_phases_of_play.csv')
    for row in phases.itertuples():
        a, b = int(row.frame_start) - first, int(row.frame_end) - first
        labels[max(0, a):min(n, b + 1)] = 'H' if row.team_in_possession_id == home_id else 'A'
    labels[period == 0] = 'X'
    length, width = float(info['pitch_length']), float(info['pitch_width'])
    home = np.array([team_of[pid] == home_id for pid in slots])
    return xy, ball, period, labels, home, (length, width), height


def load_skillcorner_match(folder, cache_dir=DEFAULT_CACHE):
    folder = Path(folder)
    match_id = folder.name
    tracking = folder / f'{match_id}_tracking_extrapolated.jsonl'
    stamp = f'{tracking.stat().st_size}_{int(tracking.stat().st_mtime)}'
    cache = Path(cache_dir) / f'{match_id}_{stamp}_z.npz'
    if cache.exists():
        with np.load(cache, allow_pickle=False) as d:
            xy, ball, period, labels, home, pitch, ball_z = (d[k] for k in ['xy', 'ball', 'period', 'labels', 'home', 'pitch', 'ball_z'])
    else:
        xy10, ball10, period10, labels10, home, pitch, z10 = _parse(folder, match_id)
        n = int(len(period10) * TARGET_FPS / SOURCE_FPS)
        xy = _resample(xy10.astype(np.float64), n).astype(np.float32)
        ball = _resample(ball10, n)
        ball_z = _resample(z10, n)
        nearest = np.minimum(np.round(np.arange(n) * SOURCE_FPS / TARGET_FPS).astype(int), len(period10) - 1)
        period, labels = period10[nearest], labels10[nearest]
        # Centre-origin metres -> 0..length x 0..width.
        xy[..., 0] += pitch[0] / 2; xy[..., 1] += pitch[1] / 2
        ball[:, 0] += pitch[0] / 2; ball[:, 1] += pitch[1] / 2
        pitch = np.array(pitch)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, xy=xy, ball=ball, period=period, labels=labels, home=home, pitch=pitch, ball_z=ball_z)
    return SkillCornerMatch(ExternalMatch(f'SC_{match_id}', xy, home, ball, period, [], float(TARGET_FPS),
                                          tuple(float(v) for v in pitch)), labels, match_id, ball_z)


def load_skillcorner_matches(root=DEFAULT_SC_ROOT, cache_dir=DEFAULT_CACHE):
    folders = sorted(p for p in Path(root).iterdir() if p.is_dir() and (p / f'{p.name}_match.json').exists())
    if not folders:
        raise FileNotFoundError(f'No SkillCorner matches in {root}; see solution/README.md for the download')
    return [load_skillcorner_match(f, cache_dir) for f in folders]
