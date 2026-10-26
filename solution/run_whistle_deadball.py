"""Do detected whistles line up with V16's dead-ball starts beyond chance? Answer-free.

Rules fixed before the run. Main check: share of graded whistles with a V16 dead stretch (at least
1 s) starting from 2 s before to 5 s after the whistle. Chance: the whistle list shifted in time
within each half by a random offset of at least 30 s (keeps the count and bunching), 2000 times.
Pass: at least twice the chance mean and above its 97.5th percentile. Reported only: coverage of
V16 dead stretches (at least 3 s) by a whistle 5 s before to 2 s after their start, the main check
by tracking coverage, and players' slowdown after whistles with no V16 dead ball within 10 s.
Whistles are used in predictions only if the main check passes. No answer labels are read.
No commit or push.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import runs
from .run import ROOT

FPS = 25
HIT_WINDOW_S = (-2, 5)
COVER_WINDOW_S = (-5, 2)
MIN_DEAD_S, LONG_DEAD_S = 1, 3
FAR_S = 10
SHIFTS, MIN_SHIFT_S = 2000, 30


def dead_starts(labels, active, min_s):
    return np.array([a for a, b in runs((labels == 'D') & active) if b - a >= min_s * FPS], dtype=int)


def any_in(times, centres, lo_s, hi_s):
    """For each centre, whether some time lies in [centre + lo_s, centre + hi_s]."""
    times = np.sort(times)
    lo = np.searchsorted(times, centres + lo_s * FPS)
    hi = np.searchsorted(times, centres + hi_s * FPS, side='right')
    return hi > lo


def shift_within_halves(times, halves, offset_s):
    out = times.copy()
    for a, b in halves:
        sel = (times >= a) & (times < b)
        out[sel] = a + (times[sel] - a + int(offset_s * FPS)) % (b - a)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--whistles', type=Path, default=ROOT/'data/runs/whistles_v3/whistles.csv')
    parser.add_argument('--labels', type=Path,
                        default=ROOT/'data/runs/sixteenth_trial_gated/predictions/v16_gated_cal.csv')
    parser.add_argument('--out', type=Path, default=ROOT/'data/runs/whistles_v3/deadball_lineup.json')
    args = parser.parse_args(argv)
    from games import GAME, half_video_blocks
    from .run_v6 import _period_keys
    n = 199319
    blocks = half_video_blocks(GAME)
    period, active = _period_keys(n, blocks)
    halves = [(a, b) for _, a, b in blocks]
    labels = pd.read_csv(args.labels)['label'].to_numpy(dtype='<U1')
    w = pd.read_csv(args.whistles)
    whistles = w.tactical_frame.to_numpy()[w.in_graded_half.to_numpy()]
    starts = dead_starts(labels, active, MIN_DEAD_S)
    long_starts = dead_starts(labels, active, LONG_DEAD_S)
    dead_frames = np.flatnonzero((labels == 'D') & active)
    f = pd.read_csv(ROOT/'data/runs/seventh_trial/diagnostics/v7_features.csv',
                    usecols=['speed_median_mps', 'players_observed'])
    speed, players = f.speed_median_mps.to_numpy(), f.players_observed.to_numpy()

    hit = lambda t: float(any_in(starts, t, *HIT_WINDOW_S).mean())
    cover = lambda t: float(any_in(t, long_starts, *COVER_WINDOW_S).mean())
    rng = np.random.RandomState(0)
    offsets = rng.uniform(MIN_SHIFT_S, min(b - a for a, b in halves) / FPS - MIN_SHIFT_S, SHIFTS)
    shifted = [shift_within_halves(whistles, halves, o) for o in offsets]
    hit_null = np.array([hit(t) for t in shifted])
    cover_null = np.array([cover(t) for t in shifted])
    observed = hit(whistles)
    passed = bool(observed >= 2 * hit_null.mean() and observed > np.percentile(hit_null, 97.5))

    full = players[whistles] >= 21
    by_coverage = {}
    for name, sel in (('21+ players', full), ('under 21 players', ~full)):
        null = np.array([float(any_in(starts, t[sel], *HIT_WINDOW_S).mean()) if sel.any() else np.nan for t in shifted])
        by_coverage[name] = {'whistles': int(sel.sum()),
                             'hit_rate': float(any_in(starts, whistles[sel], *HIT_WINDOW_S).mean()) if sel.any() else None,
                             'chance_mean': float(np.nanmean(null))}

    win = 3 * FPS
    def slowdown(times):
        vals = [np.nanmean(speed[t:t + win]) - np.nanmean(speed[t - win:t]) for t in times if win <= t < n - win]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else None
    far = ~any_in(dead_frames, whistles, -FAR_S, FAR_S)
    random_times = rng.choice(np.flatnonzero(active), 2000, replace=False)

    result = {
        'whistles_in_graded_halves': int(len(whistles)),
        'v16_dead_stretches_1s_plus': int(len(starts)), 'v16_dead_stretches_3s_plus': int(len(long_starts)),
        'main_check': {'hit_rate': observed, 'chance_mean': float(hit_null.mean()),
                       'chance_97_5pct': float(np.percentile(hit_null, 97.5)),
                       'ratio_to_chance': float(observed / max(hit_null.mean(), 1e-9)),
                       'share_of_shifts_at_or_above': float((hit_null >= observed).mean()), 'passed': passed},
        'coverage_of_v16_dead_starts': {'share': cover(whistles), 'chance_mean': float(cover_null.mean())},
        'main_check_by_tracking_coverage': by_coverage,
        'whistles_far_from_any_v16_dead': {'count': int(far.sum()),
                                           'speed_change_after_mps': slowdown(whistles[far]),
                                           'speed_change_after_whistles_near_dead_mps': slowdown(whistles[~far]),
                                           'speed_change_after_random_time_mps': slowdown(random_times)},
    }
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
