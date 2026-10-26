"""Detect referee whistles in the broadcast sound and check them without answers.

Checks: whistles near the automatic kick-off tags (every kick-off starts with a whistle); whistles
in the minute before set-piece tags compared with random times; players' median speed before and
after whistles compared with random times. No answer labels are read. No commit or push.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .run import ROOT
from .whistle_audio import FPS, detect_whistles, frame_spectra, to_tactical


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wav', type=Path, default=ROOT/'data/broadcast/BAR-ATH_broadcast_audio_22k.wav')
    parser.add_argument('--mapping', type=Path, default=ROOT/'data/runs/tenth_trial/broadcast_to_tactical.npy')
    parser.add_argument('--out', type=Path, default=ROOT/'data/runs/whistles')
    args = parser.parse_args(argv)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    from games import GAME, half_video_blocks
    from starter import load_smart_tagging
    from .run_v6 import _period_keys
    n = 199319
    period, active = _period_keys(n, half_video_blocks(GAME))
    peak, level = frame_spectra(args.wav)
    events = detect_whistles(peak, level)
    starts = to_tactical([a for a, _ in events], np.load(args.mapping))
    durations = np.array([(b - a) / FPS for a, b in events])
    keep = (starts >= 0) & (starts < n)
    starts, durations = starts[keep], durations[keep]
    in_play_window = active[starts]
    pd.DataFrame({'tactical_frame': starts, 'duration_s': durations, 'in_graded_half': in_play_window}).to_csv(
        out/'whistles.csv', index=False)
    s = starts[in_play_window]
    tags = load_smart_tagging.load()
    set_pieces = [e for e in tags if (e['code'] or '').strip().casefold() == 'set pieces' and active[min(n - 1, e['start_frame'])]]
    kickoffs = [e for e in set_pieces if 'kick off' in (e['tags'].get('Type') or '').casefold()]
    near = lambda t, a, b: bool(((s >= t + a) & (s <= t + b)).any())
    rng = np.random.RandomState(0)
    random_times = rng.choice(np.flatnonzero(active), 2000, replace=False)
    def before_share(times):
        return float(np.mean([near(t, -60 * FPS, 0) for t in times]))
    f = pd.read_csv(ROOT/'data/runs/seventh_trial/diagnostics/v7_features.csv', usecols=['speed_median_mps'])
    speed = f.speed_median_mps.to_numpy()
    def slowdown(times):
        vals = []
        for t in times:
            if 3 * FPS <= t < n - 3 * FPS:
                vals.append(np.nanmean(speed[t:t + 3 * FPS]) - np.nanmean(speed[t - 3 * FPS:t]))
        return float(np.nanmean(vals))
    result = {
        'whistles_total': int(len(starts)), 'whistles_in_graded_halves': int(len(s)),
        'per_minute_of_play': float(len(s) / (active.sum() / FPS / 60)),
        'duration_s_median': float(np.median(durations)) if len(durations) else None,
        'kickoff_tags': len(kickoffs),
        'kickoffs_with_whistle_within_3s_before_to_1s_after': sum(near(e['start_frame'], -3 * FPS, FPS) for e in kickoffs),
        'set_piece_tags': len(set_pieces),
        'set_pieces_with_whistle_in_60s_before': before_share([e['start_frame'] for e in set_pieces]),
        'random_times_with_whistle_in_60s_before': before_share(random_times[:500]),
        'speed_change_after_whistle_mps': slowdown(s),
        'speed_change_after_random_time_mps': slowdown(random_times),
    }
    (out/'whistle_checks.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
