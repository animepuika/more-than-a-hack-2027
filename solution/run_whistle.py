"""Detect referee whistles in the broadcast sound and check them without answers.

Detector v1: any sharp loud tone in 2-4.5 kHz (mono 22 kHz sound). Detector v2: the referee's tone
pair, learned from the kick-off whistles only (stereo 24 kHz sound; both channels are identical).
Detector v3: v2 keeping only events as strong as the kick-off whistles (whistle_audio.STRONG_DB).

Checks, fixed before v2 was run: whistles per minute of play (plausible 0.3-3); players' median
speed in the 3 s after a whistle minus the 3 s before, compared with random times (pass: at least
0.3 m/s more slowdown, bootstrap 95% range clear of the random value); set-piece tags other than
kick-offs with a whistle in the 30 s before, compared with random times (pass: at least twice the
random share). Kick-offs trained v2, so their hit rate is reported but is not a check. No answer
labels are read. No commit or push.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .run import ROOT
from .whistle_audio import (FPS, STRONG_DB, detect_referee_whistles, detect_whistles, frame_spectra,
                            to_tactical, tone_pair_strength)

WAVS = {'v1': 'data/broadcast/BAR-ATH_broadcast_audio_22k.wav',
        'v2': 'data/broadcast/BAR-ATH_broadcast_audio_24k_stereo.wav',
        'v3': 'data/broadcast/BAR-ATH_broadcast_audio_24k_stereo.wav'}
BEFORE_TAG_S = 30
SPEED_WINDOW_S = 3


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--detector', choices=sorted(WAVS), default='v2')
    parser.add_argument('--wav', type=Path)
    parser.add_argument('--mapping', type=Path, default=ROOT/'data/runs/tenth_trial/broadcast_to_tactical.npy')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    wav = args.wav or ROOT/WAVS[args.detector]
    out = (args.out or ROOT/f'data/runs/whistles_{args.detector}').resolve()
    out.mkdir(parents=True, exist_ok=True)
    from games import GAME, half_video_blocks
    from starter import load_smart_tagging
    from .run_v6 import _period_keys
    n = 199319
    period, active = _period_keys(n, half_video_blocks(GAME))
    if args.detector == 'v1':
        events = detect_whistles(*frame_spectra(wav))
    else:
        events = detect_referee_whistles(*tone_pair_strength(wav), STRONG_DB if args.detector == 'v3' else None)
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
    others = [e for e in set_pieces if e not in kickoffs]
    near = lambda t, a, b: bool(((s >= t + a) & (s <= t + b)).any())
    rng = np.random.RandomState(0)
    random_times = rng.choice(np.flatnonzero(active), 2000, replace=False)
    def before_share(times):
        return float(np.mean([near(t, -BEFORE_TAG_S * FPS, 0) for t in times])) if len(times) else None
    f = pd.read_csv(ROOT/'data/runs/seventh_trial/diagnostics/v7_features.csv', usecols=['speed_median_mps'])
    speed = f.speed_median_mps.to_numpy()
    w = SPEED_WINDOW_S * FPS
    def changes(times):
        vals = [np.nanmean(speed[t:t + w]) - np.nanmean(speed[t - w:t]) for t in times if w <= t < n - w]
        return np.array([v for v in vals if np.isfinite(v)])
    after_whistle, after_random = changes(s), changes(random_times)
    boot = [rng.choice(after_whistle, len(after_whistle)).mean() for _ in range(1000)] if len(after_whistle) else [np.nan]
    by_type = {}
    for e in others:
        kind = e['tags'].get('Type') or ''
        by_type.setdefault(kind, []).append(e['start_frame'])
    result = {
        'detector': args.detector,
        'whistles_total': int(len(starts)), 'whistles_in_graded_halves': int(len(s)),
        'per_minute_of_play': float(len(s) / (active.sum() / FPS / 60)),
        'duration_s_median': float(np.median(durations)) if len(durations) else None,
        'kickoff_tags_used_for_training': len(kickoffs),
        'kickoffs_with_whistle_3s_before_to_4s_after': sum(near(e['start_frame'], -3 * FPS, 4 * FPS) for e in kickoffs),
        'speed_change_after_whistle_mps': float(after_whistle.mean()) if len(after_whistle) else None,
        'speed_change_after_whistle_95pct': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        'speed_change_after_random_time_mps': float(after_random.mean()),
        'set_piece_tags_other_than_kickoff': len(others),
        f'set_pieces_with_whistle_in_{BEFORE_TAG_S}s_before': before_share([e['start_frame'] for e in others]),
        f'set_pieces_by_type_with_whistle_in_{BEFORE_TAG_S}s_before': {k: before_share(v) for k, v in by_type.items()},
        f'random_times_with_whistle_in_{BEFORE_TAG_S}s_before': before_share(random_times[:500]),
    }
    (out/'whistle_checks.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
