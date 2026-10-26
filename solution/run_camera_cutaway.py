"""Is the broadcast camera leaving the wide pitch view a dead-ball clue? Checked before any use.

Rules fixed before the run.
SkillCorner (broadcast tracking with phase-of-play labels): a frame is a cutaway when no player is
detected. Pass 1: at least 80% of cutaway frames are dead ball, and cutaways cover at least 30% of
dead-ball frames (all 20 matches pooled; per match reported). Reported caveat: SkillCorner builds its
phases partly from its own tracking, so the share of cutaway frames inside a phase that continues
through the cutaway is reported; if phases almost never cross cutaways the labels may follow the
camera by construction.
BAR-ATH (answer-free): a tactical frame is a cutaway when its broadcast frame (constant offset per
half, measured from player positions) has no valid homography. Pass 2: under 5% of certainly-live
frames (direct control, reliable ball moving faster than 2 m/s) fall in a cutaway. Reported:
cutaway share overall, around set-piece tags vs random times, overlap with V16 gated dead ball,
players' median speed in cutaways vs wide view. Not used in any prediction. No commit or push.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import runs
from .run import ROOT

OFFSETS = {1: 4014, 2: 26920}


def skillcorner_part(root):
    from .skillcorner_open import load_skillcorner_match
    folders = sorted(p for p in Path(root).iterdir() if p.is_dir() and (p / f'{p.name}_match.json').exists())
    per_match, totals = {}, np.zeros(4)
    crossing = [0, 0]
    for folder in folders:
        m = load_skillcorner_match(folder)
        keep = m.labels != 'X'
        detected = np.isfinite(m.match.player_xy_m[..., 0]).sum(axis=1)
        cut = (detected == 0) & keep
        dead = (m.labels == 'D') & keep
        both, n_cut, n_dead = (cut & dead).sum(), cut.sum(), dead.sum()
        totals += [both, n_cut, n_dead, keep.sum()]
        per_match[m.match_id] = {'cutaway_share': float(n_cut / keep.sum()), 'dead_share': float(n_dead / keep.sum()),
                                 'cutaway_is_dead': float(both / max(n_cut, 1)), 'dead_in_cutaway': float(both / max(n_dead, 1))}
        # Cutaway frames whose phase (same live label) is present both just before and just after the cutaway run.
        labels = m.labels
        for a, b in runs(cut):
            if 0 < a and b < len(labels) and labels[a - 1] in 'HA' and labels[a - 1] == labels[b]:
                crossing[0] += (labels[a:b] == labels[a - 1]).sum()
            crossing[1] += b - a
        print(f'  {m.match_id}: {per_match[m.match_id]}', flush=True)
    both, n_cut, n_dead, n_keep = totals
    precision, coverage = both / n_cut, both / n_dead
    return {'matches': len(folders), 'cutaway_share': float(n_cut / n_keep), 'dead_share': float(n_dead / n_keep),
            'cutaway_is_dead': float(precision), 'dead_in_cutaway': float(coverage),
            'pass_1': bool(precision >= 0.80 and coverage >= 0.30),
            'cutaway_frames_inside_a_phase_that_continues_through': float(crossing[0] / max(crossing[1], 1)),
            'per_match': per_match}


def bar_ath_part():
    from games import GAME, half_video_blocks
    from starter import load_smart_tagging
    from .run_v6 import _period_keys
    n, fps = 199319, 25
    period, active = _period_keys(n, half_video_blocks(GAME))
    h = pd.read_csv(ROOT/'data/broadcast/BAR-ATH_broadcast_homography.csv', usecols=['frame', 'valid'])
    valid = np.zeros(int(h.frame.max()) + 1, bool)
    valid[h.frame.to_numpy()] = h.valid.to_numpy() == 1
    tactical = np.arange(n)
    broadcast = tactical - np.where(period == 2, OFFSETS[2], OFFSETS[1])
    seen = active & (broadcast >= 0) & (broadcast < len(valid))
    cut = seen & ~valid[np.clip(broadcast, 0, len(valid) - 1)]
    f = pd.read_csv(ROOT/'data/runs/seventh_trial/diagnostics/v7_features.csv',
                    usecols=['direct_control', 'ball_reliable', 'ball_speed_mps', 'speed_median_mps'])
    live = seen & (f.direct_control.to_numpy() >= 0) & (f.ball_reliable.to_numpy() == 1) \
        & (np.nan_to_num(f.ball_speed_mps.to_numpy()) > 2)
    speed = f.speed_median_mps.to_numpy()
    v16 = pd.read_csv(ROOT/'data/runs/sixteenth_trial_gated/predictions/v16_gated_cal.csv')['label'].to_numpy(dtype='<U1')
    d = (v16 == 'D') & seen
    tags = [e['start_frame'] for e in load_smart_tagging.load()
            if (e['code'] or '').strip().casefold() == 'set pieces' and seen[min(n - 1, e['start_frame'])]]
    share_before = lambda times: float(np.mean([cut[max(0, t - 10 * fps):t].mean() for t in times]))
    rng = np.random.RandomState(0)
    random_times = rng.choice(np.flatnonzero(seen), 2000, replace=False)
    live_in_cut = float(cut[live].mean())
    return {'graded_frames_with_broadcast': int(seen.sum()), 'cutaway_share': float(cut[seen].mean()),
            'certainly_live_frames': int(live.sum()), 'certainly_live_in_cutaway': live_in_cut,
            'pass_2': bool(live_in_cut < 0.05),
            'cutaway_share_in_10s_before_set_piece_tags': share_before(tags),
            'cutaway_share_in_10s_before_random_times': share_before(random_times[random_times > 10 * fps]),
            'set_piece_tags': len(tags),
            'v16_gated_dead_share': float(d[seen].mean()),
            'v16_dead_frames_in_cutaway': float(cut[d].mean()), 'cutaway_frames_v16_dead': float(d[cut].mean()),
            'player_speed_median_mps': {'cutaway': float(np.nanmedian(speed[cut])), 'wide_view': float(np.nanmedian(speed[seen & ~cut]))}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skillcorner', type=Path, default=ROOT/'data/external/skillcorner')
    parser.add_argument('--out', type=Path, default=ROOT/'data/runs/camera_cutaway/checks.json')
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result = {'skillcorner': skillcorner_part(args.skillcorner)}
    print(json.dumps({k: v for k, v in result['skillcorner'].items() if k != 'per_match'}, indent=2), flush=True)
    result['bar_ath'] = bar_ath_part()
    print(json.dumps(result['bar_ath'], indent=2))
    args.out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
