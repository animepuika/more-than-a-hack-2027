"""V21b: pick the line-restart timing on practice copies only, then judge it on separate copies.

Fixed before the run. Grid: end on the ball seen back inside (any speed) or only when kicked/thrown
in; time limit 30/20/12 s; 0 or 2 s cut off the end of every stretch; backward calm limit for edge
restarts 1.5 or 1.2 m/s (24 combinations). Picking copies: the training-seed realistic copies
(BAR-ATH edge drop-out, hidden ball) of all ten matches; judging copies: the test-seed copies,
generated separately. Choice: among combinations whose added dead frames (beyond V16 gated) are at
least 80% truly dead on the picking copies, the largest pooled Spell F1 gain over V16 gated; if none,
the most precise one, flagged. Judged on the test copies with the unchanged rules P1 (gain at least
+0.005, better in at least 80% of bootstrap draws), P2 (added frames at least 80% truly dead) and P3
(false dead on the live reference at most 4%). BAR-ATH is not read here. No commit or push.
"""
import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as pm
from .run import ROOT

GRID = [{'end_on_inside': e, 'max_s': m, 'end_trim_s': t, 'calm_back_mps': c}
        for e, m, t, c in product((False, True), (30.0, 20.0, 12.0), (0.0, 2.0), (1.5, 1.2))]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bench', type=Path, default=ROOT/'data/runs/bench10_tuning')
    parser.add_argument('--out', type=Path, default=ROOT/'data/runs/v21b_timing/timing.json')
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    from .line_restarts import line_restart_dead
    from .run_bench10 import SEEDS, features_dict, get_bundle, load_sources
    from .run_external import V7_CONFIG, pooled
    from .run_gap_model import paired_difference
    from .run_v14 import MOVING_BALL_MPS, overlay
    from .v7_sequence import control_evidence
    v7c = json.loads(V7_CONFIG.read_text())
    saved = np.load(args.bench/'tuning_labels.npz')
    sources = load_sources()

    def prepare(condition, version):
        rows = []
        for i, s in enumerate(sources):
            m, b = get_bundle(s, i, version)
            feats = features_dict(b)
            active = m.period > 0
            live = ((control_evidence(feats, v7c) >= 0) & (feats['ball_reliable'] == 1)
                    & (np.nan_to_num(feats['ball_speed_mps']) > MOVING_BALL_MPS) & active)
            rows.append({'m': m, 'labels': s['labels'], 'active': active, 'live': live,
                         'gated': saved[f"{condition}|{s['name']}|v16_gated_cal"]})
        return rows

    def score(rows, timing):
        added = truly = false_live = live_n = 0
        preds = []
        for r in rows:
            m = r['m']
            lines = line_restart_dead(m.ball_xy_m, m.player_xy_m, m.period, m.pitch, m.fps, timing=timing)
            pred = overlay(r['gated'], lines, r['active'])
            graded = r['labels'] != 'X'
            new = (pred == 'D') & (r['gated'] != 'D') & graded
            added += new.sum(); truly += (new & (r['labels'] == 'D')).sum()
            false_live += ((pred == 'D') & r['live']).sum(); live_n += r['live'].sum()
            preds.append(pred)
        gt = pooled(r['labels'] for r in rows)
        joined, base = pooled(preds), pooled(r['gated'] for r in rows)
        f1 = lambda v: float(pm.evaluate(pm.label_codes(gt), pm.label_codes(v))['spell_f1_mean'])
        return {'added_frames': int(added), 'added_truly_dead': float(truly / max(added, 1)),
                'false_dead_on_live_ref': float(false_live / max(live_n, 1)),
                'spell_f1': f1(joined), 'spell_f1_v16_gated': f1(base)}, (gt, joined, base)

    tune_rows = prepare('tune', 'train_edge')
    grid = []
    for timing in GRID:
        result, _ = score(tune_rows, timing)
        result['gain'] = result['spell_f1'] - result['spell_f1_v16_gated']
        grid.append({'timing': timing, **result})
        print(f"{timing}: added {result['added_frames']} truly dead {result['added_truly_dead']:.3f} "
              f"false live {result['false_dead_on_live_ref']:.3f} gain {result['gain']:+.4f}", flush=True)
    precise = [g for g in grid if g['added_truly_dead'] >= 0.80]
    chosen = max(precise, key=lambda g: g['gain']) if precise else max(grid, key=lambda g: g['added_truly_dead'])
    del tune_rows
    test, (gt, joined, base) = score(prepare('realistic', 'test_edge'), chosen['timing'])
    paired = paired_difference(gt, joined, base)
    rules = {'P1': bool(paired['mean'] >= 0.005 and paired['share_first_better'] >= 0.80),
             'P2': bool(test['added_truly_dead'] >= 0.80), 'P3': bool(test['false_dead_on_live_ref'] <= 0.04)}
    report = {'grid_on_picking_copies': grid, 'chosen': chosen['timing'], 'chosen_met_precision_bar': bool(precise),
              'test_copies': {**test, 'spell_f1_change': paired}, 'rules': rules, 'seeds': SEEDS}
    args.out.write_text(json.dumps(report, indent=2, default=float) + '\n')
    print('chosen:', chosen['timing'], 'met precision bar on picking copies:', bool(precise))
    print('test copies:', json.dumps({**test, 'spell_f1_change': paired}, default=float))
    print('rules:', rules)


if __name__ == '__main__':
    main()
