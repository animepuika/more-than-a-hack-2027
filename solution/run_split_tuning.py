"""Pick the anti-split cleanup lengths on practice copies, judge them on separate copies.

Fixed before the run. Applied to V16 gated + calibrated practice predictions (run_bench10
--save-tuning, BAR-ATH edge drop-out and hidden ball). Truth: Metrica labels as they are; DFL labels
with brief opposing touches under 1 s relabelled (run_v16.controlled_possession), because the
challenge counts only controlled possession. Grid: flip 0/1/2/3/4 s, short dead 0/1/2/3 s, dead gap
0/2/4 s. Choice on the training-seed copies: the largest pooled Spell F1 gain among settings with
false dead on the live reference at most 4%. Judged on the test-seed copies: P1 gain at least +0.005,
better in at least 80% of bootstrap draws; P2 false dead at most 4%. BAR-ATH is not read here.
No commit or push.
"""
import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as pm
from .run import ROOT

GRID = [{'flip_s': f, 'short_dead_s': s, 'dead_gap_s': g}
        for f, s, g in product((0.0, 1.0, 2.0, 3.0, 4.0), (0.0, 1.0, 2.0, 3.0), (0.0, 2.0, 4.0))]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bench', type=Path, default=ROOT/'data/runs/bench10_tuning')
    parser.add_argument('--out', type=Path, default=ROOT/'data/runs/split_tuning/split_tuning.json')
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    from .run_bench10 import features_dict, get_bundle, load_sources
    from .run_external import V7_CONFIG, pooled
    from .run_gap_model import paired_difference
    from .run_v14 import MOVING_BALL_MPS
    from .run_v16 import controlled_possession
    from .split_cleanup import tidy
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
            truth = controlled_possession(s['labels'], m.period) if s['kind'] == 'dfl' else s['labels']
            rows.append({'period': m.period, 'fps': m.fps, 'truth': truth, 'live': live,
                         'pred': saved[f"{condition}|{s['name']}|v16_gated_cal"]})
        return rows

    def score(rows, setting):
        preds = [tidy(r['pred'], r['period'], r['fps'], **setting) for r in rows]
        gt, joined, base = pooled(r['truth'] for r in rows), pooled(preds), pooled(r['pred'] for r in rows)
        f1 = lambda v: float(pm.evaluate(pm.label_codes(gt), pm.label_codes(v))['spell_f1_mean'])
        live = sum(int(r['live'].sum()) for r in rows)
        false_dead = sum(int(((p == 'D') & r['live']).sum()) for p, r in zip(preds, rows)) / max(live, 1)
        return {'spell_f1': f1(joined), 'spell_f1_v16_gated': f1(base), 'false_dead_on_live_ref': false_dead}, (gt, joined, base)

    tune = prepare('tune', 'train_edge')
    grid = []
    for setting in GRID:
        result, _ = score(tune, setting)
        result['gain'] = result['spell_f1'] - result['spell_f1_v16_gated']
        grid.append({'setting': setting, **result})
        print(f"{setting}: gain {result['gain']:+.4f} false dead {result['false_dead_on_live_ref']:.3f}", flush=True)
    safe = [g for g in grid if g['false_dead_on_live_ref'] <= 0.04]
    chosen = max(safe, key=lambda g: g['gain'])
    del tune
    test, (gt, joined, base) = score(prepare('realistic', 'test_edge'), chosen['setting'])
    paired = paired_difference(gt, joined, base)
    rules = {'P1': bool(paired['mean'] >= 0.005 and paired['share_first_better'] >= 0.80),
             'P2': bool(test['false_dead_on_live_ref'] <= 0.04)}
    report = {'grid_on_picking_copies': grid, 'chosen': chosen['setting'],
              'test_copies': {**test, 'spell_f1_change': paired}, 'rules': rules}
    args.out.write_text(json.dumps(report, indent=2, default=float) + '\n')
    print('chosen:', chosen['setting'])
    print('test copies:', json.dumps({**test, 'spell_f1_change': paired}, default=float))
    print('rules:', rules)


if __name__ == '__main__':
    main()
