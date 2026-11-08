"""V23: test longer cleanup limits using practice predictions only.

Select on the saved tuning copies, evaluate one locked choice on the saved test
copies, and compare against the existing 1/3/4-second cleanup. No BAR-ATH labels
or public ground truth are loaded. These matches/copies have been used before;
this is an incremental experiment, not a new independent final evaluation.
"""
import argparse
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as pm
from .run import ROOT, sha256
from .split_cleanup import tidy


INCUMBENT = {'flip_s': 1.0, 'short_dead_s': 3.0, 'dead_gap_s': 4.0}
GRID = [dict(flip_s=1.0, short_dead_s=s, dead_gap_s=g)
        for s, g in product((3.0, 4.0, 6.0), (4.0, 6.0, 8.0))]
PROTOCOL = """# V23 protocol — written before results

Question: do longer cleanup durations improve possession spells beyond the current
flip=1 s / short-dead=3 s / dead-gap=4 s cleanup?

Use only saved leave-one-match-out V16 gated/calibrated predictions from
data/runs/bench10_tuning/tuning_labels.npz (3 Metrica and 7 DFL matches).
Keep the flip limit at 1 s; grid short-dead in {3,4,6} s and dead-gap in {4,6,8} s.
Keep source label definitions, upstream models, and all other settings fixed.

Selection: highest pooled official spellF1 on tuning-seed copies, subject to
pooled false D on the moving-ball/clear-control reference <=4%. Ties within 1e-12
favor the smallest duration changes from the incumbent. Lock one choice before
loading test-copy feature bundles. If none is safe, reject; do not relax rules.

Practice pass rules on the test-seed copies, relative to EXISTING CLEANUP:
P1 pooled spellF1 gain >=0.005;
P2 candidate wins in >=80% of 2,000 paired whole-match bootstrap draws (seed 20271027);
P3 pooled false D on the certainly-live reference <=4%, and no match increases
   that rate by more than 0.5 percentage points;
P4 pooled spellF1 change >=0 separately for Metrica and DFL;
P5 at least 6 of 10 matches improve (strict gain >1e-12);
P6 nonnegative pooled gain when scored against UNMODIFIED source labels too;
P7 reproduce the incumbent's earlier test score within 1e-12 and preserve every
   ungraded-period output label; identical repeated execution must be possible.

Do not score or apply to BAR-ATH if any practice rule fails. If all pass, a
separate answer-free application must first specify its own rules, before public
scoring. Do not change make_submission or replace the current candidate here.

Limitations: test-seed copies are different masks of the same ten practice matches,
already used in prior research, not ten new matches. Base predictions are from
leave-one-match-out models, but cleanup tuning reuses these practice identities.
DFL's controlled labels suppress sub-second touches; keeping flip_s fixed and
checking original labels limits, but does not remove, definition dependence.
The live reference is a tracking proxy, not independent truth. No practice match
tests the full tactical-plus-broadcast V20 pipeline. No match-specific external
information, manual BAR-ATH labels or private answers are allowed.
"""


def spell_counts(truth, labels):
    """Use official per-match matching; sum counts for the exact pooled F1."""
    truth = pm.label_codes(truth)
    runs = pm.valid_runs(truth)
    pred = pm.fill_abstentions(pm.label_codes(labels), runs)
    result = pm.spell_scores(truth, pred, runs)
    return np.asarray([[v['n_matched'], v['n_spurious'], v['n_missed']]
                       for v in result['per_iou'].values()], dtype=np.int64)


def counts_score(counts):
    counts = np.asarray(counts)
    tp, fp, fn = np.moveaxis(counts, -1, 0)
    denom = 2 * tp + fp + fn
    values = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)
    return np.mean(values, axis=-1)


def cluster_difference(first, second, draws=2000, seed=20271027):
    """Resample whole matches, retaining each pair's full predictions and truth."""
    first, second = np.asarray(first), np.asarray(second)
    picks = np.random.default_rng(seed).integers(0, len(first), size=(draws, len(first)))
    difference = counts_score(first[picks].sum(axis=1)) - counts_score(second[picks].sum(axis=1))
    return {'unit': 'whole_match', 'draws': draws, 'seed': seed,
            'mean': float(difference.mean()),
            'interval_90': np.percentile(difference, [5, 95]).tolist(),
            'share_first_better': float(np.mean(difference > 0))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bench', type=Path, default=ROOT/'data/runs/bench10_tuning')
    parser.add_argument('--out', type=Path, default=ROOT/'data/runs/twentythird_trial')
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if not out.is_relative_to((ROOT/'data').resolve()) or (out.exists() and any(out.iterdir())):
        raise ValueError('Choose a fresh output directory inside ignored data/')
    out.mkdir(parents=True)
    protocol = out/'PROTOCOL.md'
    protocol.write_text(PROTOCOL, encoding='utf-8')
    print('Protocol saved before loading experiment data.', flush=True)

    from .run_bench10 import features_dict, get_bundle, load_sources
    from .run_external import V7_CONFIG
    from .run_v14 import MOVING_BALL_MPS
    from .run_v16 import controlled_possession
    from .v7_sequence import control_evidence
    quality = json.loads(V7_CONFIG.read_text())
    saved_path = args.bench/'tuning_labels.npz'
    saved = np.load(saved_path, allow_pickle=False)
    print('Loading practice sources.', flush=True)
    sources = load_sources()

    def prepare(condition, version):
        rows = []
        for i, source in enumerate(sources):
            print(f"Loading {condition}: {source['name']}", flush=True)
            match, bundle = get_bundle(source, i, version)
            f = features_dict(bundle)
            live = ((match.period > 0) & (control_evidence(f, quality) >= 0)
                    & (f['ball_reliable'] == 1)
                    & (np.nan_to_num(f['ball_speed_mps']) > MOVING_BALL_MPS))
            raw = source['labels']
            truth = controlled_possession(raw, match.period, match.fps) if source['kind'] == 'dfl' else raw
            rows.append({'name': source['name'], 'kind': source['kind'], 'period': match.period,
                         'fps': match.fps, 'truth': truth, 'raw_truth': raw, 'live': live,
                         'pred': saved[f"{condition}|{source['name']}|v16_gated_cal"]})
        return rows

    def score(rows, setting, raw=False):
        results = []
        for row in rows:
            pred = tidy(row['pred'], row['period'], row['fps'], **setting)
            if len(pred) != len(row['truth']) or not np.isin(pred, ['H', 'A', 'D', 'X']).all():
                raise ValueError('Invalid cleanup prediction')
            if not np.array_equal(pred[row['period'] <= 0], row['pred'][row['period'] <= 0]):
                raise ValueError('Cleanup changed ungraded-period labels')
            result = {'name': row['name'], 'kind': row['kind'],
                      'counts': spell_counts(row['raw_truth'] if raw else row['truth'], pred),
                      'live_frames': int(row['live'].sum()),
                      'false_dead_frames': int(((pred == 'D') & row['live']).sum())}
            result['spell_f1'] = float(counts_score(result['counts']))
            result['false_dead'] = result['false_dead_frames'] / max(result['live_frames'], 1)
            results.append(result)
        return results

    def aggregate(rows):
        return {'spell_f1': float(counts_score(sum(r['counts'] for r in rows))),
                'false_dead': sum(r['false_dead_frames'] for r in rows) / max(sum(r['live_frames'] for r in rows), 1)}

    tune = prepare('tune', 'train_edge')
    grid = []
    for setting in GRID:
        result = {'setting': setting, **aggregate(score(tune, setting))}
        grid.append(result)
        print('Tuning:', result, flush=True)
    safe = [r for r in grid if r['false_dead'] <= .04]
    if not safe:
        (out/'results.json').write_text(json.dumps({'grid': grid, 'decision': 'reject_no_safe_choice'}, indent=2))
        return
    best_score = max(r['spell_f1'] for r in safe)
    chosen = min((r for r in safe if best_score - r['spell_f1'] <= 1e-12),
                 key=lambda r: (r['setting']['short_dead_s'], r['setting']['dead_gap_s']))['setting']
    (out/'locked_choice.json').write_text(json.dumps({'chosen': chosen, 'grid': grid}, indent=2)+'\n')
    print('Locked choice:', chosen, flush=True)
    del tune

    test = prepare('realistic', 'test_edge')
    incumbent, candidate = score(test, INCUMBENT), score(test, chosen)
    old, new = aggregate(incumbent), aggregate(candidate)
    delta = new['spell_f1'] - old['spell_f1']
    paired = cluster_difference([r['counts'] for r in candidate], [r['counts'] for r in incumbent])
    groups = {}
    for kind in ('metrica', 'dfl'):
        before = aggregate([r for r in incumbent if r['kind'] == kind])
        after = aggregate([r for r in candidate if r['kind'] == kind])
        groups[kind] = {'incumbent': before, 'candidate': after,
                        'gain': after['spell_f1'] - before['spell_f1']}
    raw_old, raw_new = aggregate(score(test, INCUMBENT, True)), aggregate(score(test, chosen, True))
    old_report = json.loads((ROOT/'data/runs/split_tuning/split_tuning.json').read_text())
    old_score = old_report['test_copies']['spell_f1']
    per_match = [{'name': n['name'], 'kind': n['kind'], 'incumbent': b['spell_f1'],
                  'candidate': n['spell_f1'], 'gain': n['spell_f1'] - b['spell_f1'],
                  'false_dead_before': b['false_dead'], 'false_dead_after': n['false_dead']}
                 for b, n in zip(incumbent, candidate)]
    rules = {'P1_gain': delta >= .005,
             'P2_cluster_bootstrap': paired['share_first_better'] >= .80,
             'P3_false_dead': new['false_dead'] <= .04 and all(
                 r['false_dead_after'] <= r['false_dead_before'] + .005 for r in per_match),
             'P4_each_source': all(r['gain'] >= 0 for r in groups.values()),
             'P5_six_matches': sum(r['gain'] > 1e-12 for r in per_match) >= 6,
             'P6_original_labels': raw_new['spell_f1'] >= raw_old['spell_f1'],
             'P7_incumbent_reproduced': abs(old['spell_f1'] - old_score) <= 1e-12}
    report = {'experiment': 'V23 longer cleanup limits, practice only',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'protocol_sha256': sha256(protocol), 'saved_predictions_sha256': sha256(saved_path),
              'source_sha256': {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted((ROOT/'solution').glob('*.py'))},
              'incumbent_setting': INCUMBENT, 'chosen': chosen, 'grid': grid,
              'test': {'incumbent': old, 'candidate': new, 'gain': delta, 'paired': paired,
                       'groups': groups, 'per_match': per_match,
                       'raw_labels': {'incumbent': raw_old, 'candidate': raw_new}},
              'rules': rules, 'decision': 'eligible_for_answer_free_trial' if all(rules.values()) else 'reject',
              'public_ground_truth_read': False,
              'limitations': ['Previously reused practice matches and masks; no new independent final test.',
                              'Upstream model is V16 gated; complete V20 broadcast pipeline is not tested.',
                              'Certainly-live reference is a tracking proxy, not ground truth.']}
    (out/'results.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'chosen': chosen, 'incumbent': old, 'candidate': new, 'gain': delta,
                      'paired': paired, 'groups': groups, 'rules': rules, 'decision': report['decision']}, indent=2), flush=True)
    saved.close()


if __name__ == '__main__':
    main()
