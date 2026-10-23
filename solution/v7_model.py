"""Small external-only model comparison with masked, period-aware timing scores."""
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, brier_score_loss
from threadpoolctl import threadpool_limits

from .features import runs
from .v7_labels import period_runs
from .v6_model import smooth_dead_probability


def estimator(name, seed=2027):
    if name == 'logistic':
        return make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(),
                             LogisticRegression(C=0.1, max_iter=400, random_state=seed))
    if name == 'extra_trees':
        return make_pipeline(SimpleImputer(strategy='median', add_indicator=True),
                             ExtraTreesClassifier(n_estimators=120, max_depth=14, min_samples_leaf=40,
                                                  max_features=0.8, n_jobs=4, random_state=seed))
    raise ValueError(name)


def dead_mask(score, period, config, fps=25):
    result = np.zeros(len(score), bool)
    for a, b in period_runs(period):
        if period[a] <= 0:
            continue
        result[a:b] = smooth_dead_probability(score[a:b], np.ones(b-a), fps=fps,
            start_threshold=config['dead_start_score'], continue_threshold=config['dead_continue_score'],
            minimum_s=config['dead_confirmation_s'], bridge_s=config['restart_confirmation_s'])
    return result


def interval_metrics(y, pred, period, fps=25, thresholds=(0.3, 0.5, 0.7)):
    """Match only within contiguous known-label regions of one period.

    Unknown boundaries are excluded, so timing refers to labeled interiors.
    Scores do not assert accuracy at the masked event boundaries.
    """
    y, pred, period = np.asarray(y), np.asarray(pred, bool), np.asarray(period)
    targets, predictions = [], []
    block = 0
    for a, b in period_runs(period):
        for first, end in runs(y[a:b] >= 0):
            first, end = first+a, end+a
            targets += [(s+first, e+first, block) for s, e in runs(y[first:end] == 1)]
            predictions += [(s+first, e+first, block) for s, e in runs(pred[first:end])]
            block += 1
    pairs = []
    for i, (a, b, block) in enumerate(targets):
        for j, (c, d, other) in enumerate(predictions):
            if block != other:
                continue
            intersection = max(0, min(b, d)-max(a, c))
            if intersection:
                pairs.append((intersection/(max(b, d)-min(a, c)), i, j))
    pairs.sort(reverse=True)
    scores = {}
    for threshold in thresholds:
        used_t, used_p, latencies = set(), set(), []
        for iou, i, j in pairs:
            if iou >= threshold and i not in used_t and j not in used_p:
                used_t.add(i); used_p.add(j)
                latencies.append((abs(predictions[j][0]-targets[i][0])/fps,
                                  abs(predictions[j][1]-targets[i][1])/fps))
        scores[str(threshold)] = {
            'f1': 2*len(used_t)/max(1, len(targets)+len(predictions)),
            'matched': len(used_t), 'target_segments': len(targets), 'predicted_segments': len(predictions),
            'mean_start_error_s': float(np.mean([x[0] for x in latencies])) if latencies else None,
            'mean_end_error_s': float(np.mean([x[1] for x in latencies])) if latencies else None,
        }
    return {'interval_f1_mean': float(np.mean([v['f1'] for v in scores.values()])), 'by_iou': scores}


def metrics(y, score, period, config):
    valid = y >= 0
    pred = dead_mask(score, period, config)
    p, r, f, _ = precision_recall_fscore_support(y[valid], pred[valid], average='binary', zero_division=0)
    return {'known_frames': int(valid.sum()), 'dead_precision': float(p), 'dead_recall': float(r),
            'dead_frame_f1': float(f), 'false_dead_s': float(((y == 0) & pred).sum()/25),
            'missed_dead_s': float(((y == 1) & ~pred).sum()/25),
            'brier_on_weak_labels': float(brier_score_loss(y[valid], score[valid])),
            **interval_metrics(y, pred, period)}


def fit_compare(x, y, groups, periods, config, progress=print):
    x = np.asarray(x, dtype=np.float32)
    sample = (np.arange(len(y)) % config['train_stride_frames'] == 0) & (y >= 0)
    results, oof_scores = {}, {}
    for name in config['models']:
        oof = np.full(len(y), np.nan, dtype=np.float32)
        folds = []
        for held_out in np.unique(groups):
            train = sample & (groups != held_out)
            test = groups == held_out
            if len(np.unique(y[train])) != 2:
                raise ValueError('External training fold lacks a live/dead class')
            progress(f'  {name}: holding out {held_out}; {train.sum():,} training rows')
            model = estimator(name, config['seed'])
            with threadpool_limits(limits=4):
                model.fit(x[train], y[train])
                oof[test] = model.predict_proba(x[test])[:, 1]
            folds.append({'held_out': str(held_out), **metrics(y[test], oof[test], periods[test], config)})
        results[name] = {'folds': folds,
            'mean_match_interval_f1': float(np.mean([f['interval_f1_mean'] for f in folds])),
            'mean_match_dead_frame_f1': float(np.mean([f['dead_frame_f1'] for f in folds])),
            'mean_match_false_dead_s': float(np.mean([f['false_dead_s'] for f in folds])),
            'pooled': metrics(y, oof, periods, config)}
        oof_scores[name] = oof
    selected = max(results, key=lambda name: (results[name]['mean_match_interval_f1'],
                                              results[name]['mean_match_dead_frame_f1']))
    progress(f'Selected {selected} on external match-mean interval IoU F1 only.')
    final = estimator(selected, config['seed'])
    with threadpool_limits(limits=4):
        final.fit(x[sample], y[sample])
    return final, selected, results, oof_scores
