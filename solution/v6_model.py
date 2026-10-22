"""Grouped validation and compact model set for the V6 dead-ball experiment."""

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .deadball_features import feature_columns


def estimator(model_name):
    if model_name == "logistic_core":
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(C=0.15, class_weight="balanced", max_iter=250,
                               solver="lbfgs", random_state=2027),
        )
    if model_name == "hist_core":
        return HistGradientBoostingClassifier(
            learning_rate=0.07, max_iter=140, max_leaf_nodes=17,
            min_samples_leaf=80, l2_regularization=5.0,
            class_weight="balanced", random_state=2027,
        )
    if model_name == "hist_all":
        return HistGradientBoostingClassifier(
            learning_rate=0.07, max_iter=140, max_leaf_nodes=17,
            min_samples_leaf=80, l2_regularization=5.0,
            class_weight="balanced", random_state=2027,
        )
    if model_name == "extra_trees_all":
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            ExtraTreesClassifier(
                n_estimators=120, max_depth=15, min_samples_leaf=35,
                max_features=0.85, class_weight="balanced", n_jobs=-1,
                random_state=2027,
            ),
        )
    raise ValueError("Unknown V6 candidate: {}".format(model_name))


def candidate_columns(model_name):
    groups = ("ball", "movement", "convergence") if model_name in {"logistic_core", "hist_core"} else (
        "ball", "movement", "spatial", "convergence")
    return feature_columns(groups)


def _runs(mask):
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2]))


def smooth_dead_probability(probability, period, fps=25.0, start_threshold=0.55,
                            continue_threshold=0.35, minimum_s=0.4, bridge_s=0.2):
    """Apply a short rolling mean, hysteresis, and small-gap cleanup per half."""
    probability = np.asarray(probability, dtype=np.float32)
    period = np.asarray(period)
    if probability.shape != period.shape:
        raise ValueError("probability and period arrays must have the same shape")
    window = max(1, int(round(0.2 * fps)))
    kernel = np.ones(window, dtype=np.float32) / window
    half_width = window // 2
    output = np.zeros(len(probability), dtype=bool)
    min_run = max(1, int(round(minimum_s * fps)))
    bridge = max(0, int(round(bridge_s * fps)))
    for period_id in np.unique(period):
        indexes = np.flatnonzero(period == period_id)
        if not len(indexes):
            continue
        local = probability[indexes]
        padded = np.pad(local, (half_width, max(0, window - 1 - half_width)), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")
        active = False
        below = 0
        for i, value in enumerate(smoothed):
            if not active and value >= start_threshold:
                active, below = True, 0
                output[indexes[i]] = True
            elif active and value >= continue_threshold:
                below = 0
                output[indexes[i]] = True
            elif active:
                below += 1
                output[indexes[i]] = True
                if below > bridge:
                    output[indexes[i - below + 1:i + 1]] = False
                    active, below = False, 0
        if active and below:
            output[indexes[len(indexes) - below:]] = False
        local_mask = output[indexes].copy()
        for first, stop in _runs(local_mask):
            if stop - first < min_run:
                local_mask[first:stop] = False
        if bridge:
            for (_, left_stop), (right_start, _) in zip(_runs(local_mask), _runs(local_mask)[1:]):
                if right_start - left_stop <= bridge:
                    local_mask[left_stop:right_start] = True
        output[indexes] = local_mask
    return output


def segment_metrics(target, prediction):
    """One-to-one overlap scores for contiguous dead-ball intervals."""
    target_runs, pred_runs = _runs(target), _runs(prediction)
    if not target_runs or not pred_runs:
        return {"target_segments": len(target_runs), "predicted_segments": len(pred_runs),
                "segment_precision": 0.0, "segment_recall": 0.0, "segment_f1": 0.0}
    pairs = []
    for i, (a, b) in enumerate(target_runs):
        for j, (c, d) in enumerate(pred_runs):
            overlap = max(0, min(b, d) - max(a, c))
            if overlap:
                union = max(b, d) - min(a, c)
                pairs.append((overlap / max(union, 1), i, j))
    pairs.sort(reverse=True)
    used_target, used_pred = set(), set()
    for _, i, j in pairs:
        if i not in used_target and j not in used_pred:
            used_target.add(i)
            used_pred.add(j)
    precision = len(used_pred) / len(pred_runs)
    recall = len(used_target) / len(target_runs)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"target_segments": len(target_runs), "predicted_segments": len(pred_runs),
            "segment_precision": precision, "segment_recall": recall, "segment_f1": f1}


def score_probability(target, probability, period, fps=25.0, threshold=0.5):
    target = np.asarray(target, dtype=bool)
    probability = np.asarray(probability, dtype=np.float32)
    raw = probability >= threshold
    smooth = smooth_dead_probability(probability, period, fps=fps)
    result = {
        "threshold": threshold,
        "frame_precision": float(precision_score(target, raw, zero_division=0)),
        "frame_recall": float(recall_score(target, raw, zero_division=0)),
        "frame_f1": float(f1_score(target, raw, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(target, raw)),
        "average_precision": float(average_precision_score(target, probability)),
        "smooth_frame_precision": float(precision_score(target, smooth, zero_division=0)),
        "smooth_frame_recall": float(recall_score(target, smooth, zero_division=0)),
        "smooth_frame_f1": float(f1_score(target, smooth, zero_division=0)),
    }
    result["segments"] = segment_metrics(target, smooth)
    return result, smooth


def grouped_cross_validation(x, y, groups, periods, fps=25.0, train_stride=5):
    """Leave one complete match out; no neighboring frames cross the split."""
    names = ("logistic_core", "hist_core", "hist_all", "extra_trees_all")
    results = {}
    fitted = {}
    for name in names:
        columns = candidate_columns(name)
        matrix = x.loc[:, columns].to_numpy(dtype=np.float32, copy=False)
        probability = np.full(len(y), np.nan, dtype=np.float32)
        folds = []
        for held_out in np.unique(groups):
            train = groups != held_out
            test = ~train
            sample = train & ((np.arange(len(y)) % train_stride) == 0)
            model = estimator(name)
            model.fit(matrix[sample], y[sample].astype(np.uint8))
            probability[test] = model.predict_proba(matrix[test])[:, 1]
            metrics, _ = score_probability(y[test], probability[test], periods[test], fps=fps)
            folds.append({"held_out_match": str(held_out), "frames": int(test.sum()), **metrics})
        if not np.isfinite(probability).all():
            raise RuntimeError("Grouped validation did not produce every out-of-fold probability")
        aggregate, smooth = score_probability(y, probability, periods, fps=fps)
        results[name] = {
            "features": columns,
            "out_of_fold": aggregate,
            "folds": folds,
            "probability": probability,
            "smoothed_prediction": smooth,
        }
    return results


def fit_final_model(model_name, x, y, train_stride=5):
    columns = candidate_columns(model_name)
    matrix = x.loc[:, columns].to_numpy(dtype=np.float32, copy=False)
    selected = (np.arange(len(y)) % train_stride) == 0
    model = estimator(model_name)
    model.fit(matrix[selected], np.asarray(y)[selected].astype(np.uint8))
    return model, columns
