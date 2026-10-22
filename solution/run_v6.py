"""Train V6 on public Metrica examples and test the frozen model on BAR-ATH.

Only public sample event tables create training targets. BAR-ATH public answers
are opened by the official evaluator after every prediction and manifest has
been written. Generated files remain under ignored data/runs/sixth_trial/.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import json
import platform
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from games import GAME, GAMES, asset_paths, half_video_blocks
from starter import baseline, load_smart_tagging
from .context import build_context, proxy_costs
from .data import load_tracking
from .deadball_features import build_deadball_features, feature_columns
from .external_samples import DEFAULT_SAMPLE_ROOT, ExternalMatch, load_metrica_samples, weak_dead_ball_labels
from .features import make_features, true_run_lengths
from .run import (DEFAULT_CONFIG, ROOT, read_config, sha256, write_diagnostics,
                  write_labels)
from .run_v2 import V2_CONFIG, context_config
from .run_v3 import V3_CONFIG, _stoppage_variant, read_v3_config
from .stoppages import build_stoppage_features
from .v6_model import (candidate_columns, estimator, grouped_cross_validation,
                       smooth_dead_probability)


V6_CONFIG_PATH = ROOT / "solution" / "config_v6.json"
V6_CONFIG = json.loads(V6_CONFIG_PATH.read_text(encoding="utf-8"))


def _sample_fingerprints(sample_root):
    paths = sorted((Path(sample_root) / "data").glob("Sample_Game_*/*"))
    return {path.relative_to(sample_root).as_posix(): sha256(path)
            for path in paths if path.is_file() and path.suffix.lower() in {".csv", ".txt", ".xml", ".json"}}


def _training_data(sample_root, train_stride, progress=print):
    matches = load_metrica_samples(sample_root)
    matrices, labels, groups, periods = [], [], [], []
    all_columns = feature_columns()
    for match_index, match in enumerate(matches):
        progress("  Features for {} ({:,} frames)".format(match.name, len(match)))
        target = weak_dead_ball_labels(match.events, match.period, len(match))
        raw_features = build_deadball_features(match)
        matrix = pd.DataFrame({name: raw_features[name] for name in all_columns})
        matrix = matrix.astype(np.float32, copy=False)
        matrices.append(matrix)
        labels.append(target)
        groups.append(np.full(len(match), match.name, dtype=object))
        periods.append((match_index + 1) * 10 + match.period.astype(np.int16))
        progress("    ball missing {:.1%}; weak dead-ball labels {:.1%}; event rows {:,}".format(
            1.0 - np.isfinite(match.ball_xy_m).all(axis=1).mean(), target.mean(), len(match.events)))
        del raw_features, match
        gc.collect()
    x = pd.concat(matrices, ignore_index=True)
    y = np.concatenate(labels).astype(bool)
    group = np.concatenate(groups)
    period = np.concatenate(periods)
    return x, y, group, period


def _json_cv(cv_results):
    result = {}
    for name, values in cv_results.items():
        result[name] = {
            "features": values["features"],
            "out_of_fold": values["out_of_fold"],
            "folds": values["folds"],
        }
    return result


def _period_keys(n, blocks):
    period = np.zeros(n, dtype=np.int16)
    active = np.zeros(n, dtype=bool)
    for index, (_, start, last) in enumerate(blocks, 1):
        period[start:last + 1] = index
        active[start:last + 1] = True
    return period, active


def _overlay_dead(base_labels, dead_mask):
    result = np.asarray(base_labels, dtype="<U1").copy()
    result[np.asarray(dead_mask, dtype=bool)] = "D"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLE_ROOT,
                        help="Local clone of github.com/metrica-sports/sample-data")
    parser.add_argument("--data-dir", type=Path,
                        help="Optional location of supplied BAR-ATH data")
    parser.add_argument("--out", type=Path, default=ROOT / "data/runs/sixth_trial")
    parser.add_argument("--skip-score", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if not out.is_relative_to((ROOT / "data").resolve()):
        raise ValueError("Keep V6 outputs inside the ignored data/ folder")
    if args.bootstrap < 0:
        raise ValueError("bootstrap cannot be negative")
    if out.exists() and any(out.iterdir()):
        raise ValueError("V6 output directory is not empty; choose a fresh --out folder")
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir()
    (out / "diagnostics").mkdir()

    print("Loading public sample matches and building training features.", flush=True)
    x, y, groups, sample_periods = _training_data(args.samples, V6_CONFIG["train_stride_frames"],
                                                  progress=lambda message: print(message, flush=True))
    print("Running leave-one-match-out validation; complete matches stay in one fold.", flush=True)
    cv = grouped_cross_validation(x, y, groups, sample_periods, fps=25.0,
                                  train_stride=V6_CONFIG["train_stride_frames"])
    cv_json = _json_cv(cv)
    cv_path = out / "cross_validation.json"
    cv_path.write_text(json.dumps(cv_json, indent=2) + "\n", encoding="utf-8")
    selected_name = max(cv, key=lambda name: (
        cv[name]["out_of_fold"]["segments"]["segment_f1"],
        cv[name]["out_of_fold"]["smooth_frame_f1"],
        cv[name]["out_of_fold"]["average_precision"],
    ))
    print("Candidate ranking by held-out weak-label segment F1, then smooth frame F1:", flush=True)
    for name in sorted(cv, key=lambda key: cv[key]["out_of_fold"]["segments"]["segment_f1"], reverse=True):
        score = cv[name]["out_of_fold"]
        print("  {}: segment F1 {:.3f}, smooth frame F1 {:.3f}, AP {:.3f}".format(
            name, score["segments"]["segment_f1"], score["smooth_frame_f1"],
            score["average_precision"]), flush=True)
    print("Selected {} on external-sample CV only.".format(selected_name), flush=True)
    model = estimator(selected_name)
    columns = candidate_columns(selected_name)
    matrix = x.loc[:, columns].to_numpy(dtype=np.float32, copy=False)
    sample = (np.arange(len(y)) % V6_CONFIG["train_stride_frames"]) == 0
    model.fit(matrix[sample], y[sample].astype(np.uint8))
    del matrix, x, y, groups, sample_periods, cv
    gc.collect()

    print("Loading BAR-ATH tracking; answer labels are still unopened.", flush=True)
    data_dir = str(args.data_dir.resolve()) if args.data_dir else None
    paths = asset_paths(data_dir=data_dir)
    for key in ("atd_meta", "atd_raw", "smart_tagging"):
        if not Path(paths[key]).is_file():
            raise FileNotFoundError(paths[key])
    n, fps = GAMES[GAME]["n_video_frames"], GAMES[GAME]["fps"]
    blocks = half_video_blocks(GAME)
    period_key, active = _period_keys(n, blocks)
    base_config = read_config(DEFAULT_CONFIG)
    shape_config = context_config(V2_CONFIG)
    stop_config = read_v3_config(V3_CONFIG)
    tracking = load_tracking(paths["atd_meta"], paths["atd_raw"], n, fps,
        GAMES[GAME]["pitch"], max_player_speed_mps=base_config["max_player_speed_mps"],
        walking_speed_mps=base_config["walking_speed_mps"], retain_tracks=True,
        progress=lambda count, total: print("  Tracking {:,}/{:,}".format(count, total), flush=True))
    bar_match = ExternalMatch(GAME, tracking.track_xy_m, tracking.home_slots,
        tracking.ball_xy_m, period_key, [], fps=fps, pitch=tracking.pitch)
    print("Building BAR-ATH movement, shape, ball-path and territory features.", flush=True)
    bar_features = build_deadball_features(bar_match)
    bar_x = pd.DataFrame({name: bar_features[name] for name in columns}).astype(np.float32, copy=False)
    probabilities = model.predict_proba(bar_x.to_numpy(dtype=np.float32, copy=False))[:, 1]
    ml_dead = smooth_dead_probability(
        probabilities, period_key, fps=fps,
        start_threshold=V6_CONFIG["sequence_start_probability"],
        continue_threshold=V6_CONFIG["sequence_continue_probability"],
        minimum_s=V6_CONFIG["minimum_dead_duration_s"],
        bridge_s=V6_CONFIG["bridge_dead_gap_s"],
    ) & active
    outside = np.asarray(bar_features["ball_entirely_outside"], dtype=bool)
    outside_confirmed = (true_run_lengths(outside) >= V6_CONFIG["boundary_confirm_frames"])
    margin = np.asarray(bar_features["ball_outside_margin_m"], dtype=float)
    outside_confirmed |= np.isfinite(margin) & (margin >= V6_CONFIG["single_frame_outside_margin_m"])
    geometry_dead = outside_confirmed & active

    print("Rebuilding the fixed V2/V3 references for a fair comparison.", flush=True)
    basic = make_features(tracking, blocks, base_config)
    context = build_context(tracking, basic, blocks, shape_config,
        progress=lambda count, total: print("  Team shape {:,}/{:,}".format(count, total), flush=True))
    shape_cost, shape_fit = proxy_costs(context["shape"], tracking, basic, blocks,
        shape_config, shape_config["shape_weight"])
    motion_cost, motion_fit = proxy_costs(context["motion"], tracking, basic, blocks,
        shape_config, shape_config["motion_weight"])
    v2_shape_labels, v2_shape_reasons = __import__("solution.predict", fromlist=["predict"]).predict(
        tracking, basic, blocks, base_config, extra_team_costs=shape_cost)
    v2_motion_labels, v2_motion_reasons = __import__("solution.predict", fromlist=["predict"]).predict(
        tracking, basic, blocks, base_config, extra_team_costs=motion_cost)
    v2_labels, v2_reasons = __import__("solution.predict", fromlist=["predict"]).predict(
        tracking, basic, blocks, base_config, extra_team_costs=shape_cost + motion_cost)
    tagging = load_smart_tagging.load(data_dir=data_dir)
    tag_check = load_smart_tagging.check_time_base(tagging)
    if not tag_check["agrees"]:
        raise ValueError("Automatic tagging failed the published clock checks")
    stop_clues = build_stoppage_features(tracking, basic, blocks, stop_config, tagging)
    from .run_v3 import _stoppage_variant
    v3_labels, v3_reasons, _, _ = _stoppage_variant(
        tracking, basic, blocks, base_config, shape_cost + motion_cost, stop_clues,
        stop_config, hard_override=False, use_restart_tags=True, duration_frames=None)
    baseline_labels = baseline.smooth(baseline.build(data_dir=data_dir), 50)
    models = {
        "reference_smooth50": (baseline_labels, np.full(n, "official_baseline", dtype="<U48")),
        "v2_shape": (v2_shape_labels, v2_shape_reasons),
        "v2_motion": (v2_motion_labels, v2_motion_reasons),
        "v2_shape_motion": (v2_labels, v2_reasons),
        "v3_tag_soft": (v3_labels, v3_reasons),
        "v6_external_dead_v2": (_overlay_dead(v2_labels, ml_dead),
                                 np.where(ml_dead, "v6_external_sample_deadball", v2_reasons)),
        "v6_external_dead_geometry_v2": (_overlay_dead(v2_labels, ml_dead | geometry_dead),
            np.where(ml_dead & geometry_dead, "v6_ml_plus_boundary_crossing",
            np.where(ml_dead, "v6_external_sample_deadball",
            np.where(geometry_dead, "v6_boundary_crossing", v2_reasons)))),
        "v6_external_dead_v3": (_overlay_dead(v3_labels, ml_dead),
                                 np.where(ml_dead, "v6_external_sample_deadball", v3_reasons)),
    }
    for name, (labels, _) in models.items():
        if len(labels) != n or not set(np.unique(labels)).issubset({"H", "A", "D", "X"}):
            raise ValueError("{} produced invalid labels".format(name))
        write_labels(out / "predictions" / (name + ".csv"), labels)

    diagnostic = pd.DataFrame({
        "frame": np.arange(n), "period_key": period_key, "active": active,
        "ball_seen": bar_features["ball_seen"], "ball_gap_age_s": bar_features["ball_gap_age_s"],
        "dead_probability": probabilities, "v6_ml_dead": ml_dead,
        "ball_outside_margin_m": bar_features["ball_outside_margin_m"],
        "ball_entirely_outside": bar_features["ball_entirely_outside"],
        "v6_geometry_dead": geometry_dead,
        "home_nearest_ball_m": bar_features["home_nearest_ball_m"],
        "away_nearest_ball_m": bar_features["away_nearest_ball_m"],
        "home_players_within_1_5m": bar_features["home_players_within_1_5m_last_ball"],
        "away_players_within_1_5m": bar_features["away_players_within_1_5m_last_ball"],
        "home_voronoi_share": bar_features["home_voronoi_share"],
        "home_hull_area_m2": bar_features["home_hull_area_m2"],
        "away_hull_area_m2": bar_features["away_hull_area_m2"],
        "all_converging_players": bar_features["all_converging_players"],
        "v2_shape_label": v2_shape_labels, "v2_motion_label": v2_motion_labels,
        "v2_shape_motion_label": v2_labels, "v3_label": v3_labels,
    })
    diagnostic.to_csv(out / "diagnostics" / "deadball_features.csv", index=False, float_format="%.4f")
    for model_name, (labels, reasons) in models.items():
        write_diagnostics(out / "diagnostics" / (model_name + ".csv"), labels, reasons,
                          tracking, basic)

    external_hashes = _sample_fingerprints(args.samples)
    manifest = {
        "experiment": "FirstTrial v6 — public-sample dead-ball learning and tracking geometry",
        "report_design": ("Trained four classifiers on three public Metrica matches using event-derived weak "
                          "dead-ball labels. Selected {} using leave-one-match-out validation. "
                          "The BAR-ATH scores below use only its public five-minute example.".format(selected_name)),
        "method_descriptions": {
            "reference_smooth50": "The official nearest-player reference, smoothed over 50 frames.",
            "v2_shape": "V2 proximity evidence plus team hull, width, length, center, and spread.",
            "v2_motion": "V2 proximity evidence plus track speed, acceleration, braking, and convergence clues.",
        "v2_shape_motion": "V2 team shape and player movement; close to, but below, the V2 motion-only public spell score.",
            "v3_tag_soft": "V3 soft stoppage evidence with Smart Tagging support.",
        "v6_external_dead_v2": "V2 shape-plus-motion labels with the public-sample dead-ball model overlaid.",
        "v6_external_dead_geometry_v2": "V2 shape-plus-motion labels plus learned dead-ball scores and confirmed ball-boundary crossings.",
            "v6_external_dead_v3": "V3 labels with the public-sample dead-ball model overlaid.",
        },
        "created_utc": datetime.now(timezone.utc).isoformat(), "game": GAME,
        "frames": n, "fps": fps,
        "sample_source": "https://github.com/metrica-sports/sample-data",
        "sample_input_sha256": external_hashes,
        "v6_config": V6_CONFIG,
        "selected_external_model": selected_name,
        "selected_features": columns,
        "external_sample_cross_validation": cv_json,
        "bar_ath_input_sha256": {key: sha256(paths[key]) for key in ("atd_meta", "atd_raw", "smart_tagging")},
        "base_configs": {"base": base_config, "context": shape_config, "stoppage": stop_config},
        "config_sha256": {"base": sha256(DEFAULT_CONFIG), "context": sha256(V2_CONFIG),
                          "stoppage": sha256(V3_CONFIG)},
        "proxy_fitting": {"shape": shape_fit, "movement": motion_fit, "public_answers_used": False},
        "smart_tagging_clock_check": tag_check,
        "source_sha256": {path.relative_to(ROOT).as_posix(): sha256(path)
                          for path in sorted((ROOT / "solution").glob("*.py"))},
        "training_target": ("Weak event-derived dead-ball intervals: BALL OUT, FAULT RECEIVED, CARD, explicit foul/offside/handball/goal markers to the next SET PIECE. "
                            "These are not frame-perfect human labels; unmarked live frames are weak negatives."),
        "features": {"count": len(columns), "groups": ["ball boundary and path", "player proximity and velocity", "team shape and coarse Voronoi", "player convergence"]},
        "models": {},
        "limitations": [
            "The three public examples are only three matches; leave-one-match-out scores are a model-selection signal, not a guarantee.",
            "Metrica sample dead-ball targets are event-derived weak labels. Long marker-to-restart gaps can include uncertain timing.",
            "The BAR-ATH ball is 2D; no ball height, body skeleton, or reliable named player roles are available, so goals, handballs, and offsides are not proven by geometry.",
            "The Voronoi feature is a coarse nearest-player territory share, not Spearman pitch control with reaction time and acceleration.",
            "Ball-distance thresholds of 1.0 and 1.5 m are soft clues; they do not assert possession during aerial, rolling, contested, or occluded phases.",
            "Absolute x thirds are included; attacking/defensive thirds require a verified team direction for each half.",
            "No full-match BAR-ATH answers are used to train or select the model. The official evaluator reads only the public labelled sample after predictions are saved.",
        ],
    }
    for name, (labels, reasons) in models.items():
        prediction_path = out / "predictions" / (name + ".csv")
        manifest["models"][name] = {
            "prediction_sha256": sha256(prediction_path),
            "label_counts": dict(Counter(map(str, labels))),
            "reason_counts": dict(Counter(map(str, reasons))),
        }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not args.skip_score:
        print("Scoring frozen predictions on the public five-minute BAR-ATH example.", flush=True)
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "evaluation" / "run_evaluation.py"),
            "--gt", "public", "--results", str(out / "predictions"),
            "--out", str(out / "scoring"), "--bootstrap", str(args.bootstrap), "--no-detail"],
            cwd=ROOT, check=True)
    from .report import write_report
    write_report(ROOT, out, manifest, {name: item[0] for name, item in models.items()},
                 not args.skip_score)
    print("\nLocal V6 report: {}".format(out / "comparison.html"), flush=True)
    print("Sample training data and generated outputs remain local under ignored data/.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
