"""Compare team shape, fragment motion, and automatic tags without answer leakage."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd

from games import GAME, GAMES, asset_paths, half_video_blocks
from starter import baseline, load_smart_tagging
from .context import build_context, proxy_costs, tagging_costs
from .data import load_tracking
from .features import make_features
from .predict import predict
from .report import write_report
from .run import ROOT, DEFAULT_CONFIG, read_config, sha256, write_diagnostics, write_labels

V2_CONFIG = Path(__file__).with_name("config_v2.json")
METHODS = {
    "reference_smooth50": "Official reference, unchanged.",
    "v1_gap_stop": "The first experiment, unchanged settings.",
    "v2_shape": "V1 plus team hull area, width, length, centers, and spread.",
    "v2_motion": "V1 plus fragment speed, acceleration, braking, and convergence clues.",
    "v2_shape_motion": "V1 plus both geometry and motion clues.",
    "v2_with_auto_tags": "Geometry and motion plus weak automatic attacking/defending phase priors.",
}


def context_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = set(json.loads(V2_CONFIG.read_text(encoding="utf-8")))
    if set(config) != expected:
        raise ValueError("Unexpected v2 config keys")
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in config.values()):
        raise ValueError("All v2 settings must be finite nonnegative numbers")
    integers = ("minimum_team_tracks", "minimum_motion_tracks", "velocity_window_frames", "minimum_anchors_per_team")
    if any(config[key] < 1 or int(config[key]) != config[key] for key in integers):
        raise ValueError("Track counts, smoothing window, and anchor counts must be positive integers")
    for key in integers:
        config[key] = int(config[key])
    if config["maximum_track_speed_mps"] <= 0 or config["maximum_acceleration_mps2"] <= 0:
        raise ValueError("Motion limits must be positive")
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=V2_CONFIG)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "data/runs/second_trial")
    parser.add_argument("--skip-score", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=0)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if not out.is_relative_to((ROOT / "data").resolve()):
        raise ValueError("Keep v2 outputs in the ignored data/ folder")
    if args.bootstrap < 0:
        raise ValueError("bootstrap cannot be negative")
    base, config = read_config(args.base_config), context_config(args.config)
    data_dir = str(args.data_dir.resolve()) if args.data_dir else None
    paths = asset_paths(data_dir=data_dir)
    for key in ("atd_meta", "atd_raw", "smart_tagging"):
        if not Path(paths[key]).is_file():
            raise FileNotFoundError(paths[key])
    predictions, diagnostics = out / "predictions", out / "diagnostics"
    if predictions.exists() and {p.stem for p in predictions.glob("*.csv")} - set(METHODS):
        raise ValueError("Choose a fresh output folder; unrelated predictions are already present")
    predictions.mkdir(parents=True, exist_ok=True)
    diagnostics.mkdir(exist_ok=True)
    n, fps = GAMES[GAME]["n_video_frames"], GAMES[GAME]["fps"]
    blocks = half_video_blocks(GAME)
    print("Loading tactical ATD with anonymous track fragments. No answer labels are read.", flush=True)
    tracking = load_tracking(paths["atd_meta"], paths["atd_raw"], n, fps, GAMES[GAME]["pitch"],
        max_player_speed_mps=base["max_player_speed_mps"], walking_speed_mps=base["walking_speed_mps"],
        retain_tracks=True, progress=lambda i, total: print("  Loaded {:,}/{:,}".format(i, total), flush=True))
    features = make_features(tracking, blocks, base)
    print("Building collective geometry and motion features.", flush=True)
    context = build_context(tracking, features, blocks, config,
        progress=lambda i, total: print("  Geometry {:,}/{:,}".format(i, total), flush=True))
    print("Fitting weak team prototypes from clear visible-ball anchors only.", flush=True)
    shape, shape_fit = proxy_costs(context["shape"], tracking, features, blocks, config, config["shape_weight"])
    motion, motion_fit = proxy_costs(context["motion"], tracking, features, blocks, config, config["motion_weight"])
    events = load_smart_tagging.load(data_dir=data_dir)
    time_check = load_smart_tagging.check_time_base(events)
    if not time_check["agrees"]:
        raise ValueError("Automatic tagging failed the published clock checks")
    tags, tag_info = tagging_costs(events, baseline.team_names(paths["atd_meta"]), features, config)
    strong = features["control_usable"] & np.isfinite(tracking.home_distance_m) & np.isfinite(tracking.away_distance_m)
    strong &= np.fmin(tracking.home_distance_m, tracking.away_distance_m) <= config["anchor_radius_m"]
    strong &= np.fmax(tracking.home_distance_m, tracking.away_distance_m) >= config["anchor_opponent_distance_m"]
    stop = features["outside_stop"] | features["slow_gap_stop"] | features["stationary_stop"]
    tags[strong | stop] = 0
    models = {"reference_smooth50": (baseline.smooth(baseline.build(data_dir=data_dir), 50),
                                     np.full(n, "official_baseline", dtype="<U40"))}
    for name, extra in (("v1_gap_stop", None), ("v2_shape", shape), ("v2_motion", motion),
                        ("v2_shape_motion", shape + motion), ("v2_with_auto_tags", shape + motion + tags)):
        print("Decoding " + name, flush=True)
        models[name] = predict(tracking, features, blocks, base, extra_team_costs=extra)
    context_table = {"frame": np.arange(n)}
    for family in ("shape", "motion"):
        context_table.update({name: context[family][:, i] for i, name in enumerate(context[family + "_names"])})
    context_table.update({"ball_interpolated": context["ball_interpolated"],
                          "home_shape_cost": shape[:, 0], "away_shape_cost": shape[:, 1],
                          "home_motion_cost": motion[:, 0], "away_motion_cost": motion[:, 1],
                          "home_tag_cost": tags[:, 0], "away_tag_cost": tags[:, 1],
                          "both_teams_converging": context["both_teams_converging"]})
    pd.DataFrame(context_table).to_csv(diagnostics / "context_features.csv", index=False, float_format="%.4f")
    manifest = {
        "experiment": "FirstTrial v2 — team shape and movement", "game": GAME, "frames": n, "fps": fps,
        "report_design": "Six fixed comparisons. Prototypes use noisy visible-ball anchors, never public answers. No parameter search.",
        "method_descriptions": METHODS, "base_config": base, "context_config": config,
        "base_config_sha256": sha256(args.base_config), "context_config_sha256": sha256(args.config),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "input_sha256": {key: sha256(paths[key]) for key in ("atd_meta", "atd_raw", "smart_tagging")},
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted((ROOT / "solution").glob("*.py"))},
        "proxy_fitting": {"shape": shape_fit, "motion": motion_fit, "public_answers_used": False},
        "automatic_tagging": tag_info,
        "feature_coverage": {"shape_frames": int(np.isfinite(context["shape"]).all(axis=1).sum()),
            "motion_frames_with_both_teams": int(np.isfinite(context["motion"][:, [0, 5]]).all(axis=1).sum()),
            "interpolated_ball_frames": int(context["ball_interpolated"].sum()),
            "both_teams_converging_frames": int((context["both_teams_converging"] & features["active"]).sum())},
        "limitations": [
            "The same public five-minute example was seen in v1; this is exploratory, not an independent validation set.",
            "Full-match accuracy is unknown. No winner is selected automatically.",
            "Proxy labels come from ball proximity and can be wrong; geometric costs are not calibrated probabilities.",
            "Anonymous track slots can swap identities without a detectable jump. No named-player roles are inferred.",
            "Short ball-gap interpolation estimates a location; it does not recover the real hidden ball path.",
            "Convergence can describe pressing rather than possession. Remaining duplicate detections can bias motion summaries.",
            "Automatic phase tags are noisy priors. Set-piece and goal intervals are not treated as confirmed stoppages.",
            "Stoppage rules remain the v1 rules. Broadcast alignment, video review, and named roles remain unused.",
            "All sequence inference and proxy fitting are offline and use future match observations.",
        ], "models": {},
    }
    from evaluation import possession_metrics as pm
    for name, (labels, reasons) in models.items():
        path = predictions / (name + ".csv")
        write_labels(path, labels)
        problem = pm.validate_submission(pm.read_label_csv(path), n)
        if problem:
            raise ValueError(name + ": " + problem)
        write_diagnostics(diagnostics / (name + ".csv"), labels, reasons, tracking, features)
        manifest["models"][name] = {"sha256": sha256(path), "official_format_valid": True,
            "label_counts": dict(Counter(map(str, labels))), "reason_counts": dict(Counter(map(str, reasons)))}
        print("Validated " + name, flush=True)
    # Prediction outputs and their hashes are frozen before reading public answers.
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not args.skip_score:
        subprocess.run([sys.executable, str(ROOT / "evaluation/run_evaluation.py"), "--gt", "public",
            "--results", str(predictions), "--out", str(out / "scoring"), "--bootstrap", str(args.bootstrap),
            "--no-detail"], cwd=ROOT, check=True)
    write_report(ROOT, out, manifest, {name: pair[0] for name, pair in models.items()}, not args.skip_score)
    print("Local v2 report: " + str(out / "comparison.html"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
