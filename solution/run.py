"""Run a reproducible, tactical-only comparison; all outputs stay under data/."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd

from games import GAME, GAMES, asset_paths, half_video_blocks
from starter import baseline
from .data import load_tracking
from .features import make_features
from .predict import predict, stable_gap_mask
from .report import write_report

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("config.json")
MODEL_NAMES = ("reference_smooth50", "gap_sequence", "gap_stop_sequence", "gap_stop_sequence_x")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = set(json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8")))
    if set(config) != expected:
        raise ValueError("Config keys differ from solution/config.json")
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in config.values()):
        raise ValueError("All settings must be finite numbers")
    if any(value < 0 for value in config.values()):
        raise ValueError("Settings cannot be negative")
    positive = ("control_radius_m", "distance_scale_m", "max_player_speed_mps", "max_ball_speed_mps",
                "outside_confirm_s", "slow_confirm_s", "missing_stop_after_s", "stationary_confirm_s")
    if any(config[key] <= 0 for key in positive):
        raise ValueError("Radii, scales, speed ceilings, and confirmation times must be positive")
    if config["minimum_players"] < 1 or int(config["minimum_players"]) != config["minimum_players"]:
        raise ValueError("minimum_players must be a positive integer")
    if not 0 <= config["slow_fraction"] <= 1:
        raise ValueError("slow_fraction must be between zero and one")
    return config


def write_labels(path, labels):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["frame", "label"])
        writer.writerows(enumerate(labels))


def write_diagnostics(path, labels, reasons, tracking, features):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["frame", "label", "reason", "ball_seen", "ball_reliable", "missing_seconds",
                         "observed_players", "home_distance_m", "away_distance_m",
                         "median_player_speed_mps", "outside_stop", "slow_gap_stop", "stationary_stop"])
        def number(value):
            return "" if not np.isfinite(value) else format(float(value), ".4f")
        for i, label in enumerate(labels):
            writer.writerow([i, label, reasons[i], int(features["ball_seen"][i]),
                int(features["ball_reliable"][i]), number(features["gap_s"][i]),
                int(tracking.observed_players[i]), number(tracking.home_distance_m[i]),
                number(tracking.away_distance_m[i]), number(tracking.median_player_speed_mps[i]),
                int(features["outside_stop"][i]), int(features["slow_gap_stop"][i]),
                int(features["stationary_stop"][i])])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=None, help="Asset root containing tactical/")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "runs" / "first_trial")
    parser.add_argument("--skip-score", action="store_true", help="Generate and validate predictions without reading answer labels")
    parser.add_argument("--bootstrap", type=int, default=0, help="Optional official scorer uncertainty draws")
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if not out.is_relative_to((ROOT / "data").resolve()):
        raise ValueError("Keep generated outputs inside this repository's ignored data/ folder")
    if args.bootstrap < 0:
        raise ValueError("bootstrap cannot be negative")
    config = read_config(args.config)
    paths = asset_paths(data_dir=str(args.data_dir.resolve()) if args.data_dir else None)
    for key in ("atd_meta", "atd_raw"):
        if not Path(paths[key]).is_file():
            raise FileNotFoundError("Missing {}. Put the supplied tactical files in data/tactical/".format(paths[key]))
    predictions = out / "predictions"
    if predictions.exists():
        unexpected = {path.stem for path in predictions.glob("*.csv")} - set(MODEL_NAMES)
        if unexpected:
            raise ValueError("Output folder contains other predictions; choose a new --out folder")
    predictions.mkdir(parents=True, exist_ok=True)
    (out / "diagnostics").mkdir(exist_ok=True)

    n = GAMES[GAME]["n_video_frames"]
    fps = GAMES[GAME]["fps"]
    blocks = half_video_blocks(GAME)
    print("Loading tactical tracking. No answer labels are used for prediction.", flush=True)
    tracking = load_tracking(paths["atd_meta"], paths["atd_raw"], n, fps, GAMES[GAME]["pitch"],
        max_player_speed_mps=config["max_player_speed_mps"], walking_speed_mps=config["walking_speed_mps"],
        progress=lambda count, total: print("  Loaded {:,}/{:,} frames".format(count, total), flush=True))
    features = make_features(tracking, blocks, config)
    print("Reproducing the official smooth50 reference.", flush=True)
    raw = baseline.build(data_dir=str(args.data_dir.resolve()) if args.data_dir else None)
    reference = baseline.smooth(raw, 50)
    models = {MODEL_NAMES[0]: (reference, np.full(n, "official_baseline", dtype="<U40"))}
    print("Running gap sequence comparison.", flush=True)
    models[MODEL_NAMES[1]] = predict(tracking, features, blocks, config, use_stoppages=False)
    print("Running gap + stoppage comparison.", flush=True)
    stop_labels, stop_reasons = predict(tracking, features, blocks, config, use_stoppages=True)
    models[MODEL_NAMES[2]] = (stop_labels, stop_reasons)
    x_labels, x_reasons = stop_labels.copy(), stop_reasons.copy()
    x_mask = stable_gap_mask(stop_labels, features, blocks, fps, config["max_abstention_gap_s"])
    x_labels[x_mask] = "X"
    x_reasons[x_mask] = "short_stable_gap_abstention"
    models[MODEL_NAMES[3]] = (x_labels, x_reasons)

    # Format-only validation imports no answer data. The official scorer is unmodified.
    from evaluation import possession_metrics as pm
    manifest = {
        "experiment": "FirstTrial v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "game": GAME, "fps": fps, "frames": n, "prediction_inputs": ["tactical ATD XML", "tactical ATD TXT"],
        "config": config, "config_sha256": sha256(args.config),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "input_sha256": {key: sha256(paths[key]) for key in ("atd_meta", "atd_raw")},
        "source_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in sorted((ROOT / "solution").glob("*.py"))},
        "comparison_design": "Fixed comparisons. No training, parameter search, or automatic winner selection.",
        "models": {},
    }
    for name, (labels, reasons) in models.items():
        path = predictions / (name + ".csv")
        write_labels(path, labels)
        problem = pm.validate_submission(pm.read_label_csv(path), n)
        if problem:
            raise ValueError("{} failed official validation: {}".format(name, problem))
        write_diagnostics(out / "diagnostics" / (name + ".csv"), labels, reasons, tracking, features)
        manifest["models"][name] = {"sha256": sha256(path), "official_format_valid": True,
            "label_counts": dict(Counter(map(str, labels))), "reason_counts": dict(Counter(map(str, reasons)))}
        print("Validated {}: {:,} rows".format(name, n), flush=True)
    # Freeze input/config/output fingerprints before any public answers are read.
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not args.skip_score:
        print("Scoring the completed files on the public five-minute example only.", flush=True)
        subprocess.run([sys.executable, str(ROOT / "evaluation" / "run_evaluation.py"),
            "--gt", "public", "--results", str(predictions), "--out", str(out / "scoring"),
            "--bootstrap", str(args.bootstrap), "--no-detail"], cwd=ROOT, check=True)
    write_report(ROOT, out, manifest, {name: pair[0] for name, pair in models.items()}, not args.skip_score)
    print("\nLocal report: {}".format(out / "comparison.html"))
    print("No winner was selected. No files were committed or pushed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
