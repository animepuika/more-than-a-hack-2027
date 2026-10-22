"""Compare the best prior model with a calm-gap abstention policy.

Predictions are completed and hashed before the public answer labels are opened.
The only new inference rule is inspired by the official scorer: X is filled from
the previous prediction for spell scoring, so calm missing-ball frames can hold
the previous owner while a movement scramble still uses the V2/V3 estimate.
"""

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
from .context import build_context, proxy_costs
from .data import load_tracking
from .features import make_features
from .gap_policy import apply_missing_ball_policy, chaotic_movement_mask
from .predict import predict
from .report import write_report
from .run import ROOT, DEFAULT_CONFIG, read_config, sha256, write_diagnostics, write_labels
from .run_v2 import V2_CONFIG, context_config
from .run_v3 import V3_CONFIG, _stoppage_variant, read_v3_config
from .stoppages import build_stoppage_features


V5_CONFIG = Path(__file__).with_name("config_v5.json")
METHODS = {
    "reference_smooth50": "Unchanged official nearest-player reference.",
    "v2_shape_motion": "Best prior Spell F1 method: ball evidence plus team shape and fragment movement.",
    "v3_tag_soft": "V2 shape and movement plus cautious tracking-supported stoppage and set-piece clues.",
    "v5_v2_x_all_missing": "V2 output with X on every missing-ball H/A frame; an ablation that holds the previous owner.",
    "v5_v2_x_calm": "V2 output with X on calm missing-ball frames; movement scrambles keep a forced estimate.",
    "v5_tag_x_all_missing": "V3 tagged soft output with X on every missing-ball H/A frame.",
    "v5_tag_x_calm": "V3 tagged soft output with X on calm missing-ball frames; scrambles keep a forced estimate.",
}


def read_v5_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = set(json.loads(V5_CONFIG.read_text(encoding="utf-8")))
    if set(config) != expected:
        raise ValueError("Unexpected v5 config keys")
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
           for value in config.values()):
        raise ValueError("All v5 settings must be finite nonnegative numbers")
    integers = ("minimum_reacting_players_each_team", "minimum_total_reacting_players",
                "minimum_one_team_reacting_players", "minimum_motion_tracks_per_team")
    if any(config[key] < 1 or int(config[key]) != config[key] for key in integers):
        raise ValueError("Player and track-count thresholds must be positive integers")
    positive = ("one_team_speed_threshold_mps", "both_teams_speed_threshold_mps")
    if any(config[key] <= 0 for key in positive):
        raise ValueError("Speed thresholds must be positive")
    return config


def with_policy(labels, reasons, ball_seen, active, chaos, mode, method):
    output, abstain = apply_missing_ball_policy(labels, ball_seen, active, chaos, mode)
    updated_reasons = np.asarray(reasons, dtype="<U48").copy()
    updated_reasons[abstain] = "v5_{}_missing_ball_abstention".format(mode)
    return output, updated_reasons, abstain


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=V5_CONFIG)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--context-config", type=Path, default=V2_CONFIG)
    parser.add_argument("--stoppage-config", type=Path, default=V3_CONFIG)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "data/runs/fifth_trial")
    parser.add_argument("--skip-score", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if not out.is_relative_to((ROOT / "data").resolve()):
        raise ValueError("Keep v5 outputs inside the ignored data/ folder")
    if args.bootstrap < 0:
        raise ValueError("bootstrap cannot be negative")
    base, shape_config = read_config(args.base_config), context_config(args.context_config)
    stop_config, gap_config = read_v3_config(args.stoppage_config), read_v5_config(args.config)
    data_dir = str(args.data_dir.resolve()) if args.data_dir else None
    paths = asset_paths(data_dir=data_dir)
    for key in ("atd_meta", "atd_raw", "smart_tagging"):
        if not Path(paths[key]).is_file():
            raise FileNotFoundError(paths[key])

    predictions, diagnostics = out / "predictions", out / "diagnostics"
    if predictions.exists() and {p.stem for p in predictions.glob("*.csv")} - set(METHODS):
        raise ValueError("Choose a fresh output folder; unrelated predictions are already present")
    predictions.mkdir(parents=True, exist_ok=True)
    diagnostics.mkdir(parents=True, exist_ok=True)
    n, fps = GAMES[GAME]["n_video_frames"], GAMES[GAME]["fps"]
    blocks = half_video_blocks(GAME)

    print("Loading tactical tracking. Public answer labels are not read for prediction.", flush=True)
    tracking = load_tracking(paths["atd_meta"], paths["atd_raw"], n, fps, GAMES[GAME]["pitch"],
        max_player_speed_mps=base["max_player_speed_mps"],
        walking_speed_mps=base["walking_speed_mps"], retain_tracks=True,
        progress=lambda i, total: print("  Loaded {:,}/{:,}".format(i, total), flush=True))
    features = make_features(tracking, blocks, base)
    print("Rebuilding V2 team-shape and movement evidence.", flush=True)
    context = build_context(tracking, features, blocks, shape_config,
        progress=lambda i, total: print("  Geometry {:,}/{:,}".format(i, total), flush=True))
    shape, shape_fit = proxy_costs(context["shape"], tracking, features, blocks,
                                   shape_config, shape_config["shape_weight"])
    movement, movement_fit = proxy_costs(context["motion"], tracking, features, blocks,
                                         shape_config, shape_config["motion_weight"])
    core_costs = shape + movement
    v2_labels, v2_reasons = predict(tracking, features, blocks, base,
                                    extra_team_costs=core_costs)

    events = load_smart_tagging.load(data_dir=data_dir)
    time_check = load_smart_tagging.check_time_base(events)
    if not time_check["agrees"]:
        raise ValueError("Automatic tagging failed the published clock checks")
    print("Building the prior soft stoppage comparison.", flush=True)
    clues = build_stoppage_features(tracking, features, blocks, stop_config, events)
    v3_labels, v3_reasons, stop_mask, stop_reasons = _stoppage_variant(
        tracking, features, blocks, base, core_costs, clues, stop_config,
        hard_override=False, use_restart_tags=True, duration_frames=None)

    chaos = chaotic_movement_mask(context["motion"], context["motion_track_counts"], gap_config)
    ball_seen = features["ball_seen"]
    models = {
        "reference_smooth50": (baseline.smooth(baseline.build(data_dir=data_dir), 50),
                                np.full(n, "official_baseline", dtype="<U48")),
        "v2_shape_motion": (v2_labels, v2_reasons),
        "v3_tag_soft": (v3_labels, v3_reasons),
    }
    policy_masks = {}
    for parent_name, labels, reasons in (
        ("v2", v2_labels, v2_reasons),
        ("tag", v3_labels, v3_reasons),
    ):
        for mode in ("all", "calm"):
            name = "v5_{}_x_{}".format(parent_name, "all_missing" if mode == "all" else "calm")
            output, updated_reasons, applied = with_policy(
                labels, reasons, ball_seen, features["active"], chaos, mode, name)
            models[name] = (output, updated_reasons)
            policy_masks[name] = applied

    # Ensure the carried-forward X variants remain distinct from hard dead-ball labels.
    for name, (labels, _) in models.items():
        if len(labels) != n or not set(np.unique(labels)).issubset({"H", "A", "D", "X"}):
            raise ValueError("{} produced invalid labels".format(name))
    old_runs = ROOT / "data/runs/fourth_trial/predictions"
    reproduced = {}
    for name, labels in (("reference_smooth50", models["reference_smooth50"][0]),
                         ("v2_shape_motion", v2_labels)):
        saved = old_runs / (name + ".csv")
        if saved.is_file():
            prior = pd.read_csv(saved)["label"].to_numpy(dtype=str)
            reproduced[name] = bool(np.array_equal(prior, labels))
            if not reproduced[name]:
                raise ValueError("V5 no longer reproduces saved {} predictions".format(name))
    old_v3 = old_runs / "v3_tag_soft.csv"
    if old_v3.is_file():
        prior = pd.read_csv(old_v3)["label"].to_numpy(dtype=str)
        reproduced["v3_tag_soft"] = bool(np.array_equal(prior, v3_labels))
        if not reproduced["v3_tag_soft"]:
            raise ValueError("V5 no longer reproduces saved V3 tagged soft predictions")

    gap_diagnostics = pd.DataFrame({
        "frame": np.arange(n), "active": features["active"], "ball_seen": ball_seen,
        "chaos_proxy": chaos, "v2_label": v2_labels, "v3_tag_soft_label": v3_labels,
        "v5_v2_x_all_missing": policy_masks["v5_v2_x_all_missing"],
        "v5_v2_x_calm": policy_masks["v5_v2_x_calm"],
        "v5_tag_x_all_missing": policy_masks["v5_tag_x_all_missing"],
        "v5_tag_x_calm": policy_masks["v5_tag_x_calm"],
        "home_speed_mps": context["motion"][:, 0],
        "away_speed_mps": context["motion"][:, 5],
        "home_burst_fraction": context["motion"][:, 2],
        "away_burst_fraction": context["motion"][:, 7],
        "home_braking_fraction": context["motion"][:, 3],
        "away_braking_fraction": context["motion"][:, 8],
        "home_motion_tracks": context["motion_track_counts"][:, 0],
        "away_motion_tracks": context["motion_track_counts"][:, 1],
        "tagged_stoppage_candidate": stop_mask,
        "tagged_stoppage_reason": stop_reasons,
    })
    gap_diagnostics.to_csv(diagnostics / "gap_policy.csv", index=False, float_format="%.4f")

    input_keys = ("atd_meta", "atd_raw", "smart_tagging")
    manifest = {
        "experiment": "FirstTrial v5 — calm-gap abstention and scramble guesses",
        "game": GAME, "frames": n, "fps": fps,
        "report_design": ("Four fixed X-policy comparisons against the unchanged V2 and V3 references. "
                          "No public answers are used to predict, fit, or select a model."),
        "method_descriptions": METHODS,
        "base_config": base, "context_config": shape_config,
        "stoppage_config": stop_config, "v5_config": gap_config,
        "config_sha256": {"base": sha256(args.base_config), "context": sha256(args.context_config),
                          "stoppage": sha256(args.stoppage_config), "v5": sha256(args.config)},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "input_sha256": {key: sha256(paths[key]) for key in input_keys},
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha256(p)
                          for p in sorted((ROOT / "solution").glob("*.py"))},
        "proxy_fitting": {"shape": shape_fit, "movement": movement_fit,
                          "public_answers_used": False},
        "smart_tagging_clock_check": time_check,
        "gap_policy_coverage": {
            "active_frames": int(features["active"].sum()),
            "active_ball_missing_frames": int((features["active"] & ~ball_seen).sum()),
            "active_ball_missing_with_chaos_proxy": int((features["active"] & ~ball_seen & chaos).sum()),
            "active_ball_missing_without_chaos_proxy": int((features["active"] & ~ball_seen & ~chaos).sum()),
            "v2_calm_frames_changed_to_x": int(policy_masks["v5_v2_x_calm"].sum()),
            "tagged_calm_frames_changed_to_x": int(policy_masks["v5_tag_x_calm"].sum()),
            "v2_missing_frames_changed_to_x": int(policy_masks["v5_v2_x_all_missing"].sum()),
        },
        "saved_prediction_reproduction_checks": reproduced,
        "prediction_contract": ("The output X is intended only for Spell F1's official previous-label fill. "
                                "X remains a miss for frame-level macro F1."),
        "limitations": [
            "The public example has 7,500 graded frames and 17 possessions; public scores cannot establish a full-match winner.",
            "Only 11.8% of the public window has missing ball coordinates, compared with about 43% across the full tactical tracking file.",
            "The scramble proxy is based on anonymous track speed and acceleration summaries; it does not locate a hidden ball or prove a tackle/turnover.",
            "X during a long calm gap carries the previous predicted label in Spell F1; a true turnover inside the gap may therefore be delayed.",
            "The V5 variants are fixed comparisons, not a parameter search. HMM and pitch-control variants are excluded because they fragmented spells in V4.",
            "All sequence methods are offline and use future observations; full-match ground truth is unavailable to participants.",
        ],
        "models": {},
    }

    from evaluation import possession_metrics as pm
    for name, (labels, reasons) in models.items():
        path = predictions / (name + ".csv")
        write_labels(path, labels)
        error = pm.validate_submission(pm.read_label_csv(path), n)
        if error:
            raise ValueError(name + ": " + error)
        write_diagnostics(diagnostics / (name + ".csv"), labels, reasons, tracking, features)
        manifest["models"][name] = {"sha256": sha256(path), "official_format_valid": True,
            "label_counts": dict(Counter(map(str, labels))),
            "reason_counts": dict(Counter(map(str, reasons)))}
        print("Validated " + name, flush=True)

    # Freeze all predictions and input/code hashes before the public answers are opened.
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not args.skip_score:
        subprocess.run([sys.executable, str(ROOT / "evaluation/run_evaluation.py"), "--gt", "public",
            "--results", str(predictions), "--out", str(out / "scoring"),
            "--bootstrap", str(args.bootstrap), "--no-detail"], cwd=ROOT, check=True)
    write_report(ROOT, out, manifest, {name: pair[0] for name, pair in models.items()},
                 not args.skip_score)
    print("Local v5 report: " + str(out / "comparison.html"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
