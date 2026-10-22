"""Compare Spearman-inspired reach costs and a fixed-prior HMM smoother."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import hmmlearn
import numpy as np
import pandas as pd

from games import GAME, GAMES, asset_paths, half_video_blocks
from starter import baseline, load_smart_tagging
from .context import MOTION_NAMES, SHAPE_NAMES, build_context, estimate_ball, fragment_motion, proxy_costs
from .data import load_tracking
from .features import make_features
from .hmm_smoothing import smooth_costs
from .pitch_control import probability_costs, team_reach_probabilities
from .predict import LABELS, evidence_costs, predict
from .report import write_report
from .run import ROOT, DEFAULT_CONFIG, read_config, sha256, write_diagnostics, write_labels
from .run_v2 import context_config as read_context_config
from .run_v3 import read_v3_config
from .stoppages import build_stoppage_features, enhanced_dead_mask

V2_CONFIG = Path(__file__).with_name("config_v2.json")
V3_CONFIG = Path(__file__).with_name("config_v3.json")
V4_CONFIG = Path(__file__).with_name("config_v4.json")

METHODS = {
    "reference_smooth50": "Unchanged official nearest-player baseline.",
    "v2_shape_motion": "Current best spell-score reference from shape and movement.",
    "v3_tag_soft": "V3's best spell-score method: soft tracking-supported dead-ball costs with set-piece support.",
    "v4_pitch_control": "V2 costs plus acceleration-aware player reach probabilities where a ball target is available.",
    "v4_hmm": "HMM decoding of raw V2 evidence with fixed persistence priors.",
    "v4_pitch_control_hmm": "Acceleration-aware reach costs followed by fixed-prior HMM decoding.",
    "v4_tagged_stoppage_hmm": "V3's soft tagged stoppage costs decoded by the fixed-prior HMM.",
}


def load_verified_v3_context(root, frames, config_path, input_hashes):
    """Reuse V3 geometry only when the saved inputs, settings, and source match."""
    out = Path(root) / "data/runs/third_trial"
    manifest_path = out / "manifest.json"
    feature_path = out / "diagnostics/v3_context_features.csv"
    if not manifest_path.is_file() or not feature_path.is_file():
        return None, {"reused": False, "reason": "saved V3 feature cache is absent"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("game") != GAME or manifest.get("frames") != frames:
        return None, {"reused": False, "reason": "saved V3 cache belongs to a different game or frame count"}
    if manifest.get("context_config_sha256") != sha256(config_path):
        return None, {"reused": False, "reason": "saved V3 cache uses different geometry settings"}
    if manifest.get("input_sha256") != input_hashes:
        return None, {"reused": False, "reason": "saved V3 cache uses different input files"}
    if manifest.get("source_sha256", {}).get("solution/context.py") != sha256(Path(root) / "solution/context.py"):
        return None, {"reused": False, "reason": "saved V3 cache uses different feature code"}

    columns = ["frame"] + ["home_" + name for name in SHAPE_NAMES] + [
        "away_" + name for name in SHAPE_NAMES] + ["home_" + name for name in MOTION_NAMES] + [
        "away_" + name for name in MOTION_NAMES]
    table = pd.read_csv(feature_path, usecols=columns)
    if len(table) != frames or not np.array_equal(table["frame"].to_numpy(), np.arange(frames)):
        return None, {"reused": False, "reason": "saved V3 feature rows do not align to tracking frames"}
    shape_names = [side + "_" + name for side in ("home", "away") for name in SHAPE_NAMES]
    motion_names = [side + "_" + name for side in ("home", "away") for name in MOTION_NAMES]
    context = {
        "shape": table[shape_names].to_numpy(dtype=np.float32),
        "motion": table[motion_names].to_numpy(dtype=np.float32),
        "shape_names": shape_names,
        "motion_names": motion_names,
    }
    return context, {"reused": True, "path": str(feature_path),
                     "sha256": sha256(feature_path), "source_run": "third_trial"}


def read_v4_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = set(json.loads(V4_CONFIG.read_text(encoding="utf-8")))
    if set(config) != expected:
        raise ValueError("Unexpected v4 config keys")
    list_keys = {"same_confidence_mix", "wrong_confidence_mix"}
    numeric = {key: value for key, value in config.items() if key not in list_keys}
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
           for value in numeric.values()):
        raise ValueError("V4 settings must be finite nonnegative numbers")
    for key in list_keys:
        values = np.asarray(config[key], dtype=float)
        if values.shape != (3,) or not np.isfinite(values).all() or (values < 0).any() or not np.isclose(values.sum(), 1.0):
            raise ValueError("{} must contain three probabilities summing to one".format(key))
    if config["reaction_time_s"] < 0 or config["max_acceleration_mps2"] <= 0 or config["max_speed_mps"] <= 0:
        raise ValueError("Player reaction, acceleration, and speed settings are invalid")
    if config["arrival_uncertainty_s"] <= 0 or config["control_horizon_s"] <= 0:
        raise ValueError("Pitch-control time settings must be positive")
    if config["medium_confidence_margin"] < config["weak_confidence_margin"]:
        raise ValueError("HMM confidence margins must be ordered")
    if config["pitch_control_weight"] < 0 or config["maximum_pitch_control_cost"] < 0:
        raise ValueError("Pitch-control cost settings must be nonnegative")
    return config


def unique_track_positions(tracking, duplicate_distance_m, margin_m=3.0):
    """Remove out-of-pitch points and near-duplicate same-team detections."""
    xy = tracking.track_xy_m.copy().astype(np.float32)
    length, width = tracking.pitch
    invalid = (~np.isfinite(xy).all(axis=2) | (xy[:, :, 0] < -margin_m) |
               (xy[:, :, 0] > length + margin_m) | (xy[:, :, 1] < -margin_m) |
               (xy[:, :, 1] > width + margin_m))
    xy[invalid] = np.nan
    for slots in (tracking.home_slots, ~tracking.home_slots):
        indices = np.flatnonzero(slots)
        keep = np.isfinite(xy[:, indices]).all(axis=2)
        for left in range(len(indices)):
            for right in range(left + 1, len(indices)):
                distance = np.linalg.norm(xy[:, indices[left]] - xy[:, indices[right]], axis=1)
                duplicate = keep[:, left] & keep[:, right] & (distance < duplicate_distance_m)
                keep[duplicate, right] = False
        xy[:, indices] = np.where(keep[:, :, None], xy[:, indices], np.nan)
    return xy


def pitch_control_costs(tracking, features, blocks, context_config, dead_config,
                        v4_config, base_config):
    """Build weak per-frame H/A costs from a measured/interpolated ball target."""
    xy = unique_track_positions(tracking, context_config["duplicate_distance_m"],
                                context_config["position_margin_m"])
    velocity, _, _ = fragment_motion(xy, blocks, tracking.fps, context_config)
    targets, interpolated = estimate_ball(tracking, features, blocks, context_config)
    probabilities, supported = team_reach_probabilities(
        xy, velocity, targets, tracking.home_slots, v4_config)

    distance = np.linalg.norm(xy - targets[:, None, :], axis=2)
    home_distance = np.min(np.where(tracking.home_slots[None, :], distance, np.inf), axis=1)
    away_distance = np.min(np.where((~tracking.home_slots)[None, :], distance, np.inf), axis=1)
    nearest = np.minimum(home_distance, away_distance)
    margin = np.abs(home_distance - away_distance)
    strong_direct = np.isfinite(nearest) & (nearest <= base_config["control_radius_m"])
    strong_direct &= margin >= base_config["control_margin_m"]
    eligible = features["active"] & supported & ~strong_direct
    costs = probability_costs(probabilities, eligible,
                              v4_config["pitch_control_weight"],
                              v4_config["maximum_pitch_control_cost"])
    metadata = {
        "finite_ball_target_frames": int(np.isfinite(targets).all(axis=1).sum()),
        "short_gap_interpolated_target_frames": int(interpolated.sum()),
        "supported_team_reach_frames": int(supported.sum()),
        "eligible_weak_prior_frames": int(eligible.sum()),
        "clear_direct_ball_control_frames_protected": int((features["active"] & supported & strong_direct).sum()),
        "fixed_physics": {key: v4_config[key] for key in (
            "reaction_time_s", "max_acceleration_mps2", "max_speed_mps",
            "arrival_uncertainty_s", "control_horizon_s", "pitch_control_weight")},
    }
    diagnostic = {
        "frame": np.arange(len(tracking)),
        "ball_target_interpolated": interpolated,
        "team_reach_supported": supported,
        "pitch_control_eligible": eligible,
        "home_reach_probability": probabilities[:, 0],
        "away_reach_probability": probabilities[:, 1],
        "home_cost": costs[:, 0],
        "away_cost": costs[:, 1],
    }
    return costs, probabilities, metadata, diagnostic


def hmm_labels(costs, base_labels, blocks, config):
    states = smooth_costs(costs, blocks, config)
    labels = np.full(len(costs), "X", dtype="<U1")
    reasons = np.full(len(costs), "outside_scored_halves", dtype="<U40")
    for _, start, last in blocks:
        stop = last + 1
        if np.all(base_labels[start:stop] == "X"):
            continue
        labels[start:stop] = LABELS[states[start:stop]]
        reasons[start:stop] = "hmm_fixed_prior_decode"
        raw = np.argmin(costs[start:stop], axis=1)
        changed = states[start:stop] != raw
        reasons[start:stop][changed] = "hmm_smoothed_raw_evidence"
    return labels, reasons


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=V4_CONFIG)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--context-config", type=Path, default=V2_CONFIG)
    parser.add_argument("--stoppage-config", type=Path, default=V3_CONFIG)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "data/runs/fourth_trial")
    parser.add_argument("--skip-score", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if not out.is_relative_to((ROOT / "data").resolve()):
        raise ValueError("Keep v4 outputs in the ignored data/ folder")
    if args.bootstrap < 0:
        raise ValueError("bootstrap cannot be negative")

    base = read_config(args.base_config)
    context_config = read_context_config(args.context_config)
    dead_config = read_v3_config(args.stoppage_config)
    v4_config = read_v4_config(args.config)
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
    print("Loading tactical ATD. No public answer labels are read.", flush=True)
    tracking = load_tracking(paths["atd_meta"], paths["atd_raw"], n, fps, GAMES[GAME]["pitch"],
        max_player_speed_mps=base["max_player_speed_mps"],
        walking_speed_mps=base["walking_speed_mps"], retain_tracks=True,
        progress=lambda i, total: print("  Loaded {:,}/{:,}".format(i, total), flush=True))
    features = make_features(tracking, blocks, base)
    input_hashes = {key: sha256(paths[key]) for key in ("atd_meta", "atd_raw", "smart_tagging")}
    context, context_cache = load_verified_v3_context(ROOT, n, args.context_config, input_hashes)
    if context is None:
        print("Building team-shape and movement evidence.", flush=True)
        context = build_context(tracking, features, blocks, context_config,
            progress=lambda i, total: print("  Geometry {:,}/{:,}".format(i, total), flush=True))
    else:
        print("Reusing verified V3 team-shape and movement features.", flush=True)
    shape, shape_fit = proxy_costs(context["shape"], tracking, features, blocks,
                                   context_config, context_config["shape_weight"])
    motion, motion_fit = proxy_costs(context["motion"], tracking, features, blocks,
                                     context_config, context_config["motion_weight"])
    core = shape + motion
    print("Calculating acceleration-aware reach probabilities for eligible ball targets.", flush=True)
    physics, probabilities, physics_metadata, physics_diagnostic = pitch_control_costs(
        tracking, features, blocks, context_config, dead_config, v4_config, base)
    v2_costs = evidence_costs(tracking, features, base, use_stoppages=True)
    v2_costs[:, :2] += core
    v4_costs = v2_costs.copy()
    v4_costs[:, :2] += physics

    print("Building soft stoppage candidates for the combination check.", flush=True)
    events = load_smart_tagging.load(data_dir=data_dir)
    time_check = load_smart_tagging.check_time_base(events)
    if not time_check["agrees"]:
        raise ValueError("Automatic tagging failed the published clock checks")
    clues = build_stoppage_features(tracking, features, blocks, dead_config, events)
    tagged_stop_mask, _ = enhanced_dead_mask(clues, blocks, fps, dead_config,
                                             use_restart_tags=True)
    tagged_costs = core.copy()
    tagged_costs[tagged_stop_mask] += dead_config["new_dead_team_cost"]
    tagged_v2_costs = evidence_costs(tracking, features, base, use_stoppages=True)
    tagged_v2_costs[:, :2] += tagged_costs

    reference = baseline.smooth(baseline.build(data_dir=data_dir), 50)
    v2 = predict(tracking, features, blocks, base, extra_team_costs=core)
    v3_soft, v3_reasons = predict(tracking, features, blocks, base,
                                  extra_team_costs=tagged_costs)
    v4_physics = predict(tracking, features, blocks, base,
                         extra_team_costs=core + physics)
    v4_hmm, v4_hmm_reasons = hmm_labels(v2_costs, v2[0], blocks, v4_config)
    v4_physics_hmm, v4_physics_hmm_reasons = hmm_labels(v4_costs, v2[0], blocks, v4_config)
    v4_stoppage_hmm, v4_stoppage_hmm_reasons = hmm_labels(
        tagged_v2_costs, v3_soft, blocks, v4_config)
    models = {
        "reference_smooth50": (reference, np.full(n, "official_baseline", dtype="<U40")),
        "v2_shape_motion": v2,
        "v3_tag_soft": (v3_soft, v3_reasons),
        "v4_pitch_control": v4_physics,
        "v4_hmm": (v4_hmm, v4_hmm_reasons),
        "v4_pitch_control_hmm": (v4_physics_hmm, v4_physics_hmm_reasons),
        "v4_tagged_stoppage_hmm": (v4_stoppage_hmm, v4_stoppage_hmm_reasons),
    }

    prior_predictions = {
        "reference_smooth50": ROOT / "data/runs/third_trial/predictions/reference_smooth50.csv",
        "v2_shape_motion": ROOT / "data/runs/third_trial/predictions/v2_shape_motion.csv",
        "v3_tag_soft": ROOT / "data/runs/third_trial/predictions/v3_tag_soft.csv",
    }
    prior_checks = {}
    for name, path in prior_predictions.items():
        if path.is_file():
            old = pd.read_csv(path)["label"].to_numpy(dtype=str)
            prior_checks[name] = bool(np.array_equal(old, models[name][0]))
            if not prior_checks[name]:
                raise ValueError("Saved {} predictions changed".format(name))

    pd.DataFrame(physics_diagnostic).to_csv(diagnostics / "v4_pitch_control.csv", index=False,
                                            float_format="%.5f")
    manifest = {
        "experiment": "FirstTrial v4 — physics reach and HMM smoothing",
        "game": GAME, "frames": n, "fps": fps,
        "report_design": ("Seven fixed comparisons: prior V2/V3 references, an acceleration-aware reach prior, "
                          "a fixed-persistence categorical HMM, and two combinations. No parameter search or automatic winner."),
        "method_descriptions": METHODS,
        "base_config": base, "context_config": context_config,
        "stoppage_config": dead_config, "v4_config": v4_config,
        "config_sha256": {"base": sha256(args.base_config), "context": sha256(args.context_config),
                          "stoppage": sha256(args.stoppage_config), "v4": sha256(args.config)},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "hmmlearn": hmmlearn.__version__},
        "input_sha256": input_hashes,
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha256(p)
                          for p in sorted((ROOT / "solution").glob("*.py"))},
        "proxy_fitting": {"shape": shape_fit, "motion": motion_fit, "public_answers_used": False},
        "verified_v3_context_cache": context_cache,
        "pitch_control": physics_metadata,
        "hmm": {"state_order": ["H", "A", "D"], "observation_type": "raw per-frame relative evidence category and confidence",
                "training": "No fitting on the public labels; fixed transition and emission assumptions from config_v4.json.",
                "transition_matrix": [[v4_config["team_self_transition"], v4_config["team_to_opponent_transition"], v4_config["team_to_dead_transition"]],
                                      [v4_config["team_to_opponent_transition"], v4_config["team_self_transition"], v4_config["team_to_dead_transition"]],
                                      [v4_config["dead_to_team_transition"], v4_config["dead_to_team_transition"], v4_config["dead_self_transition"]]]},
        "automatic_tagging_clock_check": time_check,
        "tagged_stoppage_candidate_frames": int(tagged_stop_mask.sum()),
        "saved_prediction_reproduction_checks": prior_checks,
        "limitations": [
            "The pitch-control feature needs an observed or short-gap interpolated ball coordinate; it cannot solve a long gap with no location estimate.",
            "This is a Spearman-inspired reach-probability adaptation, not the full Potential Pitch Control Field: ball flight drag, passing choice, offside, and player-specific attributes are omitted.",
            "The team probability normalizes independent player reach chances; it is a weak ranking feature, not a calibrated possession probability.",
            "HMM transitions and emissions are hand-set, not trained; the public example has only 17 spells and is too small for reliable parameter fitting.",
            "All scores reuse the five-minute public example seen in V1-V3 and are not independent validation.",
            "Track slots are anonymous fragments. Player names and stable role identities are not inferred.",
            "The sequence models use future frames and are offline methods.",
            "No full-match answer labels are available, so full-match accuracy and the winning feature combination remain unknown.",
        ],
        "models": {},
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
            "label_counts": dict(Counter(map(str, labels))),
            "reason_counts": dict(Counter(map(str, reasons)))}
        print("Validated " + name, flush=True)

    # Freeze all predictions and hashes before the public labels are opened.
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not args.skip_score:
        subprocess.run([sys.executable, str(ROOT / "evaluation/run_evaluation.py"), "--gt", "public",
            "--results", str(predictions), "--out", str(out / "scoring"),
            "--bootstrap", str(args.bootstrap), "--no-detail"], cwd=ROOT, check=True)
    write_report(ROOT, out, manifest, {name: pair[0] for name, pair in models.items()},
                 not args.skip_score)
    print("Local v4 report: " + str(out / "comparison.html"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
