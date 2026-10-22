"""Compare restart-aware stoppage clues and duration decoding on fixed methods."""

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
from .predict import predict
from .report import write_report
from .run import ROOT, DEFAULT_CONFIG, read_config, sha256, write_diagnostics, write_labels
from .stoppages import build_stoppage_features, enhanced_dead_mask

V2_CONFIG = Path(__file__).with_name("config_v2.json")
V3_CONFIG = Path(__file__).with_name("config_v3.json")

METHODS = {
    "reference_smooth50": "Unchanged official nearest-player baseline.",
    "v2_shape": "Shape-only comparison, unchanged from v2.",
    "v2_shape_motion": "Shape plus movement, unchanged from v2.",
    "v3_soft": "Shape and movement plus a soft cost for tracking-supported dead-ball candidates.",
    "v3_soft_duration": "Soft dead-ball costs plus minimum-duration sequence decoding.",
    "v3_hard": "Force tracking-supported dead-ball candidates to D after sequence decoding.",
    "v3_hard_duration": "Minimum-duration decoding plus forced tracking-supported D labels.",
    "v3_tag_soft": "Soft dead-ball costs with cautious automatic set-piece support.",
    "v3_tag_soft_duration": "Tagged soft dead-ball costs plus minimum-duration decoding.",
    "v3_tag_hard": "Force D when tracking evidence is supported by an automatic set-piece window.",
    "v3_tag_hard_duration": "Tagged forced-D candidates plus minimum-duration decoding.",
}


def read_v3_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = set(json.loads(V3_CONFIG.read_text(encoding="utf-8")))
    if set(config) != expected:
        raise ValueError("Unexpected v3 config keys")
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
           for value in config.values()):
        raise ValueError("All v3 settings must be finite nonnegative numbers")
    integers = ("minimum_speed_tracks",)
    if any(config[key] < 1 or int(config[key]) != config[key] for key in integers):
        raise ValueError("Minimum track count must be a positive integer")
    fractions = ("quiet_walking_fraction", "ball_cluster_coverage", "restart_walking_fraction")
    if any(not 0 <= config[key] <= 1 for key in fractions):
        raise ValueError("Fraction settings must be between zero and one")
    positive = ("quiet_confirmation_s", "ball_cluster_window_s", "stationary_confirmation_s",
                "missing_confirmation_s", "restart_hint_confirmation_s", "ball_travel_window_s",
                "restart_confirmation_s", "minimum_team_spell_s", "minimum_dead_spell_s")
    if any(config[key] <= 0 for key in positive):
        raise ValueError("Confirmation windows and minimum durations must be positive")
    if config["maximum_dead_hold_s"] < 0:
        raise ValueError("Dead-ball continuity cannot be negative")
    return config


def _stoppage_variant(tracking, features, blocks, base, context_costs, clues, config,
                      hard_override, use_restart_tags, duration_frames):
    mask, reasons = enhanced_dead_mask(clues, blocks, tracking.fps, config,
                                       use_restart_tags=use_restart_tags)
    costs = context_costs.copy()
    if not hard_override:
        costs[mask] += config["new_dead_team_cost"]
    labels, model_reasons = predict(tracking, features, blocks, base,
                                    extra_team_costs=costs,
                                    minimum_duration_frames=duration_frames)
    if hard_override:
        labels[mask] = "D"
        model_reasons[mask] = reasons[mask]
    else:
        is_team = mask & (labels != "D")
        is_dead = mask & (labels == "D")
        model_reasons[is_team] = "v3_soft_stoppage_cost"
        model_reasons[is_dead] = "v3_soft_stoppage_decoded_dead"
    return labels, model_reasons, mask, reasons


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=V3_CONFIG)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--context-config", type=Path, default=V2_CONFIG)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "data/runs/third_trial")
    parser.add_argument("--skip-score", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if not out.is_relative_to((ROOT / "data").resolve()):
        raise ValueError("Keep v3 outputs in the ignored data/ folder")
    if args.bootstrap < 0:
        raise ValueError("bootstrap cannot be negative")
    base = read_config(args.base_config)
    config = read_v3_config(args.config)
    # Reuse v2's strict schema and range checks for the shape/motion feature settings.
    from .run_v2 import context_config as read_context_config
    context_config = read_context_config(args.context_config)
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
    print("Loading tactical tracking. No answer labels are read.", flush=True)
    tracking = load_tracking(paths["atd_meta"], paths["atd_raw"], n, fps, GAMES[GAME]["pitch"],
        max_player_speed_mps=base["max_player_speed_mps"],
        walking_speed_mps=base["walking_speed_mps"], retain_tracks=True,
        progress=lambda i, total: print("  Loaded {:,}/{:,}".format(i, total), flush=True))
    features = make_features(tracking, blocks, base)
    print("Building v2 team-shape and movement costs.", flush=True)
    context = build_context(tracking, features, blocks, context_config,
        progress=lambda i, total: print("  Geometry {:,}/{:,}".format(i, total), flush=True))
    shape, shape_fit = proxy_costs(context["shape"], tracking, features, blocks,
                                   context_config, context_config["shape_weight"])
    motion, motion_fit = proxy_costs(context["motion"], tracking, features, blocks,
                                     context_config, context_config["motion_weight"])
    core = shape + motion
    events = load_smart_tagging.load(data_dir=data_dir)
    time_check = load_smart_tagging.check_time_base(events)
    if not time_check["agrees"]:
        raise ValueError("Automatic tagging failed the published clock checks")
    print("Building persistent dead-ball and restart clues.", flush=True)
    clues = build_stoppage_features(tracking, features, blocks, config, events)
    minimum_frames = [int(math.ceil(config["minimum_team_spell_s"] * fps)),
                      int(math.ceil(config["minimum_team_spell_s"] * fps)),
                      int(math.ceil(config["minimum_dead_spell_s"] * fps))]

    reference = baseline.smooth(baseline.build(data_dir=data_dir), 50)
    models = {"reference_smooth50": (reference, np.full(n, "official_baseline", dtype="<U40"))}
    models["v2_shape"] = predict(tracking, features, blocks, base, extra_team_costs=shape)
    models["v2_shape_motion"] = predict(tracking, features, blocks, base, extra_team_costs=core)
    for tagged in (False, True):
        for hard in (False, True):
            suffix = "tag_" if tagged else ""
            mode = "hard" if hard else "soft"
            base_name = "v3_" + suffix + mode
            print("Decoding " + base_name, flush=True)
            labels, reasons, _, _ = _stoppage_variant(
                tracking, features, blocks, base, core, clues, config,
                hard, tagged, None)
            models[base_name] = (labels, reasons)
            duration_name = base_name + "_duration"
            print("Decoding " + duration_name, flush=True)
            duration_labels, duration_reasons, _, _ = _stoppage_variant(
                tracking, features, blocks, base, core, clues, config,
                hard, tagged, minimum_frames)
            models[duration_name] = (duration_labels, duration_reasons)

    # Verify that the two repeated v2 reference methods remain byte-for-byte stable locally.
    prior = ROOT / "data/runs/second_trial/predictions"
    prior_checks = {}
    for name in ("reference_smooth50", "v2_shape", "v2_shape_motion"):
        old = prior / (name + ".csv")
        if old.is_file():
            old_labels = pd.read_csv(old)["label"].to_numpy(dtype=str)
            prior_checks[name] = bool(np.array_equal(old_labels, models[name][0]))
            if not prior_checks[name]:
                raise ValueError("{} no longer reproduces the saved v2 predictions".format(name))

    from evaluation import possession_metrics as pm
    context_table = {"frame": np.arange(n)}
    for family in ("shape", "motion"):
        context_table.update({name: context[family][:, i]
                              for i, name in enumerate(context[family + "_names"])})
    context_table.update({
        "quiet": clues["quiet"], "stationary_ball": clues["stationary"],
        "missing_ball_and_quiet": clues["missing_quiet"],
        "stationary_ball_and_quiet": clues["stationary_quiet"],
        "live_restart": clues["live_restart"], "set_piece_window": clues["tag_window"],
        "ball_cluster_span_m": clues["ball_span_m"],
        "ball_travel_speed_mps": clues["ball_travel_speed_mps"],
        "dead_candidate_without_tags": enhanced_dead_mask(clues, blocks, fps, config)[0],
        "dead_candidate_with_tags": enhanced_dead_mask(clues, blocks, fps, config,
                                                        use_restart_tags=True)[0],
    })
    pd.DataFrame(context_table).to_csv(diagnostics / "v3_context_features.csv", index=False,
                                      float_format="%.4f")
    manifest = {
        "experiment": "FirstTrial v3 — stoppage handling and spell stability",
        "game": GAME, "frames": n, "fps": fps,
        "report_design": ("Fixed ablations: compare soft vs forced dead-ball handling, minimum-duration "
                          "decoding, and cautious set-piece support. No parameter search or automatic winner."),
        "method_descriptions": METHODS, "base_config": base,
        "context_config": context_config, "v3_config": config,
        "minimum_duration_frames": minimum_frames,
        "base_config_sha256": sha256(args.base_config),
        "context_config_sha256": sha256(args.context_config),
        "v3_config_sha256": sha256(args.config),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__},
        "input_sha256": {key: sha256(paths[key])
                         for key in ("atd_meta", "atd_raw", "smart_tagging")},
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha256(p)
                          for p in sorted((ROOT / "solution").glob("*.py"))},
        "proxy_fitting": {"shape": shape_fit, "motion": motion_fit,
                          "public_answers_used": False},
        "smart_tagging_clock_check": time_check,
        "stoppage_detection": {
            "candidate_frames_without_tags": int(enhanced_dead_mask(clues, blocks, fps, config)[0].sum()),
            "candidate_frames_with_set_piece_support": int(enhanced_dead_mask(
                clues, blocks, fps, config, use_restart_tags=True)[0].sum()),
            "set_piece_window_frames": int(clues["tag_window"].sum()),
            "new_dead_team_cost": config["new_dead_team_cost"],
            "hard_override_is_compared_separately": True,
        },
        "saved_v2_reproduction_checks": prior_checks,
        "limitations": [
            "Every score uses the same five-minute public example already used in v1 and v2; this is not independent validation.",
            "There are only 17 labelled possessions in that example. Small score differences may come from one possession.",
            "Set-piece tags are automatic phase windows. They are not exact whistles, fouls, cards, substitutions, or confirmed dead-ball labels.",
            "Collective stillness and ball position can misread slow live play. Hard overrides can force false D labels.",
            "The minimum-duration decoder uses future frames and is an offline method, not a live real-time predictor.",
            "Track slots are anonymous fragments; no named-player roles or permanent identities are inferred.",
            "No full-match labels are available, so full-match accuracy and the winning component combination remain unknown.",
        ], "models": {},
    }
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

    # Freeze predictions, settings, inputs, and source hashes before the public labels are opened.
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not args.skip_score:
        subprocess.run([sys.executable, str(ROOT / "evaluation/run_evaluation.py"), "--gt", "public",
            "--results", str(predictions), "--out", str(out / "scoring"),
            "--bootstrap", str(args.bootstrap), "--no-detail"], cwd=ROOT, check=True)
    write_report(ROOT, out, manifest, {name: pair[0] for name, pair in models.items()},
                 not args.skip_score)
    print("Local v3 report: " + str(out / "comparison.html"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
