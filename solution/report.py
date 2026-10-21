"""A local comparison page. Answer labels are read only after prediction/scoring."""

import csv
from html import escape
import json
from pathlib import Path

import numpy as np

COLORS = {"H": "#448aff", "A": "#ef6376", "D": "#8a96a6", "X": "#e9b84c"}


def timeline_svg(names, labels, start, stop, fps):
    width, left, right, row_h = 1040, 205, 25, 42
    plot_width = width - left - right
    height = 50 + row_h * len(names)
    elements = ['<svg viewBox="0 0 {} {}" role="img" aria-label="Public five-minute label comparison">'.format(width, height)]
    for row, (name, values) in enumerate(zip(names, labels)):
        y = 25 + row * row_h
        elements.append('<text x="0" y="{}" fill="#d7e2ee" font-size="14">{}</text>'.format(y + 16, escape(name)))
        local = np.asarray(values)[start:stop]
        boundaries = np.r_[0, np.flatnonzero(local[1:] != local[:-1]) + 1, len(local)]
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            x = left + a / len(local) * plot_width
            w = (b - a) / len(local) * plot_width
            label = str(local[a])
            title = '{}: {:.2f}–{:.2f} s from the public-window start'.format(label, a / fps, b / fps)
            elements.append('<rect x="{:.3f}" y="{}" width="{:.3f}" height="24" fill="{}"><title>{}</title></rect>'.format(
                x, y, w, COLORS[label], escape(title)))
    elements.append('<text x="{}" y="{}" fill="#a8b7ca" font-size="12">0 seconds</text>'.format(left, height - 4))
    elements.append('<text x="{}" y="{}" fill="#a8b7ca" text-anchor="end" font-size="12">{:.0f} seconds</text>'.format(
        width - right, height - 4, (stop - start) / fps))
    elements.append('</svg>')
    return "\n".join(elements)


def write_report(root, out, manifest, model_labels, scoring_enabled):
    """Prediction has already finished and its hashes are fixed before this is called."""
    out = Path(out)
    rows = []
    svg = ""
    if scoring_enabled:
        with (out / "scoring" / "scoreboard.csv").open(newline="", encoding="utf-8") as handle:
            by_name = {row["submission"]: row for row in csv.DictReader(handle)}
        for name in model_labels:
            row = by_name[name]
            rows.append({"model": name, "spellF1": float(row["spell_f1_mean"]),
                         "macroF1": float(row["macro_f1"]), "public_possessions": int(row["poss"])})
        # This read is reporting-only. It cannot affect any prediction above.
        import pandas as pd
        gt = pd.read_csv(Path(root) / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv")["label"].to_numpy()
        covered = np.flatnonzero(gt != "X")
        if len(covered):
            names = ["Public correct labels"] + list(model_labels)
            values = [gt] + list(model_labels.values())
            svg = timeline_svg(names, values, int(covered[0]), int(covered[-1]) + 1, manifest["fps"])

    summary = {
        "experiment": "FirstTrial v1", "comparison_scope": "public five-minute example only" if scoring_enabled else "not scored",
        "automatically_selected_winner": False, "models": rows,
        "limitations": [
            "Only five minutes of correct labels are available; these scores do not prove full-match performance.",
            "Thresholds are initial engineering choices, not trained or calibrated values.",
            "The sequence model uses future observations as well as past observations (offline analysis).",
            "Slow collective movement plus a long ball gap can still mistake slow live play for a stoppage.",
            "Turnover timing inside a fully unobserved gap remains uncertain.",
            "Broadcast, Smart Tagging, player roles, and convex-hull features are not used in v1.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    heading = "| Method | Public spellF1 | Public macroF1 | Public possessions |\n|---|---:|---:|---:|"
    table = "\n".join("| {model} | {spellF1:.4f} | {macroF1:.4f} | {public_possessions} |".format(**row) for row in rows)
    notes = "\n".join("- " + note for note in summary["limitations"])
    markdown = "# FirstTrial v1 — local experiment\n\nNo model was trained and no parameter search was performed.\n\n"
    markdown += (heading + "\n" + table + "\n\n") if rows else "Scoring was skipped.\n\n"
    markdown += "## What the four methods test\n\n"
    markdown += "- `reference_smooth50`: unchanged official baseline.\n"
    markdown += "- `gap_sequence`: carry team evidence through missing observations using a sequence model.\n"
    markdown += "- `gap_stop_sequence`: add several stoppage clues to that same model.\n"
    markdown += "- `gap_stop_sequence_x`: same decoded states, with X only inside short, stable gaps.\n\n"
    markdown += "## Limits\n\n" + notes + "\n\nResults and diagnostic files are local and ignored by Git.\n"
    (out / "REPORT.md").write_text(markdown, encoding="utf-8")

    html_rows = "".join('<tr><td>{}</td><td>{:.4f}</td><td>{:.4f}</td><td>{}</td></tr>'.format(
        escape(row["model"]), row["spellF1"], row["macroF1"], row["public_possessions"]) for row in rows)
    legend = " ".join('<span style="color:{}">● {}</span>'.format(color, name) for name, color in COLORS.items())
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FirstTrial v1 comparison</title><style>
body{margin:0;background:#0d1522;color:#d7e2ee;font:16px/1.6 system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:32px 24px}h1{line-height:1.2}h2{margin-top:30px}.note{background:#273042;padding:16px;border-left:4px solid #e9b84c;border-radius:8px}table{width:100%;border-collapse:collapse;margin:20px 0}td,th{text-align:left;padding:10px;border-bottom:1px solid #344054}th{color:#92c7ff}.timeline{overflow-x:auto;padding:12px 0}.timeline svg{width:100%;min-width:780px}.legend{display:flex;gap:22px}.muted{color:#a8b7ca}code{color:#92c7ff}li{margin:8px 0}</style><main>
<h1>FirstTrial v1</h1><p class="muted">Tactical tracking only · four fixed comparisons · no training or parameter search</p>
<p class="note"><strong>These are public-example scores.</strong> Five minutes is too little to choose a full-match winner. Keep every result, including worse ones.</p>'''
    html += '<table><thead><tr><th>Method</th><th>spellF1</th><th>macroF1</th><th>Public possessions</th></tr></thead><tbody>' + html_rows + '</tbody></table>' if rows else '<p>Scoring was skipped.</p>'
    if svg:
        html += '<h2>Where the labels differ</h2><div class="legend">' + legend + '</div><div class="timeline">' + svg + '</div>'
        html += '<p class="muted">Each colored block is a period with the same label. More tiny blocks mean more switching. X is shown as written in the CSV; the main scorer fills it before counting spells.</p>'
    html += '<h2>What to check next</h2><ul>' + ''.join('<li>' + escape(note) + '</li>' for note in summary['limitations']) + '</ul>'
    html += '<p class="muted">Each method has a frame-by-frame explanation under <code>diagnostics/</code>. Inputs, settings, package versions, and file hashes are in <code>manifest.json</code>. Nothing was pushed.</p></main></html>'
    (out / "comparison.html").write_text(html, encoding="utf-8")
    return summary
