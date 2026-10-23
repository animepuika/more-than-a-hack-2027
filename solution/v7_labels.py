"""Event-derived weak supervision with unknown spans and explicit live actions.

Event semantics: Metrica sample-data/documentation/events-definitions.pdf.
RECOVERY may restart play; GOAL KICK is an attempt subtype, not necessarily a
SET PIECE. The targets are not corrected-tracking ground truth.
"""
import re
import numpy as np


def period_runs(period):
    period = np.asarray(period)
    if not len(period):
        return []
    edges = np.r_[0, np.flatnonzero(period[1:] != period[:-1]) + 1, len(period)]
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def _kind(event):
    return str(event.get('type', '')).strip().upper()


def _subtype(event):
    return re.sub(r'[-_]+', ' ', str(event.get('subtype', '')).upper()).split()


def is_stop(event):
    kind, words = _kind(event), _subtype(event)
    if kind in {'BALL OUT', 'FAULT RECEIVED', 'OFFSIDE', 'GOAL'}:
        return True
    if 'OFFSIDE' in words or 'FORCED' in words or ('END' in words and 'HALF' in words):
        return True
    return kind == 'SHOT' and ('GOAL' in words or 'OUT' in words)


def is_live_action(event):
    return _kind(event) in {'PASS', 'RECOVERY', 'SET PIECE', 'SHOT', 'BALL LOST', 'BALL OUT'}


def build_targets(events, period, fps=25.0, boundary_margin_s=0.2, live_bridge_s=8.0):
    """Return -1 unknown, 0 weak live, 1 weak dead, plus an audit summary.

Stop timestamps use event end; restart timestamps use first subsequent ball
action start. Label interiors only; ambiguous boundaries are excluded from fit.
Live examples come from actions/short event chains, never all unmarked frames.
"""
    period = np.asarray(period)
    y = np.full(len(period), -1, dtype=np.int8)
    live = np.zeros(len(period), bool)
    dead = np.zeros(len(period), bool)
    margins = np.zeros(len(period), bool)
    margin = int(round(boundary_margin_s * fps))
    max_bridge = int(round(live_bridge_s * fps))
    episodes = []
    for a, b in period_runs(period):
        if period[a] <= 0:
            continue
        local = []
        for e in events:
            if e.get('period') != period[a] or e.get('start_frame') is None:
                continue
            start = int(e['start_frame']) - 1
            if a <= start < b:
                end = max(start, int(e.get('end_frame') or e['start_frame']) - 1)
                local.append((start, min(end, b-1), e))
        local.sort(key=lambda row: (row[0], row[1]))
        actions = [(s, e, row) for s, e, row in local if is_live_action(row)]
        stops = [(e, row) for s, e, row in local if is_stop(row)]
        for s, e, row in actions:
            # Events ending with a whistle are live only up to, not at, that end.
            end = e if is_stop(row) else e + 1
            live[s:max(s+1, end)] = True
        for (s, e, row), (ns, _, _) in zip(actions, actions[1:]):
            if 0 <= ns-e <= max_bridge and not any(e <= t < ns for t, _ in stops):
                live[e:ns+1] = True
        action_starts = np.array([s for s, _, _ in actions], dtype=int)
        for stop, row in stops:
            future = action_starts[action_starts > stop]
            end_half = 'END' in _subtype(row) and 'HALF' in _subtype(row)
            if len(future):
                restart = int(future[0])
            elif end_half:
                restart = b
            else:
                continue  # no evidence of the next boundary: leave unknown
            dead[stop:restart] = True
            episodes.append((stop, restart))
            margins[max(a, stop-margin):min(b, stop+margin+1)] = True
            margins[max(a, restart-margin):min(b, restart+margin+1)] = True
    conflict = live & dead
    y[live & ~dead] = 0
    y[dead & ~live] = 1
    y[margins | conflict] = -1
    pass_frames = [int(e['start_frame'])-1 for e in events if _kind(e) == 'PASS'
                   and e.get('start_frame') is not None and 0 < int(e['start_frame']) <= len(y)]
    if np.any(y[pass_frames] == 1):
        raise AssertionError('A pass start is still labeled dead')
    return y, {
        'frames': len(y), 'weak_live': int((y == 0).sum()),
        'weak_dead': int((y == 1).sum()), 'unknown': int((y < 0).sum()),
        'conflict_frames_excluded': int(conflict.sum()),
        'boundary_frames_excluded': int(margins.sum()),
        'stop_to_action_candidates': len(episodes), 'pass_starts_labeled_dead': 0,
        'boundary_margin_s': boundary_margin_s, 'maximum_live_bridge_s': live_bridge_s,
    }
