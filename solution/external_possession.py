"""Event-derived H/A/D/X possession labels for the Metrica sample matches.

This is an approximate external test set, not corrected-tracking ground truth.
Dead periods reuse the V7 stop/restart rules; live frames take the team of the
latest controlling event. CHALLENGE, BALL OUT, FAULT RECEIVED and CARD carry a
team but do not show control, so they never change the owner. Frames before a
period's first event and after its END HALF marker are ungraded (X).
"""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from .v7_labels import is_live_action, is_stop, period_runs, _kind, _subtype

CONTROL_EVENTS = {'PASS', 'CARRY', 'RECOVERY', 'SET PIECE', 'SHOT', 'BALL LOST'}


def attach_teams(matches, sample_root):
    """Add 'team' ('H'/'A'/None) to each loaded event, in loader order."""
    data = Path(sample_root)/'data'
    for match in matches:
        if match.name == 'Sample_Game_3':
            game = data/'Sample_Game_3'
            score = ET.parse(game/'Sample_Game_3_metadata.xml').getroot().find('.//Score')
            home_id, away_id = score.get('idLocalTeam'), score.get('idVisitingTeam')
            rows = json.loads((game/'Sample_Game_3_events.json').read_text(encoding='utf-8'))['data']
            codes = {home_id: 'H', away_id: 'A'}
            teams = [codes.get((row.get('team') or {}).get('id')) for row in rows]
        else:
            path = data/match.name/(match.name+'_RawEventsData.csv')
            codes = {'Home': 'H', 'Away': 'A'}
            teams = [codes.get(str(team)) for team in pd.read_csv(path)['Team']]
        if len(teams) != len(match.events):
            raise ValueError(f'{match.name}: event team count differs from loaded events')
        for event, team in zip(match.events, teams):
            event['team'] = team
    return matches


def is_control(event):
    return _kind(event) in CONTROL_EVENTS and event.get('team') in ('H', 'A')


def possession_labels(events, period):
    """Return per-frame 'H'/'A'/'D'/'X' labels and an audit summary.

    Event frames are one-based and inclusive, as in v7_labels.build_targets.
    A stop marker opens D at its end frame; the next live action's start
    closes it. Possession changes at the start of the new team's first
    controlling event. Leading live frames of an episode take the owner of its
    first controlling event, because every restart has a clear possessor.
    """
    period = np.asarray(period)
    labels = np.full(len(period), 'X', dtype='<U1')
    audit = {'dead_intervals': 0, 'unrestarted_stops': 0, 'live_frames_without_owner': 0}
    for a, b in period_runs(period):
        if period[a] <= 0:
            continue
        local = []
        for e in events:
            if e.get('period') != period[a] or e.get('start_frame') is None:
                continue
            start = int(e['start_frame'])-1
            if a <= start < b:
                end = max(start, int(e.get('end_frame') or e['start_frame'])-1)
                local.append((start, min(end, b-1), e))
        if not local:
            continue
        local.sort(key=lambda row: (row[0], row[1]))
        first, last = local[0][0], b
        action_starts = np.array([s for s, _, e in local if is_live_action(e)], dtype=int)
        dead = np.zeros(b-a, bool)
        for _, stop, e in local:
            if not is_stop(e):
                continue
            words = _subtype(e)
            if 'END' in words and 'HALF' in words:
                last = min(last, stop)
                continue
            future = action_starts[action_starts > stop]
            if len(future):
                dead[stop-a:int(future[0])-a] = True
                audit['dead_intervals'] += 1
            else:
                last = min(last, stop)
                audit['unrestarted_stops'] += 1
        owner = np.full(b-a, '', dtype='<U1')
        for start, _, e in local:
            if is_control(e):
                owner[start-a] = e['team']
        # Forward fill inside each live episode, then backfill its leading frames.
        graded = np.arange(a, b)
        graded = (graded >= first) & (graded < last)
        for lo, hi in _runs(~dead & graded):
            current = ''
            for t in range(lo, hi):
                current = owner[t] or current
                owner[t] = current
            known = np.flatnonzero(owner[lo:hi] != '')
            if len(known):
                owner[lo:lo+known[0]] = owner[lo+known[0]]
        segment = labels[a:b]
        live = ~dead & graded
        segment[live] = owner[live]
        segment[dead & graded] = 'D'
        missing = live & (owner == '')
        segment[missing] = 'X'
        audit['live_frames_without_owner'] += int(missing.sum())
    counts = {k: int((labels == k).sum()) for k in 'HADX'}
    return labels, {**audit, 'label_counts': counts}


def _runs(mask):
    padded = np.r_[False, np.asarray(mask, bool), False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2]))
