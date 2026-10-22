"""Load Metrica's public sample matches and derive weak dead-ball targets.

The sample data is kept under the ignored ``data/external`` directory. Event
windows are supervision signals, not frame-perfect ground truth: a stop marker
starts a candidate dead-ball interval and a SET PIECE event ends it.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


SAMPLE_REPOSITORY = "https://github.com/metrica-sports/sample-data"
DEFAULT_SAMPLE_ROOT = Path(__file__).resolve().parents[1] / "data/external/metrica-sample-data"
PITCH_M = (105.0, 68.0)
FPS = 25.0


@dataclass
class ExternalMatch:
    name: str
    player_xy_m: np.ndarray  # frames x track slots x (x, y)
    home_slots: np.ndarray
    ball_xy_m: np.ndarray   # frames x (x, y)
    period: np.ndarray
    events: list
    fps: float = FPS
    pitch: tuple = PITCH_M

    def __len__(self):
        return len(self.ball_xy_m)


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _event_is_stop_marker(event):
    kind = event["type"].strip().upper()
    subtype = event["subtype"].strip().upper()
    if kind in {"BALL OUT", "FAULT RECEIVED", "CARD", "GOAL"}:
        return True
    if kind in {"OFFSIDE", "HAND BALL", "HANDBALL"}:
        return True
    # Do not treat SHOT subtypes such as ON TARGET-GOAL as scored goals.
    return kind == "SHOT" and subtype.strip() in {"GOAL", "GOAL SCORED", "SCORED"}


def weak_dead_ball_labels(events, period, n_frames):
    """Build conservative stop-to-restart targets from synchronized event logs.

    Frame identifiers in Metrica event logs are one-based. The stop marker's
    final frame is included; the first SET PIECE frame is live and excluded.
    Frames around unmarked periods remain negative only as a weak assumption,
    so callers should describe the labels as event-derived weak supervision.
    """
    labels = np.zeros(n_frames, dtype=bool)
    by_period = {}
    for event in events:
        if event.get("start_frame") is None or event.get("period") is None:
            continue
        by_period.setdefault(int(event["period"]), []).append(event)

    for period_id, period_events in by_period.items():
        period_events.sort(key=lambda row: (row.get("start_frame", 0), row.get("end_frame", 0)))
        restarts = [event for event in period_events if event["type"].strip().upper() == "SET PIECE"]
        markers = [event for event in period_events if _event_is_stop_marker(event)]
        previous_restart = -1
        for restart in restarts:
            restart_frame = int(restart["start_frame"])
            candidates = [event for event in markers
                          if previous_restart < int(event["start_frame"]) < restart_frame]
            if candidates:
                marker = min(candidates, key=lambda row: row["start_frame"])
                stop_frame = marker.get("end_frame") or marker.get("start_frame")
                # Convert one-based inclusive event frames to [start, stop).
                start = max(0, int(stop_frame) - 1)
                stop = min(n_frames, restart_frame - 1)
                if start < stop:
                    labels[start:stop] = True
            previous_restart = restart_frame

        # END HALF is an explicit stoppage marker and may have no later restart.
        end_markers = [event for event in period_events
                       if "END HALF" in event["subtype"].upper()]
        period_rows = np.flatnonzero(np.asarray(period) == period_id)
        if end_markers:
            if len(period_rows):
                marker = max(end_markers, key=lambda row: row["start_frame"])
                start = max(0, int(marker.get("end_frame") or marker["start_frame"]) - 1)
                labels[start:int(period_rows[-1]) + 1] = True

    return labels


def _standard_events(path):
    frame = pd.read_csv(path).fillna("")
    events = []
    for values in frame.to_dict("records"):
        start = _number(values.get("Start Frame"))
        end = _number(values.get("End Frame"))
        events.append({
            "type": str(values.get("Type", "")),
            "subtype": str(values.get("Subtype", "")),
            "period": int(_number(values.get("Period")) or 0),
            "start_frame": int(start) if np.isfinite(start) else None,
            "end_frame": int(end) if np.isfinite(end) else None,
        })
    return events


def _load_csv_game(game_dir, game_id):
    prefix = "Sample_Game_{}".format(game_id)
    team_rows = []
    ball_xy = None
    period = None
    for team_index, team in enumerate(("Home", "Away")):
        path = game_dir / (prefix + "_RawTrackingData_{}_Team.csv".format(team))
        if not path.is_file():
            raise FileNotFoundError(path)
        table = pd.read_csv(path, header=[0, 1, 2])
        frame_period = pd.to_numeric(table.iloc[:, 0], errors="coerce").to_numpy()
        frame_ids = pd.to_numeric(table.iloc[:, 1], errors="coerce").to_numpy()
        players = table.iloc[:, 3:-2].to_numpy(dtype=np.float64).reshape(len(table), -1, 2)
        local_ball = table.iloc[:, -2:].to_numpy(dtype=np.float64)
        if team_index == 0:
            period, ball_xy = frame_period.astype(np.int16), local_ball
            first_frame_ids = frame_ids
        elif (not np.array_equal(frame_ids, first_frame_ids) or
              not np.allclose(local_ball, ball_xy, equal_nan=True)):
            raise ValueError("{} home and away timelines/ball tracks do not align".format(prefix))
        team_rows.append(players)

    player_xy = np.concatenate(team_rows, axis=1)
    home_slots = np.r_[np.ones(team_rows[0].shape[1], dtype=bool),
                       np.zeros(team_rows[1].shape[1], dtype=bool)]
    scale = np.asarray(PITCH_M, dtype=np.float32)
    events = _standard_events(game_dir / (prefix + "_RawEventsData.csv"))
    return ExternalMatch(prefix, player_xy * scale, home_slots, ball_xy * scale,
                         period, events)


def _load_game3(game_dir):
    prefix = "Sample_Game_3"
    metadata_path = game_dir / (prefix + "_metadata.xml")
    tracking_path = game_dir / (prefix + "_tracking.txt")
    events_path = game_dir / (prefix + "_events.json")
    if not all(path.is_file() for path in (metadata_path, tracking_path, events_path)):
        raise FileNotFoundError("Sample Game 3 tracking, metadata, and events are required")

    root = ET.parse(metadata_path).getroot()
    player_teams = {player.get("id"): player.get("teamId")
                    for player in root.findall(".//Players/Player")}
    channel_teams = {}
    for channel in root.iter():
        if not channel.tag.endswith("PlayerChannel"):
            continue
        match = re.fullmatch(r"player(\d+)_x", channel.get("id", ""))
        player_id = channel.get("playerId")
        if match and player_id in player_teams:
            channel_teams[int(match.group(1))] = player_teams[player_id]
    # Metadata also lists substitutes who have no coordinate channel in the
    # tracking text. The text itself has 22 anonymous on-pitch slots.
    slot_count = 22
    if not set(range(1, slot_count + 1)).issubset(channel_teams):
        raise ValueError("Sample Game 3 metadata does not map all 22 tracking slots")
    team_ids = sorted(set(channel_teams.values()))
    if len(team_ids) != 2:
        raise ValueError("Sample Game 3 metadata must describe two teams")
    # The EPTS text contains 22 active tracks in team-grouped order; roster
    # metadata includes substitutes but does not map those 22 columns one to
    # one. The first 11 and final 11 tracks form the two sides in the file.
    home_slots = np.r_[np.ones(11, dtype=bool), np.zeros(11, dtype=bool)]

    frames, players, balls = [], [], []
    with tracking_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            parts = line.strip().split(":")
            if len(parts) != 3:
                raise ValueError("Unexpected Sample Game 3 row {}".format(line_no))
            frames.append(int(parts[0]))
            player_rows = [[_number(axis) for axis in value.split(",")]
                           for value in parts[1].split(";")]
            if len(player_rows) != slot_count or any(len(row) != 2 for row in player_rows):
                raise ValueError("Unexpected player coordinate count in Sample Game 3 row {}".format(line_no))
            players.append(np.asarray(player_rows, dtype=np.float32))
            ball_values = [_number(value) for value in parts[2].split(",")]
            if len(ball_values) < 2:
                raise ValueError("Unexpected ball coordinate count in Sample Game 3 row {}".format(line_no))
            balls.append(ball_values[:2])

    frames = np.asarray(frames)
    if not np.array_equal(frames, np.arange(1, len(frames) + 1)):
        raise ValueError("Sample Game 3 tracking frame IDs are not consecutive")
    n_frames = len(frames)
    config = {item.findtext("Name"): item.findtext("Value")
              for item in root.findall(".//ProviderParameter")}
    first_start = int(config.get("first_half_start") or 1)
    first_end = int(config.get("first_half_end") or 0)
    period = np.where(frames <= first_end, 1, 2).astype(np.int8)

    raw_events = json.loads(events_path.read_text(encoding="utf-8"))["data"]
    events = []
    for row in raw_events:
        subtype = row.get("subtypes")
        if isinstance(subtype, list):
            subtype_name = " ".join(str(item.get("name", "")) for item in subtype)
        elif isinstance(subtype, dict):
            subtype_name = str(subtype.get("name", ""))
        else:
            subtype_name = ""
        start = row.get("start") or {}
        end = row.get("end") or {}
        events.append({
            "type": str((row.get("type") or {}).get("name", "")),
            "subtype": subtype_name,
            "period": int(row.get("period") or 0),
            "start_frame": int(start["frame"]) if start.get("frame") is not None else None,
            "end_frame": int(end["frame"]) if end.get("frame") is not None else None,
        })

    scale = np.asarray(PITCH_M, dtype=np.float32)
    return ExternalMatch(prefix, np.asarray(players, dtype=np.float32) * scale, home_slots,
                         np.asarray(balls, dtype=np.float32) * scale, period, events)


def load_metrica_samples(sample_root=DEFAULT_SAMPLE_ROOT):
    """Load all three synchronized public Metrica sample games."""
    sample_root = Path(sample_root)
    data_dir = sample_root / "data"
    if not (sample_root / ".git").is_dir() and not data_dir.is_dir():
        raise FileNotFoundError(
            "Metrica public samples missing. Clone {} into {} before running V6.".format(
                SAMPLE_REPOSITORY, sample_root))
    return [
        _load_csv_game(data_dir / "Sample_Game_1", 1),
        _load_csv_game(data_dir / "Sample_Game_2", 2),
        _load_game3(data_dir / "Sample_Game_3"),
    ]
