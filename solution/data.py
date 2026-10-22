"""Read the provided tactical tracking in one streaming pass, without kloppy.

Only aggregate frame features are retained. Track slots are never treated as
named players, and no ground-truth or event labels are read here.
"""

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


@dataclass
class Tracking:
    fps: float
    pitch: tuple
    ball_xy_m: np.ndarray
    home_distance_m: np.ndarray
    away_distance_m: np.ndarray
    observed_players: np.ndarray
    speed_players: np.ndarray
    median_player_speed_mps: np.ndarray
    walking_fraction: np.ndarray
    track_xy_m: np.ndarray = None
    home_slots: np.ndarray = None

    def __len__(self):
        return len(self.ball_xy_m)


def load_tracking(meta_path, raw_path, n_frames, fps, pitch,
                  max_player_speed_mps=12.0, walking_speed_mps=1.4, progress=None,
                  retain_tracks=False):
    """Fail on frame/schema mismatches instead of silently shifting predictions."""
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    slots = list(root.iter("Player"))
    if len(teams) != 2 or not slots:
        raise ValueError("Tracking metadata must contain two teams and track slots")
    team_ids = [team.get("id") for team in teams]
    if len(set(team_ids)) != 2 or None in team_ids:
        raise ValueError("Tracking metadata has invalid team IDs")
    slot_ids = [slot.get("teamId") for slot in slots]
    if any(team_id not in team_ids for team_id in slot_ids):
        raise ValueError("A track slot has an unknown team")
    home_slots = np.array([team_id == team_ids[0] for team_id in slot_ids])
    export_fps = float(root.findtext(".//FrameRate"))
    if not np.isclose(export_fps, fps):
        raise ValueError("Tracking frame rate differs from the reference frame rate")

    ball_xy = np.full((n_frames, 2), np.nan, dtype=np.float64)
    home_dist = np.full(n_frames, np.nan)
    away_dist = np.full(n_frames, np.nan)
    observed = np.zeros(n_frames, dtype=np.int16)
    speed_count = np.zeros(n_frames, dtype=np.int16)
    median_speed = np.full(n_frames, np.nan)
    walking_fraction = np.full(n_frames, np.nan)
    scale = np.asarray(pitch, dtype=float)
    track_xy = np.full((n_frames, len(slots), 2), np.nan, dtype=np.float32) if retain_tracks else None
    count = 0

    with Path(raw_path).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            parts = line.strip().split(":")
            if len(parts) != 3 or index >= n_frames:
                raise ValueError("Unexpected tracking row structure/count at row {}".format(index + 1))
            if int(parts[0]) != index + 1:
                raise ValueError("Tracking IDs must be consecutive and start at 1")
            values = np.fromstring(parts[1].replace(";", ","), sep=",")
            ball = np.fromstring(parts[2], sep=",")
            if values.size != len(slots) * 3 or ball.size != 3:
                raise ValueError("Unexpected track/ball field count at row {}".format(index + 1))
            players = values.reshape(-1, 3)
            xy = players[:, :2] * scale
            if track_xy is not None:
                track_xy[index] = xy
            valid = np.isfinite(xy).all(axis=1)
            observed[index] = int(valid.sum())
            speed = players[:, 2]
            reliable_speed = valid & np.isfinite(speed) & (speed >= 0) & (speed <= max_player_speed_mps)
            speed_count[index] = int(reliable_speed.sum())
            if reliable_speed.any():
                good_speed = speed[reliable_speed]
                median_speed[index] = float(np.median(good_speed))
                walking_fraction[index] = float(np.mean(good_speed <= walking_speed_mps))
            if np.isfinite(ball[:2]).all():
                ball_xy[index] = ball[:2] * scale
                distance = np.linalg.norm(xy - ball_xy[index], axis=1)
                if (valid & home_slots).any():
                    home_dist[index] = float(distance[valid & home_slots].min())
                if (valid & ~home_slots).any():
                    away_dist[index] = float(distance[valid & ~home_slots].min())
            count += 1
            if progress is not None and count % 50000 == 0:
                progress(count, n_frames)

    if count != n_frames:
        raise ValueError("Expected {} tracking rows, got {}".format(n_frames, count))
    return Tracking(float(fps), tuple(pitch), ball_xy, home_dist, away_dist,
                    observed, speed_count, median_speed, walking_fraction, track_xy,
                    home_slots if retain_tracks else None)
