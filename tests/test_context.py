"""Geometry, fragment continuity, and weak-evidence checks on invented data."""

import json
from pathlib import Path
import unittest

import numpy as np

from solution.context import (build_context, estimate_ball, fragment_motion,
                              hull_area, proxy_costs, tagging_costs, unique_positions)
from solution.data import Tracking
from solution.features import make_features
from solution.predict import predict

ROOT = Path(__file__).resolve().parents[1]
BASE = json.loads((ROOT / "solution/config.json").read_text())
CONFIG = json.loads((ROOT / "solution/config_v2.json").read_text())


def sample(n=100):
    data = Tracking(25, (100, 60), np.tile([40.0, 30.0], (n, 1)),
                    np.full(n, 0.5), np.full(n, 5.0), np.full(n, 22),
                    np.full(n, 22), np.full(n, 3.0), np.zeros(n))
    blocks = [(1, 0, n - 1)]
    return data, blocks


class GeometryTest(unittest.TestCase):
    def test_hull_handles_duplicates_interior_points_and_collinear_points(self):
        points = np.array([[0, 0], [10, 0], [10, 5], [0, 5], [2, 2], [0, 0]])
        self.assertEqual(hull_area(points), 50)
        self.assertEqual(hull_area(points + [100, 50]), 50)
        self.assertEqual(hull_area(points * 2), 200)
        self.assertEqual(hull_area([[0, 0], [1, 1], [2, 2]]), 0)

    def test_close_detections_do_not_expand_player_count(self):
        positions = unique_positions(np.array([[0, 0], [0.1, 0.1], [10, 5]]), 0.5)
        self.assertEqual(len(positions), 2)

    def test_both_teams_need_enough_tracks_for_shape(self):
        data, blocks = sample(25)
        home = np.column_stack((np.arange(8) * 3.0 + 10, np.arange(8) % 3 * 4.0 + 10))
        away = home + [30, 10]
        data.track_xy_m = np.tile(np.vstack([home, away]), (25, 1, 1)).astype(np.float32)
        data.home_slots = np.array([True] * 8 + [False] * 8)
        data.track_xy_m[15:, 12:] = np.nan
        context = build_context(data, make_features(data, blocks, BASE), blocks, CONFIG)
        self.assertTrue(np.isfinite(context["shape"][10]).all())
        self.assertTrue(np.isnan(context["shape"][20]).all())


class FragmentMovementTest(unittest.TestCase):
    def test_constant_velocity_has_zero_acceleration_after_smoothing(self):
        xy = np.zeros((50, 1, 2), dtype=np.float32)
        xy[:, 0, 0] = np.arange(50) * 2 / 25
        velocity, speed, acceleration = fragment_motion(xy, [(1, 0, 49)], 25, CONFIG)
        np.testing.assert_allclose(velocity[10:, 0, 0], 2, atol=1e-4)
        np.testing.assert_allclose(speed[10:, 0], 2, atol=1e-4)
        np.testing.assert_allclose(acceleration[10:, 0], 0, atol=1e-3)

    def test_gap_jump_and_half_boundaries_reset_motion(self):
        xy = np.zeros((90, 1, 2), dtype=np.float32)
        xy[:, 0, 0] = np.arange(90) / 25
        xy[20:22] = np.nan
        xy[40:, 0, 0] += 50
        _, speed, _ = fragment_motion(xy, [(1, 0, 54), (2, 60, 89)], 25, CONFIG)
        for start in (22, 40, 60):
            with self.subTest(start=start):
                self.assertTrue(np.isnan(speed[start:start + 5]).all())
                self.assertAlmostEqual(speed[start + 6, 0], 1, places=3)
        self.assertTrue(np.isnan(speed[55:60]).all())

    def test_acceleration_and_braking_signs(self):
        for sign, initial in ((1, 1), (-1, 9)):
            t = np.arange(50) / 25
            xy = np.zeros((50, 1, 2), dtype=np.float32)
            xy[:, 0, 0] = initial * t + sign * 0.5 * 4 * t ** 2
            _, _, accel = fragment_motion(xy, [(1, 0, 49)], 25, CONFIG)
            np.testing.assert_allclose(accel[15:35, 0], sign * 4, atol=0.003)


class MissingBallTest(unittest.TestCase):
    def test_interpolation_is_short_bounded_and_does_not_cross_halves(self):
        data, _ = sample()
        data.ball_xy_m[20:25] = np.nan
        data.ball_xy_m[0:3] = np.nan
        data.ball_xy_m[47:54] = np.nan
        blocks = [(1, 0, 49), (2, 50, 99)]
        features = make_features(data, blocks, BASE)
        ball, estimated = estimate_ball(data, features, blocks, CONFIG)
        self.assertTrue(estimated[20:25].all())
        self.assertFalse(estimated[:3].any())
        self.assertFalse(estimated[47:54].any())
        self.assertTrue(np.isnan(ball[47:54]).all())

    def test_stoppage_or_implausible_ball_travel_blocks_interpolation(self):
        data, blocks = sample()
        data.ball_xy_m[20:25] = np.nan
        features = make_features(data, blocks, BASE)
        features["outside_stop"][22] = True
        self.assertFalse(estimate_ball(data, features, blocks, CONFIG)[1].any())
        features["outside_stop"][:] = False
        data.ball_xy_m[25] = [90, 50]
        self.assertFalse(estimate_ball(data, features, blocks, CONFIG)[1].any())


class WeakEvidenceTest(unittest.TestCase):
    def test_proxy_fitting_uses_visible_anchors_and_leaves_strong_frames_untouched(self):
        data, blocks = sample()
        data.home_distance_m[50:] = 5
        data.away_distance_m[50:] = 0.5
        values = np.r_[np.zeros(50), np.full(50, 10)][:, None]
        data.ball_xy_m[[10, 60]] = np.nan
        data.home_distance_m[[10, 60]] = np.nan
        data.away_distance_m[[10, 60]] = np.nan
        features = make_features(data, blocks, BASE)
        config = dict(CONFIG, minimum_anchors_per_team=2)
        costs, metadata = proxy_costs(values, data, features, blocks, config, 0.35)
        self.assertGreater(costs[10, 1], costs[10, 0])
        self.assertGreater(costs[60, 0], costs[60, 1])
        self.assertTrue(np.all(costs[[0, 25, 50, 75]] == 0))
        self.assertEqual(metadata[0]["usable_features"], 1)

    def test_missing_anchors_do_not_invent_a_shape_prior(self):
        data, blocks = sample()
        features = make_features(data, blocks, BASE)
        costs, _ = proxy_costs(np.ones((100, 2)), data, features, blocks, CONFIG, 0.35)
        self.assertTrue(np.all(costs == 0))

    def test_defending_tags_are_inverted_and_unknown_or_restart_tags_are_ignored(self):
        features = {"active": np.ones(100, dtype=bool)}
        def event(code, team, start, stop):
            return {"code": code, "tags": {"Team": team}, "start_frame": start, "end_frame": stop}
        events = [event("DEFENDING IN MIDDLE THIRD", "Home Club", 0, 20),
                  event("BUILD UP", "Home Club", 20, 40),
                  event("SET PIECES", "Home Club", 40, 60),
                  event("GOALS", "Away Club", 60, 80),
                  event("BUILD UP", "N/A", 80, 100)]
        costs, info = tagging_costs(events, ("Home Club", "Away Club"), features, CONFIG)
        self.assertGreater(costs[10, 0], costs[10, 1])
        self.assertGreater(costs[30, 1], costs[30, 0])
        self.assertTrue(np.all(costs[40:] == 0))
        self.assertEqual(info["automatic_events_used"], 2)

    def test_conflicting_team_tags_cancel(self):
        features = {"active": np.ones(10, dtype=bool)}
        events = [{"code": "BUILD UP", "tags": {"Team": team}, "start_frame": 0, "end_frame": 10}
                  for team in ("Home", "Away")]
        costs, _ = tagging_costs(events, ("Home", "Away"), features, CONFIG)
        self.assertTrue(np.all(costs == 0))

    def test_zero_extra_costs_preserve_the_original_prediction(self):
        data, blocks = sample()
        features = make_features(data, blocks, BASE)
        first = predict(data, features, blocks, BASE)[0]
        second = predict(data, features, blocks, BASE, extra_team_costs=np.zeros((100, 2)))[0]
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
