"""V6 feature and weak-label checks use small, known tracking examples."""

import unittest
from types import SimpleNamespace

import numpy as np

from solution.deadball_features import build_deadball_features
from solution.external_samples import weak_dead_ball_labels
from solution.v6_model import segment_metrics, smooth_dead_probability


class WeakEventLabelTests(unittest.TestCase):
    def test_ball_out_to_restart_is_positive_and_challenge_is_not_a_whistle(self):
        events = [
            {"type": "SET PIECE", "subtype": "KICK OFF", "period": 1,
             "start_frame": 1, "end_frame": 1},
            {"type": "BALL OUT", "subtype": "", "period": 1,
             "start_frame": 10, "end_frame": 12},
            {"type": "CHALLENGE", "subtype": "TACKLE-FAULT-WON", "period": 1,
             "start_frame": 15, "end_frame": 15},
            {"type": "SHOT", "subtype": "ON TARGET-GOAL", "period": 1,
             "start_frame": 18, "end_frame": 20},
            {"type": "SET PIECE", "subtype": "THROW IN", "period": 1,
             "start_frame": 30, "end_frame": 30},
        ]
        labels = weak_dead_ball_labels(events, np.ones(40), 40)
        self.assertTrue(labels[11])
        self.assertTrue(labels[28])
        self.assertFalse(labels[29])
        self.assertFalse(labels[:11].any())

    def test_fault_received_starts_a_separate_dead_ball_interval(self):
        events = [
            {"type": "FAULT RECEIVED", "subtype": "", "period": 2,
             "start_frame": 5, "end_frame": 6},
            {"type": "SET PIECE", "subtype": "FREE KICK", "period": 2,
             "start_frame": 20, "end_frame": 20},
        ]
        labels = weak_dead_ball_labels(events, np.full(25, 2), 25)
        self.assertTrue(labels[5:19].all())
        self.assertFalse(labels[:5].any())
        self.assertFalse(labels[19:].any())


class TrackingFeatureTests(unittest.TestCase):
    def test_velocity_boundary_zones_and_missing_ball_path_guard(self):
        fps, n, players = 25.0, 80, 22
        xy = np.zeros((n, players, 2), dtype=np.float32)
        # Place two realistic teams on opposite sides and keep them on-pitch.
        for p in range(players):
            xy[:, p, 0] = (20 + (p % 11) * 2) if p < 11 else (65 + (p % 11) * 2)
            xy[:, p, 1] = 15 + (p % 11) * 3
        xy[:, 0, 0] += np.arange(n, dtype=np.float32) * 0.2
        ball = np.column_stack([np.full(n, 80.0), np.full(n, 34.0)]).astype(np.float32)
        ball[30, 0] = 105.2  # full ball has crossed the goal line
        ball[45:51] = np.nan
        match = SimpleNamespace(player_xy_m=xy, ball_xy_m=ball,
                                home_slots=np.r_[np.ones(11, bool), np.zeros(11, bool)],
                                pitch=(105.0, 68.0), fps=fps)
        features = build_deadball_features(match)
        self.assertAlmostEqual(float(features["speed_max_mps"][10]), 5.0, delta=0.2)
        self.assertEqual(float(features["ball_entirely_outside"][30]), 1.0)
        self.assertEqual(float(features["ball_middle_third"][10]), 0.0)
        self.assertTrue(np.isnan(features["ball_path_length_0.4s_m"][47]))
        self.assertTrue(np.isfinite(features["home_hull_area_m2"][25]))
        self.assertTrue(np.isfinite(features["home_voronoi_share"][25]))


class SequencePostprocessingTests(unittest.TestCase):
    def test_hysteresis_bridges_short_holes_but_drops_single_spikes(self):
        probability = np.r_[np.zeros(20), np.full(30, 0.8), np.full(3, 0.1),
                           np.full(25, 0.8), np.zeros(20)]
        period = np.ones(len(probability), dtype=np.int8)
        prediction = smooth_dead_probability(probability, period, fps=25)
        self.assertTrue(prediction[25:75].all())
        spike = np.zeros(30)
        spike[10] = 0.99
        self.assertFalse(smooth_dead_probability(spike, np.ones(30), fps=25).any())

    def test_segment_metric_matches_one_to_one(self):
        target = np.zeros(30, dtype=bool)
        target[2:8] = True
        target[18:25] = True
        prediction = np.zeros(30, dtype=bool)
        prediction[3:7] = True
        prediction[19:24] = True
        result = segment_metrics(target, prediction)
        self.assertEqual(result["segment_precision"], 1.0)
        self.assertEqual(result["segment_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
