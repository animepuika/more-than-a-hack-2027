"""Regression cases from the audit, without BAR-ATH answer labels."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from solution.v7_labels import build_targets
from solution.v7_features import build_features, model_columns
from solution.v7_model import interval_metrics
from solution.v7_sequence import control_memory, joint_decode

CONFIG = json.loads((Path(__file__).resolve().parents[1]/'solution/config_v7.json').read_text())


def event(kind, start, end=None, subtype='', period=1):
    return dict(type=kind, start_frame=start, end_frame=end or start, subtype=subtype, period=period)


def match(n=100):
    xy = np.tile(np.array([[15+p*3, 10+p*4] for p in range(11)] +
                         [[65+p*2, 10+p*4] for p in range(11)], np.float32), (n, 1, 1))
    return SimpleNamespace(player_xy_m=xy, home_slots=np.r_[np.ones(11, bool), np.zeros(11, bool)],
        ball_xy_m=np.tile([30., 30.], (n, 1)), period=np.ones(n, int), fps=25., pitch=(105., 68.))


def evidence(owner):
    owner = np.asarray(owner)
    n = len(owner)
    return {'home_nearest_ball_m': np.where(owner == 0, .5, 4.),
            'away_nearest_ball_m': np.where(owner == 1, .5, 4.),
            'ball_reliable': (owner >= 0).astype(float), 'home_players': np.full(n, 11),
            'away_players': np.full(n, 11), 'home_ball_relative_speed_mps': np.zeros(n),
            'away_ball_relative_speed_mps': np.zeros(n)}


class LabelRepairTests(unittest.TestCase):
    def test_goal_kick_and_recovery_end_dead_without_set_piece(self):
        es = [event('BALL OUT', 10, 12), event('PASS', 40, 44, 'GOAL KICK'),
              event('PASS', 55, 60), event('FAULT RECEIVED', 70), event('RECOVERY', 90)]
        y, audit = build_targets(es, np.ones(110), boundary_margin_s=0)
        self.assertTrue((y[12:39] == 1).all())
        self.assertEqual(y[39], -1)  # boundary excluded even with zero extra padding
        self.assertEqual(y[42], 0)
        self.assertEqual(y[89], -1)
        self.assertEqual(audit['pass_starts_labeled_dead'], 0)

    def test_goal_subtype_and_unknown_tail(self):
        es = [event('SHOT', 10, 20, 'HEAD-ON TARGET-GOAL'), event('SET PIECE', 50),
              event('PASS', 50, 60)]
        y, _ = build_targets(es, np.ones(100), boundary_margin_s=0)
        self.assertEqual(y[30], 1)
        self.assertEqual(y[55], 0)
        self.assertEqual(y[99], -1)

    def test_card_and_handball_without_whistle_are_not_stop_targets(self):
        es = [event('CARD', 10), event('CHALLENGE', 20, subtype='HANDBALL'), event('SET PIECE', 80)]
        y, _ = build_targets(es, np.ones(100))
        self.assertFalse((y == 1).any())

    def test_no_target_crosses_a_period(self):
        y, _ = build_targets([event('BALL OUT', 10, 12), event('PASS', 65, 70, period=2)],
                             np.r_[np.ones(50), np.full(50, 2)])
        self.assertFalse((y[:50] == 1).any())


class FeatureRepairTests(unittest.TestCase):
    def test_v4_direct_control_guard_survives_absent_slots(self):
        from solution.run_v4 import pitch_control_costs
        from solution.data import Tracking
        folder = Path(__file__).resolve().parents[1]/'solution'
        c4 = json.loads((folder/'config_v4.json').read_text())
        c2 = json.loads((folder/'config_v2.json').read_text())
        base = json.loads((folder/'config.json').read_text())
        xy = np.tile([[.5, 0], [np.nan, np.nan], [4., 0], [np.nan, np.nan]], (2, 1, 1))
        tracking = Tracking(25., (105., 68.), np.zeros((2, 2)), np.full(2, .5), np.full(2, 4.),
                            np.full(2, 2), np.full(2, 2), np.zeros(2), np.ones(2), xy,
                            np.array([1, 1, 0, 0], bool))
        with patch('solution.run_v4.unique_track_positions', return_value=xy), \
             patch('solution.run_v4.fragment_motion', return_value=(np.zeros_like(xy), None, None)), \
             patch('solution.run_v4.estimate_ball', return_value=(np.zeros((2, 2)), np.zeros(2, bool))), \
             patch('solution.run_v4.team_reach_probabilities', return_value=(np.tile([.2, .8], (2, 1)), np.ones(2, bool))):
            costs, _, meta, _ = pitch_control_costs(tracking, {'active': np.ones(2, bool)},
                                                   [(1, 0, 1)], c2, {}, c4, base)
        self.assertEqual(meta['clear_direct_ball_control_frames_protected'], 2)
        np.testing.assert_array_equal(costs, np.zeros((2, 2)))

    def test_half_discontinuity_does_not_make_acceleration_or_stale_ball(self):
        m = match(100); m.period[50:] = 2
        m.player_xy_m[50:, :, 0] += 8
        m.ball_xy_m[50:] = np.nan
        f, _ = build_features(m, CONFIG)
        self.assertTrue(np.isnan(f['home_near_last_ball'][50]))
        valid = f['acceleration_q90_mps2'][np.isfinite(f['acceleration_q90_mps2'])]
        self.assertTrue((valid <= CONFIG['maximum_acceleration_mps2']).all())

    def test_old_ball_and_low_coverage_are_not_geometry_evidence(self):
        m = match(); m.ball_xy_m[10:] = np.nan; m.player_xy_m[70:, :8] = np.nan
        f, _ = build_features(m, CONFIG)
        self.assertTrue(np.isnan(f['home_near_last_ball'][40]))
        self.assertTrue(np.isnan(f['home_hull_area_m2'][80]))

    def test_dead_features_are_invariant_to_team_swap(self):
        m = match(60)
        f1, _ = build_features(m, CONFIG)
        m.home_slots = ~m.home_slots
        f2, _ = build_features(m, CONFIG)
        np.testing.assert_allclose(model_columns(f1).to_numpy(), model_columns(f2).to_numpy(),
                                   equal_nan=True, atol=1e-4)


class ScoreRepairTests(unittest.TestCase):
    def test_one_frame_overlap_is_not_a_match(self):
        y = np.zeros(1000, np.int8); y[:500] = 1
        p = np.zeros(1000, bool); p[499:900] = True
        self.assertEqual(interval_metrics(y, p, np.ones(1000))['interval_f1_mean'], 0)

    def test_unknowns_and_half_edges_are_respected(self):
        y = np.r_[np.ones(10), np.full(10, -1), np.ones(10)].astype(np.int8)
        out = interval_metrics(y, np.ones(30, bool), np.r_[np.ones(20), np.full(10, 2)])
        self.assertEqual(out['interval_f1_mean'], 1)
        self.assertEqual(out['by_iou']['0.5']['target_segments'], 2)


class TransferTests(unittest.TestCase):
    def test_brief_opponent_touch_does_not_change_control(self):
        owners = np.r_[np.zeros(30), np.ones(3), np.zeros(30)]
        labels, _, _ = control_memory(evidence(owners), np.ones(len(owners)), CONFIG)
        self.assertTrue((labels == 'H').all())

    def test_confirmed_control_backdates_and_resets_after_dead(self):
        owners = np.r_[np.zeros(30), np.ones(30), np.zeros(30)]
        dead = np.zeros(90, bool); dead[60:70] = True
        labels, _, _ = control_memory(evidence(owners), np.ones(90), CONFIG, dead=dead)
        self.assertTrue((labels[:30] == 'H').all())
        self.assertTrue((labels[30:60] == 'A').all())
        self.assertTrue((labels[60:70] == 'D').all())
        self.assertTrue((labels[70:] == 'H').all())

    def test_missing_ball_does_not_create_dead(self):
        owners = np.r_[np.zeros(25), np.full(80, -1), np.zeros(25)]
        labels, _, _ = control_memory(evidence(owners), np.ones(len(owners)), CONFIG)
        self.assertTrue((labels == 'H').all())

    def test_equal_team_context_cannot_raise_dead_score(self):
        f = evidence(np.zeros(60))
        labels, score = joint_decode(f, np.full(60, .05), np.ones(60), CONFIG, context_logit=np.zeros(60))
        self.assertTrue((labels == 'H').all())
        np.testing.assert_allclose(score, .05)


if __name__ == '__main__':
    unittest.main()
