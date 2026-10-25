"""V7 control memory with a fallback team during long ball gaps."""
import json
from pathlib import Path
import unittest
import numpy as np

from solution.gap_fallback import gap_fallback_memory, long_gap_mask, midpoint_fallback
from solution.v7_sequence import control_memory

CONFIG = json.loads((Path(__file__).resolve().parents[1]/'solution/config_v7.json').read_text())


def evidence(owner):
    """owner 0/1 controls the ball; -1 means the ball is missing."""
    owner = np.asarray(owner)
    n = len(owner)
    return {'home_nearest_ball_m': np.where(owner == 0, .5, 4.),
            'away_nearest_ball_m': np.where(owner == 1, .5, 4.),
            'ball_reliable': (owner >= 0).astype(float), 'home_players': np.full(n, 11),
            'away_players': np.full(n, 11), 'home_ball_relative_speed_mps': np.zeros(n),
            'away_ball_relative_speed_mps': np.zeros(n)}


class GapFallbackTests(unittest.TestCase):
    def test_long_gap_mask_resets_per_period(self):
        reliable = np.array([1, 0, 0, 0, 0, 1, 0, 0], bool)
        period = np.array([1, 1, 1, 1, 2, 2, 2, 2])
        self.assertEqual(long_gap_mask(reliable, period, 2).astype(int).tolist(),
                         [0, 1, 1, 1, 0, 0, 1, 1])

    def test_matches_v7_when_no_gap_is_long(self):
        owners = [0]*30 + [-1]*5 + [1]*30 + [-1]*3 + [0]*40
        fallback = np.full(len(owners), 'A')
        expected, _, _ = control_memory(evidence(owners), np.ones(len(owners)), CONFIG)
        labels, gaps = gap_fallback_memory(evidence(owners), np.ones(len(owners)), CONFIG, fallback, 2.0)
        self.assertFalse(gaps.any())
        self.assertEqual(labels.tolist(), expected.tolist())

    def test_long_gap_follows_fallback_and_hands_over_ownership(self):
        owners = [0]*30 + [-1]*60 + [1]*30
        fallback = np.array(['H']*30 + ['A']*90)
        v7, _, _ = control_memory(evidence(owners), np.ones(120), CONFIG)
        labels, _ = gap_fallback_memory(evidence(owners), np.ones(120), CONFIG, fallback, 2.0)
        self.assertEqual(''.join(v7), 'H'*90 + 'A'*30)
        self.assertEqual(''.join(labels), 'H'*30 + 'A'*90)

    def test_dead_frames_stay_dead_inside_long_gap(self):
        owners = [0]*30 + [-1]*60 + [0]*30
        dead = np.zeros(120, bool); dead[50:70] = True
        labels, _ = gap_fallback_memory(evidence(owners), np.ones(120), CONFIG,
                                        np.full(120, 'A'), 2.0, dead=dead)
        self.assertTrue((labels[50:70] == 'D').all())
        self.assertTrue((labels[30:50] == 'A').all() and (labels[70:90] == 'A').all())


class MidpointFallbackTests(unittest.TestCase):
    def test_switches_at_gap_midpoint_when_owners_differ(self):
        owners = [0]*30 + [-1]*60 + [1]*30
        held = np.array(['H']*120)
        fallback = midpoint_fallback(evidence(owners), np.ones(120), CONFIG, held, 2.0)
        self.assertEqual(''.join(fallback[30:90]), 'H'*30 + 'A'*30)
        self.assertTrue((fallback[:30] == '').all() and (fallback[90:] == '').all())
        labels, _ = gap_fallback_memory(evidence(owners), np.ones(120), CONFIG, fallback, 2.0)
        self.assertEqual(''.join(labels), 'H'*60 + 'A'*60)

    def test_brief_control_after_gap_is_not_the_new_owner(self):
        owners = [0]*30 + [-1]*60 + [1]*5 + [0]*25
        fallback = midpoint_fallback(evidence(owners), np.ones(120), CONFIG, np.array(['H']*120), 2.0)
        self.assertTrue((fallback[30:90] == 'H').all())

    def test_dead_ball_after_gap_stops_the_search(self):
        owners = [0]*30 + [-1]*60 + [2]*15 + [1]*15  # 2: ball visible, nobody in control
        dead = np.zeros(120, bool); dead[95:105] = True
        fallback = midpoint_fallback(evidence(owners), np.ones(120), CONFIG, np.array(['H']*120), 2.0,
                                     dead=dead)
        self.assertTrue((fallback[30:90] == 'H').all())


if __name__ == '__main__':
    unittest.main()
