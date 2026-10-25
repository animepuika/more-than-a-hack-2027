"""Restart-formation features on synthetic player layouts."""
import unittest
import numpy as np

from solution.deadball_formation import _formation, formation_features

PITCH = (105., 68.)
HOME = np.r_[np.ones(11, bool), np.zeros(11, bool)]


def layout(home_xy, away_xy, n=1):
    return np.tile(np.r_[home_xy, away_xy][None].astype(np.float32), (n, 1, 1))


class FormationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(1)
        self.home = np.c_[rng.uniform(5, 50, 11), rng.uniform(5, 63, 11)]
        self.away = np.c_[rng.uniform(55, 100, 11), rng.uniform(5, 63, 11)]

    def features(self, xy, speed=None):
        speed = np.zeros(xy.shape[:2], np.float32) if speed is None else speed
        return {k: float(v[0]) for k, v in _formation(xy, HOME, speed, PITCH).items()}

    def test_kick_off_has_everyone_in_own_half(self):
        f = self.features(layout(self.home, self.away))
        self.assertEqual(f['own_half_share_min'], 1.0)
        pushed = self.home.copy(); pushed[:5, 0] = 70
        self.assertLess(self.features(layout(pushed, self.away))['own_half_share_min'], 1.0)

    def test_corner_taker_and_throw_in_lines(self):
        home = self.home.copy(); home[0] = (104.5, 0.5); home[1] = (40, -0.8)
        f = self.features(layout(home, self.away))
        self.assertLess(f['corner_flag_min_m'], 1.0)
        self.assertLess(f['touchline_min_m'], 0)
        self.assertEqual(f['players_outside'], 1)

    def test_wall_counts_still_close_teammates_only(self):
        away = self.away.copy(); away[:4] = [(80, 30), (80, 31), (80, 32), (80, 33)]
        speed = np.zeros((1, 22), np.float32)
        self.assertGreaterEqual(self.features(layout(self.home, away), speed)['still_cluster_max'], 4)
        speed[0, 11:15] = 5.0
        self.assertLess(self.features(layout(self.home, away), speed)['still_cluster_max'], 4)

    def test_goal_kick_box_counts_by_side(self):
        home = self.home.copy(); home[:, 0] = np.maximum(home[:, 0], 20)   # nobody else in the box
        home[:3] = [(3, 30), (6, 40), (10, 25)]   # home defends the left goal
        away = self.away.copy()
        f = self.features(layout(home, away))
        self.assertEqual((f['busy_box_defenders'], f['busy_box_attackers']), (3, 0))

    def test_features_cover_only_active_periods(self):
        xy = layout(self.home, self.away, n=60)
        table = formation_features(xy, HOME, np.r_[np.zeros(10, int), np.ones(50, int)], PITCH, 25)
        self.assertTrue(table.iloc[:10].isna().all().all())
        self.assertFalse(table['own_half_share_min'].iloc[10:].isna().any())


if __name__ == '__main__':
    unittest.main()
