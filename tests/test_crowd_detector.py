"""Mixed-crowd stoppage detector on synthetic layouts."""
import unittest
import numpy as np

from solution.crowd_detector import CROWD_MIN_S, crowd_dead_mask

PITCH = (105., 68.)
HOME = np.r_[np.ones(11, bool), np.zeros(11, bool)]


def spread_out(n):
    rng = np.random.RandomState(3)
    base = np.c_[rng.uniform(5, 100, 22), rng.uniform(5, 63, 22)]
    base[:, 0] = np.linspace(5, 100, 22)  # nobody close to anyone else
    return np.tile(base[None].astype(np.float32), (n, 1, 1))


class CrowdTests(unittest.TestCase):
    def test_sustained_still_mixed_crowd_is_dead(self):
        n = 250
        xy = spread_out(n)
        crowd = [(50, 30), (51, 31), (52, 30), (50, 32), (53, 31), (51, 29)]
        xy[50:200, [0, 1, 2, 11, 12, 13]] = np.array(crowd, np.float32)
        mask = crowd_dead_mask(xy, HOME, np.ones(n, int), PITCH, 25)
        self.assertTrue(mask[60:190].all())
        self.assertFalse(mask[:45].any() or mask[205:].any())

    def test_one_team_cluster_or_short_crowd_is_not_dead(self):
        n = 250
        xy = spread_out(n)
        xy[50:200, [0, 1, 2, 3, 4, 5]] = np.array([(50, 30), (51, 31), (52, 30), (50, 32), (53, 31), (51, 29)], np.float32)
        self.assertFalse(crowd_dead_mask(xy, HOME, np.ones(n, int), PITCH, 25).any())
        xy = spread_out(n)
        short = int(CROWD_MIN_S * 25) - 10
        xy[50:50+short, [0, 1, 2, 11, 12, 13]] = np.array([(50, 30), (51, 31), (52, 30), (50, 32), (53, 31), (51, 29)], np.float32)
        self.assertFalse(crowd_dead_mask(xy, HOME, np.ones(n, int), PITCH, 25).any())

    def test_moving_crowd_is_not_dead(self):
        n = 250
        xy = spread_out(n)
        group = np.array([(50, 30), (51, 31), (52, 30), (50, 32), (53, 31), (51, 29)], np.float32)
        for t in range(50, 200):
            xy[t, [0, 1, 2, 11, 12, 13]] = group + [0.12 * (t - 50), 0]  # 3 m/s together
        self.assertFalse(crowd_dead_mask(xy, HOME, np.ones(n, int), PITCH, 25).any())


if __name__ == '__main__':
    unittest.main()
