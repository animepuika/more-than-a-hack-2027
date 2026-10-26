"""High-ball features and labels."""
import unittest
import numpy as np

from solution.aerial_model import aerial_features, training_rows


class AerialTests(unittest.TestCase):
    def test_steady_long_flight_looks_different_from_slowing_roll(self):
        n = 100
        t = np.arange(n)
        flight = np.c_[10 + 0.8 * t, np.full(n, 30.)]                 # 20 m/s, constant
        roll = np.c_[10 + 0.8 * t * np.exp(-t / 30), np.full(n, 30.)]  # decelerating
        players = np.full((n, 4, 2), 80.)
        a = aerial_features(flight, players, np.ones(n, int), 25)
        r = aerial_features(roll, players, np.ones(n, int), 25)
        self.assertLess(a['speed_std_12'].iloc[50], r['speed_std_12'].iloc[50])
        self.assertGreater(a['travel_25'].iloc[50], r['travel_25'].iloc[50])
        self.assertAlmostEqual(float(a['straightness_25'].iloc[50]), 1.0, places=2)

    def test_labels_only_where_height_is_known(self):
        table = aerial_features(np.ones((20, 2)), np.full((20, 2, 2), 5.), np.ones(20, int), 25)
        z = np.r_[np.full(10, np.nan), np.full(5, 0.3), np.full(5, 3.0)]
        x, y = training_rows(table, z, stride=1)
        self.assertEqual(len(x), 10)
        self.assertEqual(y.tolist(), [0]*5 + [1]*5)


if __name__ == '__main__':
    unittest.main()
