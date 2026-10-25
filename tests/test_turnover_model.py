"""Touch timing and decoding for the turnover-aware possession model."""
import unittest
import numpy as np

from solution.turnover_model import _age, _shift, decode


class TurnoverModelTests(unittest.TestCase):
    def test_close_age_counts_since_and_until_contact(self):
        close = np.array([0, 1, 0, 0, 1, 0], bool)
        self.assertEqual(np.nan_to_num(_age(close, True), nan=-1).tolist(), [-1, 0, 1, 2, 0, 1])
        self.assertEqual(np.nan_to_num(_age(close, False), nan=-1).tolist(), [1, 0, 2, 1, 0, -1])

    def test_shift_stays_inside_period(self):
        values = np.arange(10, dtype=np.float32)
        self.assertEqual(np.nan_to_num(_shift(values, 2, 2, 6), nan=-1).tolist(), [4, 5, -1, -1])
        self.assertEqual(np.nan_to_num(_shift(values, -1, 2, 6), nan=-1).tolist(), [-1, 2, 3, 4])

    def test_low_switch_cost_keeps_a_short_confident_spell(self):
        p = np.r_[np.full(50, 0.9), np.full(15, 0.05), np.full(50, 0.9)]
        period, dead = np.ones(len(p), int), np.zeros(len(p), bool)
        self.assertIn('A', decode(p, period, dead, 0.25))
        self.assertNotIn('A', decode(p, period, dead, 4.0))

    def test_dead_frames_are_forced_and_reset_the_team(self):
        p = np.r_[np.full(30, 0.9), np.full(10, 0.5), np.full(30, 0.1)]
        dead = np.zeros(70, bool); dead[30:40] = True
        labels = decode(p, np.ones(70, int), dead, 4.0)
        self.assertEqual(''.join(labels), 'H'*30 + 'D'*10 + 'A'*30)


if __name__ == '__main__':
    unittest.main()
