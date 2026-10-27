"""Anti-split cleanup rules."""
import unittest
import numpy as np

from solution.split_cleanup import tidy

FPS = 25


def seq(*parts):
    return np.array([label for label, seconds in parts for _ in range(int(seconds * FPS))], dtype='<U1')


class TidyTests(unittest.TestCase):
    def setUp(self):
        self.period = lambda labels: np.ones(len(labels), int)

    def test_nothing_changes_with_zero_lengths(self):
        labels = seq(('H', 5), ('A', 1), ('H', 5), ('D', 1), ('H', 5))
        self.assertTrue((tidy(labels, self.period(labels), FPS) == labels).all())

    def test_short_flip_goes_back(self):
        labels = seq(('H', 5), ('A', 1), ('H', 5))
        self.assertTrue((tidy(labels, self.period(labels), FPS, flip_s=2) == 'H').all())

    def test_long_flip_stays(self):
        labels = seq(('H', 5), ('A', 3), ('H', 5))
        self.assertTrue((tidy(labels, self.period(labels), FPS, flip_s=2) == labels).all())

    def test_flip_before_a_stoppage_goes_back(self):
        labels = seq(('H', 5), ('A', 1), ('D', 5))
        out = tidy(labels, self.period(labels), FPS, flip_s=2)
        self.assertTrue((out[:6 * FPS] == 'H').all())

    def test_short_dead_piece_inside_a_possession_is_dropped(self):
        labels = seq(('H', 5), ('D', 1), ('H', 5))
        self.assertTrue((tidy(labels, self.period(labels), FPS, short_dead_s=2) == 'H').all())

    def test_short_dead_piece_at_a_change_is_kept(self):
        labels = seq(('H', 5), ('D', 1), ('A', 5))
        self.assertTrue((tidy(labels, self.period(labels), FPS, short_dead_s=2) == labels).all())

    def test_dead_stretches_close_together_are_joined(self):
        labels = seq(('H', 5), ('D', 4), ('A', 1), ('D', 4), ('H', 5))
        out = tidy(labels, self.period(labels), FPS, dead_gap_s=2)
        self.assertTrue((out[5 * FPS:14 * FPS] == 'D').all())

    def test_halves_are_separate(self):
        labels = seq(('H', 5), ('A', 1), ('H', 5))
        period = np.ones(len(labels), int); period[5 * FPS:] = 2
        out = tidy(labels, period, FPS, flip_s=2)
        self.assertTrue((out[5 * FPS:6 * FPS] == 'A').all())  # the flip starts the second half


if __name__ == '__main__':
    unittest.main()
