"""Replaying a missing-player count sequence onto tracking."""
import unittest
import numpy as np

from solution.player_masking import drop_players


class PlayerMaskingTests(unittest.TestCase):
    def test_hidden_count_follows_sequence_and_players_persist(self):
        xy = np.ones((200, 22, 2), np.float32)
        missing = np.r_[np.zeros(50, int), np.full(100, 3), np.ones(50, int)]
        out = drop_players(xy, np.ones(200, int), missing, seed=0)
        hidden = np.isnan(out).any(axis=2)
        counts = hidden.sum(axis=1)
        self.assertEqual(sorted(set(counts.tolist())), [0, 1, 3])
        rows = np.flatnonzero(counts == 3)
        same = [np.array_equal(hidden[rows[i]], hidden[rows[i+1]]) for i in range(len(rows) - 1)
                if rows[i+1] == rows[i] + 1]
        self.assertTrue(all(same))   # the same players stay hidden while the count holds
        self.assertTrue(np.isfinite(xy).all())   # input untouched

    def test_never_hides_more_than_are_visible(self):
        xy = np.full((10, 5, 2), np.nan, np.float32); xy[:, :2] = 1
        out = drop_players(xy, np.ones(10, int), np.full(10, 4), seed=1)
        self.assertTrue(np.isnan(out).all())


class EdgeWeightTests(unittest.TestCase):
    def test_players_near_lines_vanish_far_more_often(self):
        rng = np.random.RandomState(0)
        xy = np.zeros((2000, 22, 2), np.float32)
        xy[:, :11] = [50, 34]                       # centre of the pitch
        xy[:, 11:] = [104, 34]                      # 1 m from the goal line
        missing = np.tile([0, 1], 1000)             # a fresh choice every other frame
        out = drop_players(xy, np.ones(2000, int), missing, seed=3, pitch=(105., 68.))
        hidden = np.isnan(out).any(axis=2)
        self.assertGreater(hidden[:, 11:].sum(), 2 * hidden[:, :11].sum())

    def test_uniform_default_is_unchanged(self):
        xy = np.ones((50, 22, 2), np.float32)
        a = drop_players(xy, np.ones(50, int), np.full(50, 2), seed=4)
        b = drop_players(xy, np.ones(50, int), np.full(50, 2), seed=4)
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b))


if __name__ == '__main__':
    unittest.main()
