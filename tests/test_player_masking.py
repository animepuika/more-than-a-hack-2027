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


if __name__ == '__main__':
    unittest.main()
