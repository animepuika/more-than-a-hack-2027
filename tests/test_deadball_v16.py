"""V16 ball clues and realistic hiding."""
import unittest
import numpy as np

from solution.deadball_v16 import ball_clues, hide_ball_realistic


class BallClueTests(unittest.TestCase):
    def test_still_ball_at_corner_then_kick(self):
        ball = np.r_[np.tile([[104.5, 0.5]], (60, 1)), np.c_[104.5 - 0.5*np.arange(1, 41), np.full(40, 0.5)]]
        clues = ball_clues(ball, np.ones(100, int), (105., 68.), 25)
        self.assertLess(clues['ball_corner_m'].iloc[10], 1)
        self.assertGreater(clues['ball_still_age'].iloc[55], 40)
        self.assertEqual(clues['ball_kick_from_still'].iloc[62], 1)
        self.assertLess(clues['ball_touchline_m'].iloc[10], 1)

    def test_clues_are_blank_where_ball_missing(self):
        ball = np.tile([[50., 30.]], (50, 1)); ball[10:20] = np.nan
        clues = ball_clues(ball, np.ones(50, int), (105., 68.), 25)
        self.assertTrue(clues.iloc[10:20].isna().all().all())

    def test_realistic_hiding_reaches_both_shares(self):
        ball = np.ones((400, 2))
        labels = np.array(['H']*200 + ['D']*200)
        out = hide_ball_realistic(ball, labels, np.ones(400, int), np.array([10]), seed=0,
                                  live_share=0.3, dead_share=0.35)
        hidden = np.isnan(out).any(axis=1)
        self.assertGreaterEqual(hidden[:200].mean(), 0.3)
        self.assertGreaterEqual(hidden[200:].mean(), 0.35)


class ControlledPossessionTests(unittest.TestCase):
    def test_brief_opposing_touch_is_not_a_change(self):
        from solution.run_v16 import controlled_possession
        labels = np.array(list('HHHHHAAHHHHH' + 'HHHHA' + 'DDD' + 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'))
        out = controlled_possession(labels, np.ones(len(labels), int), fps=5, min_s=1.0)
        self.assertEqual(''.join(out[:12]), 'H'*12)          # 2-frame touch between H spells
        self.assertEqual(''.join(out[12:17]), 'HHHHH')       # touch just before a stoppage
        self.assertTrue((out[20:] == 'A').all())             # a real possession after the restart stays


if __name__ == '__main__':
    unittest.main()
