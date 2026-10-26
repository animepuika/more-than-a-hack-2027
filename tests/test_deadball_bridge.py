"""Bridging dead ball across low-coverage blind spots."""
import unittest
import numpy as np

from solution.deadball_bridge import bridge_blind_spots

PITCH = (105., 68.)


def scenario(before_dead, after_dead, ball_out=False):
    n = 100
    dead = np.zeros(n, bool)
    covered = np.ones(n, bool); covered[40:60] = False
    if before_dead: dead[30:40] = True
    if after_dead: dead[60:70] = True
    ball = np.tile([[50., 34.]], (n, 1))
    if ball_out:
        ball[38] = [104.8, 34.]; ball[39:] = np.nan
    return dead, covered, ball


class BridgeTests(unittest.TestCase):
    def test_dead_on_both_sides_fills(self):
        dead, covered, ball = scenario(True, True)
        self.assertTrue(bridge_blind_spots(dead, covered, np.ones(100, int), ball, PITCH)[40:60].all())

    def test_live_on_both_sides_stays_live(self):
        dead, covered, ball = scenario(False, False, ball_out=True)
        self.assertFalse(bridge_blind_spots(dead, covered, np.ones(100, int), ball, PITCH)[40:60].any())

    def test_mixed_needs_the_ball_to_have_gone_out(self):
        dead, covered, ball = scenario(False, True)
        self.assertFalse(bridge_blind_spots(dead, covered, np.ones(100, int), ball, PITCH)[40:60].any())
        dead, covered, ball = scenario(False, True, ball_out=True)
        self.assertTrue(bridge_blind_spots(dead, covered, np.ones(100, int), ball, PITCH)[40:60].all())


if __name__ == '__main__':
    unittest.main()
