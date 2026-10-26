"""Line-restart dead ball: exits, waiting takers, restarts."""
import unittest
import numpy as np

from solution.line_restarts import line_restart_dead

PITCH = (105.0, 68.0)
FPS = 25


def scene(n=1500):
    """Ten players jogging back and forth in midfield; the ball moving back and forth through the middle."""
    t = np.arange(n) / FPS
    players = np.stack([np.tile(30 + 4 * np.arange(10.0), (n, 1)),
                        np.tile(34 + 8 * np.sin(t / 2)[:, None], (1, 10))], axis=2)
    ball = np.stack([52.5 + 30 * np.sin(t / 4), np.full(n, 34.0)], axis=1)
    return ball, players, np.ones(n, int)


class LineRestartTests(unittest.TestCase):
    def test_live_play_inside_the_pitch_adds_nothing(self):
        ball, players, period = scene()
        self.assertFalse(line_restart_dead(ball, players, period, PITCH, FPS).any())

    def test_ball_over_touchline_then_throw_in(self):
        ball, players, period = scene()
        ball[500:510] = [[60.0, 68.0 + 0.2 * k] for k in range(10)]   # crosses the touchline
        ball[510:800] = np.nan                                          # out of view
        players[520:, :, 1] = players[519, :, 1]                        # players stand
        ball[800:] = np.nan
        ball[800:830] = np.c_[np.full(30, 60.0), 67.0 - np.arange(30) * 8.0 / FPS]  # thrown in
        dead = line_restart_dead(ball, players, period, PITCH, FPS)
        on = np.flatnonzero(dead)
        self.assertEqual(on[0], 503)            # first frame 0.5 m beyond, two frames in a row
        self.assertTrue(770 <= on[-1] <= 801)  # ends at the throw (or the players speeding up)

    def test_ball_briefly_out_and_straight_back_adds_nothing(self):
        ball, players, period = scene()
        ball[500:504] = [[60.0, 68.6]] * 4
        self.assertFalse(line_restart_dead(ball, players, period, PITCH, FPS).any())

    def test_waiting_taker_without_ball(self):
        ball, players, period = scene()
        ball[600:900] = np.nan
        players[600:900, 0] = [70.0, 68.5]   # was on the pitch, now standing beyond the touchline
        players[620:880, 1:, 1] = players[619, 1:, 1]
        dead = line_restart_dead(ball, players, period, PITCH, FPS)
        self.assertTrue(dead[650:850].all())

    def test_substitute_warming_up_is_ignored(self):
        ball, players, period = scene()
        players = np.concatenate([players, np.full((len(players), 1, 2), [[50.0, 70.0]])], axis=1)  # never on the pitch
        ball[600:900] = np.nan
        self.assertFalse(line_restart_dead(ball, players, period, PITCH, FPS).any())

    def test_throw_in_seen_only_when_it_is_taken(self):
        ball, players, period = scene()
        ball[480:800] = np.nan                                          # the exit is never seen
        players[500:, :, 1] = players[499, :, 1]                        # players stand
        ball[800:] = np.nan
        ball[800:830] = np.c_[np.full(30, 60.0), 67.9 - np.arange(30) * 8.0 / FPS]
        dead = line_restart_dead(ball, players, period, PITCH, FPS)
        self.assertTrue(dead[530:800].all())
        self.assertFalse(dead[801:].any())

    def test_corner_kick(self):
        ball, players, period = scene()
        players[500:, :, 1] = players[499, :, 1]
        ball[480:900] = [[104.4, 67.4]] * 420                           # ball placed at the corner arc
        ball[900:] = np.nan
        ball[900:925] = np.c_[104.4 - np.arange(25) * 15.0 / FPS, 67.4 - np.arange(25) * 5.0 / FPS]
        dead = line_restart_dead(ball, players, period, PITCH, FPS)
        self.assertTrue(dead[700:899].all())

    def test_live_pass_along_the_touchline_adds_nothing(self):
        ball, players, period = scene()
        ball[600:640] = np.c_[40 + np.arange(40) * 0.3, np.full(40, 67.9)]  # dribble on the line
        ball[640:660] = np.c_[52 + np.zeros(20), 67.9 - np.arange(20) * 8.0 / FPS]
        self.assertFalse(line_restart_dead(ball, players, period, PITCH, FPS).any())

    def test_timing_options_shorten_stretches(self):
        ball, players, period = scene()
        ball[500:510] = [[60.0, 68.0 + 0.2 * k] for k in range(10)]
        ball[510:] = np.nan
        ball[700:720] = [[60.0, 50.0]] * 20                              # ball seen back inside, not yet kicked
        players[520:, :, 1] = players[519, :, 1]
        base = line_restart_dead(ball, players, period, PITCH, FPS)
        early = line_restart_dead(ball, players, period, PITCH, FPS, timing={'end_on_inside': True})
        capped = line_restart_dead(ball, players, period, PITCH, FPS, timing={'max_s': 12.0, 'end_trim_s': 2.0})
        self.assertEqual(int(base.sum()), 30 * FPS)
        self.assertEqual(np.flatnonzero(early)[-1], 699)
        self.assertEqual(int(capped.sum()), 10 * FPS)


if __name__ == '__main__':
    unittest.main()
