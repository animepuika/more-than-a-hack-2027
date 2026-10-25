"""Event-derived external possession labels, on synthetic event logs."""
from types import SimpleNamespace
import unittest
import numpy as np

from solution.ball_masking import gap_lengths, hide_live_ball
from solution.external_possession import possession_labels
from solution.run_external import baseline_codes, merge_short_spells, pooled


def event(kind, start, end=None, team='H', subtype='', period=1):
    return dict(type=kind, start_frame=start, end_frame=end or start, subtype=subtype,
                period=period, team=team)


def text(labels):
    return ''.join(labels)


class ExternalPossessionTests(unittest.TestCase):
    def test_owner_changes_at_new_team_control_not_at_challenge(self):
        es = [event('SET PIECE', 1, team='H', subtype='KICK OFF'), event('PASS', 1, 4),
              event('CHALLENGE', 6, team='A', subtype='GROUND-LOST'),
              event('BALL LOST', 7, team='H'), event('RECOVERY', 9, team='A'),
              event('PASS', 10, 12, team='A')]
        labels, _ = possession_labels(es, np.ones(14, int))
        self.assertEqual(text(labels), 'HHHHHHHHAAAAAA')

    def test_stop_to_restart_is_dead_and_restart_team_owns(self):
        es = [event('PASS', 1, 3, team='H'), event('BALL OUT', 4, 6, team='H'),
              event('SET PIECE', 10, team='A', subtype='THROW IN'), event('PASS', 10, 12, team='A')]
        labels, audit = possession_labels(es, np.ones(14, int))
        self.assertEqual(text(labels), 'HHHHHDDDDAAAAA')
        self.assertEqual(audit['dead_intervals'], 1)

    def test_before_first_event_and_after_end_half_are_ungraded(self):
        es = [event('SET PIECE', 3, team='A', subtype='KICK OFF'), event('PASS', 3, 5, team='A'),
              event('BALL OUT', 8, 8, team=None, subtype='END HALF')]
        labels, _ = possession_labels(es, np.ones(12, int))
        self.assertEqual(text(labels), 'XXAAAAAXXXXX')

    def test_periods_are_independent(self):
        es = [event('PASS', 1, 2, team='H', period=1), event('PASS', 6, 7, team='A', period=2)]
        labels, _ = possession_labels(es, np.r_[np.ones(4, int), np.full(4, 2)])
        self.assertEqual(text(labels), 'HHHHXAAA')

    def test_episode_without_controlling_event_stays_ungraded(self):
        es = [event('CHALLENGE', 1, team='H'), event('CHALLENGE', 3, team='A')]
        labels, audit = possession_labels(es, np.ones(5, int))
        self.assertEqual(text(labels), 'XXXXX')
        self.assertEqual(audit['live_frames_without_owner'], 5)


class ExternalBenchHelperTests(unittest.TestCase):
    def test_merge_short_spells_only_between_same_team(self):
        labels = np.array(list('HHHAHHHDAHH'))
        self.assertEqual(text(merge_short_spells(labels, 2)), 'HHHHHHHDAHH')

    def test_pooled_matches_are_separated_by_ungraded_frame(self):
        self.assertEqual(text(pooled([np.array(list('HH')), np.array(list('HA'))])), 'HHXHA')

    def test_baseline_codes_use_nearest_team_and_nobody_without_ball(self):
        tracking = SimpleNamespace(home_distance_m=np.array([1., 3., np.nan, np.nan]),
                                   away_distance_m=np.array([2., 1., 1., np.nan]))
        self.assertEqual(baseline_codes(tracking).tolist(), [0, 1, 1, -1])


class BallMaskingTests(unittest.TestCase):
    def test_gap_lengths_reset_per_period_and_skip_inactive(self):
        seen = np.array([1, 0, 0, 1, 0, 0, 0, 0], bool)
        period = np.array([1, 1, 1, 1, 1, 2, 2, 0])
        self.assertEqual(gap_lengths(seen, period).tolist(), [2, 1, 2])

    def test_hiding_reaches_target_without_crossing_periods(self):
        ball = np.ones((40, 2))
        live = np.ones(40, bool)
        period = np.r_[np.ones(20, int), np.full(20, 2)]
        masked, hidden, added = hide_live_ball(ball, live, period, np.array([30]), 0.6, seed=0)
        self.assertGreaterEqual(hidden.mean(), 0.6)
        self.assertTrue(np.isnan(masked[hidden]).all() and np.isfinite(masked[~hidden]).all())
        self.assertGreater(added, 1)  # 30-frame gaps are clipped at each 20-frame period

    def test_zero_target_changes_nothing(self):
        ball = np.ones((5, 2))
        masked, hidden, added = hide_live_ball(ball, np.ones(5, bool), np.ones(5, int), [3], 0, 1)
        self.assertEqual((hidden.sum(), added), (0, 0))


if __name__ == '__main__':
    unittest.main()
