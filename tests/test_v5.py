"""V5 abstention policy tests using small synthetic inputs."""

import unittest

import numpy as np

from solution.gap_policy import apply_missing_ball_policy, chaotic_movement_mask


CONFIG = {
    "minimum_reacting_players_each_team": 1,
    "minimum_total_reacting_players": 3,
    "minimum_one_team_reacting_players": 3,
    "one_team_speed_threshold_mps": 3.5,
    "both_teams_speed_threshold_mps": 3.5,
    "minimum_motion_tracks_per_team": 5,
}


class ChaosMaskTest(unittest.TestCase):
    def test_three_reacting_tracks_across_both_teams_flag_chaos(self):
        motion = np.zeros((1, 10), dtype=float)
        motion[0, 2] = 2 / 11
        motion[0, 7] = 1 / 11
        counts = np.array([[11, 11]])
        self.assertTrue(chaotic_movement_mask(motion, counts, CONFIG)[0])

    def test_one_team_burst_needs_three_players_and_fast_movement(self):
        motion = np.zeros((2, 10), dtype=float)
        motion[:, 0] = [3.6, 2.0]
        motion[:, 2] = [3 / 11, 3 / 11]
        motion[:, 5] = [2.0, 2.0]
        counts = np.full((2, 2), 11)
        np.testing.assert_array_equal(chaotic_movement_mask(motion, counts, CONFIG), [True, False])

    def test_missing_motion_coverage_is_not_called_chaotic(self):
        motion = np.full((1, 10), np.nan)
        counts = np.array([[4, 11]])
        self.assertFalse(chaotic_movement_mask(motion, counts, CONFIG)[0])


class MissingBallPolicyTest(unittest.TestCase):
    def test_calm_policy_abstains_only_on_missing_non_dead_frames(self):
        labels = np.array(["H", "H", "A", "D", "A", "X"])
        seen = np.array([True, False, False, False, True, False])
        active = np.array([True, True, True, True, True, False])
        chaos = np.array([False, False, True, False, False, False])
        output, mask = apply_missing_ball_policy(labels, seen, active, chaos, "calm")
        np.testing.assert_array_equal(mask, [False, True, False, False, False, False])
        np.testing.assert_array_equal(output, ["H", "X", "A", "D", "A", "X"])

    def test_all_missing_policy_does_not_change_visible_dead_or_inactive(self):
        labels = np.array(["H", "A", "D", "H"])
        seen = np.array([False, False, False, False])
        active = np.array([True, True, True, False])
        chaos = np.array([True, False, False, False])
        output, mask = apply_missing_ball_policy(labels, seen, active, chaos, "all")
        np.testing.assert_array_equal(mask, [True, True, False, False])
        np.testing.assert_array_equal(output, ["X", "X", "D", "H"])

    def test_inputs_are_not_mutated_and_shapes_are_checked(self):
        labels = np.array(["H"])
        output, _ = apply_missing_ball_policy(labels, [False], [True], [False], "calm")
        np.testing.assert_array_equal(labels, ["H"])
        np.testing.assert_array_equal(output, ["X"])
        with self.assertRaises(ValueError):
            apply_missing_ball_policy(labels, [False, False], [True], [False], "calm")


if __name__ == "__main__":
    unittest.main()
