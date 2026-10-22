import json
from pathlib import Path
import unittest

import numpy as np

from solution.hmm_smoothing import make_model, observation_symbols, smooth_costs
from solution.pitch_control import (intercept_probability, probability_costs,
                                    team_reach_probabilities, time_to_intercept)


CONFIG = json.loads((Path(__file__).parents[1] / "solution/config_v4.json").read_text(encoding="utf-8"))


class PitchControlMathTest(unittest.TestCase):
    def test_time_to_intercept_includes_reaction_and_acceleration(self):
        position = np.array([0.0, 0.0])
        target = np.array([2.0, 0.0])
        stationary = time_to_intercept(position, np.array([0.0, 0.0]), target)
        moving_toward = time_to_intercept(position, np.array([2.0, 0.0]), target)
        moving_away = time_to_intercept(position, np.array([-2.0, 0.0]), target)
        self.assertGreater(stationary, 0.7)
        self.assertLess(moving_toward, stationary)
        self.assertGreater(moving_away, stationary)

    def test_arrival_probability_is_monotone_and_nan_safe(self):
        position = np.array([[0.0, 0.0], [np.nan, 0.0]])
        velocity = np.zeros((2, 2))
        target = np.array([[0.0, 0.0], [1.0, 0.0]])
        probability = intercept_probability(position, velocity, target, horizon_s=1.0)
        self.assertGreater(probability[0], 0.5)
        self.assertLess(probability[1], 0.01)
        self.assertTrue(np.isfinite(probability).all())

    def test_team_probabilities_are_normalized_and_unsupported_targets_are_neutral(self):
        xy = np.array([[[0.0, 0.0], [4.0, 0.0]],
                       [[np.nan, np.nan], [np.nan, np.nan]]])
        velocity = np.zeros_like(xy)
        targets = np.array([[0.0, 0.0], [np.nan, np.nan]])
        probs, supported = team_reach_probabilities(
            xy, velocity, targets, np.array([True, False]), CONFIG)
        self.assertTrue(supported[0])
        self.assertFalse(supported[1])
        np.testing.assert_allclose(probs.sum(axis=1), 1.0)
        np.testing.assert_allclose(probs[1], [0.5, 0.5])
        self.assertGreater(probs[0, 0], probs[0, 1])

    def test_costs_are_weak_and_zero_when_ineligible(self):
        probs = np.array([[0.9, 0.1], [0.1, 0.9]])
        cost = probability_costs(probs, np.array([True, False]), weight=0.2)
        self.assertGreater(cost[0, 1], cost[0, 0])
        np.testing.assert_array_equal(cost[1], [0.0, 0.0])


class HMMSmoothingTest(unittest.TestCase):
    def test_unknown_observations_do_not_force_home(self):
        costs = np.zeros((4, 3))
        symbols = observation_symbols(costs, CONFIG["weak_confidence_margin"],
                                     CONFIG["medium_confidence_margin"])
        np.testing.assert_array_equal(symbols, np.full(4, 9))

    def test_transition_and_emission_rows_are_probabilities(self):
        model = make_model(CONFIG)
        np.testing.assert_allclose(model.transmat_.sum(axis=1), 1.0)
        np.testing.assert_allclose(model.emissionprob_.sum(axis=1), 1.0)

    def test_hmm_suppresses_one_frame_flip_but_keeps_sustained_change(self):
        costs = np.tile([0.0, 2.0, 4.0], (50, 1))
        costs[20] = [3.0, 0.0, 4.0]
        costs[30:45] = [3.0, 0.0, 4.0]
        blocks = [(1, 0, 49)]
        result = smooth_costs(costs, blocks, CONFIG)
        self.assertEqual(result[20], 0)
        self.assertTrue(np.all(result[30:45] == 1))

    def test_each_half_is_decoded_independently(self):
        costs = np.tile([0.0, 2.0, 4.0], (60, 1))
        costs[30:] = [2.0, 0.0, 4.0]
        blocks = [(1, 0, 29), (2, 30, 59)]
        result = smooth_costs(costs, blocks, CONFIG)
        self.assertTrue(np.all(result[:30] == 0))
        self.assertTrue(np.all(result[30:] == 1))


if __name__ == "__main__":
    unittest.main()
