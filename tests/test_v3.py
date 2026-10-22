"""Version 3 duration and stoppage behavior on invented tracking only."""

import json
from itertools import product
from pathlib import Path
import tempfile
import unittest

import numpy as np

from solution.duration import decode_minimum_duration
from solution.data import Tracking
from solution.features import make_features
from solution.predict import decode_sequence
from solution.run_v3 import read_v3_config
from solution.stoppages import (build_stoppage_features, enhanced_dead_mask,
                                restart_windows)

ROOT = Path(__file__).resolve().parents[1]
BASE = json.loads((ROOT / "solution/config.json").read_text(encoding="utf-8"))
V3 = json.loads((ROOT / "solution/config_v3.json").read_text(encoding="utf-8"))


def sample(n=120, speed=0.5, walking=1.0):
    return Tracking(25, (100, 60), np.tile([40.0, 30.0], (n, 1)),
                    np.full(n, 5.0), np.full(n, 8.0), np.full(n, 22),
                    np.full(n, 22), np.full(n, speed), np.full(n, walking))


class DurationDecoderTest(unittest.TestCase):
    def test_short_interior_turnover_is_removed_but_sustained_turnover_survives(self):
        costs = np.tile([0.0, 5.0, 8.0], (30, 1))
        costs[10:12] = [5.0, 0.0, 8.0]
        costs[20:] = [5.0, 0.0, 8.0]
        states = decode_minimum_duration(costs, 4.0, 2.5, [5, 5, 4])
        self.assertTrue(np.all(states[:20] == 0))
        self.assertTrue(np.all(states[20:] == 1))

    def test_duration_one_matches_unconstrained_minimum_cost_path(self):
        costs = np.array([[0, 5, 10], [5, 0, 10], [0, 5, 10], [5, 0, 10]], dtype=float)
        np.testing.assert_array_equal(decode_minimum_duration(costs, 1.5, 1.0, [1, 1, 1]),
                                      decode_sequence(costs, 1.5, 1.0))

    def test_end_segments_may_be_short_and_input_is_validated(self):
        costs = np.array([[0, 10, 10], [0, 10, 10], [10, 0, 10]], dtype=float)
        states = decode_minimum_duration(costs, 1.0, 1.0, [5, 5, 4])
        self.assertEqual(states[-1], 1)
        with self.assertRaises(ValueError):
            decode_minimum_duration(costs, 1.0, 1.0, [2, 0, 1])

    def test_decoder_matches_brute_force_minimum_cost_with_duration_rules(self):
        rng = np.random.default_rng(43)
        costs = rng.uniform(0, 4, size=(7, 3))
        team_switch, dead_switch = 1.7, 1.1
        minimum = [2, 3, 2]
        actual = decode_minimum_duration(costs, team_switch, dead_switch, minimum)
        transition = np.array([[0, team_switch, dead_switch],
                               [team_switch, 0, dead_switch],
                               [dead_switch, dead_switch, 0]])
        best = float("inf")
        for path in product(range(3), repeat=len(costs)):
            starts = [0] + [i for i in range(1, len(path)) if path[i] != path[i - 1]]
            stops = starts[1:] + [len(path)]
            if any(stop - start < minimum[path[start]]
                   for start, stop in zip(starts, stops)
                   if start != 0 and stop != len(path)):
                continue
            value = sum(costs[i, state] for i, state in enumerate(path))
            value += sum(transition[path[i - 1], path[i]] for i in range(1, len(path)))
            best = min(best, value)
        actual_value = sum(costs[i, state] for i, state in enumerate(actual))
        actual_value += sum(transition[actual[i - 1], actual[i]] for i in range(1, len(actual)))
        self.assertAlmostEqual(actual_value, best)


class V3StoppageTest(unittest.TestCase):
    def test_slow_stationary_ball_needs_persistent_evidence(self):
        data = sample(120)
        blocks = [(1, 0, 119)]
        features = make_features(data, blocks, BASE)
        clues = build_stoppage_features(data, features, blocks, V3)
        mask, _ = enhanced_dead_mask(clues, blocks, data.fps, V3)
        self.assertFalse(mask[:19].any())
        self.assertTrue(mask[60:100].all())

    def test_long_missing_ball_with_fast_players_is_not_a_dead_ball(self):
        data = sample(120, speed=3.0, walking=0.0)
        data.ball_xy_m[20:100] = np.nan
        data.home_distance_m[20:100] = np.nan
        data.away_distance_m[20:100] = np.nan
        blocks = [(1, 0, 119)]
        features = make_features(data, blocks, BASE)
        clues = build_stoppage_features(data, features, blocks, V3)
        mask, _ = enhanced_dead_mask(clues, blocks, data.fps, V3)
        self.assertFalse(mask.any())

    def test_live_restart_breaks_dead_ball_latch_and_confirmation_timer(self):
        n = 100
        clues = {name: np.zeros(n, dtype=bool) for name in
                 ("quiet", "stationary", "missing_quiet", "stationary_quiet", "live_restart",
                  "missing_seed", "stationary_seed", "tag_seed", "outside_seed", "tag_window")}
        clues["missing_quiet"][10:90] = True
        clues["missing_seed"][15] = True
        clues["live_restart"][50] = True
        mask, _ = enhanced_dead_mask(clues, [(1, 0, 99)], 25, V3)
        self.assertTrue(mask[10:50].all())
        self.assertFalse(mask[50:].any())

    def test_set_piece_windows_are_padded_and_clipped_to_scored_halves(self):
        events = [{"code": "SET PIECES", "start_frame": 10, "end_frame": 20},
                  {"code": "BUILD UP", "start_frame": 50, "end_frame": 60}]
        mask = restart_windows(events, [(1, 0, 99)], 100, 25, 1.0)
        self.assertTrue(mask[:45].all())
        self.assertFalse(mask[45:].any())

    def test_v3_config_has_valid_duration_and_fraction_settings(self):
        self.assertEqual(read_v3_config(ROOT / "solution/config_v3.json"), V3)
        bad = dict(V3, ball_cluster_coverage=1.1)
        parent = ROOT / "data" / "local_checks" / "test_tmp"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as folder:
            path = Path(folder) / "bad_v3_config.json"
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Fraction"):
                read_v3_config(path)


if __name__ == "__main__":
    unittest.main()
