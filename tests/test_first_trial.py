"""Behavior checks using invented tracking, never match labels or match positions."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from solution.data import Tracking, load_tracking
from solution.features import make_features
from solution.predict import decode_sequence, predict, stable_gap_mask
from solution.run import DEFAULT_CONFIG, read_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))


def tracking(n=250, slow=False):
    """A hypothetical home-controlled ball moving across a 100 x 60 m pitch."""
    return Tracking(25.0, (100.0, 60.0),
                    np.column_stack((20 + np.arange(n) * 0.03, np.full(n, 30.0))),
                    np.full(n, 0.5), np.full(n, 5.0), np.full(n, 22),
                    np.full(n, 22), np.full(n, 0.5 if slow else 3.0),
                    np.full(n, 1.0 if slow else 0.0))


class LocalFilesTest(unittest.TestCase):
    def setUp(self):
        parent = ROOT / "data" / "local_checks" / "test_tmp"
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def write_tracking(self, lines):
        meta = self.folder / "synthetic.xml"
        raw = self.folder / "synthetic.txt"
        meta.write_text('<Root><FrameRate>25</FrameRate><Team id="h"/>'
                        '<Team id="a"/><Player teamId="h"/><Player teamId="a"/></Root>',
                        encoding="utf-8")
        raw.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return meta, raw

    def test_loader_scales_coordinates_and_keeps_team_mapping(self):
        paths = self.write_tracking(["1:0.2,0.5,2;0.8,0.5,4:0.21,0.5,1",
                                     "2:NaN,NaN,NaN;0.8,0.5,4:NaN,NaN,NaN"])
        data = load_tracking(*paths, 2, 25, (100, 60))
        np.testing.assert_allclose(data.ball_xy_m[0], [21, 30])
        self.assertAlmostEqual(data.home_distance_m[0], 1)
        self.assertAlmostEqual(data.away_distance_m[0], 59)
        self.assertEqual(data.observed_players[1], 1)
        self.assertTrue(np.isnan(data.home_distance_m[1]))
        self.assertEqual(data.median_player_speed_mps[0], 3)

    def test_loader_rejects_frame_shift(self):
        paths = self.write_tracking(["2:0.2,0.5,2;0.8,0.5,4:0.21,0.5,1"])
        with self.assertRaisesRegex(ValueError, "consecutive"):
            load_tracking(*paths, 1, 25, (100, 60))

    def test_loader_rejects_missing_columns(self):
        paths = self.write_tracking(["1:0.2,0.5,2:0.21,0.5,1"])
        with self.assertRaisesRegex(ValueError, "field count"):
            load_tracking(*paths, 1, 25, (100, 60))

    def test_zero_confirmation_time_is_rejected(self):
        config = dict(CONFIG, outside_confirm_s=0)
        path = self.folder / "config.json"
        path.write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "positive"):
            read_config(path)


class PossessionBehaviorTest(unittest.TestCase):
    def labels(self, data, blocks=None, **options):
        blocks = blocks or [(1, 0, len(data) - 1)]
        features = make_features(data, blocks, CONFIG)
        labels, reasons = predict(data, features, blocks, CONFIG, **options)
        return labels, features, reasons

    def test_long_missing_ball_with_moving_players_is_not_automatically_dead(self):
        data = tracking()
        data.ball_xy_m[30:220] = np.nan
        data.home_distance_m[30:220] = np.nan
        data.away_distance_m[30:220] = np.nan
        labels, features, _ = self.labels(data)
        self.assertTrue(np.all(labels == "H"))
        self.assertFalse(features["slow_gap_stop"].any())

    def test_slow_players_with_moving_controlled_ball_stay_live(self):
        labels, features, _ = self.labels(tracking(slow=True))
        self.assertTrue(np.all(labels == "H"))
        self.assertFalse(features["stationary_stop"].any())
        self.assertFalse(features["slow_gap_stop"].any())

    def test_persistent_outside_ball_can_mark_dead_play(self):
        data = tracking()
        data.ball_xy_m[80:180, 1] = -1.0
        labels, features, _ = self.labels(data)
        self.assertTrue(features["outside_stop"][100])
        self.assertEqual(labels[100], "D")
        self.assertEqual(labels[30], "H")
        self.assertEqual(labels[220], "H")

    def test_confirmed_slow_missing_gap_is_a_separate_testable_hypothesis(self):
        data = tracking(slow=True)
        data.ball_xy_m[30:220] = np.nan
        data.home_distance_m[30:220] = np.nan
        data.away_distance_m[30:220] = np.nan
        labels, features, _ = self.labels(data)
        without_stops, _, _ = self.labels(data, use_stoppages=False)
        self.assertTrue(features["slow_gap_stop"][120])
        self.assertEqual(labels[120], "D")
        self.assertEqual(without_stops[120], "H")

    def test_each_half_starts_independently_and_ungraded_frames_are_x(self):
        data = tracking(250)
        data.home_distance_m[130:] = 5
        data.away_distance_m[130:] = 0.5
        data.ball_xy_m[130:135] = np.nan
        blocks = [(1, 10, 100), (2, 130, 239)]
        labels, features, _ = self.labels(data, blocks)
        self.assertTrue(np.all(labels[10:101] == "H"))
        self.assertTrue(np.all(labels[130:240] == "A"))
        self.assertTrue(np.all(labels[101:130] == "X"))
        self.assertAlmostEqual(features["gap_s"][130], 0.04)

    def test_half_without_control_or_stoppage_evidence_abstains(self):
        data = tracking(100)
        data.ball_xy_m[:] = np.nan
        data.home_distance_m[:] = data.away_distance_m[:] = np.nan
        labels, _, reasons = self.labels(data)
        self.assertTrue(np.all(labels == "X"))
        self.assertTrue(np.all(reasons == "insufficient_evidence_in_half"))

    def test_clear_team_swap_does_not_favor_home(self):
        home, _, _ = self.labels(tracking())
        swapped = tracking()
        swapped.home_distance_m, swapped.away_distance_m = swapped.away_distance_m, swapped.home_distance_m
        away, _, _ = self.labels(swapped)
        self.assertTrue(np.all(home == "H"))
        self.assertTrue(np.all(away == "A"))

    def test_single_frame_opposition_is_smoothed_but_sustained_turnover_remains(self):
        costs = np.tile([0.0, 3.0, 4.0], (150, 1))
        costs[25] = [3, 0, 4]
        costs[75:] = [3, 0, 4]
        states = decode_sequence(costs, 4, 2.5)
        self.assertEqual(states[25], 0)
        self.assertTrue(np.all(states[75:] == 1))


class AbstentionBehaviorTest(unittest.TestCase):
    def mask(self, labels, missing, blocks=None, stop_frames=()):
        n = len(labels)
        features = {name: np.zeros(n, dtype=bool) for name in
                    ("ball_seen", "control_usable", "outside_stop", "slow_gap_stop", "stationary_stop")}
        features["ball_seen"][:] = True
        features["ball_seen"][missing] = False
        features["control_usable"] = features["ball_seen"].copy()
        features["outside_stop"][list(stop_frames)] = True
        return stable_gap_mask(np.array(list(labels)), features,
                               blocks or [(1, 0, n - 1)], 25, 0.12)

    def test_short_internal_same_team_gap_can_use_x(self):
        np.testing.assert_array_equal(self.mask("HHHHHH", slice(2, 4)),
                                      [False, False, True, True, False, False])

    def test_turnover_gap_does_not_use_x(self):
        self.assertFalse(self.mask("HHHAAA", slice(2, 4)).any())

    def test_dead_gap_does_not_use_x(self):
        self.assertFalse(self.mask("DDDDDD", slice(2, 4)).any())

    def test_leading_trailing_and_long_gaps_do_not_use_x(self):
        for missing in (slice(0, 2), slice(4, 6), slice(1, 5)):
            with self.subTest(missing=missing):
                self.assertFalse(self.mask("HHHHHH", missing).any())

    def test_stop_cue_blocks_x(self):
        self.assertFalse(self.mask("HHHHHH", slice(2, 4), stop_frames=[2]).any())

    def test_x_never_bridges_half_boundary(self):
        self.assertFalse(self.mask("HHHHHH", slice(2, 4),
                                  blocks=[(1, 0, 2), (2, 3, 5)]).any())


if __name__ == "__main__":
    unittest.main()
