"""Ball-free gap model: symmetry of features and the gap decode."""
import json
from pathlib import Path
import unittest
import numpy as np

from solution.gap_model import ballfree_features, model_fallback
from tests.test_gap_fallback import evidence

CONFIG = json.loads((Path(__file__).resolve().parents[1]/'solution/config_v7.json').read_text())


def formation(n=200, shift=0.0):
    rng = np.random.RandomState(0)
    home = np.c_[rng.uniform(10, 50, 11), rng.uniform(5, 63, 11)]
    away = np.c_[rng.uniform(55, 95, 11), rng.uniform(5, 63, 11)]
    xy = np.tile(np.r_[home, away][None], (n, 1, 1)).astype(np.float32)
    xy[:, :11, 0] += np.linspace(0, shift, n)[:, None]  # home pushes forward over time
    return xy, np.r_[np.ones(11, bool), np.zeros(11, bool)]


class GapModelTests(unittest.TestCase):
    def test_orientations_mirror_when_teams_and_pitch_are_swapped(self):
        xy, home = formation(shift=10)
        mirrored = xy[:, ::-1].copy()
        mirrored[:, :, 0] = 105 - mirrored[:, :, 0]
        us_home, us_away = ballfree_features(xy, home, np.ones(len(xy), int), (105., 68.), 25)
        swap_home, swap_away = ballfree_features(mirrored, home, np.ones(len(xy), int), (105., 68.), 25)
        np.testing.assert_allclose(us_home.to_numpy(), swap_away.to_numpy(), atol=1e-4, equal_nan=True)
        self.assertGreater(us_home['us_forward_mean_trend_next'].iloc[50], 0)

    def test_gap_decode_switches_where_model_is_confident(self):
        owners = [0]*30 + [-1]*100 + [1]*30
        p_home = np.r_[np.full(80, 0.9), np.full(80, 0.1)]
        fallback = model_fallback(p_home, evidence(owners), np.ones(160), CONFIG, np.array(['H']*160), 2.0)
        self.assertEqual(''.join(fallback[30:130]), 'H'*50 + 'A'*50)

    def test_anchor_holds_owner_without_confident_evidence(self):
        owners = [0]*30 + [-1]*100 + [0]*30
        fallback = model_fallback(np.full(160, 0.45), evidence(owners), np.ones(160), CONFIG,
                                  np.array(['H']*160), 2.0)
        self.assertTrue((fallback[30:130] == 'H').all())


if __name__ == '__main__':
    unittest.main()
