"""Cutaway dead ball: trimming, restart from tracking, homography mapping."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

from solution.camera_cutaway import cutaway_dead, cutaway_mask, stop_at_restart, trim


class CutawayTests(unittest.TestCase):
    def test_trim_drops_edges_and_short_leftovers(self):
        cut = np.zeros(1000, bool); cut[100:400] = True; cut[600:820] = True
        out = trim(cut, 25)
        self.assertEqual(np.flatnonzero(out)[[0, -1]].tolist(), [175, 249])
        self.assertFalse(out[600:820].any())  # 8.8 s minus 9 s of trimming leaves nothing

    def test_restart_ends_dead_ball(self):
        mask = np.zeros(300, bool); mask[50:250] = True
        speed = np.full(300, 0.8); speed[150:] = 3.0
        out = stop_at_restart(mask, speed, 25)
        self.assertEqual(np.flatnonzero(out)[[0, -1]].tolist(), [50, 162])

    def test_slow_players_keep_whole_trimmed_stretch(self):
        cut = np.zeros(1000, bool); cut[100:400] = True
        out = cutaway_dead(cut, np.full(1000, 0.9), 25)
        self.assertEqual(int(out.sum()), 300 - 75 - 150)

    def test_mapping_uses_half_offsets(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'h.csv'
            valid = np.ones(100, int); valid[20:30] = 0; valid[60:70] = 0
            pd.DataFrame({'frame': np.arange(100), 'valid': valid}).to_csv(path, index=False)
            period = np.r_[np.ones(50, int), np.full(70, 2)]
            cut, seen = cutaway_mask(path, period, np.ones(120, bool), {1: 5, 2: 40})
        self.assertEqual(np.flatnonzero(cut).tolist(), list(range(25, 35)) + list(range(60, 70)) + list(range(100, 110)))
        self.assertFalse(seen[:5].any())


if __name__ == '__main__':
    unittest.main()
