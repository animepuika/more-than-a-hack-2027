"""V13 keeps V12 outside long gaps and re-decodes gaps with the V10 model."""
import json
from pathlib import Path
import unittest
import numpy as np

from solution.run_v13 import combine
from tests.test_gap_fallback import evidence

CONFIG = json.loads((Path(__file__).resolve().parents[1]/'solution/config_v7.json').read_text())


class CombineTests(unittest.TestCase):
    def test_only_long_gap_frames_change(self):
        owners = [0]*30 + [-1]*100 + [1]*30 + [-1]*10 + [1]*30
        v12 = np.array(list('H'*130 + 'A'*70))
        p_home = np.r_[np.full(80, 0.9), np.full(120, 0.1)]
        out = combine(v12, p_home, evidence(owners), np.ones(200), CONFIG)
        self.assertEqual(''.join(out[:130]), 'H'*80 + 'A'*50)
        self.assertTrue((out[130:] == v12[130:]).all())  # the 10-frame gap is short

    def test_dead_frames_are_kept(self):
        owners = [0]*30 + [-1]*100 + [0]*30
        v12 = np.array(list('H'*60 + 'D'*20 + 'H'*80))
        out = combine(v12, np.full(160, 0.1), evidence(owners), np.ones(160), CONFIG)
        self.assertTrue((out[60:80] == 'D').all())


if __name__ == '__main__':
    unittest.main()
