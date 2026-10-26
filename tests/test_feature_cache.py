"""Feature cache round trip and invalidation."""
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import pandas as pd

from solution import feature_cache


def match(value=1.0):
    return SimpleNamespace(name='M', player_xy_m=np.full((5, 2, 2), value, np.float32), ball_xy_m=np.zeros((5, 2)),
                           period=np.ones(5, int), home_slots=np.array([True, False]), fps=25.0, pitch=(105., 68.))


class FeatureCacheTests(unittest.TestCase):
    def test_round_trip_and_reuse(self):
        calls = []
        def compute(m):
            calls.append(1)
            return {'table': pd.DataFrame({'a': [1., 2.], 'b': [3., np.nan]}), 'labels': np.array(['H', 'D'])}
        with tempfile.TemporaryDirectory() as folder, patch.object(feature_cache, 'CACHE_ROOT', Path(folder)):
            first = feature_cache.cached('t', match(), ['feature_cache.py'], compute)
            second = feature_cache.cached('t', match(), ['feature_cache.py'], compute)
            third = feature_cache.cached('t', match(2.0), ['feature_cache.py'], compute)  # different input
        self.assertEqual(len(calls), 2)
        self.assertEqual(list(second['table'].columns), ['a', 'b'])
        self.assertTrue(all(type(c) is str for c in second['table'].columns))  # sklearn rejects numpy str_
        self.assertTrue(np.isnan(second['table']['b'][1]))
        self.assertEqual(second['labels'].tolist(), ['H', 'D'])
        self.assertEqual(third['labels'].tolist(), first['labels'].tolist())


if __name__ == '__main__':
    unittest.main()
