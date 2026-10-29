"""Optional video-input export must not change possession-stage results."""
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from solution.external_samples import ExternalMatch
from solution import make_submission


class OptionalVideoExportTests(unittest.TestCase):
    def test_export_preserves_results_and_records_verifiable_training_metadata(self):
        n = 3
        period = np.ones(n, dtype=int)
        home = np.array([True, False])
        players = np.zeros((n, 2, 2), dtype=np.float32)
        ball = np.zeros((n, 2), dtype=np.float32)
        pitch = (105.3, 68.)

        def match(name):
            return ExternalMatch(name, players, home, ball, period, [], 25, pitch)

        sources = [match('practice_one'), match('practice_two')]
        labels = [np.array(['H', 'A', 'H']) for _ in sources]
        tables = (pd.DataFrame({'test_feature': np.zeros(n)}),
                  pd.DataFrame({'test_feature': np.zeros(n)}))
        turnover = SimpleNamespace(test_model='turnover')
        patches = [
            patch.object(make_submission, 'metrica_sources', return_value=(sources, labels)),
            patch('solution.turnover_model.fit', return_value=turnover),
            patch('solution.gap_model.fit', return_value=SimpleNamespace(test_model='gap')),
            patch('solution.gap_model.ballfree_features', return_value=tables),
            patch('solution.v7_features.build_features', return_value=({}, players)),
            patch('solution.ball_masking.hide_live_ball', return_value=(ball, np.zeros(n, bool))),
            patch('solution.turnover_model.touch_features', return_value=tables),
            patch('solution.turnover_model.home_probability', return_value=np.full(n, .6)),
            patch('solution.gap_model.home_probability', return_value=np.full(n, .4)),
            patch('solution.turnover_model.decode', return_value=np.array(['H', 'A', 'H'])),
        ]
        args = (match('unseen_target'), {}, np.array(['H', 'A', 'H']), period,
                np.ones(n, dtype=bool), {}, [], lambda message: None)
        # Mock numerical training/inference: this checks export plumbing and
        # provenance without retraining models or touching competition inputs.
        with ExitStack() as stack:
            for mocked in patches:
                stack.enter_context(mocked)
            with TemporaryDirectory() as directory:
                base = Path(directory)
                normal = make_submission.stage_possession(*args)
                self.assertEqual(list(base.iterdir()), [])
                destination = base/'nested/video_inputs/turnover.joblib'
                exported = make_submission.stage_possession(*args, model_path=destination)
                for expected, actual in zip(normal, exported):
                    np.testing.assert_array_equal(actual, expected)
                self.assertEqual(len(normal), 3)
                saved = joblib.load(destination)
                self.assertEqual(saved['turn'].test_model, turnover.test_model)
                metadata = json.loads((destination.parent/'model.json').read_text(encoding='utf8'))
                with destination.open('rb') as binary:
                    digest = hashlib.file_digest(binary, 'sha256').hexdigest()
                self.assertEqual(metadata['sha256'], digest)
                self.assertEqual(metadata['training_matches'], ['practice_one', 'practice_two'])
                self.assertNotIn('unseen_target', metadata['training_matches'])


if __name__ == '__main__':
    unittest.main()
