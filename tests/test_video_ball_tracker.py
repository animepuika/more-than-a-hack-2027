import unittest
import numpy as np
from solution.video_recovery.tracker import BallIdentityTracker


def box(x, y, confidence=.8):
    return [x-2, y-2, x+2, y+2, confidence]


class IdentityAssociationTests(unittest.TestCase):
    def test_near_lower_confidence_beats_distant_distractor(self):
        tracker = BallIdentityTracker([50, 50], np.eye(3))
        selected = tracker.step([box(140, 50, .99), box(54, 50, .3)], np.eye(3))
        np.testing.assert_allclose(selected, box(54, 50, .3))
        self.assertEqual(tracker.last_reason, 'observed')

    def test_stationary_ball_remains_valid(self):
        tracker = BallIdentityTracker([50, 50], np.eye(3))
        for _ in range(20):
            selected = tracker.step([box(50, 50)], np.eye(3))
            self.assertIsNotNone(selected)
            np.testing.assert_allclose(tracker.velocity, [0, 0])

    def test_camera_translation_transports_seed_before_gate(self):
        tracker = BallIdentityTracker([50, 50], np.eye(3))
        translated = np.array([[1., 0., 100.], [0., 1., 30.], [0., 0., 1.]])
        selected = tracker.step([box(150, 80)], translated)
        self.assertIsNotNone(selected)
        np.testing.assert_allclose(tracker.position, [150, 80])
        np.testing.assert_allclose(tracker.velocity, [0, 0])

    def test_misses_never_emit_internal_predictions(self):
        tracker = BallIdentityTracker([50, 50], np.eye(3), velocity=[2, 0])
        for expected_x in (52, 54, 56):
            self.assertIsNone(tracker.step([], np.eye(3)))
            np.testing.assert_allclose(tracker.position, [expected_x, 50])
        selected = tracker.step([box(58, 50)], np.eye(3))
        self.assertIsNotNone(selected)
        np.testing.assert_allclose(tracker.velocity, [2, 0])

    def test_eleven_misses_require_explicit_reseed(self):
        tracker = BallIdentityTracker([50, 50], np.eye(3))
        for _ in range(10):
            self.assertIsNone(tracker.step([], np.eye(3)))
        self.assertFalse(tracker.lost)
        self.assertIsNone(tracker.step([], np.eye(3)))
        self.assertTrue(tracker.lost)
        self.assertIsNone(tracker.step([box(50, 50)], np.eye(3)))
        self.assertTrue(tracker.seed([50, 50], np.eye(3)))
        self.assertIsNotNone(tracker.step([box(50, 50)], np.eye(3)))

    def test_invalid_current_homography_loses_identity(self):
        tracker = BallIdentityTracker([50, 50], np.eye(3))
        self.assertIsNone(tracker.step([box(50, 50)], np.zeros((3, 3))))
        self.assertTrue(tracker.lost)
        self.assertIsNone(tracker.step([box(50, 50)], np.eye(3)))

    def test_invalid_candidate_is_not_emitted(self):
        tracker = BallIdentityTracker([50, 50], np.eye(3))
        self.assertIsNone(tracker.step([[52, 52, 48, 48, .8]], np.eye(3)))


if __name__ == '__main__':
    unittest.main()
