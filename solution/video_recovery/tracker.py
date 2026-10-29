"""Causal identity association seeded by a supplied observed ball position.

Never emits internal predictions as recovered observations. No global reinitial-
ization, image annotations, reference matching, size/paint bans, or look-ahead.
Fixed gates/velocity mixing are research hypotheses, not fitted parameters.
"""
import numpy as np


def _homography(value):
    try:
        matrix = np.asarray(value, dtype=float)
        if (matrix.shape == (3, 3) and np.isfinite(matrix).all()
                and np.linalg.matrix_rank(matrix) == 3):
            return matrix.copy()
    except (TypeError, ValueError, np.linalg.LinAlgError):
        pass
    return None


def _project(point, matrix):
    projected = matrix @ np.r_[point, 1.]
    if not np.isfinite(projected).all() or abs(projected[2]) < 1e-9:
        return None
    return projected[:2] / projected[2]


class BallIdentityTracker:
    """Associate detections around an explicit seed in consecutive frame steps.

    Args: seed_pixel is image (x,y); H_seed maps pitch coordinates into the
    seed image; optional velocity is pixels per frame in that seed image.
    Call step once per consecutive frame, including frames with no detections.
    step returns an observed detection row [x1,y1,x2,y2,confidence] or None.

    ``position`` and ``velocity`` are INTERNAL state and can be predictions.
    They must never be exported as observed recovery when step returns None.
    """

    def __init__(self, seed_pixel, H_seed, velocity=None):
        self.position = None
        self.velocity = None
        self.H = None
        self.misses = 0
        self.lost = True
        self.last_reason = 'uninitialized'
        self.seed(seed_pixel, H_seed, velocity)

    def seed(self, pixel, H, velocity=None):
        """Explicitly reset from a new supplied observation, not a prediction."""
        matrix = _homography(H)
        try:
            point = np.asarray(pixel, dtype=float)
            speed = np.zeros(2) if velocity is None else np.asarray(velocity, dtype=float)
            valid = (point.shape == (2,) and speed.shape == (2,)
                     and np.isfinite(point).all() and np.isfinite(speed).all())
        except (ValueError, TypeError):
            valid = False
        self.misses = 0
        if matrix is None or not valid:
            self.lost = True
            self.last_reason = 'invalid_seed'
            self.position = self.velocity = self.H = None
            return False
        self.position = point.copy()
        self.velocity = speed.copy()
        self.H = matrix
        self.lost = False
        self.last_reason = 'seeded'
        return True

    def step(self, candidates, H_current):
        """Return one actual box or abstain; internal missed-frame state is causal."""
        if self.lost:
            self.last_reason = 'lost_requires_seed'
            return None
        current = _homography(H_current)
        if current is None:
            self.lost = True
            self.last_reason = 'invalid_homography'
            return None
        try:
            warp = current @ np.linalg.inv(self.H)
            position = _project(self.position, warp)
            endpoint = _project(self.position + self.velocity, warp)
        except np.linalg.LinAlgError:
            position = endpoint = None
        if position is None or endpoint is None:
            self.lost = True
            self.last_reason = 'invalid_projection'
            return None
        velocity = endpoint - position
        expected = position + velocity
        gate = 25. + 8. * self.misses
        try:
            boxes = np.asarray(candidates, dtype=float).reshape(-1, 5)
        except (TypeError, ValueError):
            boxes = np.empty((0, 5))
        valid = (np.isfinite(boxes).all(axis=1)
                 & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                 & (boxes[:, 4] >= 0) & (boxes[:, 4] <= 1))
        boxes = boxes[valid]
        if len(boxes):
            centers = (boxes[:, :2] + boxes[:, 2:4]) / 2
            distances = np.linalg.norm(centers - expected, axis=1)
            allowed = np.flatnonzero(distances <= gate)
        else:
            allowed = np.array([], dtype=int)
        self.H = current
        if len(allowed):
            score = distances[allowed] / gate - .1 * boxes[allowed, 4]
            index = allowed[int(np.argmin(score))]
            measurement = centers[index]
            self.velocity = .7 * velocity + .3 * (measurement - position)
            self.position = measurement.copy()
            self.misses = 0
            self.last_reason = 'observed'
            return boxes[index].copy()
        self.position = expected
        self.velocity = velocity
        self.misses += 1
        self.lost = self.misses > 10
        self.last_reason = 'too_many_misses' if self.lost else 'no_candidate_in_gate'
        return None
