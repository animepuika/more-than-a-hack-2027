"""Physics/uncertainty, independent small-path oracle, and output safety."""
import contextlib
import io
import itertools
from pathlib import Path
import tempfile
import unittest
import numpy as np
from solution.video_recovery.association import associate, deduplicate
from solution.video_recovery.full_match import _output_path, main as recovery_main
from solution.video_recovery.expanded_score import _fresh_out, main as score_main

ROOT = Path(__file__).resolve().parents[1]
H = np.diag([105.3, 68., 1.])  # Pixel coordinates equal metres in synthetic cases.


def _box(x=40., y=30., confidence=.99):
    return [x-1, y-1, x+1, y+1, confidence]


def _oracle(boxes):
    """Enumerate every tiny trajectory, independently of the sparse DP."""
    all_paths = []
    for choices in itertools.product(*[[None, *b] for b in boxes]):
        nodes = [(-1, -1, np.array([40., 30.]), None)]
        nodes += [(k*5, k, (np.array(b[:2])+b[2:4])/2, b) for k, b in enumerate(choices) if b is not None]
        nodes.append((11, len(boxes), np.array([40., 30.]), None))
        cost = 0.
        for left, right in zip(nodes[:-1], nodes[1:]):
            dt = (right[0]-left[0])/25.; distance = np.linalg.norm(right[2]-left[2])
            if distance > 70*dt+2:
                cost = np.inf; break
            cost += 2*(right[1]-left[1]-1)+.25*(distance/(20*dt+2))**2
            if right[3] is not None: cost -= np.log(right[3][4])
        all_paths.append((cost, choices))
    best_cost, best = min(all_paths, key=lambda row: row[0])
    selected = []
    for k, box in enumerate(best):
        if box is None:
            selected.append(None); continue
        point = (np.array(box[:2])+box[2:4])/2
        alternatives = [cost for cost, path in all_paths if path[k] is None or
                        np.linalg.norm((np.array(path[k][:2])+path[k][2:4])/2-point) > 20]
        selected.append(box if min(alternatives)-best_cost >= .5 else None)
    return best_cost, selected


class FullMatchAssociationTests(unittest.TestCase):
    def run_graph(self, boxes):
        return associate([0, 5, 10], boxes, [H]*3, -1, [40., 30.], H, 11, [40., 30.], H)

    def test_stationary_ball_is_allowed(self):
        b = _box()
        self.assertEqual(self.run_graph([[b]]*3)['selected'], [b]*3)

    def test_no_detection_never_exports_prediction(self):
        self.assertEqual(self.run_graph([[], [], []])['selected'], [None]*3)

    def test_implausible_jump_is_rejected(self):
        self.assertEqual(self.run_graph([[_box(100)], [], []])['selected'], [None]*3)

    def test_small_paths_match_independent_exhaustive_oracle(self):
        rng = np.random.default_rng(20261030)
        for _ in range(30):
            boxes = [[_box(* (np.array([40., 30.])+rng.normal(0, 4, 2)), float(rng.uniform(.1, .99)))
                      for _ in range(2)] for _ in range(3)]
            cost, selected = _oracle(boxes); result = self.run_graph(boxes)
            self.assertAlmostEqual(result['path_cost'], cost, places=10)
            self.assertEqual(result['selected'], selected)

    def test_equal_distant_paths_abstain(self):
        scaled_H = np.diag([1053., 680., 1.])
        boxes = [[[390, 275, 410, 295, .99], [390, 305, 410, 325, .99]]]*3
        result = associate([0, 5, 10], boxes, [scaled_H]*3, -1, [400, 300], scaled_H,
                           11, [400, 300], scaled_H)
        self.assertEqual(result['selected'], [None]*3)

    def test_long_missing_section_does_not_invent_observations(self):
        frames = list(range(0, 101, 5)); boxes = [[] for _ in frames]; boxes[10] = [_box()]
        result = associate(frames, boxes, [H]*len(frames), -1, [40, 30], H, 102, [40, 30], H)
        self.assertEqual(sum(b is not None for b in result['selected']), 1)

    def test_tile_duplicates_keep_higher_confidence(self):
        result = deduplicate([_box(confidence=.5), _box(confidence=.8)])
        np.testing.assert_equal(result, [_box(confidence=.8)])


class FullMatchSafetyTests(unittest.TestCase):
    def test_output_paths_stay_under_ignored_data(self):
        for guard in (_output_path, _fresh_out):
            with self.assertRaises(ValueError): guard(ROOT/'outside_data_test')

    def test_scoring_refuses_nonempty_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'data') as directory:
            marker = Path(directory)/'keep.txt'; marker.write_text('keep')
            with self.assertRaises(ValueError): _fresh_out(directory)
            self.assertEqual(marker.read_text(), 'keep')

    def test_help_does_not_need_gpu_or_read_answers(self):
        for main in (recovery_main, score_main):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as stopped:
                main(['--help'])
            self.assertEqual(stopped.exception.code, 0)


if __name__ == '__main__': unittest.main()
