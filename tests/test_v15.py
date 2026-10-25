"""Set-piece tag windows for V15."""
import unittest
import numpy as np

from solution.run_v15 import TAG_AFTER_S, TAG_BEFORE_S, tag_window


class TagWindowTests(unittest.TestCase):
    def test_window_covers_preparation_before_each_set_piece_only(self):
        events = [{'code': 'SET PIECES', 'start_frame': 1000, 'end_frame': 1100, 'tags': {}},
                  {'code': 'BUILD UP', 'start_frame': 3000, 'end_frame': 3500, 'tags': {}}]
        window = tag_window(events, 4000, 25)
        self.assertTrue(window[1000 - int(TAG_BEFORE_S*25):1000 + int(TAG_AFTER_S*25)].all())
        self.assertFalse(window[:1000 - int(TAG_BEFORE_S*25)].any())
        self.assertFalse(window[1013:].any())


if __name__ == '__main__':
    unittest.main()
