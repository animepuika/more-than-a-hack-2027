"""Whistle detector on synthetic sound."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.io import wavfile

from solution.whistle_audio import detect_whistles, frame_spectra, to_tactical


class WhistleTests(unittest.TestCase):
    def test_tone_in_noise_is_found_once(self):
        rate = 22050
        rng = np.random.RandomState(0)
        t = np.arange(rate * 30) / rate
        sound = 0.05 * rng.randn(len(t))
        blast = (t >= 12.0) & (t < 12.6)
        sound[blast] += 0.5 * np.sin(2 * np.pi * 3200 * t[blast])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'a.wav'
            wavfile.write(path, rate, (sound * 32767).astype(np.int16))
            peak, level = frame_spectra(path)
        events = detect_whistles(peak, level)
        self.assertEqual(len(events), 1)
        a, b = events[0]
        self.assertTrue(295 <= a <= 305 and 310 <= b <= 320)

    def test_mapping_uses_measured_frames_then_half_offset(self):
        mapping = np.full(100, -1); mapping[10:20] = np.arange(10, 20) + 4014; mapping[80:90] = np.arange(80, 90) + 26920
        self.assertEqual(to_tactical([15, 25, 85, 95], mapping).tolist(), [15 + 4014, 25 + 4014, 85 + 26920, 95 + 26920])


if __name__ == '__main__':
    unittest.main()
