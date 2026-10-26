"""Referee-whistle detector on the broadcast sound track, mapped to tactical frames.

The broadcast MP4 is an official competition input; its sound is analysed by code only, with
settings fixed before any check: one spectrum per broadcast video frame (25 per second); a
frame is whistle-like when the 2.0-4.5 kHz band holds a sharp tone (peak at least PEAK_DB above
the band median) and the band is at least LOUD_DB above its own rolling background. Events last
MIN_S..MAX_S and merge across gaps under MERGE_S. Broadcast frames map to tactical frames with the
alignment measured from player positions (constant offset per half). Commentary is not used.
"""
import numpy as np
from scipy.io import wavfile
from scipy.ndimage import median_filter

from .features import runs

BAND_HZ = (2000.0, 4500.0)
PEAK_DB = 15.0
LOUD_DB = 8.0
BACKGROUND_S = 10.0
MIN_S, MAX_S, MERGE_S = 0.16, 1.6, 0.3
FPS = 25
WINDOW = 1024


def frame_spectra(wav_path):
    """Per-video-frame band peakiness (dB) and band level (dB) from a mono WAV."""
    rate, audio = wavfile.read(wav_path, mmap=True)
    hop = rate // FPS
    n = len(audio) // hop
    freqs = np.fft.rfftfreq(WINDOW, 1 / rate)
    band = (freqs >= BAND_HZ[0]) & (freqs <= BAND_HZ[1])
    window = np.hanning(WINDOW).astype(np.float32)
    peak_db = np.zeros(n, np.float32)
    level_db = np.zeros(n, np.float32)
    chunk = 20000
    for start in range(0, n, chunk):
        idx = np.arange(start, min(n, start + chunk))
        offsets = idx[:, None] * hop + np.arange(WINDOW)[None]
        offsets = np.minimum(offsets, len(audio) - 1)
        frames = audio[offsets].astype(np.float32) / 32768.0 * window
        power = np.abs(np.fft.rfft(frames, axis=1))[:, band] ** 2 + 1e-12
        peak_db[idx] = 10 * np.log10(power.max(axis=1) / np.median(power, axis=1))
        level_db[idx] = 10 * np.log10(power.sum(axis=1))
    return peak_db, level_db


def detect_whistles(peak_db, level_db):
    """Return [(first_frame, last_frame_exclusive)] of whistle events in broadcast frames."""
    background = median_filter(level_db, size=int(BACKGROUND_S * FPS), mode='nearest')
    candidate = (peak_db >= PEAK_DB) & (level_db - background >= LOUD_DB)
    # Merge short breaks inside one blast.
    merged = candidate.copy()
    for a, b in runs(~candidate):
        if 0 < a and b < len(candidate) and (b - a) < MERGE_S * FPS:
            merged[a:b] = True
    return [(int(a), int(b)) for a, b in runs(merged) if MIN_S * FPS <= (b - a) <= MAX_S * FPS]


def to_tactical(frames, mapping, offsets=(4014, 26920)):
    """Map broadcast frames to tactical frames: the measured per-frame mapping where available,
    otherwise the half's constant offset (chosen by the nearest aligned broadcast frame)."""
    mapping = np.asarray(mapping)
    aligned = np.flatnonzero(mapping >= 0)
    out = []
    for f in frames:
        if 0 <= f < len(mapping) and mapping[f] >= 0:
            out.append(int(mapping[f]))
            continue
        near = aligned[np.argmin(np.abs(aligned - f))]
        offset = mapping[near] - near
        out.append(int(f + (offsets[0] if abs(offset - offsets[0]) < abs(offset - offsets[1]) else offsets[1])))
    return np.array(out)
