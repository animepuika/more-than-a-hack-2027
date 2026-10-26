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


# Detector v2: the referee's whistle is a pair of steady tones (a high tone near 4.3 kHz and a
# second tone about 10% lower), measured on the kick-off whistles (automatic kick-off tags) only.
# Crowd whistling is a single tone. The broadcast's two channels are identical, so the sound is
# treated as mono. All settings below were fixed before any check was run.
HIGH_HZ = (4100.0, 4450.0)
RATIO = (1.08, 1.14)
TONE_DB = 18.0
REF_BAND_HZ = (1500.0, 11500.0)
V2_WINDOW = 2048
V2_MIN_S, V2_MAX_S = 0.2, 3.0
# v3: keep an event only if its strongest moment reaches STRONG_DB. The clear kick-off whistles
# peaked at 34.9-46.3 dB; 30 dB leaves a 5 dB margin for whistles further from the microphones.
STRONG_DB = 30.0


def tone_pair_strength(wav_path):
    """Per-video-frame strength (dB above the band median) of the high tone and of its lower partner."""
    rate, audio = wavfile.read(wav_path, mmap=True)
    if audio.ndim == 2:
        audio = audio[:, 0]
    hop = rate // FPS
    n = len(audio) // hop
    freqs = np.fft.rfftfreq(V2_WINDOW, 1 / rate)
    ref = (freqs >= REF_BAND_HZ[0]) & (freqs <= REF_BAND_HZ[1])
    high = np.flatnonzero((freqs >= HIGH_HZ[0]) & (freqs <= HIGH_HZ[1]))
    window = np.hanning(V2_WINDOW).astype(np.float32)
    high_db = np.zeros(n, np.float32)
    low_db = np.zeros(n, np.float32)
    chunk = 10000
    for start in range(0, n, chunk):
        idx = np.arange(start, min(n, start + chunk))
        offsets = np.minimum(idx[:, None] * hop + np.arange(V2_WINDOW)[None], len(audio) - 1)
        power = np.abs(np.fft.rfft(audio[offsets].astype(np.float32) / 32768.0 * window, axis=1)) ** 2 + 1e-12
        floor = np.median(power[:, ref], axis=1)
        best = high[power[:, high].argmax(axis=1)]
        high_db[idx] = 10 * np.log10(power[np.arange(len(idx)), best] / floor)
        # Strongest bin in the partner range below each frame's high tone.
        lo_edge = np.searchsorted(freqs, freqs[best] / RATIO[1])
        hi_edge = np.searchsorted(freqs, freqs[best] / RATIO[0], side='right')
        width = int((hi_edge - lo_edge).max())
        cols = np.minimum(lo_edge[:, None] + np.arange(width)[None], len(freqs) - 1)
        inside = cols < hi_edge[:, None]
        partner = np.where(inside, power[np.arange(len(idx))[:, None], cols], 0).max(axis=1)
        low_db[idx] = 10 * np.log10(partner / floor + 1e-12)
    return high_db, low_db


def detect_referee_whistles(high_db, low_db, strong_db=None):
    """Return [(first_frame, last_frame_exclusive)] where both whistle tones are present; with
    strong_db, only events whose strongest frame has both tones at least that loud."""
    candidate = (high_db >= TONE_DB) & (low_db >= TONE_DB)
    merged = candidate.copy()
    for a, b in runs(~candidate):
        if 0 < a and b < len(candidate) and (b - a) < MERGE_S * FPS:
            merged[a:b] = True
    pair = np.minimum(high_db, low_db)
    return [(int(a), int(b)) for a, b in runs(merged) if V2_MIN_S * FPS <= (b - a) <= V2_MAX_S * FPS
            and (strong_db is None or pair[a:b].max() >= strong_db)]


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
