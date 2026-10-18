"""Offline (non-causal) reference F0 tracker: Praat's autocorrelation pitch via
parselmouth.

This is the *measuring stick* shared by eval_vc.py (F0 PCC), Stage 03 (accuracy
of the causal YIN tracker), Stage 05 (pitch-shift ratio) and Stage 09
(intonation controls), so they all agree. It never runs in the product.

The 50-1000 Hz range covers low male voices up to female voices shifted by
+12 semitones. PCC is computed on log-F0, over frames voiced in both signals.
"""
import numpy as np

from . import audio

HOP_S = 0.01
FLOOR_HZ = 50.0
CEILING_HZ = 1000.0
TRACKER = f"Praat autocorrelation pitch (parselmouth), {FLOOR_HZ:g}-{CEILING_HZ:g} Hz, 10 ms hop"


def track(x, sr=audio.SAMPLE_RATE):
    """F0 in Hz per 10 ms frame (0 = unvoiced). Frame i is centred near
    i * 10 ms + Praat's start offset, identical for equal-length inputs."""
    import parselmouth
    snd = parselmouth.Sound(np.asarray(x, dtype=np.float64), sampling_frequency=sr)
    pitch = snd.to_pitch_ac(time_step=HOP_S, pitch_floor=FLOOR_HZ, pitch_ceiling=CEILING_HZ)
    return np.nan_to_num(pitch.selected_array["frequency"], nan=0.0)


def envelope_db(x, sr=audio.SAMPLE_RATE):
    """10 ms log-RMS envelope, for alignment."""
    hop = int(sr * HOP_S)
    n = len(x) // hop
    frames = np.asarray(x[:n * hop], np.float64).reshape(n, hop)
    return 10.0 * np.log10(np.mean(frames ** 2, axis=1) + 1e-10)


def best_lag(src, out, sr=audio.SAMPLE_RATE, max_lag_s=0.2):
    """Frames by which `out` lags `src` (converted audio carries the model's
    algorithmic latency), from the envelope cross-correlation."""
    a, b = envelope_db(src, sr), envelope_db(out, sr)
    max_lag = int(round(max_lag_s / HOP_S))
    best, best_score = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        x, y = (a[:len(a) - lag], b[lag:]) if lag >= 0 else (a[-lag:], b[:len(b) + lag])
        n = min(len(x), len(y))
        if n < 10:
            continue
        x, y = x[:n] - x[:n].mean(), y[:n] - y[:n].mean()
        score = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
        if score > best_score:
            best, best_score = lag, score
    return best


def pcc(f0_src, f0_out, lag=0):
    """Pearson correlation of log-F0 on frames voiced in both (NaN if < 10)."""
    a, b = (f0_src[:len(f0_src) - lag], f0_out[lag:]) if lag >= 0 else (f0_src[-lag:], f0_out[:len(f0_out) + lag])
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    voiced = (a > 0) & (b > 0)
    if voiced.sum() < 10:
        return float("nan")
    return float(np.corrcoef(np.log(a[voiced]), np.log(b[voiced]))[0, 1])


def stats(f0):
    """Median F0 (Hz) and std of log-F0 over voiced frames (Stage 09 checks)."""
    v = f0[f0 > 0]
    if len(v) < 10:
        return float("nan"), float("nan")
    return float(np.median(v)), float(np.std(np.log(v)))
