"""Student-input augmentation for content-encoder distillation (Milestone 2 /
Stage 04). Only the student hears it: labels always come from the clean teacher
pass (denoising distillation).

Chain, in mic order, each step optional and random:
  RIR reverb (p 0.4)    the RIR is cut at its direct-path peak (so no delay is
                        added against the labels), truncated to 1 s, unit energy,
                        applied by FFT convolution truncated to the input length
                        (causal: output n uses input <= n), dry RMS restored
  MUSAN noise (p 0.6)   a random segment of a random noise/music file at an SNR
                        uniform in 5-30 dB
  mild EQ (p 0.5)       one RBJ biquad: peaking, low shelf or high shelf,
                        +-6 dB, run by lfilter (causal)
  gain (p 1.0)          uniform +-10 dB, then clipped to [-1, 1] like an ADC

numpy/scipy only: it runs inside DataLoader workers.
"""
import dataclasses

import numpy as np
import soundfile as sf

SR = 16000


@dataclasses.dataclass
class AugmentConfig:
    noise_p: float = 0.6
    snr_db: tuple = (5.0, 30.0)
    rir_p: float = 0.4
    rir_max_s: float = 1.0
    eq_p: float = 0.5
    eq_gain_db: float = 6.0
    gain_p: float = 1.0
    gain_db: float = 10.0


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if len(x) else 0.0


def read_segment(path, n, rng):
    """n samples from a random offset of a 16 kHz file, looped if shorter."""
    info = sf.info(str(path))
    if info.samplerate != SR:
        raise ValueError(f"{path}: {info.samplerate} Hz, expected {SR}")
    start = int(rng.integers(0, max(1, info.frames - n + 1)))
    x, _ = sf.read(str(path), start=start, frames=n, dtype="float32", always_2d=True)
    x = x[:, 0]
    if len(x) < n:
        x = np.resize(x, n) if len(x) else np.zeros(n, np.float32)
    return x


def add_noise(x, noise, snr_db):
    ps, pn = rms(x), rms(noise)
    if ps == 0.0 or pn == 0.0:
        return x
    return x + noise * (ps / (pn * 10.0 ** (snr_db / 20.0)))


def prepare_rir(rir, max_len):
    rir = np.asarray(rir, np.float64)
    if rir.ndim == 2:
        rir = rir[:, 0]
    rir = rir[int(np.argmax(np.abs(rir))):][:max_len]
    e = np.sqrt(np.sum(rir * rir))
    return rir / e if e > 0 else None


def reverb(x, rir):
    from scipy.signal import fftconvolve
    y = fftconvolve(x, rir)[:len(x)]
    r = rms(y)
    return (y * (rms(x) / r)).astype(np.float32) if r > 0 else x


def biquad(kind, f0, gain_db, q, sr=SR):
    """RBJ Audio-EQ-Cookbook coefficients (b, a), a[0] = 1."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / sr
    cw, alpha = np.cos(w0), np.sin(w0) / (2 * q)
    if kind == "peaking":
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    else:
        sa = 2 * np.sqrt(A) * alpha
        sign = 1 if kind == "lowshelf" else -1
        b = [A * ((A + 1) - sign * (A - 1) * cw + sa), sign * 2 * A * ((A - 1) - sign * (A + 1) * cw),
             A * ((A + 1) - sign * (A - 1) * cw - sa)]
        a = [(A + 1) + sign * (A - 1) * cw + sa, -sign * 2 * ((A - 1) + sign * (A + 1) * cw),
             (A + 1) + sign * (A - 1) * cw - sa]
    b, a = np.array(b), np.array(a)
    return b / a[0], a / a[0]


def random_eq(x, rng, max_gain_db):
    from scipy.signal import lfilter
    kind = ("peaking", "lowshelf", "highshelf")[int(rng.integers(0, 3))]
    f0 = {"peaking": np.exp(rng.uniform(np.log(150), np.log(6000))),
          "lowshelf": rng.uniform(100, 400), "highshelf": rng.uniform(2000, 6000)}[kind]
    q = rng.uniform(0.5, 2.0) if kind == "peaking" else 0.707
    b, a = biquad(kind, f0, rng.uniform(-max_gain_db, max_gain_db), q)
    return lfilter(b, a, x).astype(np.float32)


class Augmenter:
    def __init__(self, cfg=None, noise_paths=(), rir_paths=()):
        self.cfg = cfg or AugmentConfig()
        self.noise_paths, self.rir_paths = list(noise_paths), list(rir_paths)

    def __call__(self, x, rng):
        c = self.cfg
        x = np.asarray(x, np.float32)
        if self.rir_paths and rng.random() < c.rir_p:
            path = self.rir_paths[int(rng.integers(0, len(self.rir_paths)))]
            rir, sr = sf.read(str(path), dtype="float32", always_2d=True)
            rir = prepare_rir(rir, int(c.rir_max_s * sr)) if sr == SR else None
            if rir is not None:
                x = reverb(x, rir)
        if self.noise_paths and rng.random() < c.noise_p:
            path = self.noise_paths[int(rng.integers(0, len(self.noise_paths)))]
            x = add_noise(x, read_segment(path, len(x), rng), rng.uniform(*c.snr_db))
        if rng.random() < c.eq_p:
            x = random_eq(x, rng, c.eq_gain_db)
        if rng.random() < c.gain_p:
            x = x * 10.0 ** (rng.uniform(-c.gain_db, c.gain_db) / 20.0)
        return np.clip(x, -1.0, 1.0).astype(np.float32)
