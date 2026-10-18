"""Causal pitch & energy features (Milestone 2 / Stage 03).

Spec: Documents/Milestone 2/03_causal_pitch_energy.md. Every 20 ms frame gets
f0, aperiodicity, soft voicing and log-energy from a causal YIN tracker, plus
the held log-f0 and the whitened (z-scored) log-f0 and log-energy.

src/dsp/PitchTracker.hpp, RunningNorm.hpp and Biquad.hpp are the C++ port of
this file: same constants, same order of operations, double precision inside.
The golden vectors in tests/data/pitch/ (python_training/make_pitch_golden.py)
pin the two together, and both test suites load the constants from that
folder's header.json. Change a constant or a step in both files at once and
regenerate the vectors.

There is one code path: `extract()` loops `PitchTracker.push()`, so offline
features equal streaming features by construction. The vocoder (Stage 05) is
trained on this tracker's output for the target speaker, errors included.
"""
import math
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import lfilter

# Population priors (start values of the whitening EMAs): log-f0 (natural log
# of Hz) over voiced frames, log-energy over all frames. The voiced decision
# uses z_energy, so the log-f0 prior is iterated to a fixed point.
# PROVISIONAL: measured with `eval_pitch.py priors` on LibriSpeech dev-clean
# (5.4 h, 40 speakers, 20 F / 20 M); recompute on train-clean-100 once Stage 01
# has prepared it, then regenerate the golden vectors.
LOGF0_PRIOR_MEAN = 5.087
LOGF0_PRIOR_STD = 0.427
ENERGY_PRIOR_MEAN = -8.500
ENERGY_PRIOR_STD = 3.829

FEATURES = ("f0_hz", "aperiodicity", "voicing", "log_energy", "logf0_held", "z_logf0", "z_energy")


@dataclass(frozen=True)
class PitchConfig:
    sample_rate: int = 16000
    frame_samples: int = 320           # 20 ms hop
    window_samples: int = 640          # 40 ms: two periods at 50 Hz
    f0_min_hz: float = 55.0
    f0_max_hz: float = 1000.0
    thresholds: tuple = (0.05, 0.10, 0.15)   # soft voicing (StreamVC)
    f0_threshold: float = 0.10                # YIN's absolute threshold (dip choice)
    # The voiced/unvoiced decision behind f0_hz, the held log-f0 and the log-f0
    # whitening. 0.10 alone misses a quarter of voiced speech frames (74%
    # decision accuracy on dev-clean); see the Stage 03 Verification section.
    voiced_max_aperiodicity: float = 0.40
    voiced_min_z_energy: float = -0.5
    hpf_hz: float = 70.0
    hpf_q: float = 1.0 / math.sqrt(2.0)
    energy_eps: float = 1e-10
    silence_mean_square: float = 1e-8  # -80 dBFS: below this the window is unvoiced
    norm_tau_s: float = 4.0
    logf0_std_floor: float = 0.05
    energy_std_floor: float = 0.5
    hold_relax_s: float = 0.25
    logf0_prior_mean: float = LOGF0_PRIOR_MEAN
    logf0_prior_std: float = LOGF0_PRIOR_STD
    energy_prior_mean: float = ENERGY_PRIOR_MEAN
    energy_prior_std: float = ENERGY_PRIOR_STD
    lookahead_frames: int = 0

    def __post_init__(self):
        if self.lookahead_frames < 0:
            raise ValueError(f"lookahead_frames must be >= 0 (got {self.lookahead_frames})")
        if self.f0_threshold not in self.thresholds:
            raise ValueError("f0_threshold must be one of the voicing thresholds")
        if self.integration_samples < 1 or self.tau_min < 2:
            raise ValueError("window too short for the f0 search range")

    # Lag search range [tau_min, tau_max] in samples.
    @property
    def tau_min(self):
        return int(math.ceil(self.sample_rate / self.f0_max_hz))

    @property
    def tau_max(self):
        return int(self.sample_rate // self.f0_min_hz)

    @property
    def integration_samples(self):
        """YIN's fixed integration window W: every lag compares x[j] with
        x[j + tau] for j < W, so the whole window is used at tau_max."""
        return self.window_samples - self.tau_max

    @property
    def window_offset(self):
        """Samples by which the analysis window extends past the frame end.
        0 for L = 0 (past window); with L >= 1 the window is centred on the
        frame, which costs no latency because the encoder already waits L
        frames."""
        return self.frame_samples // 2 if self.lookahead_frames >= 1 else 0

    @property
    def history_samples(self):
        """Filtered samples kept: the window for frame t ends at
        end(t) + window_offset, and the newest sample is end(t + L)."""
        return self.window_samples + self.lookahead_frames * self.frame_samples - self.window_offset

    @property
    def frame_seconds(self):
        return self.frame_samples / self.sample_rate

    @property
    def norm_alpha(self):
        return 1.0 - math.exp(-self.frame_seconds / self.norm_tau_s)

    @property
    def hold_decay(self):
        return math.exp(-self.frame_seconds / self.hold_relax_s)

    def hpf_coefficients(self):
        return highpass_biquad(self.hpf_hz, self.sample_rate, self.hpf_q)


def highpass_biquad(fc, sr, q):
    """RBJ cookbook 2nd-order high-pass, normalised: (b0, b1, b2), (1, a1, a2)."""
    w0 = 2.0 * math.pi * fc / sr
    cosw = math.cos(w0)
    alpha = math.sin(w0) / (2.0 * q)
    a0 = 1.0 + alpha
    b = ((1.0 + cosw) / 2.0 / a0, -(1.0 + cosw) / a0, (1.0 + cosw) / 2.0 / a0)
    a = (1.0, -2.0 * cosw / a0, (1.0 - alpha) / a0)
    return b, a


class RunningNorm:
    """EMA mean and standard deviation, started from population priors.

    The update includes the current value before it is normalised (still
    causal). std never drops below `floor`, so near-constant input can't blow
    z up.
    """

    def __init__(self, prior_mean, prior_std, alpha, floor):
        self.alpha = float(alpha)
        self.floor = float(floor)
        self.mean = float(prior_mean)
        self.var = float(prior_std) ** 2

    def update(self, x):
        delta = x - self.mean
        self.mean += self.alpha * delta
        self.var = (1.0 - self.alpha) * (self.var + self.alpha * delta * delta)

    @property
    def std(self):
        return max(math.sqrt(self.var), self.floor)

    def z(self, x):
        return (x - self.mean) / self.std


@dataclass
class PitchFrame:
    f0_hz: float          # 0 on unvoiced frames
    aperiodicity: float   # min CMND over the search range (1 for silence)
    voicing: float        # {0, 1/3, 2/3, 1}
    log_energy: float     # ln(mean square + eps) of the filtered frame
    logf0_held: float
    z_logf0: float
    z_energy: float

    def as_tuple(self):
        return tuple(getattr(self, k) for k in FEATURES)


def yin(window, cfg):
    """(aperiodicity, f0 estimate in Hz) of one analysis window (float64). The
    estimate is 0 only for a silent window; voicing is decided by the caller."""
    w = cfg.integration_samples
    tau_min, tau_max = cfg.tau_min, cfg.tau_max
    if float(np.dot(window, window)) < cfg.silence_mean_square * len(window):
        return 1.0, 0.0
    # Difference function d(tau) for tau = 0..tau_max, then the cumulative-mean
    # normalised difference (CMND) d'(0) = 1, d'(tau) = d(tau) tau / sum d(1..tau).
    lagged = sliding_window_view(window, w)[: tau_max + 1]
    d = ((lagged - window[:w]) ** 2).sum(axis=1)
    cum = np.cumsum(d[1:])
    cmnd = np.ones(tau_max + 1)
    taus = np.arange(1, tau_max + 1, dtype=np.float64)
    nz = cum > 0.0
    cmnd[1:][nz] = d[1:][nz] * taus[nz] / cum[nz]

    search = cmnd[tau_min: tau_max + 1]
    aperiodicity = float(search.min())
    if aperiodicity < cfg.f0_threshold:
        # First dip below the threshold, then down to its local minimum.
        tau = tau_min + int(np.argmax(search < cfg.f0_threshold))
        while tau + 1 <= tau_max and cmnd[tau + 1] < cmnd[tau]:
            tau += 1
    else:
        tau = tau_min + int(np.argmin(search))   # YIN's fallback: the global minimum
    shift = 0.0
    if tau + 1 <= tau_max:
        a, b, c = cmnd[tau - 1], cmnd[tau], cmnd[tau + 1]
        den = a - 2.0 * b + c
        if den > 0.0:
            shift = 0.5 * (a - c) / den
    return aperiodicity, cfg.sample_rate / (tau + shift)


class PitchTracker:
    """Streaming tracker: one 20 ms frame in, one PitchFrame out.

    With lookahead L, push(frame t) returns the features of frame t - L (the
    first L outputs describe the zero history before the stream). All state is
    in the attributes; nothing depends on the future.
    """

    def __init__(self, cfg=None):
        self.cfg = cfg or PitchConfig()
        c = self.cfg
        self.b, self.a = c.hpf_coefficients()
        self.zi = np.zeros(2)
        self.history = np.zeros(c.history_samples)
        self.logf0_norm = RunningNorm(c.logf0_prior_mean, c.logf0_prior_std, c.norm_alpha,
                                      c.logf0_std_floor)
        self.energy_norm = RunningNorm(c.energy_prior_mean, c.energy_prior_std, c.norm_alpha,
                                       c.energy_std_floor)
        self.logf0_held = c.logf0_prior_mean

    def push(self, frame):
        c = self.cfg
        x = np.asarray(frame, dtype=np.float64)
        if x.shape != (c.frame_samples,):
            raise ValueError(f"expected {c.frame_samples} samples, got {x.shape}")
        y, self.zi = lfilter(self.b, self.a, x, zi=self.zi)
        h = self.history
        h[:-c.frame_samples] = h[c.frame_samples:]
        h[-c.frame_samples:] = y

        n = len(h)
        frame_end = n - c.lookahead_frames * c.frame_samples
        cur = h[frame_end - c.frame_samples: frame_end]
        win_end = frame_end + c.window_offset
        window = h[win_end - c.window_samples: win_end]

        log_energy = math.log(float(np.dot(cur, cur)) / c.frame_samples + c.energy_eps)
        self.energy_norm.update(log_energy)
        z_energy = self.energy_norm.z(log_energy)

        aperiodicity, f0 = yin(window, c)
        voicing = sum(aperiodicity < t for t in c.thresholds) / len(c.thresholds)
        voiced = (f0 > 0.0 and aperiodicity < c.voiced_max_aperiodicity
                  and z_energy > c.voiced_min_z_energy)
        if voiced:
            self.logf0_held = math.log(f0)
            self.logf0_norm.update(self.logf0_held)
        else:
            # Hold and relax towards the running mean: interpolating across the
            # gap would need the future.
            f0 = 0.0
            m = self.logf0_norm.mean
            self.logf0_held = m + (self.logf0_held - m) * c.hold_decay
        return PitchFrame(f0, aperiodicity, voicing, log_energy, self.logf0_held,
                          self.logf0_norm.z(self.logf0_held), z_energy)


def extract(x, cfg=None):
    """Features of a whole signal, (frames, len(FEATURES)) float64, frame t
    aligned with samples [t*hop, (t+1)*hop). Zero-pads the last partial frame
    and the L lookahead frames, then drops the L leading outputs."""
    cfg = cfg or PitchConfig()
    hop, la = cfg.frame_samples, cfg.lookahead_frames
    x = np.asarray(x, dtype=np.float64)
    frames = -(-len(x) // hop)
    padded = np.zeros((frames + la) * hop)
    padded[: len(x)] = x
    tracker = PitchTracker(cfg)
    out = [tracker.push(padded[i * hop: (i + 1) * hop]).as_tuple() for i in range(frames + la)]
    return np.asarray(out[la:], dtype=np.float64).reshape(frames, len(FEATURES))


def window_centre_seconds(cfg, frames):
    """Centre time of each frame's analysis window (for comparing against a
    reference tracker): end(t) + window_offset - window / 2."""
    t = np.arange(frames)
    return ((t + 1) * cfg.frame_samples + cfg.window_offset - cfg.window_samples / 2) / cfg.sample_rate


def map_to_target(z_logf0, z_energy, voice, scale=1.0, shift_st=0.0):
    """Whitened source features → the target voice's domain (C++ run-time path):
    logf0_tgt = mu + scale * sigma * z + shift_st * ln2 / 12, energy_tgt =
    mu_e + sigma_e * z_e. `voice` is a voices.json entry; scale and shift_st
    are the Stage 09 intonation and pitch-shift controls."""
    logf0 = voice["logf0_mean"] + scale * voice["logf0_std"] * np.asarray(z_logf0) \
        + shift_st * math.log(2.0) / 12.0
    energy = voice["energy_mean"] + voice["energy_std"] * np.asarray(z_energy)
    return logf0, energy


def header(cfg=None):
    """The constants both implementations load in their tests (header.json)."""
    cfg = cfg or PitchConfig()
    b, a = cfg.hpf_coefficients()
    return {
        "sample_rate": cfg.sample_rate,
        "frame_samples": cfg.frame_samples,
        "window_samples": cfg.window_samples,
        "f0_min_hz": cfg.f0_min_hz,
        "f0_max_hz": cfg.f0_max_hz,
        "tau_min": cfg.tau_min,
        "tau_max": cfg.tau_max,
        "integration_samples": cfg.integration_samples,
        "thresholds": list(cfg.thresholds),
        "f0_threshold": cfg.f0_threshold,
        "voiced_max_aperiodicity": cfg.voiced_max_aperiodicity,
        "voiced_min_z_energy": cfg.voiced_min_z_energy,
        "hpf_hz": cfg.hpf_hz,
        "hpf_q": cfg.hpf_q,
        "hpf_b": list(b),
        "hpf_a": list(a),
        "energy_eps": cfg.energy_eps,
        "silence_mean_square": cfg.silence_mean_square,
        "norm_tau_s": cfg.norm_tau_s,
        "logf0_std_floor": cfg.logf0_std_floor,
        "energy_std_floor": cfg.energy_std_floor,
        "hold_relax_s": cfg.hold_relax_s,
        "logf0_prior_mean": cfg.logf0_prior_mean,
        "logf0_prior_std": cfg.logf0_prior_std,
        "energy_prior_mean": cfg.energy_prior_mean,
        "energy_prior_std": cfg.energy_prior_std,
        "features": list(FEATURES),
    }
