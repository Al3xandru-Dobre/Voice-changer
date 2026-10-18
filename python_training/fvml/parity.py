"""Compare a test signal/tensor with a reference.

Used for streaming-vs-offline (Stages 04-05) and PyTorch-vs-ORT (Stage 06)
checks. The metrics are max |Δ|, SNR (dB), argmax agreement (logits), the first
diverging index, and the largest sample-to-sample step (a click shows up as a
step larger than the reference's own).
"""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class ParityReport:
    shape: tuple
    max_abs: float
    mean_abs: float
    snr_db: float                  # inf when identical
    first_divergence: tuple | None  # first index with |Δ| > atol, or None
    argmax_agreement: float | None  # fraction of positions whose argmax over `axis` agrees
    max_step_ref: float | None      # largest |x[t+1] - x[t]| along the last axis
    max_step_test: float | None

    def to_dict(self):
        d = asdict(self)
        d["shape"] = list(self.shape)
        d["first_divergence"] = None if self.first_divergence is None else list(self.first_divergence)
        return d


def snr_db(reference, test):
    ref = np.asarray(reference, dtype=np.float64)
    err = np.asarray(test, dtype=np.float64) - ref
    noise = float(np.sum(err * err))
    if noise == 0.0:
        return float("inf")
    return 10.0 * np.log10(max(float(np.sum(ref * ref)), 1e-300) / noise)


def first_divergence(reference, test, atol=0.0):
    bad = np.abs(np.asarray(test, np.float64) - np.asarray(reference, np.float64)) > atol
    if not bad.any():
        return None
    return tuple(int(i) for i in np.unravel_index(int(np.argmax(bad)), bad.shape))


def argmax_agreement(reference, test, axis=-1):
    return float(np.mean(np.argmax(reference, axis=axis) == np.argmax(test, axis=axis)))


def max_step(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.max(np.abs(np.diff(x, axis=-1)))) if x.shape[-1] > 1 else 0.0


def step_percentile(x, q=99.9):
    """Percentile of |x[t+1] - x[t]|: the Stage 09 click metric."""
    return float(np.percentile(np.abs(np.diff(np.asarray(x, np.float64), axis=-1)), q))


def compare(reference, test, *, atol=0.0, argmax_axis=None, steps=True):
    ref, out = np.asarray(reference), np.asarray(test)
    if ref.shape != out.shape:
        raise ValueError(f"shape mismatch: reference {ref.shape}, test {out.shape}")
    diff = np.abs(out.astype(np.float64) - ref.astype(np.float64))
    return ParityReport(
        shape=ref.shape,
        max_abs=float(diff.max()) if diff.size else 0.0,
        mean_abs=float(diff.mean()) if diff.size else 0.0,
        snr_db=snr_db(ref, out),
        first_divergence=first_divergence(ref, out, atol),
        argmax_agreement=None if argmax_axis is None else argmax_agreement(ref, out, argmax_axis),
        max_step_ref=max_step(ref) if steps and ref.size else None,
        max_step_test=max_step(out) if steps and out.size else None,
    )
