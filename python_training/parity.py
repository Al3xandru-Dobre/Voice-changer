#!/usr/bin/env python3
"""Compare two signals or tensors: max |Δ|, SNR, argmax agreement, first divergence.

Inputs are .wav/.flac (compared at their own sample rate, mono) or .npy. Later
stages call fvml.parity.compare() directly; this CLI is for ad-hoc checks and
for files written by C++ (fv_vc_file) or ORT runs.

Usage:
    parity.py reference.wav test.wav [--atol 1e-4] [--min-snr 50] [--max-abs 1e-4]
    parity.py ref_logits.npy ort_logits.npy --argmax-axis -1 --min-argmax 0.999
    parity.py a.npy b.npy --json report.json

Exit status: 0 = within every given threshold (or none given), 1 = outside one.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import soundfile as sf

from fvml import parity


def load(path):
    p = pathlib.Path(path)
    if p.suffix == ".npy":
        return np.load(p)
    x, _ = sf.read(str(p), dtype="float32", always_2d=True)
    return x.mean(axis=1)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("reference")
    p.add_argument("test")
    p.add_argument("--atol", type=float, default=0.0, help="divergence tolerance (default 0)")
    p.add_argument("--argmax-axis", type=int, default=None, help="axis for argmax agreement (logits)")
    p.add_argument("--max-abs", type=float, help="fail if max |Δ| exceeds this")
    p.add_argument("--min-snr", type=float, help="fail if SNR (dB) is below this")
    p.add_argument("--min-argmax", type=float, help="fail if argmax agreement is below this")
    p.add_argument("--trim", action="store_true", help="cut both to the shorter length (signals)")
    p.add_argument("--json", help="also write the report here")
    args = p.parse_args()

    ref, test = load(args.reference), load(args.test)
    if args.trim and ref.ndim == test.ndim == 1:
        n = min(len(ref), len(test))
        ref, test = ref[:n], test[:n]
    r = parity.compare(ref, test, atol=args.atol, argmax_axis=args.argmax_axis)
    for k, v in r.to_dict().items():
        print(f"{k:18s}: {v}")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(r.to_dict(), indent=2) + "\n", encoding="utf-8")

    failures = []
    if args.max_abs is not None and r.max_abs > args.max_abs:
        failures.append(f"max |Δ| {r.max_abs:.3g} > {args.max_abs:g}")
    if args.min_snr is not None and r.snr_db < args.min_snr:
        failures.append(f"SNR {r.snr_db:.1f} dB < {args.min_snr:g} dB")
    if args.min_argmax is not None and (r.argmax_agreement or 0.0) < args.min_argmax:
        failures.append(f"argmax agreement {r.argmax_agreement} < {args.min_argmax:g}")
    print("verdict           : " + ("; ".join(failures) if failures else "within thresholds"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
