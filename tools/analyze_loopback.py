#!/usr/bin/env python3
"""Analyze a recorded loopback of a pure test tone for glitches.

The recording should contain a steady sine (default 440 Hz) that passed
through fluxvoice. Any dropout, click or sample discontinuity shows up as a dip
in the tone's envelope or a spike in its instantaneous frequency; a wrong
sample-rate path shows up as a frequency offset ("chipmunk" bug).

Usage:
    analyze_loopback.py recording.wav [--freq 440]
    analyze_loopback.py --self-test        # prove the detector catches glitches

Exit status: 0 = clean, 1 = glitches / wrong pitch / no tone found.

Limitation: a skip or repeat of (almost) a whole number of tone periods leaves
the phase continuous and is invisible with a pure tone; such events still show
up in fluxvoice's overrun/underrun counters.
Requires numpy.
"""
import argparse
import sys
import wave

import numpy as np

TRIM_S = 0.1            # ignore tone onset/offset transients
DIP_RATIO = 0.5         # envelope below 50 % of median = dropout
FREQ_TOL_HZ = 25.0      # 1 ms-smoothed instantaneous-frequency spike threshold
PITCH_TOL_HZ = 1.0      # mean frequency must be within this of the test tone


def read_wav_mono(path):
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            sys.exit(f"{path}: expected 16-bit PCM WAV")
        fs, ch = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64)
    return fs, x.reshape(-1, ch).mean(axis=1) / 32768.0


def analytic(x):
    """Analytic signal via FFT (numpy-only Hilbert transform)."""
    n = len(x)
    spec = np.fft.fft(x)
    h = np.zeros(n)
    h[0] = 1.0
    if n % 2 == 0:
        h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[1:(n + 1) // 2] = 2.0
    return np.fft.ifft(spec * h)


def event_count(mask):
    """Number of contiguous True runs."""
    m = np.concatenate(([False], mask, [False])).astype(np.int8)
    return int(np.count_nonzero(np.diff(m) == 1))


def analyze(fs, x, freq):
    # Tone extent from a 10 ms moving RMS. (Not the FFT envelope: the FFT is
    # circular, so a tone running to the end of the file would wrap into
    # sample 0 and make leading silence look like a dropout.)
    win10 = max(1, fs // 100)
    rms = np.sqrt(np.convolve(x * x, np.ones(win10) / win10, mode="same"))
    if rms.max() < 1e-3:
        return {"ok": False, "reason": "no tone found (silence)"}

    # Span from the tone's first to last appearance. Gaps *inside* the span are
    # exactly what we are looking for, so don't split on them.
    active = np.flatnonzero(rms > 0.25 * np.percentile(rms, 99))
    trim = int(TRIM_S * fs)
    a, b = int(active[0]) + trim, int(active[-1]) - trim
    if b - a < fs:
        return {"ok": False, "reason": "tone shorter than 1 s"}
    seg = x[a:b]

    z = analytic(seg)
    env = np.abs(z)
    inst_f = np.diff(np.unwrap(np.angle(z))) * fs / (2 * np.pi)
    win = max(1, fs // 1000)  # 1 ms smoothing
    inst_f = np.convolve(inst_f, np.ones(win) / win, mode="same")
    edge = 4 * win            # Hilbert edge effects at segment ends
    inst_f = inst_f[edge:-edge]
    env_c = env[edge:-edge]

    med_env = float(np.median(env_c))
    dips = env_c < DIP_RATIO * med_env
    spikes = np.abs(inst_f - freq) > FREQ_TOL_HZ
    mean_f = float(np.median(inst_f))

    res = {
        "sample_rate": fs,
        "tone_seconds": round((b - a) / fs, 2),
        "mean_freq_hz": round(mean_f, 2),
        "level_dbfs": round(20 * np.log10(med_env + 1e-12), 1),
        "dropout_events": event_count(dips),
        "click_events": event_count(spikes & ~dips[:len(spikes)]),
    }
    reasons = []
    if abs(mean_f - freq) > PITCH_TOL_HZ:
        reasons.append(f"pitch {mean_f:.2f} Hz != {freq} Hz (sample-rate path wrong?)")
    if res["dropout_events"]:
        reasons.append(f"{res['dropout_events']} dropout(s)")
    if res["click_events"]:
        reasons.append(f"{res['click_events']} click/discontinuity event(s)")
    res["ok"] = not reasons
    res["reason"] = "; ".join(reasons) or "clean"
    return res


def self_test():
    fs, f = 16000, 440.0
    t = np.arange(fs * 5) / fs
    clean = 0.25 * np.sin(2 * np.pi * f * t)
    pad = np.zeros(fs // 2)

    def run(sig):
        return analyze(fs, np.concatenate((pad, sig, pad)), f)

    gap = clean.copy()
    gap[2 * fs:2 * fs + 160] = 0.0                      # 10 ms dropout
    skip = np.concatenate((clean[:2 * fs], clean[2 * fs + 13:]))  # 13 dropped samples
    chip = 0.25 * np.sin(2 * np.pi * f * 1.02 * t)      # 2 % rate error

    lead = analyze(fs, np.concatenate((np.zeros(fs), clean)), f)  # tone runs to EOF
    cases = [("clean", clean, True), ("10 ms gap", gap, False),
             ("13-sample skip", skip, False), ("2 % rate error", chip, False)]
    ok = True
    for name, sig, expect_ok in cases:
        r = run(sig)
        good = r["ok"] == expect_ok
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {name:15s} -> {r['reason']}")
    good = lead["ok"]
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  {'lead-in, to EOF':15s} -> {lead['reason']}")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("wav", nargs="?")
    p.add_argument("--freq", type=float, default=440.0)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        return self_test()
    if not args.wav:
        p.error("wav file required")
    fs, x = read_wav_mono(args.wav)
    r = analyze(fs, x, args.freq)
    for k, v in r.items():
        print(f"{k:15s}: {v}")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
