"""Target-voice recording preprocessing (Milestone 2 / Stage 10).

Pure numpy/scipy helpers driven by prepare_target_voice.py: high-pass,
loudness normalisation, an energy VAD with merge/pad/split rules, room-tone
SNR, clipping detection on the raw samples, deterministic held-out selection
and take-name parsing. Nothing here imports torch or an ML model, so the unit
tests run on CPU with synthetic audio.
"""
import math
import pathlib
import re

import numpy as np

FRAME_S = 0.02          # VAD / envelope frame


def take_pattern(speaker):
    """``<speaker>_sNN_TTTT`` — e.g. me_s01_0001 -> (session s01, take 0001)."""
    return re.compile(rf"^{re.escape(speaker)}_(s\d{{2}})_(\d{{4}})$")


def room_tone_pattern(speaker):
    """``<speaker>_sNN_roomtone`` — 30 s of room tone at the session start."""
    return re.compile(rf"^{re.escape(speaker)}_(s\d{{2}})_roomtone$")


def parse_take(path, speaker, pattern=None):
    """(session, take) of a take file name, e.g. me_s01_0001.wav -> ("s01", "0001").

    `pattern` overrides the default (recording rigs may rename files). Names
    that match neither the take nor the room-tone pattern are an error that
    lists the expected pattern."""
    stem = pathlib.Path(path).stem
    m = (pattern or take_pattern(speaker)).match(stem)
    if m:
        return m.group(1), m.group(2)
    raise ValueError(
        f"unexpected take name '{stem}': expected <speaker>_sNN_TTTT.wav "
        f"(e.g. {speaker}_s01_0001.wav); room tone is <speaker>_sNN_roomtone.wav")


def parse_room_tone(path, speaker, pattern=None):
    """Session id of a room-tone file, e.g. me_s01_roomtone.wav -> "s01"."""
    stem = pathlib.Path(path).stem
    m = (pattern or room_tone_pattern(speaker)).match(stem)
    if m:
        return m.group(1)
    raise ValueError(
        f"unexpected room-tone name '{stem}': expected "
        f"<speaker>_sNN_roomtone.wav (e.g. {speaker}_s01_roomtone.wav)")


def highpass(x, sr, hz=65.0):
    """4th-order Butterworth high-pass. Offline data prep: zero phase
    (sosfiltfilt) is fine and keeps the VAD envelope aligned."""
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, hz, "highpass", fs=sr, output="sos")
    return sosfiltfilt(sos, np.asarray(x, dtype=np.float64)).astype(np.float32)


def loudness_normalize(x, sr, target_lufs=-23.0):
    """Integrated-loudness normalisation -> (y, measured_lufs, gain_db).

    Raises ValueError on (near-)silence (< -70 LUFS): normalising it would
    just amplify noise."""
    import pyloudnorm
    meter = pyloudnorm.Meter(sr)
    loud = float(meter.integrated_loudness(np.asarray(x, dtype=np.float64)))
    if not math.isfinite(loud) or loud < -70.0:
        raise ValueError(f"take is (near-)silent ({loud:.1f} LUFS); nothing to normalise")
    gain_db = float(target_lufs - loud)
    return (np.asarray(x, np.float32) * 10.0 ** (gain_db / 20.0)).astype(np.float32), loud, gain_db


def frame_db(x, sr, frame_s=FRAME_S):
    """10*log10(mean square + 1e-12) per frame of `frame_s` seconds."""
    n = max(1, int(round(sr * frame_s)))
    m = len(x) // n
    if m == 0:
        return np.zeros(0, dtype=np.float64)
    f = np.asarray(x[: m * n], dtype=np.float64).reshape(m, n)
    return 10.0 * np.log10(np.mean(f * f, axis=1) + 1e-12)


def noise_floor_db(room_tone, sr):
    """Median frame level (dB) of a room-tone recording."""
    db = frame_db(room_tone, sr)
    if len(db) == 0:
        raise ValueError("room tone is empty")
    return float(np.median(db))


def estimate_snr_db(segment, sr, noise_floor):
    """SNR of a segment against a room-tone floor (dB). Speech power is the
    mean square of the segment's frames above floor + 10 dB."""
    db = frame_db(segment, sr)
    if len(db) == 0:
        return float("-inf")
    p_n = 10.0 ** (noise_floor / 10.0)
    sel = db >= noise_floor + 10.0
    if not sel.any():
        return float("-inf")
    p_s = float(np.mean(10.0 ** (db[sel] / 10.0)))
    return float(10.0 * np.log10(max(p_s - p_n, 1e-12) / p_n))


def is_clipped(raw, threshold=0.999):
    """True if any raw sample hits the converter's full scale."""
    return bool(np.any(np.abs(np.asarray(raw)) >= threshold))


def vad_segments(x, sr, noise_floor_dbv, margin_db=10.0, min_pause_s=0.3,
                 pad_s=0.1, min_s=2.0, max_s=12.0):
    """Energy VAD -> [(start, end)] in samples.

    Speech = frame_db >= floor + margin. Regions separated by < min_pause are
    merged, then padded by pad_s. Regions longer than max_s are split at the
    lowest-energy frame, each split leaving >= min_s on both sides (recursing
    until every region fits). Regions shorter than min_s are dropped."""
    x = np.asarray(x)
    frame_n = max(1, int(round(sr * FRAME_S)))
    db = frame_db(x, sr, FRAME_S)
    speech = db >= noise_floor_dbv + margin_db
    runs, i = [], 0
    while i < len(speech):
        if not speech[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(speech) and speech[j + 1]:
            j += 1
        runs.append((i * frame_n, (j + 1) * frame_n))
        i = j + 1
    gap = int(round(min_pause_s * sr))
    merged = []
    for a, b in runs:
        if merged and a - merged[-1][1] < gap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    pad = int(round(pad_s * sr))
    regions = [(max(0, a - pad), min(len(x), b + pad)) for a, b in merged]

    lo = int(round(min_s * sr))
    hi = int(round(max_s * sr))
    out = []

    def split(a, b):
        if b - a <= hi:
            out.append((a, b))
            return
        f_lo = int(math.ceil((a + lo) / frame_n))
        f_hi = int(math.floor((b - lo) / frame_n))
        if f_hi < f_lo:
            # No low-energy point leaves >= min_s on both sides: hard split
            # into equal parts, each <= max_s (parts < min_s are dropped by
            # the caller's filter — a kept region is never over max_s).
            n = max(2, math.ceil((b - a) / hi))
            step = (b - a) / n
            for j in range(n):
                s, e = a + round(j * step), a + round((j + 1) * step)
                if e - s >= lo:
                    out.append((s, e))
            return
        k = f_lo + int(np.argmin(db[f_lo:f_hi + 1]))
        split(a, k * frame_n)
        split(k * frame_n, b)

    for a, b in regions:
        split(a, b)
    return [(a, b) for a, b in out if b - a >= lo]


def select_heldout(clips, seconds=300.0):
    """Deterministic held-out selection -> {"heldout": {utt_id}, "guard":
    {utt_id}, "mode": "take" | "take-tail"}. `clips` are dicts with utt_id,
    duration, session, take, seg. Guard clips are excluded from both train
    and held-out so no training clip is adjacent to a held-out clip.

    Mode "take" (preferred): whole takes, sorted by (session, take) with
    evenly spaced picks grown until >= seconds — a held-out reference never
    shares a take with a training clip. Accepted only when the held-out
    total stays <= max(2 * seconds, seconds + 60) and >= 50 % of the audio
    remains for training; otherwise one long take would either overshoot
    badly or hold out the whole dataset.

    Mode "take-tail" (fallback): evenly spaced takes, each contributing the
    contiguous tail of its highest-seg clips worth >= seconds / n; the clip
    just before each tail becomes a guard."""
    from .evallist import evenly_spaced
    takes = {}
    for c in clips:
        takes.setdefault((c["session"], c["take"]), []).append(c)
    if sum(len(v) for v in takes.values()) < 2:
        raise ValueError("held-out selection needs at least 2 segments")
    keys = sorted(takes)
    take_dur = {k: sum(c["duration"] for c in takes[k]) for k in keys}
    total_dur = sum(take_dur.values())

    picked = []
    held = 0.0
    for n in range(1, len(keys) + 1):
        picked = evenly_spaced(keys, n)
        held = sum(take_dur[k] for k in picked)
        if held >= seconds:
            break
    if held >= seconds and held <= max(2.0 * seconds, seconds + 60.0) \
            and held <= 0.5 * total_dur:
        return {"heldout": {c["utt_id"] for k in picked for c in takes[k]},
                "guard": set(), "mode": "take"}

    for n in range(1, len(keys) + 1):
        picked = evenly_spaced(keys, n)
        share = seconds / len(picked)
        held_ids, guard_ids, held = set(), set(), 0.0
        for k in picked:
            segs = sorted(takes[k], key=lambda c: c["seg"])
            n_tail = 0
            tail_dur = 0.0
            while n_tail < len(segs) and tail_dur < share:
                tail_dur += segs[len(segs) - 1 - n_tail]["duration"]
                n_tail += 1
            held_ids.update(c["utt_id"] for c in segs[len(segs) - n_tail:])
            held += tail_dur
            if n_tail < len(segs):       # guard = clip just before the tail
                guard_ids.add(segs[len(segs) - n_tail - 1]["utt_id"])
        if held >= seconds or n == len(keys):
            return {"heldout": held_ids, "guard": guard_ids, "mode": "take-tail"}
