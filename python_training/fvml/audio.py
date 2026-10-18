"""Audio I/O: everything in training and evaluation is 16 kHz mono float32.

soundfile reads/writes, soxr resamples. torchaudio's I/O (torchcodec + FFmpeg
since 2.9) is deliberately not used.
"""
import numpy as np
import soundfile as sf
import soxr

SAMPLE_RATE = 16000


def resample(x, sr_in, sr_out=SAMPLE_RATE):
    if sr_in == sr_out:
        return np.asarray(x, dtype=np.float32)
    return soxr.resample(x, sr_in, sr_out, quality="VHQ").astype(np.float32, copy=False)


def load(path, sr=SAMPLE_RATE):
    """Mono float32 at `sr` (channels averaged, resampled if needed)."""
    x, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    return resample(x.mean(axis=1), file_sr, sr)


def save(path, x, sr=SAMPLE_RATE, subtype="FLOAT"):
    sf.write(str(path), np.asarray(x, dtype=np.float32), sr, subtype=subtype)


def info(path):
    """(duration seconds, sample rate, channels) from the header only."""
    i = sf.info(str(path))
    return i.frames / i.samplerate, i.samplerate, i.channels


def concat(paths, seconds, sr=SAMPLE_RATE):
    """Concatenate utterances until `seconds` of audio (parity runs need long
    streams, e.g. 60 s of dev-clean)."""
    need = int(round(seconds * sr))
    parts, have = [], 0
    for p in paths:
        x = load(p, sr)
        parts.append(x)
        have += len(x)
        if have >= need:
            return np.concatenate(parts)[:need]
    raise ValueError(f"only {have / sr:.1f} s available, {seconds} s requested")
