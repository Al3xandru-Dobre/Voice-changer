#!/usr/bin/env python3
"""Write the Stage 03 golden vectors to tests/data/pitch/.

The Python tracker (fvml/features.py) and its C++ port (src/dsp/PitchTracker.hpp)
are both tested against these files:

  header.json        every constant of the tracker + the case list
  <input>.pcm16      input signal, 16 kHz mono, little-endian int16 (read as
                     int16 / 32768, so both languages see identical floats)
  <input>_L<k>.f32   expected features for lookahead k: frames x 7 float32,
                     little-endian, row-major, columns as header["features"]

Inputs: `synthetic` (sawtooth glide 80->400 Hz, voiced/unvoiced alternation,
white noise, digital silence, a clipped segment) and `libri` (LibriSpeech
test-clean 61-70968-0058, CC BY 4.0; needs prepare_data.py --datasets eval).

Usage:
    make_pitch_golden.py            # (re)write tests/data/pitch/
    make_pitch_golden.py --check    # exit 1 if the files differ from a fresh run
"""
import argparse
import json
import sys

import numpy as np

from fvml import paths
from fvml import features as F

OUT_DIR = paths.REPO_ROOT / "tests" / "data" / "pitch"
SR = 16000
LOOKAHEADS = (0, 1, 2)
LIBRI_UTT = "61-70968-0058"
LIBRI_PATH = "raw/LibriSpeech/test-clean/61/70968/61-70968-0058.flac"


def _saw(freq_hz, seconds, amp):
    """Naive sawtooth with a (possibly time-varying) frequency."""
    f = np.broadcast_to(np.asarray(freq_hz, dtype=np.float64), (int(seconds * SR),))
    phase = np.cumsum(f) / SR
    return amp * (2.0 * (phase - np.floor(phase)) - 1.0)


def synthetic():
    rng = np.random.default_rng(20261018)
    n = int(1.2 * SR)
    glide = _saw(80.0 * (400.0 / 80.0) ** (np.arange(n) / n), 1.2, 0.5)
    alternation = np.concatenate([np.concatenate([_saw(150.0, 0.1, 0.4),
                                                  0.1 * rng.standard_normal(int(0.1 * SR))])
                                  for _ in range(3)])
    noise = 0.1 * rng.standard_normal(int(0.3 * SR))
    silence = np.zeros(int(0.2 * SR))
    clipped = np.clip(3.0 * np.sin(2 * np.pi * 220.0 * np.arange(int(0.4 * SR)) / SR), -1.0, 1.0)
    x = np.concatenate([glide, alternation, noise, silence, clipped])
    return np.clip(np.round(x * 32767.0), -32768, 32767).astype("<i2")


def libri():
    import soundfile as sf
    path = paths.resolve(LIBRI_PATH)
    if not path.is_file():
        sys.exit(f"missing {path}: run prepare_data.py --datasets eval")
    x, sr = sf.read(str(path), dtype="int16")
    assert sr == SR and x.ndim == 1
    return x.astype("<i2")


def as_float(pcm):
    return pcm.astype(np.float64) / 32768.0


def build():
    """{file name: bytes} for the whole folder."""
    inputs = {"synthetic": synthetic(), "libri": libri()}
    files, cases = {}, []
    for name, pcm in inputs.items():
        files[f"{name}.pcm16"] = pcm.tobytes()
        for la in LOOKAHEADS:
            feats = F.extract(as_float(pcm), F.PitchConfig(lookahead_frames=la))
            files[f"{name}_L{la}.f32"] = feats.astype("<f4").tobytes()
            cases.append(f"{name}_L{la}")
    head = F.header()
    head["cases"] = cases
    head["libri_utterance"] = LIBRI_UTT
    files["header.json"] = (json.dumps(head, indent=2) + "\n").encode()
    return files


def _read(path):
    data = path.read_bytes()
    return data.replace(b"\r\n", b"\n") if path.suffix == ".json" else data


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--check", action="store_true", help="compare instead of writing")
    p.add_argument("--out-dir", default=str(OUT_DIR))
    args = p.parse_args()
    out = paths.pathlib.Path(args.out_dir)
    files = build()
    if args.check:
        bad = [n for n, b in files.items() if not (out / n).is_file() or _read(out / n) != b]
        for n in bad:
            print(f"stale: {n}")
        return 1 if bad else 0
    out.mkdir(parents=True, exist_ok=True)
    for n, b in files.items():
        (out / n).write_bytes(b)
    print(f"wrote {len(files)} files ({sum(map(len, files.values())) / 1024:.0f} KiB) to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
