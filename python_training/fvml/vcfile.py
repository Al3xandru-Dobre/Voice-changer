"""Run the shipped fv_vc_file converter (Milestone 2 / Stage 09 DoD).

Gates measure the shipped runtime, not PyTorch eager (MEMORY.md rule 11):
evaluation scripts drive the same fv_vc_file binary the product uses.

find_vc_file() resolves the binary from $FLUXVOICE_VC_FILE, else the newest
fv_vc_file[.exe] under any build*/ directory of the repo. convert() runs one
file; convert_list() runs an eval list into an output directory and writes
timing.csv (seconds per conversion, for fvml.evaluate's RTF).
"""
import os
import pathlib
import subprocess
import tempfile
import time

import soundfile as sf

from . import audio, paths


def find_vc_file(repo_root=None):
    """Path to an ORT-enabled fv_vc_file, or None.

    $FLUXVOICE_VC_FILE wins; otherwise the newest fv_vc_file[.exe] in any
    build*/ tree under the repo root."""
    env = os.environ.get("FLUXVOICE_VC_FILE")
    if env and pathlib.Path(env).is_file():
        return pathlib.Path(env)
    root = pathlib.Path(repo_root) if repo_root else paths.REPO_ROOT
    cands = []
    for build in root.glob("build*"):
        if not build.is_dir():
            continue
        for p in build.rglob("fv_vc_file*"):
            if p.is_file() and p.suffix in ("", ".exe"):
                cands.append(p)
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def convert(vc_file, bundle, in_wav, out_wav, voice=None, args=()):
    """One conversion through fv_vc_file. `in_wav` must be 16 kHz mono WAV
    (FLAC and friends go through convert_list's scratch stage)."""
    vc_file = pathlib.Path(vc_file)
    if not vc_file.is_file():
        raise FileNotFoundError(f"fv_vc_file not found at {vc_file}")
    info = sf.info(str(in_wav))
    if info.samplerate != audio.SAMPLE_RATE or info.channels != 1 \
            or info.format.upper() != "WAV":
        raise ValueError(f"{in_wav}: expected 16 kHz mono WAV "
                         f"(got {info.samplerate} Hz, {info.channels} ch, {info.format})")
    cmd = [str(vc_file), "--model-dir", str(bundle)]
    if voice:
        cmd += ["--voice", str(voice)]
    cmd += [str(a) for a in args]
    cmd += [str(in_wav), str(out_wav)]
    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"fv_vc_file failed ({proc.returncode}) for {in_wav}:\n"
                           f"{proc.stdout}{proc.stderr}")
    return time.monotonic() - t0


def _is_16k_wav(path):
    try:
        info = sf.info(str(path))
    except Exception:
        return False
    return (info.samplerate == audio.SAMPLE_RATE and info.channels == 1
            and info.format.upper() == "WAV")


def convert_list(vc_file, bundle, rows, out_dir, voice=None, args=(), limit=None,
                 runner=convert):
    """Convert every eval-list row -> <out_dir>/<utt_id>.wav + timing.csv.

    Rows are dicts with utt_id and path (relative to the data root or
    absolute). Non-16 kHz-WAV sources (FLAC clips, other rates) are first
    written as float WAV in a scratch directory so the converter's input rule
    always holds. `runner` is injectable for tests; it receives
    (vc_file, bundle, in_wav, out_wav, voice, args) and returns seconds."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timing = []
    selected = list(rows)[:limit] if limit else list(rows)
    with tempfile.TemporaryDirectory(prefix="fv_vc_file_") as scratch:
        scratch = pathlib.Path(scratch)
        for i, r in enumerate(selected):
            src = paths.resolve(r["path"])
            in_wav = src if _is_16k_wav(src) else None
            if in_wav is None:
                x = audio.load(src)
                in_wav = scratch / f"in_{i}.wav"
                audio.save(in_wav, x)
            dst = out_dir / f"{r['utt_id']}.wav"
            seconds = runner(vc_file, bundle, in_wav, dst, voice, tuple(args))
            timing.append((r["utt_id"], seconds))
    with open(out_dir / "timing.csv", "w", encoding="utf-8", newline="") as f:
        f.write("utt_id,seconds\n")
        for utt_id, s in timing:
            f.write(f"{utt_id},{s:.4f}\n")
    return timing
