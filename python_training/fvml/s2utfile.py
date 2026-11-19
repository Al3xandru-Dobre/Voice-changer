"""Run the shipped fv_s2ut_file translator (Milestone 3 / Stages 09-11 DoD).

Gates measure the shipped runtime, not PyTorch eager (MEMORY.md rule 11):
Mode B evaluation scripts drive the same fv_s2ut_file binary
(tools/fv_s2ut_file.cpp, InferencePipeline + OrtModel) the product runs.

  find_s2ut_file()  $FLUXVOICE_S2UT_FILE, else the newest fv_s2ut_file[.exe]
                    under any build*/ directory of the repo, else None
  convert()         one 16 kHz mono WAV -> one translated WAV (+ optional
                    --dump-trace JSONL); returns the wall-clock seconds
  convert_list()    an eval list -> <out_dir>/<utt_id>.wav + timing.csv, and
                    with trace=True one <utt_id>.trace.jsonl per utterance
                    (eval_s2ut.py picks those up for AL / AL_CA)

The binary's interface (16 kHz mono input):

  fv_s2ut_file --model-dir <bundle> --src-lang <code> --tgt-lang <code>
      [--voice <name>] [--voice-b <name>] [--wait-k K] [--adaptive]
      [--frames-per-step 1|2|4] [--preset P] [--param name=value]...
      [--change T:name=value]... [--dump-trace path.jsonl] [--raw]
      [--no-prosody-ctx] in.wav out.wav

Every flag after the languages and the voice goes through `extra_args`.
"""
import pathlib
import subprocess
import tempfile
import time

import soundfile as sf

from . import audio, paths
from .units import safe_name
from .vcfile import _is_16k_wav, find_tool

BINARY = "fv_s2ut_file"
ENV_VAR = "FLUXVOICE_S2UT_FILE"
NOT_FOUND = (f"no {BINARY}: set {ENV_VAR} or build an ORT-enabled tree "
             "(cmake --build <build> --target fv_s2ut_file)")


def find_s2ut_file(repo_root=None):
    """Path to an fv_s2ut_file binary, or None ($FLUXVOICE_S2UT_FILE wins)."""
    return find_tool(BINARY, ENV_VAR, repo_root)


def _languages(src, tgt):
    src, tgt = str(src or "").strip(), str(tgt or "").strip()
    if not src or not tgt:
        raise ValueError("fv_s2ut_file needs both a source and a target language")
    if src == tgt:
        raise ValueError(f"source and target language must differ (both '{src}'): "
                         "same-language conversion is Mode A (fv_vc_file)")
    return src, tgt


def build_command(s2ut_file, bundle, in_wav, out_wav, src, tgt, voice=None,
                  extra_args=(), trace=None):
    """The fv_s2ut_file argv for one translation."""
    src, tgt = _languages(src, tgt)
    cmd = [str(s2ut_file), "--model-dir", str(bundle), "--src-lang", src, "--tgt-lang", tgt]
    if voice:
        cmd += ["--voice", str(voice)]
    if trace:
        cmd += ["--dump-trace", str(trace)]
    cmd += [str(a) for a in extra_args]
    return cmd + [str(in_wav), str(out_wav)]


def convert(bundle, in_wav, out_wav, src, tgt, voice=None, extra_args=(), *,
            s2ut_file=None, trace=None):
    """Translate one file through fv_s2ut_file -> wall-clock seconds.

    `in_wav` must be a 16 kHz mono WAV (convert_list stages FLAC and other
    rates first). `trace` adds --dump-trace <trace>. A non-zero exit raises
    RuntimeError carrying the tool's output."""
    tool = pathlib.Path(s2ut_file) if s2ut_file else find_s2ut_file()
    if tool is None:
        raise FileNotFoundError(NOT_FOUND)
    if not tool.is_file():
        raise FileNotFoundError(f"{BINARY} not found at {tool}")
    cmd = build_command(tool, bundle, in_wav, out_wav, src, tgt, voice, extra_args, trace)
    info = sf.info(str(in_wav))
    if info.samplerate != audio.SAMPLE_RATE or info.channels != 1 \
            or info.format.upper() != "WAV":
        raise ValueError(f"{in_wav}: expected 16 kHz mono WAV "
                         f"(got {info.samplerate} Hz, {info.channels} ch, {info.format})")
    t0 = time.monotonic()
    # UTF-8 whatever the console codepage: voice display names are not ASCII.
    proc = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"{BINARY} failed ({proc.returncode}) for {in_wav} "
                           f"({src}->{tgt}):\n{proc.stdout}{proc.stderr}")
    return time.monotonic() - t0


def _retag_trace(trace_path, utt_id):
    """fv_s2ut_file tags trace lines with the input file stem; restore the
    list's utt_id when the staged input had to use a filesystem-safe stem."""
    import json
    p = pathlib.Path(trace_path)
    if not p.is_file():
        return
    lines = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            obj["utt_id"] = utt_id
            line = json.dumps(obj)
        lines.append(line)
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def convert_list(bundle, rows, out_dir, src=None, tgt=None, voice=None, extra_args=(),
                 limit=None, runner=None, trace=False, s2ut_file=None):
    """Translate every eval-list row -> <out_dir>/<utt_id>.wav + timing.csv.

    Rows are S2UT list rows (utt_id, src_lang, tgt_lang, src_path) or Mode A
    source rows (utt_id, path); `src` / `tgt` override the row languages and
    are required for rows without them. Paths are relative to the data root or
    absolute. Inputs that are not a 16 kHz mono WAV named <utt_id>.wav are
    staged as one in a scratch directory, so the converter's input rule holds
    and its trace lines carry the utt_id. `runner` is injectable for tests; it
    is called as runner(bundle, in_wav, out_wav, src, tgt, voice=..,
    extra_args=.., trace=.., s2ut_file=..) and returns seconds."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = list(rows)[:limit] if limit else list(rows)
    if runner is None:
        runner = convert
        s2ut_file = s2ut_file or find_s2ut_file()
        if s2ut_file is None:
            raise FileNotFoundError(NOT_FOUND)
    timing = []
    with tempfile.TemporaryDirectory(prefix="fv_s2ut_file_") as scratch:
        scratch = pathlib.Path(scratch)
        for r in selected:
            uid = str(r["utt_id"])
            row_src, row_tgt = _languages(src or r.get("src_lang"), tgt or r.get("tgt_lang"))
            source = paths.resolve(r.get("src_path") or r["path"])
            stem = safe_name(uid)
            in_wav = source if (_is_16k_wav(source) and source.stem == stem) else None
            if in_wav is None:
                in_wav = scratch / f"{stem}.wav"
                audio.save(in_wav, audio.load(source))
            dst = out_dir / f"{uid}.wav"
            if dst.resolve() == pathlib.Path(in_wav).resolve():
                raise ValueError(f"{uid}: the output {dst} would overwrite its source")
            trace_path = out_dir / f"{uid}.trace.jsonl" if trace else None
            seconds = runner(bundle, in_wav, dst, row_src, row_tgt, voice=voice,
                             extra_args=tuple(extra_args), trace=trace_path,
                             s2ut_file=s2ut_file)
            if trace_path is not None and stem != uid:
                _retag_trace(trace_path, uid)
            timing.append((uid, float(seconds)))
    with open(out_dir / "timing.csv", "w", encoding="utf-8", newline="") as f:
        f.write("utt_id,seconds\n")
        for utt_id, s in timing:
            f.write(f"{utt_id},{s:.4f}\n")
    return timing
