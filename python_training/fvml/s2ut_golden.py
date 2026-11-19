"""Mode B golden reference for fv_s2ut_file (Milestone 3 / Stage 07).

Spec: Documents/Milestone 3/07_cpp_translator_runtime_offline_runner.md,
"Golden parity check". CLI: python_training/s2ut_golden.py.

make()  runs the C++ Mode B hop (src/pipeline/InferencePipeline.cpp,
        State::process_s2ut + State::chunk_step, raw output:
        PipelineOptions.output_chain = false) step for step in Python ONNX
        Runtime, configured like OrtModel (CPU, one thread, sequential,
        ORT_ENABLE_ALL, no spinning, denormals as zero), with the
        fvml/waitk_policy.py twins of WaitKPolicy and UnitProsodyFifo. It
        writes <out>/input.wav, output.wav (32-bit float mono), trace.jsonl and
        meta.json.
check() compares a `fv_s2ut_file --raw --no-prosody-ctx --dump-trace` run with
        a golden: the same chunk steps with identical decisions, VAD verdicts,
        writes, policy counters, raw units, <EOS> and utterance ends (100%),
        and the same output length with max |delta| <= atol on a non-silent
        golden (MEMORY.md rule 11, BUG-017).

One hop (frames_per_step x 20 ms frames):
  1. per frame, VAD: the chunk is speech when any frame has
     10 log10(mean(x^2) + 1e-16) > -55 dBFS (double precision, sequential sum,
     like the C++ loop);
  2. the encoder runs on every hop (states carried); its `latents` fill a
     chunk_frames accumulator and a full chunk runs one chunk step;
  3. frames_per_step frames are popped from the FIFO: none -> a silent hop
     without a vocoder call; else the frames' (z_f0, voicing, aperiodicity,
     z_energy) are mapped into the voice's domain with the Stage 09 controls
     at their defaults, the vocoder runs, and the samples of BUG-028 pad
     frames are zeroed.
A chunk step is StreamingTranslator.push_chunk's order (fvml/s2ut_stream.py):

    flushes_before = policy.flushes_completed      # before on_chunk_energy()
    policy.on_chunk_energy(speech, chunk_ms)
    if policy.source_wanted():
        translator_enc; policy.notify_source_unit()
        if adaptive: policy.on_source_readiness(float32(sigmoid(readiness_logit)))
    decision = policy.decide()
    for w in range(policy.max_writes_this_step()):
        if w > 0 and policy.decide() == READ: break  # <EOS> in a pause: caught up
        if fifo.free() < chunk_frames: break          # back-pressure, the write stays owed
        translator_dec; fifo.push_chunk(); fifo.pad_to_multiple(frames_per_step)
        policy.notify_target_unit(eos)
        if policy.flushes_completed != flushes_before: break

plus what only the C++ worker does: the FIFO with BUG-028 pads, the
translator's own `prosody` rows in it, readiness only when translator_enc has
that output (as the C++ readiness_from_logit: a plain sigmoid, while
s2ut_stream.py now maps 2 x sigmoid, BUG-029). A completed flush zeroes both
translators and returns prev_units to <BOS>.

The input is zero-padded to whole hops (scheduled --change controls apply
before the first hop starting at or after their time, like fv_s2ut_file).
Zero hops follow while the pipeline is busy (policy active, FIFO not empty, or
a partial chunk that saw speech) or the last output sample is audible
(|x| > 0.001), at most max_tail_hops of them (default: tail_flush_ms +
(max_flush_steps + k + 2) chunks).

prosody_ctx is always [0, 0]. fv_s2ut_file's default feeds the utterance's
pitch-tracker summary (src/dsp/PitchTracker.hpp), whose Python port
(fvml/features.py) matches it to a tolerance only, so goldens are made for
--no-prosody-ctx runs.

trace.jsonl: one object per chunk step, keys TRACE_KEYS. `t_src_ms` is the
input consumed when the step ran (tail zeros included), `t_tgt_frames_emitted`
the valid target frames queued (pads excluded), `elapsed_ms` the compute time
of the hop that ran the step (informational), `units` the raw translator_dec
units of every write (specials included), `segment` the utterances completed
before the step. Loadable by fvml.s2ut_eval.load_traces.

Standard library only at import time, in check() and in self_test(), so a C++
CTest leg can compare a golden with a plain `python -I`. make() imports numpy,
onnxruntime and onnx.
"""
import array
import hashlib
import json
import math
import pathlib
import platform
import struct
import sys
import tempfile
import time
from dataclasses import dataclass

from . import contract as fc
from .waitk_policy import Decision, UnitProsodyFifo, WaitKConfig, WaitKPolicy, _f32

SCHEMA = 1
KIND = "fluxvoice.s2ut_golden"
GOLDEN_FILES = ("input.wav", "output.wav", "trace.jsonl", "meta.json")

# S2utOptions defaults (src/pipeline/InferencePipeline.hpp) and Stage 09 controls
# at their VoiceParams defaults (src/voice/VoiceParams.hpp kParams).
SPEECH_THRESHOLD_DBFS = -55.0
EOS_THRESHOLD = 0.5
FIFO_CAPACITY_FRAMES = 64
MAX_FIFO_FRAMES = 4096            # InferencePipeline.cpp kMaxFifoFrames
DEFAULT_WAIT_K = 3                # LangCatalog.hpp kDefaultWaitK
INTONATION = 1.0
PITCH_SHIFT_ST = 0.0
WHISPER = 0.0
VIBRATO = 0.0                     # vibrato_depth 0: the LFO adds exactly 0
LN2 = 0.69314718055994530942
FLT_MAX = 3.4028234663852886e38

AUDIBLE_SAMPLE = 0.001           # fv_s2ut_file: tail hops continue while |last sample| > this
# fv_s2ut_file --change controls the golden reproduces: VoiceParams kParams
# (name -> kind, min, max). The others change audio-chain/voice state.
CHANGE_PARAMS = {"wait_k": ("integer", 1.0, 10.0), "adaptive_policy": ("toggle", 0.0, 1.0)}

DEFAULT_ATOL = 1e-5               # Stage 07 DoD: output max |delta| <= 1e-5
SILENT_PEAK_FACTOR = 100.0        # a golden peaking below 100 x atol makes |delta| vacuous

TRACE_KEYS = ("utt_id", "step", "t_src_ms", "t_tgt_frames_emitted", "elapsed_ms", "decision",
              "speech", "encoded", "writes", "t_src", "t_tgt", "units", "eos", "utterance_end",
              "cap_end", "pad_frames", "segment")
# check(): must be present and identical in every step ...
REQUIRED_KEYS = ("decision", "speech", "encoded", "writes", "t_src", "t_tgt", "units", "eos",
                 "utterance_end", "t_tgt_frames_emitted")
# ... compared when the test trace has them too (elapsed_ms and utt_id never).
EXTRA_KEYS = ("step", "segment", "pad_frames", "cap_end", "t_src_ms")
BOOL_KEYS = ("speech", "encoded", "eos", "utterance_end", "cap_end")
DECISIONS = tuple(d.name for d in Decision)

assert array.array("f").itemsize == 4 and array.array("h").itemsize == 2


class GoldenError(Exception):
    """Invalid input, bundle or option: no golden is written."""


# -- WAV I/O (twin of tools/fv_vc_file.cpp read_mono_wav / write_mono_wav) -------
def read_wav(path):
    """(sample_rate, array('f')) of a mono RIFF/WAVE file, 16-bit PCM (int16 /
    32768, exact in float32) or 32-bit IEEE float, also as
    WAVE_FORMAT_EXTENSIBLE. Same chunk walk and rejections as the C++ reader."""
    path = pathlib.Path(path)
    try:
        buf = path.read_bytes()
    except OSError:
        raise GoldenError(f"cannot open input WAV: {path}") from None
    if len(buf) < 44:
        raise GoldenError(f"input WAV is too short: {path}")
    if buf[0:4] != b"RIFF" or buf[8:12] != b"WAVE":
        raise GoldenError(f"input file is not a RIFF/WAVE file: {path}")
    fmt = data = None
    pos = 12
    while pos + 8 <= len(buf):
        chunk_id = buf[pos:pos + 4]
        size = struct.unpack_from("<I", buf, pos + 4)[0]
        payload = pos + 8
        if payload + size > len(buf):
            break
        if chunk_id == b"fmt ":
            if size < 16:
                raise GoldenError(f"malformed fmt chunk in {path}")
            tag, channels, rate = struct.unpack_from("<HHI", buf, payload)
            bits = struct.unpack_from("<H", buf, payload + 14)[0]
            if tag == 0xFFFE and size >= 40:
                tag = struct.unpack_from("<H", buf, payload + 24)[0]
            fmt = (tag, channels, rate, bits)
        elif chunk_id == b"data":
            data = buf[payload:payload + size]
            break
        pos = payload + size + (size & 1)
    if fmt is None or data is None:
        raise GoldenError(f"WAV file missing fmt or data chunk: {path}")
    tag, channels, rate, bits = fmt
    if channels != 1:
        raise GoldenError(f"WAV must be mono (got {channels} channels)")
    if tag == 1 and bits == 16:
        pcm = array.array("h")
        pcm.frombytes(data[:len(data) // 2 * 2])
        if sys.byteorder == "big":
            pcm.byteswap()
        return rate, array.array("f", [s / 32768.0 for s in pcm])
    if tag == 3 and bits == 32:
        x = array.array("f")
        x.frombytes(data[:len(data) // 4 * 4])
        if sys.byteorder == "big":
            x.byteswap()
        return rate, x
    raise GoldenError(f"unsupported WAV format (format_tag={tag}, bits_per_sample={bits}; "
                      "expected 16-bit PCM or 32-bit float)")


def write_wav_f32(path, sample_rate, samples):
    """32-bit float mono WAV with the C++ writer's 44-byte header. `samples`
    is an array('f'), a numpy array or any iterable of floats."""
    if hasattr(samples, "astype"):                       # numpy, without importing it
        data = samples.astype("<f4").tobytes()
    else:
        x = array.array("f", samples)
        if sys.byteorder == "big":
            x.byteswap()
        data = x.tobytes()
    rate = int(sample_rate)
    header = (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE" + b"fmt "
              + struct.pack("<IHHIIHH", 16, 3, 1, rate, rate * 4, 4, 32)
              + b"data" + struct.pack("<I", len(data)))
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + data)


# -- trace I/O -------------------------------------------------------------------
def read_trace(path):
    """Chunk steps of a JSONL trace: one step object per line, or
    {"utt_id": ..., "steps": [...]} lines (the forms load_traces reads)."""
    path = pathlib.Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        raise GoldenError(f"cannot read trace: {path}") from None
    steps = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise GoldenError(f"{path}:{n}: invalid JSON ({e.msg})") from None
        if isinstance(obj, dict) and isinstance(obj.get("steps"), list):
            steps.extend(obj["steps"])
        elif isinstance(obj, dict):
            steps.append(obj)
        else:
            raise GoldenError(f"{path}:{n}: expected a JSON object per line")
    if not all(isinstance(s, dict) for s in steps):
        raise GoldenError(f"{path}: every step must be a JSON object")
    return steps


def write_trace(path, steps):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for s in steps:
            f.write(json.dumps(s, separators=(", ", ": ")) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# -- small rules shared with the C++ worker ---------------------------------------
def default_max_tail_hops(policy_config, hop, sample_rate):
    """Zero hops allowed after the input: ceil((tail_flush_ms + (max_flush_steps
    + k + 2) x chunk_ms) / hop_ms), in exact integer arithmetic."""
    c = policy_config
    chunk_ms = c.chunk_frames * c.frame_ms
    num = (c.tail_flush_ms + (c.max_flush_steps + c.k + 2) * chunk_ms) * int(sample_rate)
    return -(-num // (int(hop) * 1000))


def fifo_capacity(requested, chunk):
    """InferencePipeline.cpp fifo_capacity(): >= 2 chunks, a multiple of a chunk."""
    if requested > MAX_FIFO_FRAMES:
        raise GoldenError(f"Mode B FIFO capacity exceeds {MAX_FIFO_FRAMES} frames")
    want = max(requested, 2 * chunk)
    return (want + chunk - 1) // chunk * chunk


def readiness_from_logit(logit):
    """InferencePipeline.cpp readiness_from_logit (double sigmoid -> float);
    exp overflow gives readiness 0 like std::exp's inf."""
    try:
        return _f32(1.0 / (1.0 + math.exp(-float(logit))))
    except OverflowError:
        return 0.0


def frame_is_speech(frame, threshold_dbfs):
    """Raw-level VAD of one frame (floats): sequential double sum of x^2,
    exactly the C++ loop (float x float is exact in double)."""
    ms = 0.0
    for v in frame:
        ms += v * v
    ms /= len(frame)
    return 10.0 * math.log10(ms + 1e-16) > threshold_dbfs


def parse_change(text):
    """`<seconds>:<name>=<value>` -> (seconds, name, value) like fv_s2ut_file's
    --change (tools/ToolCli.hpp parse_change_option + Presets.cpp
    parse_param_assignment + VoiceParams clamp_param), for CHANGE_PARAMS only.
    Values are clamped to the control's range; wait_k rounds half away from 0."""
    head, sep, assignment = str(text).partition(":")
    name, eq, rhs = assignment.partition("=")
    if not sep:
        raise GoldenError(f"Invalid --change '{text}' (expected seconds:name=value)")
    try:
        seconds = float(head)
    except ValueError:
        seconds = math.nan
    if not math.isfinite(seconds) or seconds < 0.0:
        raise GoldenError(f"Invalid --change '{text}' (bad seconds)")
    if not eq or not name or not rhs:
        raise GoldenError(f"--change: expected name=value, got '{assignment}'")
    if name not in CHANGE_PARAMS:
        raise GoldenError(f"--change: the golden reproduces {' and '.join(CHANGE_PARAMS)} "
                          f"changes only (got '{name}')")
    kind, lo, hi = CHANGE_PARAMS[name]
    if kind == "toggle" and rhs in ("on", "off", "true", "false"):
        v = 1.0 if rhs in ("on", "true") else 0.0
    else:
        try:
            v = float(rhs)
        except ValueError:
            v = math.nan
        if not math.isfinite(v):
            raise GoldenError(f"--change: expected name=value ('{rhs}' is not a finite number)")
        v = _f32(v)
    v = min(max(v, lo), hi)
    v = float(int(_f32(v + 0.5))) if kind == "integer" else (1.0 if v >= 0.5 else 0.0)
    return seconds, name, v


def resolve_lang(langs_doc, code_or_id):
    """LangCatalog::find(): a language code, else a decimal id (<= 5 digits)."""
    for entry in langs_doc["langs"]:
        if entry["code"] == code_or_id:
            return entry
    s = str(code_or_id)
    if not s or len(s) > 5 or not all("0" <= ch <= "9" for ch in s):
        return None
    for entry in langs_doc["langs"]:
        if entry["id"] == int(s):
            return entry
    return None


def locate_bundle(model_dir, frames_per_step=0):
    """ModelBundle.cpp locate_s2ut_bundle(): frames_per_step 0 picks the first
    complete encoder_s<n>/vocoder_s<n> pair in 1, 2, 4 order."""
    d = pathlib.Path(model_dir)
    if frames_per_step not in (0, 1, 2, 4):
        raise GoldenError("frames-per-step must be 1, 2 or 4")
    if not d.is_dir():
        raise GoldenError(f"model bundle directory not found: {d}")
    for fps in (1, 2, 4):
        if frames_per_step and frames_per_step != fps:
            continue
        enc, voc = d / f"encoder_s{fps}.onnx", d / f"vocoder_s{fps}.onnx"
        if enc.is_file() and voc.is_file():
            break
    else:
        variant = f" s{frames_per_step}" if frames_per_step else ""
        raise GoldenError(f"no complete encoder/vocoder bundle variant{variant} in {d}")
    paths = {"encoder": enc, "translator_enc": d / "translator_enc.onnx",
             "translator_dec": d / "translator_dec.onnx", "vocoder": voc,
             "voices": d / "voices.json", "langs": d / "langs.json"}
    if not (paths["translator_enc"].is_file() and paths["translator_dec"].is_file()):
        raise GoldenError("bundle has no translator models (translator_enc.onnx / "
                          f"translator_dec.onnx) in {d}")
    if not paths["langs"].is_file():
        raise GoldenError(f"Mode B bundle has no langs.json in {d}")
    return paths, fps


# -- ONNX Runtime ------------------------------------------------------------------
def ort_session(path):
    """CPU session configured like OrtModel.cpp (and fvml.export.ort_session,
    which needs torch): one thread, sequential, ORT_ENABLE_ALL, no spinning,
    denormals flushed to zero."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    so.add_session_config_entry("session.inter_op.allow_spinning", "0")
    so.add_session_config_entry("session.set_denormal_as_zero", "1")
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


class OrtStep:
    """One contract model stepped in ONNX Runtime: every state_out.X is fed back
    as state_in.X (OrtModel's KVStateCache) and reset() zeroes the states."""

    _NP = {"f32": "float32", "i64": "int64", "f16": "float16"}

    def __init__(self, path):
        self.path = pathlib.Path(path)
        try:
            self.contract, proto = fc.load_onnx(self.path)
        except fc.ContractError as e:
            raise GoldenError(f"{self.path.name}: {e}") from None
        self.graph_inputs, self.graph_outputs = fc.graph_io(proto)
        self.input_names = {t.name for t in self.graph_inputs}
        by_name = {t.name: t for t in self.graph_inputs}
        self.state_specs = [by_name[fc.STATE_IN + s] for s in self.contract.states]
        try:
            self.session = ort_session(self.path)
        except Exception as e:      # noqa: BLE001 - ORT pybind raises RuntimeError subclasses
            raise GoldenError(f"{self.path.name}: ONNX Runtime refused the model: "
                              f"{str(e).splitlines()[0] if str(e) else type(e).__name__}") from None
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.steps = 0
        self.reset()

    def reset(self):
        import numpy as np
        self.state = {t.name: np.zeros(t.shape, self._NP[t.dtype]) for t in self.state_specs}

    def step(self, feeds):
        try:
            values = self.session.run(self.output_names, {**feeds, **self.state})
        except Exception as e:      # noqa: BLE001
            raise GoldenError(f"{self.path.name}: Run failed at step {self.steps}: {e}") from None
        out = dict(zip(self.output_names, values))
        for s in self.contract.states:
            self.state[fc.STATE_IN + s] = out.pop(fc.STATE_OUT + s)
        self.steps += 1
        return out


# -- the Mode B pipeline twin -----------------------------------------------------
@dataclass
class GoldenOptions:
    """What `make` exposes of fv_s2ut_file's options. The rest is fixed at the
    C++ defaults: S2utOptions (VAD -55 dBFS, eos_prob > 0.5, 64-frame FIFO),
    WaitKPolicy::Config, VoiceParams, output_chain = false, prosody_ctx = [0, 0]."""
    src_lang: str
    tgt_lang: str
    voice: str | None = None            # None: the bundle's first voice
    wait_k: int | None = None           # None: langs.json default_wait_k for the pair, else 3
    frames_per_step: int = 4            # fv_s2ut_file's default; 0: first complete of 1, 2, 4
    adaptive: bool = False
    max_tail_hops: int | None = None    # None: default_max_tail_hops()
    # Scheduled Stage 09 policy controls, fv_s2ut_file --change: ((seconds,
    # "wait_k" | "adaptive_policy", value), ...) as parse_change() returns them.
    changes: tuple = ()


class ModeBPipeline:
    """InferencePipeline (Mode B, raw output) in Python ORT; process() is one
    hop, run() a whole file. Chunk steps are appended to `steps` in the
    trace.jsonl schema."""

    def __init__(self, model_dir, options, utt_id="utt"):
        import numpy as np
        self._np = np
        self.model_dir = pathlib.Path(model_dir)
        self.options = options
        self.utt_id = str(utt_id)
        self.paths, self.fps = locate_bundle(model_dir, int(options.frames_per_step or 0))
        self.enc = OrtStep(self.paths["encoder"])
        self.tenc = OrtStep(self.paths["translator_enc"])
        self.tdec = OrtStep(self.paths["translator_dec"])
        self.voc = OrtStep(self.paths["vocoder"])
        ec, tc, dc, vc = (m.contract for m in (self.enc, self.tenc, self.tdec, self.voc))
        try:
            fc.validate_s2ut_bundle(ec, tc, dc, vc)
            fc.validate_s2ut_shared_io(tc, self.tenc.graph_outputs, dc, self.tdec.graph_inputs)
        except fc.ContractError as e:
            raise GoldenError(str(e)) from None
        if ec.frames_per_step != self.fps:
            raise GoldenError("bundle filename variant disagrees with model frames_per_step")
        self._check_geometry(ec, tc, vc)

        self.sample_rate = ec.sample_rate
        self.frame_samples = ec.frame_samples
        self.hop = ec.hop
        self.chunk = tc.chunk_frames
        self.latent_dim = ec.latent_dim
        self.cond_names = tuple(n for n in ("logf0", "voicing", "aperiodicity", "energy")
                                if n in vc.cond_inputs)
        self.has_spk = "spk_emb" in self.voc.input_names
        self.has_readiness = tc.has_readiness_output
        self.has_prosody = dc.has_prosody_output
        self.shared = tuple(tc.shared_mem_states)
        self.bos = dc.special_units["bos"]
        self.speech_threshold = _f32(SPEECH_THRESHOLD_DBFS)
        self.eos_threshold = _f32(EOS_THRESHOLD)

        self._load_langs(tc)
        self._load_voice(vc)
        k = self.wait_k
        if not 1 <= k <= 10:
            raise GoldenError("wait-k must be in 1..10")
        self.policy = WaitKPolicy(WaitKConfig(
            k=k, adaptive_mode=bool(options.adaptive), chunk_frames=self.chunk,
            frame_ms=self.frame_samples * 1000 // self.sample_rate))
        self.chunk_ms = self.policy.chunk_ms
        self.fifo = UnitProsodyFifo(fifo_capacity(FIFO_CAPACITY_FRAMES, self.chunk), vc.vocab,
                                    dc.special_units["eos"])
        self.max_tail_hops = (default_max_tail_hops(self.policy.config, self.hop, self.sample_rate)
                              if options.max_tail_hops is None else int(options.max_tail_hops))
        if self.max_tail_hops < 0:
            raise GoldenError("max-tail-hops must be >= 0")
        self.changes = sorted(((float(s), str(n), float(v)) for s, n, v in options.changes),
                              key=lambda c: c[0])             # stable: same-time order kept
        for _, name, _ in self.changes:
            if name not in CHANGE_PARAMS:
                raise GoldenError(f"unsupported --change control '{name}'")

        neutral = [0.0] * (self.chunk * UnitProsodyFifo.PROSODY_DIMS)
        neutral[1::UnitProsodyFifo.PROSODY_DIMS] = [1.0] * self.chunk   # mean pitch, voiced
        self.neutral_prosody = neutral
        self.src_lang_arr = np.asarray([self.src_lang["id"]], np.int64)
        self.tgt_lang_arr = np.asarray([self.tgt_lang["id"]], np.int64)
        self.prosody_ctx = np.zeros((1, 2), np.float32)
        self.chunk_latents = np.zeros((self.chunk, self.latent_dim), np.float32)
        self.chunk_fill = 0
        self.chunk_speech = False
        self.prev_units = np.full(self.chunk, self.bos, np.int64)
        self.utt_eos = False
        self.chunk_index = 0
        self.segment = 0
        self.input_samples = 0
        self.steps = []
        self.stats = {"reads": 0, "writes": 0, "flushes": 0, "utterances": 0, "eos_ends": 0,
                      "cap_ends": 0, "dec_steps": 0, "voc_steps": 0, "fifo_stalls": 0,
                      "pad_frames": 0}

    # -- load-time checks (State / ModeBState constructors) -------------------
    @staticmethod
    def _check_geometry(ec, tc, vc):
        if ec.sample_rate % 50 != 0 or vc.cond_frame_samples != ec.sample_rate // 50:
            raise GoldenError("Mode A requires 20 ms pitch conditioning frames")
        if vc.overlap_samples != 0:
            raise GoldenError("Mode B requires a stateful vocoder without overlap-add "
                              "(overlap_samples = 0)")
        if vc.cond_frame_samples != vc.frame_samples:
            raise GoldenError("Mode B requires one conditioning frame per content frame")
        if not 1 <= tc.chunk_frames <= 16:
            raise GoldenError("Mode B supports translator chunk_frames 1..16")

    def _load_langs(self, tc):
        try:
            doc = json.loads(self.paths["langs"].read_text(encoding="utf-8"))
            fc.validate_langs(doc, tc)
        except (OSError, ValueError) as e:      # ContractError and JSONDecodeError are ValueErrors
            raise GoldenError(f"langs.json: {e}") from None
        o = self.options
        self.src_lang = resolve_lang(doc, o.src_lang)
        self.tgt_lang = resolve_lang(doc, o.tgt_lang)
        for given, entry, side in ((o.src_lang, self.src_lang, "src"),
                                   (o.tgt_lang, self.tgt_lang, "tgt")):
            if entry is None:
                codes = ", ".join(e["code"] for e in doc["langs"])
                raise GoldenError(f"unknown --{side}-lang '{given}' (bundle languages: {codes})")
        if self.src_lang["id"] == self.tgt_lang["id"]:
            raise GoldenError("src_lang and tgt_lang must differ")
        pair = f"{self.src_lang['code']}->{self.tgt_lang['code']}"
        self.wait_k = int(o.wait_k) if o.wait_k is not None else \
            int(doc.get("default_wait_k", {}).get(pair, DEFAULT_WAIT_K))

    def _load_voice(self, vc):
        np = self._np
        try:
            doc = json.loads(self.paths["voices"].read_text(encoding="utf-8"))
            fc.validate_voices(doc, vc)
        except OSError:
            raise GoldenError(f"cannot open voices.json: {self.paths['voices']}") from None
        except ValueError as e:
            raise GoldenError(f"voices.json: {e}") from None
        voices = doc["voices"]
        names = [v["name"] for v in voices]
        name = self.options.voice or names[0]
        if name not in names:
            raise GoldenError(f"unknown voice '{name}' (bundle voices: {', '.join(names)})")
        self.voice_index = names.index(name)
        v = voices[self.voice_index]
        self.voice = {k: float(v[k]) for k in ("logf0_mean", "logf0_std", "energy_mean",
                                               "energy_std")}
        self.voice_name = name
        self.spk_emb = np.asarray(v["embedding"], np.float32)[None]

    # -- one hop -----------------------------------------------------------------
    def idle(self):
        """InferencePipeline::s2ut_idle()."""
        return (not self.policy.utterance_active() and self.fifo.empty()
                and not (self.chunk_fill > 0 and self.chunk_speech))

    def _map_frame(self, z_f0, voicing, aperiodicity, z_energy):
        """map_frame() with the Stage 09 defaults, same order of operations."""
        v = self.voice
        logf0 = (v["logf0_mean"] + INTONATION * v["logf0_std"] * z_f0
                 + PITCH_SHIFT_ST * LN2 / 12.0) + VIBRATO
        return {"logf0": logf0, "voicing": voicing * (1.0 - WHISPER),
                "aperiodicity": aperiodicity,
                "energy": v["energy_mean"] + v["energy_std"] * z_energy}

    def process(self, hop_audio):
        """State::process_s2ut(): one hop of input in, one hop of output out."""
        np = self._np
        x = np.ascontiguousarray(hop_audio, np.float32).reshape(-1)
        if x.size != self.hop:
            raise GoldenError("pipeline requires exactly one input/output hop")
        fs = self.frame_samples
        hop_index = self.input_samples // self.hop
        for i in range(self.fps):
            if frame_is_speech(x[i * fs:(i + 1) * fs].tolist(), self.speech_threshold):
                self.chunk_speech = True
        self.input_samples += self.hop
        latents = self.enc.step({"audio": x[None]})["latents"]
        self.chunk_latents[self.chunk_fill:self.chunk_fill + self.fps] = \
            np.asarray(latents, np.float32).reshape(self.fps, self.latent_dim)
        self.chunk_fill += self.fps
        if self.chunk_fill == self.chunk:
            self.chunk_fill = 0
            self._chunk_step()

        out = np.zeros(self.hop, np.float32)
        popped = self.fifo.pop_hop_with_pads(self.fps)
        if popped is not None:
            units, rows, pads = popped
            real = pads.index(True) if True in pads else self.fps
            feeds = {"units": np.asarray([units], np.int64)}
            cond = {n: np.zeros((1, self.fps), np.float32) for n in self.cond_names}
            for i, row in enumerate(rows):
                mapped = self._map_frame(*row)
                for n in self.cond_names:
                    value = mapped[n]
                    if not math.isfinite(value) or abs(value) > FLT_MAX:
                        raise GoldenError(f"non-finite model conditioning ({n}) at hop {hop_index}")
                    cond[n][0, i] = value                    # double -> float, nearest
            feeds.update(cond)
            if self.has_spk:
                feeds["spk_emb"] = self.spk_emb
            audio = self.voc.step(feeds)["audio"]
            self.stats["voc_steps"] += 1
            out[:] = np.asarray(audio, np.float32).reshape(-1)[:self.hop]
            out[real * fs:] = 0.0                            # BUG-028: pads render as silence
        if not np.all(np.isfinite(out)):
            raise GoldenError(f"vocoder produced non-finite audio at hop {hop_index}")
        return out

    def _chunk_step(self):
        """State::chunk_step(): the Stage 04 caller protocol for one chunk."""
        np = self._np
        p = self.policy
        flushes_before = p.flushes_completed        # before on_chunk_energy(), like the C++
        speech = self.chunk_speech
        self.chunk_speech = False
        if speech:                                  # utt_eos = "<EOS> seen in a pause"
            self.utt_eos = False
        p.on_chunk_energy(speech, self.chunk_ms)
        step = {"step": self.chunk_index, "segment": self.segment, "encoded": False,
                "writes": 0, "frames": 0, "pads": 0, "eos": False, "cap_end": False, "units": []}
        self.chunk_index += 1
        if p.source_wanted():
            out = self.tenc.step({"latents": self.chunk_latents[None].copy(),
                                  "src_lang": self.src_lang_arr})
            step["encoded"] = True
            p.notify_source_unit()
            if self.has_readiness:
                logit = float(np.asarray(out["readiness_logit"]).reshape(-1)[0])
                if p.config.adaptive_mode:
                    p.on_source_readiness(readiness_from_logit(logit))
        decision = p.decide()
        self.stats[{Decision.READ: "reads", Decision.WRITE: "writes",
                    Decision.FLUSH: "flushes"}[decision]] += 1

        completed_in_loop = last_eos = False
        for w in range(p.max_writes_this_step()):    # 0 on READ, 1 on FLUSH
            # An adaptive double write must still be wanted after the first: an
            # <EOS> in a pause marks the utterance caught up (decide() -> READ).
            if w > 0 and p.decide() == Decision.READ:
                break
            if self.fifo.free() < self.chunk:        # back-pressure: the write stays owed
                self.stats["fifo_stalls"] += 1
                break
            feeds = {"prev_units": self.prev_units[None].copy(), "tgt_lang": self.tgt_lang_arr,
                     "prosody_ctx": self.prosody_ctx,
                     "src_pos": self.tenc.state[fc.STATE_IN + "src_pos"]}
            for m in self.shared:
                feeds[m] = self.tenc.state[fc.STATE_IN + m]
            out = self.tdec.step(feeds)
            self.stats["dec_steps"] += 1
            units = np.asarray(out["units"], np.int64).reshape(-1)
            step["units"] += [int(u) for u in units]
            prosody = (np.asarray(out["prosody"], np.float32).reshape(-1).tolist()
                       if self.has_prosody else self.neutral_prosody)
            pushed = self.fifo.push_chunk(units.tolist(), prosody)
            pads = self.fifo.pad_to_multiple(self.fps)
            step["frames"] += pushed.pushed
            step["pads"] += pads
            self.stats["pad_frames"] += pads
            eos_prob = float(np.asarray(out["eos_prob"], np.float32).reshape(-1)[0])
            eos = pushed.eos or eos_prob > self.eos_threshold
            self.prev_units = units.copy()           # the raw emission is the next input
            step["eos"] = step["eos"] or eos
            # Exactly when the policy records "caught up" (an <EOS> in a pause).
            if eos and not p.flushing() and p.utterance_active() and p.silence_ms > 0:
                self.utt_eos = True
            step["writes"] += 1
            p.notify_target_unit(eos)
            if p.flushes_completed != flushes_before:
                completed_in_loop, last_eos = True, eos
                break

        utterance_end = p.flushes_completed != flushes_before
        if utterance_end:
            self.stats["utterances"] += 1
            if (last_eos if completed_in_loop else self.utt_eos):
                self.stats["eos_ends"] += 1
            elif completed_in_loop:
                step["cap_end"] = True                # flush bound / max_flush_steps (BUG-030)
                self.stats["cap_ends"] += 1
        self.steps.append({
            "utt_id": self.utt_id, "step": step["step"],
            "t_src_ms": self.input_samples * 1000 / self.sample_rate,
            "t_tgt_frames_emitted": step["frames"], "elapsed_ms": 0.0,   # set by run()
            "decision": decision.name, "speech": speech, "encoded": step["encoded"],
            "writes": step["writes"], "t_src": p.t_src, "t_tgt": p.t_tgt, "units": step["units"],
            "eos": step["eos"], "utterance_end": utterance_end, "cap_end": step["cap_end"],
            "pad_frames": step["pads"], "segment": step["segment"]})
        if utterance_end:
            self._reset_translator()
            self.segment += 1

    def _reset_translator(self):
        """ModeBState::reset_translator(): zero both translators, back to <BOS>."""
        self.tenc.reset()
        self.tdec.reset()
        self.prev_units = self._np.full(self.chunk, self.bos, self._np.int64)
        self.utt_eos = False

    # -- a whole file ------------------------------------------------------------
    def apply_change(self, name, value):
        """InferencePipeline::set_params() for one policy control: wait_k retunes
        k in place, adaptive_policy switches the mode (bank to 0) only when it
        differs. No translator reset."""
        if name == "wait_k":
            self.policy.set_k(int(value))
        elif name == "adaptive_policy":
            if self.policy.config.adaptive_mode != (value >= 0.5):
                self.policy.set_adaptive_mode(value >= 0.5)
        else:
            raise GoldenError(f"unsupported --change control '{name}'")

    def run(self, samples):
        """fv_s2ut_file's file loop: the input zero-padded to whole hops, then
        zero hops while the pipeline is busy or the last output sample is
        audible, at most max_tail_hops of them; scheduled changes apply before
        the first hop starting at or after their time. Returns the output,
        hops x hop samples; each step's elapsed_ms is its hop's compute time."""
        np = self._np
        x = np.asarray(samples, np.float32).reshape(-1)
        if x.size == 0:
            raise GoldenError("input WAV has no samples")
        hops_in = -(-x.size // self.hop)
        padded = np.zeros(hops_in * self.hop, np.float32)
        padded[:x.size] = x
        zeros = np.zeros(self.hop, np.float32)
        audible = _f32(AUDIBLE_SAMPLE)
        out, h, tail, last, pending = [], 0, 0, 0.0, list(self.changes)
        while h < hops_in or ((not self.idle() or abs(last) > audible)
                              and tail < self.max_tail_hops):
            if h >= hops_in:
                tail += 1
            while pending and h * self.hop >= pending[0][0] * self.sample_rate:
                self.apply_change(*pending.pop(0)[1:])
            first_step = len(self.steps)
            t0 = time.perf_counter()
            y = self.process(padded[h * self.hop:(h + 1) * self.hop] if h < hops_in else zeros)
            hop_ms = round(1e3 * (time.perf_counter() - t0), 4)
            for s in self.steps[first_step:]:
                s["elapsed_ms"] = hop_ms
            out.append(y)
            last = float(y[-1])
            h += 1
        self.stats.update(input_hops=hops_in, tail_hops=tail,
                          tail_capped=not self.idle() or abs(last) > audible,
                          fifo_overflows=self.fifo.overflows)
        return np.concatenate(out)


def make(model_dir, in_wav, out_dir, options, log=print):
    """Translate `in_wav` with the pipeline twin and write the golden files.
    Returns meta.json's content."""
    import numpy as np
    import onnxruntime as ort
    in_wav, out_dir = pathlib.Path(in_wav), pathlib.Path(out_dir)
    rate, samples = read_wav(in_wav)
    pipe = ModeBPipeline(model_dir, options, utt_id=in_wav.stem)
    if rate != pipe.sample_rate:
        raise GoldenError(f"Sample-rate mismatch: input WAV is {rate} Hz, the bundle runs at "
                          f"{pipe.sample_rate} Hz")
    x = np.frombuffer(samples.tobytes(), np.float32)
    y = pipe.run(x)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_wav_f32(out_dir / "input.wav", rate, samples)
    write_wav_f32(out_dir / "output.wav", rate, y)
    write_trace(out_dir / "trace.jsonl", pipe.steps)
    s = pipe.stats
    if s["tail_capped"]:
        log(f"warning: still busy (or audible) after {pipe.max_tail_hops} tail hops; the "
            "golden ends early (raise --max-tail-hops)")
    if s["fifo_stalls"]:
        log(f"note: {s['fifo_stalls']} write(s) deferred by FIFO back-pressure")
    c = pipe.policy.config
    # fv_s2ut_file caps the tail in ms: ceil(ms / hop_ms) hops.
    tail_ms = pipe.max_tail_hops * pipe.hop * 1000 / pipe.sample_rate
    tail_args = [] if options.max_tail_hops is None else ["--max-tail-ms", f"{tail_ms:g}"]
    changes = [{"seconds": t, "name": n, "value": v} for t, n, v in pipe.changes]
    change_args = [a for c in changes
                   for a in ("--change", f"{c['seconds']!r}:{c['name']}={c['value']:g}")]
    cpp = ["fv_s2ut_file", "--model-dir", str(pathlib.Path(model_dir)),
           "--src-lang", pipe.src_lang["code"], "--tgt-lang", pipe.tgt_lang["code"],
           "--voice", pipe.voice_name, "--wait-k", str(pipe.wait_k),
           "--frames-per-step", str(pipe.fps)] + (["--adaptive"] if options.adaptive else []) \
        + tail_args + change_args + ["--raw", "--no-prosody-ctx", "--dump-trace",
                                     "<cpp_trace.jsonl>", str(out_dir / "input.wav"),
                                     "<cpp_out.wav>"]
    meta = {
        "schema": SCHEMA, "kind": KIND,
        "options": {
            "src_lang": pipe.src_lang["code"], "src_lang_id": pipe.src_lang["id"],
            "tgt_lang": pipe.tgt_lang["code"], "tgt_lang_id": pipe.tgt_lang["id"],
            "voice": pipe.voice_name, "voice_index": pipe.voice_index,
            "wait_k": pipe.wait_k, "frames_per_step": pipe.fps, "adaptive": bool(options.adaptive),
            "prosody_ctx": False, "output_chain": False, "max_tail_hops": pipe.max_tail_hops,
            "changes": changes,
            "speech_threshold_dbfs": SPEECH_THRESHOLD_DBFS, "eos_threshold": EOS_THRESHOLD,
            "fifo_capacity_frames": pipe.fifo.capacity,
            "policy": {"k": c.k, "tail_flush_ms": c.tail_flush_ms,
                       "max_flush_ratio": c.max_flush_ratio, "max_flush_steps": c.max_flush_steps,
                       "adaptive_mode": c.adaptive_mode,
                       "readiness_threshold": c.readiness_threshold,
                       "adaptive_extra_wait": c.adaptive_extra_wait,
                       "chunk_frames": c.chunk_frames, "frame_ms": c.frame_ms},
            "controls": {"intonation": INTONATION, "pitch_shift": PITCH_SHIFT_ST,
                         "whisper": WHISPER, "vibrato_depth": 0.0, "timbre_blend": 0.0}},
        "geometry": {"sample_rate": pipe.sample_rate, "frame_samples": pipe.frame_samples,
                     "hop": pipe.hop, "chunk_frames": pipe.chunk, "chunk_ms": pipe.chunk_ms,
                     "latent_dim": pipe.latent_dim},
        "bundle": {"dir": str(pathlib.Path(model_dir)),
                   "files": {p.name: sha256(p) for p in pipe.paths.values()}},
        "input": {"source": str(in_wav), "samples": len(samples),
                  "sha256": sha256(out_dir / "input.wav")},
        "output": {"samples": int(y.size), "hops": s["input_hops"] + s["tail_hops"],
                   "input_hops": s["input_hops"], "tail_hops": s["tail_hops"],
                   "tail_capped": s["tail_capped"], "peak": float(np.max(np.abs(y))),
                   "sha256": sha256(out_dir / "output.wav")},
        "chunk_steps": len(pipe.steps),
        "stats": {k: s[k] for k in ("reads", "writes", "flushes", "utterances", "eos_ends",
                                    "cap_ends", "dec_steps", "voc_steps", "fifo_stalls",
                                    "pad_frames", "fifo_overflows")},
        "runtime": {"onnxruntime": ort.__version__, "numpy": np.__version__,
                    "python": platform.python_version(),
                    "platform": f"{platform.system()} {platform.machine()}"},
        "cpp_command": cpp,
        "check_command": ["s2ut_golden.py", "check", "--golden", str(out_dir),
                          "--trace", "<cpp_trace.jsonl>", "--wav", "<cpp_out.wav>"],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


# -- check -------------------------------------------------------------------------
_MISSING = ("missing",)


def _norm(key, v):
    """Comparable form of a trace value; ('invalid', repr) for a wrong type."""
    if key == "decision":
        return v if isinstance(v, str) and v in DECISIONS else ("invalid", repr(v))
    if key in BOOL_KEYS:
        if isinstance(v, bool):
            return v
        return bool(v) if isinstance(v, int) and v in (0, 1) else ("invalid", repr(v))
    if key == "units":
        ok = isinstance(v, list) and all(isinstance(u, int) and not isinstance(u, bool) for u in v)
        return list(v) if ok else ("invalid", repr(v))
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return ("invalid", repr(v))
    if key == "t_src_ms":
        return round(float(v), 3)
    return int(v) if float(v).is_integer() else ("invalid", repr(v))


def compare_traces(golden, test):
    """Per-step comparison. Returns (mismatches [(index, key, golden, test)],
    {key: [identical, compared]})."""
    keys = REQUIRED_KEYS + EXTRA_KEYS
    counts = {k: [0, 0] for k in keys}
    mismatches = []
    for i, (g, t) in enumerate(zip(golden, test)):
        for key in keys:
            if key not in REQUIRED_KEYS and (key not in g or key not in t):
                continue
            gv = _norm(key, g[key]) if key in g else _MISSING
            tv = _norm(key, t[key]) if key in t else _MISSING
            counts[key][1] += 1
            if gv == tv and not (isinstance(gv, tuple) and gv[:1] in (("invalid",), _MISSING)):
                counts[key][0] += 1
            else:
                mismatches.append((i, key, g.get(key, "<missing>"), t.get(key, "<missing>")))
    return mismatches, counts


def compare_audio(golden, test, atol):
    """max |delta| (NaN/inf count as failures), first index beyond atol, SNR,
    golden peak, over the common length; plain Python floats."""
    n = min(len(golden), len(test))
    max_abs = err2 = ref2 = peak = 0.0
    first_bad = None
    nonfinite = 0
    for i in range(n):
        a, b = golden[i], test[i]
        d = abs(b - a)
        if not math.isfinite(d):
            nonfinite += 1
            d = math.inf
        if d > max_abs:
            max_abs = d
        if first_bad is None and not d <= atol:
            first_bad = i
        err2 += d * d
        ref2 += a * a
        peak = max(peak, abs(a))
    for a in golden[n:]:
        peak = max(peak, abs(a))
    if err2 == 0.0:
        snr = math.inf
    elif not math.isfinite(err2):
        snr = -math.inf
    else:
        snr = 10.0 * math.log10(max(ref2, 1e-300) / err2)
    return {"golden_samples": len(golden), "test_samples": len(test), "compared": n,
            "max_abs": max_abs, "first_bad": first_bad, "nonfinite": nonfinite, "snr_db": snr,
            "golden_peak": peak, "rel": max_abs / peak if peak > 0 else math.inf}


def _short(v, width=72):
    s = json.dumps(v) if not isinstance(v, str) else repr(v)
    return s if len(s) <= width else s[:width - 3] + "..."


def check(golden_dir, trace_path, wav_path, atol=DEFAULT_ATOL, allow_silent=False,
          max_report=12, out=print):
    """Compare a C++ run (trace + wav) with a golden. Prints a report through
    `out` and returns (passed, summary dict)."""
    golden_dir = pathlib.Path(golden_dir)
    failures = []
    try:
        g_steps = read_trace(golden_dir / "trace.jsonl")
        t_steps = read_trace(trace_path)
        g_rate, g_wav = read_wav(golden_dir / "output.wav")
        t_rate, t_wav = read_wav(wav_path)
    except GoldenError as e:
        out(f"s2ut golden check: cannot compare: {e}")
        out("RESULT: FAIL")
        return False, {"error": str(e)}
    meta = {}
    meta_path = golden_dir / "meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
    o = meta.get("options", {})
    out("s2ut golden check")
    desc = (f"{o.get('src_lang', '?')}->{o.get('tgt_lang', '?')}, k={o.get('wait_k', '?')}, "
            f"s{o.get('frames_per_step', '?')}{', adaptive' if o.get('adaptive') else ''}"
            if o else "no meta.json")
    out(f"  golden    {golden_dir} ({desc})")
    out(f"  test      trace {trace_path}")
    out(f"            wav   {wav_path}")

    ok_steps = len(g_steps) == len(t_steps)
    out(f"  steps     golden {len(g_steps)}, test {len(t_steps)}"
        f"{'' if ok_steps else '   MISMATCH'}")
    if not ok_steps:
        failures.append(f"chunk steps {len(t_steps)} != {len(g_steps)}")
    mismatches, counts = compare_traces(g_steps, t_steps)
    for key in REQUIRED_KEYS + EXTRA_KEYS:
        same, compared = counts[key]
        if key in EXTRA_KEYS and compared == 0:
            out(f"  {key:22s} not in the test trace (skipped)")
            continue
        flag = "" if same == compared else "   MISMATCH"
        out(f"  {key:22s} {same}/{compared} identical{flag}")
        if same != compared:
            failures.append(f"{key}: {compared - same} step(s) differ")
    if mismatches:
        first = mismatches[0][0]
        g = g_steps[first]
        out(f"  first divergence: step {first} (t_src_ms {g.get('t_src_ms')}, "
            f"golden {g.get('decision')} x{g.get('writes')})")
        for i, key, gv, tv in mismatches[:max_report]:
            out(f"    step {i:5d} {key:22s} golden {_short(gv)}  test {_short(tv)}")
        if len(mismatches) > max_report:
            out(f"    ... {len(mismatches) - max_report} more")

    a = compare_audio(g_wav, t_wav, atol)
    if g_rate != t_rate:
        failures.append(f"sample rate {t_rate} != {g_rate}")
    if a["golden_samples"] != a["test_samples"]:
        failures.append(f"output length {a['test_samples']} != {a['golden_samples']} samples")
    if a["first_bad"] is not None:
        hop = meta.get("geometry", {}).get("hop")
        where = f"sample {a['first_bad']}" + (f" (hop {a['first_bad'] // hop})" if hop else "")
        failures.append(f"waveform max|delta| {a['max_abs']:.3g} > atol {atol:g} from {where}"
                        + (f", {a['nonfinite']} non-finite" if a["nonfinite"] else ""))
    silent = not a["golden_peak"] > SILENT_PEAK_FACTOR * atol
    if silent and not allow_silent:
        failures.append(f"golden output peak {a['golden_peak']:.3g} <= {SILENT_PEAK_FACTOR:g} x "
                        "atol: the waveform check would be vacuous (BUG-017; --allow-silent)")
    out(f"  waveform  golden {a['golden_samples']}, test {a['test_samples']} samples "
        f"@ {t_rate} Hz; max|delta| {a['max_abs']:.3g} (atol {atol:g}, "
        f"{100 * a['rel']:.3g}% of golden peak {a['golden_peak']:.3g}), SNR {a['snr_db']:.1f} dB")
    passed = not failures
    out("RESULT: PASS" if passed else "RESULT: FAIL - " + "; ".join(failures))
    return passed, {"passed": passed, "failures": failures, "steps": [len(g_steps), len(t_steps)],
                    "fields": counts, "audio": a, "mismatches": len(mismatches)}


# -- self-test (standard library only) -----------------------------------------------
def self_test(out=print):
    n = [0]

    def expect(cond, what):
        n[0] += 1
        if not cond:
            raise AssertionError(what)

    def raises(fn, text):
        try:
            fn()
        except GoldenError as e:
            n[0] += 1
            if text not in str(e):
                raise AssertionError(f"expected {text!r} in {e!s}") from None
        else:
            raise AssertionError(f"no GoldenError ({text})")

    quiet = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        # WAV: float32 round trip, PCM16 (also WAVE_FORMAT_EXTENSIBLE + odd chunks).
        x = array.array("f", [0.0, 0.5, -0.25, 1e-8, -1.0])
        write_wav_f32(tmp / "a.wav", 16000, x)
        expect(read_wav(tmp / "a.wav") == (16000, x), "float32 WAV round trip")
        expect(len((tmp / "a.wav").read_bytes()) == 44 + 4 * len(x), "44-byte header")
        pcm = struct.pack("<5h", 0, 1, -1, 32767, -32768)

        def riff(fmt_payload, extra=b"", data=pcm):
            body = b"WAVE" + extra + b"fmt " + struct.pack("<I", len(fmt_payload)) + fmt_payload
            body += b"data" + struct.pack("<I", len(data)) + data
            return b"RIFF" + struct.pack("<I", len(body)) + body

        fmt16 = struct.pack("<HHIIHH", 1, 1, 16000, 32000, 2, 16)
        ext = struct.pack("<HHIIHHHHI", 0xFFFE, 1, 16000, 32000, 2, 16, 22, 16, 4) + \
            struct.pack("<H", 1) + b"\x00" * 14
        want = array.array("f", [0.0, 1 / 32768, -1 / 32768, 32767 / 32768, -1.0])
        for name, blob in (("pcm", riff(fmt16)), ("ext", riff(ext)),
                           ("odd", riff(fmt16, extra=b"LIST" + struct.pack("<I", 3) + b"abc\x00"))):
            (tmp / f"{name}.wav").write_bytes(blob)
            expect(read_wav(tmp / f"{name}.wav") == (16000, want), f"PCM16 WAV ({name})")
        stereo = struct.pack("<HHIIHH", 1, 2, 16000, 64000, 4, 16)
        for name, blob, text in (
                ("stereo", riff(stereo), "WAV must be mono (got 2 channels)"),
                ("u8", riff(struct.pack("<HHIIHH", 1, 1, 16000, 16000, 1, 8)),
                 "unsupported WAV format (format_tag=1, bits_per_sample=8"),
                ("nodata", riff(fmt16, data=b"").replace(b"data", b"junk"),
                 "missing fmt or data chunk"),
                ("short", b"RIFF", "too short"),
                ("riff", b"RIFX" + b"\x00" * 60, "not a RIFF/WAVE file")):
            (tmp / f"{name}.wav").write_bytes(blob)
            raises(lambda p=tmp / f"{name}.wav": read_wav(p), text)
        raises(lambda: read_wav(tmp / "absent.wav"), "cannot open input WAV")

        # Small rules.
        cfg = WaitKConfig().sanitized()
        expect([default_max_tail_hops(cfg, 320 * f, 16000) for f in (1, 2, 4)] == [168, 84, 42],
               "default tail hops")
        expect(fifo_capacity(64, 4) == 64 and fifo_capacity(1, 4) == 8 and
               fifo_capacity(10, 4) == 12, "FIFO capacity rounding")
        raises(lambda: fifo_capacity(4097, 4), "exceeds 4096")
        expect(readiness_from_logit(0.0) == 0.5 and readiness_from_logit(-1e4) == 0.0
               and readiness_from_logit(1e4) == 1.0, "readiness sigmoid")
        expect(frame_is_speech([0.25] * 320, -55.0) and not frame_is_speech([0.0] * 320, -55.0)
               and not frame_is_speech([1e-3] * 320, -55.0)
               and frame_is_speech([2e-3] * 320, -55.0), "frame VAD (-60 / -54 dBFS)")
        langs = {"langs": [{"code": "en", "id": 0}, {"code": "es", "id": 1}]}
        expect(resolve_lang(langs, "es")["id"] == 1 and resolve_lang(langs, "0")["code"] == "en"
               and resolve_lang(langs, "7") is None and resolve_lang(langs, "+1") is None
               and resolve_lang(langs, "000001") is None, "LangCatalog::find twin")

        # check(): a synthetic golden against itself and perturbed copies.
        def step(i, decision="WRITE", units=(3, 3, 3, 3), **kw):
            s = {"utt_id": "u", "step": i, "t_src_ms": 80.0 * (i + 1),
                 "t_tgt_frames_emitted": len(units), "elapsed_ms": 0.1, "decision": decision,
                 "speech": True, "encoded": decision != "FLUSH", "writes": len(units) // 4,
                 "t_src": i + 1, "t_tgt": i - 1, "units": list(units), "eos": False,
                 "utterance_end": False, "cap_end": False, "pad_frames": 0, "segment": 0}
            s.update(kw)
            return s

        steps = [step(0, "READ", ()), step(1, "READ", ()), step(2), step(3, "FLUSH", (6,) * 4,
                                                                         eos=True,
                                                                         utterance_end=True)]
        wav = array.array("f", [0.0] * 2560 + [0.16875] * 1280 + [0.3375] * 1280)
        golden = tmp / "golden"
        golden.mkdir()
        write_trace(golden / "trace.jsonl", steps)
        write_wav_f32(golden / "output.wav", 16000, wav)
        (golden / "meta.json").write_text(json.dumps({"geometry": {"hop": 1280}, "options": {
            "src_lang": "es", "tgt_lang": "de", "wait_k": 3, "frames_per_step": 4}}))

        def run(trace_steps=steps, samples=wav, atol=DEFAULT_ATOL, wrap=False, **kw):
            write_trace(tmp / "t.jsonl", [{"utt_id": "u", "steps": trace_steps}] if wrap
                        else trace_steps)
            write_wav_f32(tmp / "t.wav", 16000, samples)
            quiet.clear()
            return check(golden, tmp / "t.jsonl", tmp / "t.wav", atol=atol, out=quiet.append,
                         **kw)

        passed, summary = run()
        expect(passed and quiet[-1] == "RESULT: PASS", "golden vs itself")
        expect(summary["fields"]["units"] == [4, 4], "units compared on every step")
        expect(run(wrap=True)[0], "{utt_id, steps} lines are accepted")
        cpp_style = [dict(s, speech=int(s["speech"]), t_src_ms=int(s["t_src_ms"]), elapsed_ms=9.9,
                          utt_id="input") for s in steps]
        expect(run(cpp_style)[0], "0/1 booleans, integer t_src_ms, other utt_id/elapsed_ms")
        no_extras = [{k: v for k, v in s.items() if k in REQUIRED_KEYS} for s in steps]
        expect(run(no_extras)[0] and any("not in the test trace" in line for line in quiet),
               "extra keys are optional")
        for what, bad in (
                ("units", [*steps[:2], step(2, units=(3, 3, 3, 4)), steps[3]]),
                ("decision", [*steps[:2], step(2, "FLUSH"), steps[3]]),
                ("eos", [*steps[:3], dict(steps[3], eos=False)]),
                ("utterance_end", [*steps[:3], dict(steps[3], utterance_end=False)]),
                ("t_tgt_frames_emitted", [*steps[:2], step(2, t_tgt_frames_emitted=3), steps[3]]),
                ("speech", [dict(steps[0], speech=False), *steps[1:]]),
                ("segment", [*steps[:3], dict(steps[3], segment=1)]),
                ("pad_frames", [*steps[:3], dict(steps[3], pad_frames=3)]),
                ("t_src_ms", [*steps[:3], dict(steps[3], t_src_ms=330.0)]),
                ("writes", [*steps[:3], {k: v for k, v in steps[3].items() if k != "writes"}]),
                ("t_src", [*steps[:3], dict(steps[3], t_src="4")]),
                ("chunk steps", steps[:3])):
            passed, _ = run(bad)
            expect(not passed and what in quiet[-1], f"a {what} change fails")
        expect(not run(samples=wav[:-1])[0] and "output length" in quiet[-1], "length")
        bumped = array.array("f", wav)
        bumped[3000] += 1e-3
        expect(not run(samples=bumped)[0] and "hop 2" in quiet[-1], "waveform beyond atol")
        bumped[3000] = wav[3000] + 4e-6
        expect(run(samples=bumped)[0], "waveform within atol")
        bumped[3000] = math.nan
        expect(not run(samples=bumped)[0] and "non-finite" in quiet[-1], "NaN output")
        write_wav_f32(golden / "output.wav", 16000, [0.0] * len(wav))
        expect(not run(samples=[0.0] * len(wav))[0] and "vacuous" in quiet[-1], "silent golden")
        expect(run(samples=[0.0] * len(wav), allow_silent=True)[0], "--allow-silent")
        (golden / "output.wav").unlink()
        expect(not run()[0] and "cannot compare" in quiet[0], "missing golden file")
    out(f"s2ut_golden self-test: {n[0]} checks passed")
    return 0
