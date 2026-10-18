"""Stateful ONNX export, ONNX Runtime streaming and parity (Milestone 2 /
Stage 06).

Spec: Documents/Milestone 2/06_stateful_onnx_export.md. The CLI is
python_training/export_stateful_onnx.py.

A step graph runs one hop of a model at batch 1 with static shapes: the
contract's inputs and outputs (fvml.contract), plus every StreamingConv1d cache
the hop touches as a `state_in.<path>` input and a `state_out.<path>` output.
<path> is the conv's qualified module name (`trunk.blocks.0.down`), so names are
stable and change only with the module tree; their order is registration order
(fvml.streaming.state_layout). The caches are the whole recurrent state. There
is no step or position counter to freeze into the trace, and parity over many
hops (not one) is what proves it.

Encoder graphs run the trunk and one delay head (head index -> lookahead L) and
emit `units`, the argmax of the head's logits. While exporting, the logits are a
graph output too. The shipped file drops them, since contract v1 allows no
extra output, and parity adds them back in memory (with_outputs) to measure
|Δlogit|.

Vocoder graphs wrap fvml.vocoder.UnitVocoder (Stage 05) through this much of
its API:
  - `from_checkpoint(path) -> (model, ckpt)`; `model.contract(frames_per_step,
    license)` builds the contract from the stored config, its states being the
    model's state layout;
  - `model(content, **cond)` offline and `model.step(content, **cond, state=)`
    -> (audio, new state) for one hop, `cond` keyed by the contract's
    cond_inputs (logf0, voicing, aperiodicity, energy: (1, T_cond); spk_emb:
    (1, D));
  - `model.voices_json(speakers, speaker_stats, extra_metadata)`, fed from the
    checkpoint's `speakers` (speaker-table row names, in row order),
    `speaker_stats` ({name: logf0/energy mean/std}) and optional
    `speaker_metadata` ({name: display/license/source/attribution}); the
    optional `license` key overrides the config's.
"""
import contextlib
import hashlib
import io
import json
import pathlib
import time
import warnings

import numpy as np
import torch
import torch.nn as nn

from . import contract as fc
from . import parity as fp
from .contract import TensorSpec
from .streaming import StreamState

# ONNX Runtime 1.31 (Stage 07's pinned build) reads at most opset 26 and IR 13;
# onnx 1.23 can write IR 14, so the IR is capped explicitly.
MAX_OPSET = 26
MAX_IR_VERSION = 13
TORCHSCRIPT_OPSET = 18          # the legacy exporter's fallback opset
EXPORTERS = ("auto", "dynamo", "torchscript")

# Definition of done (Stage 06).
PARITY_SECONDS = 60.0           # > 2x any attention / KV window
GOLDEN_SECONDS = 10.0
MIN_UNIT_AGREEMENT = 0.999
LOGIT_ATOL = 1e-4               # |Δlogit| above this counts as the first divergence
MIN_VOCODER_SNR_DB = 50.0
COMPUTE_BUDGET = 0.5            # encoder + vocoder p95 <= 50% of the hop

TORCH_DTYPES = {"f32": torch.float32, "i64": torch.int64, "f16": torch.float16}
NP_DTYPES = {"f32": np.float32, "i64": np.int64, "f16": np.float16}


class ExportError(RuntimeError):
    """A model that cannot become a contract-v1 step graph, or a file that
    fails its checks after export."""


# -- step graphs ---------------------------------------------------------------
class StepGraph(nn.Module):
    """One hop of a model with flat tensor I/O, which is what torch.onnx.export
    traces: forward(*inputs, *states) -> (*outputs, *new_states), in the order
    of input_names() / output_names().

    `step(inputs, state)` maps {input name: tensor} and {state name: tensor} to
    ({output name: tensor}, {state name: new tensor}). `debug_outputs` are extra
    outputs that the shipped file drops (finalize)."""

    def __init__(self, model, step, inputs, outputs, init_state, debug_outputs=()):
        super().__init__()
        self.model = model
        self.step_fn = step
        self.inputs = list(inputs)
        self.outputs = list(outputs) + list(debug_outputs)
        self.debug_outputs = [t.name for t in debug_outputs]
        self.states = list(init_state)
        self.init_state = dict(init_state)

    def input_names(self):
        return [t.name for t in self.inputs] + [fc.STATE_IN + s for s in self.states]

    def output_names(self):
        return [t.name for t in self.outputs] + [fc.STATE_OUT + s for s in self.states]

    def example(self):
        xs = [torch.zeros(t.shape, dtype=TORCH_DTYPES[t.dtype]) for t in self.inputs]
        return tuple(xs + [self.init_state[s].clone() for s in self.states])

    def state_floats(self):
        return sum(int(t.numel()) for t in self.init_state.values())

    def forward(self, *args):
        n = len(self.inputs)
        if len(args) != n + len(self.states):
            raise TypeError(f"step graph takes {n} inputs + {len(self.states)} states, "
                            f"got {len(args)} tensors")
        outs, new = self.step_fn({t.name: a for t, a in zip(self.inputs, args[:n])},
                                 dict(zip(self.states, args[n:])))
        return tuple([outs[t.name] for t in self.outputs] + [new[s] for s in self.states])


def head_index(model, head):
    """Head index from an int / "1" or a tokenizer name ("student-L1" = the
    head whose lookahead is 1)."""
    lookahead = tuple(model.cfg.lookahead)
    text = str(head)
    if text.startswith("student-L"):
        L = int(text.removeprefix("student-L"))
        if L not in lookahead:
            raise ExportError(f"no head with lookahead {L}; heads have {lookahead}")
        return lookahead.index(L)
    i = int(text)
    if i not in range(len(lookahead)):
        raise ExportError(f"no head {i}; heads are 0..{len(lookahead) - 1}")
    return i


def encoder_contract(model, head, frames_per_step, license):
    """Contract of a ContentEncoder exported with delay head `head`."""
    from . import content_encoder as ce
    graph_states = [n for n in model.init_state() if n.startswith(_head_prefixes(head))]
    return make_contract({"role": "encoder", "sample_rate": ce.SAMPLE_RATE,
                          "frame_samples": ce.FRAME_SAMPLES,
                          "lookahead_frames": model.cfg.lookahead[head],
                          "content_kind": "discrete", "vocab": model.cfg.vocab,
                          "license": license}, frames_per_step, graph_states)


def _head_prefixes(head):
    return ("trunk.", f"heads.{head}.")


def encoder_graph(model, head, frames_per_step):
    """Step graph of a ContentEncoder: trunk + head `head` -> units (+ logits).
    The other heads are left out, with their caches."""
    from . import content_encoder as ce
    prefixes = _head_prefixes(head)
    init = {n: t for n, t in model.init_state().items() if n.startswith(prefixes)}
    conv_names = model.stream_names()
    trunk, head_module = model.trunk, model.heads[head]

    def step(ins, state):
        stream = StreamState(conv_names, state)
        latents = trunk(ins["audio"].unsqueeze(1), stream)
        logits = head_module(latents.transpose(1, 2), stream)
        return {"units": logits.argmax(-1), "logits": logits}, stream.result()

    return StepGraph(model, step,
                     [TensorSpec("audio", "f32", (1, frames_per_step * ce.FRAME_SAMPLES))],
                     [TensorSpec("units", "i64", (1, frames_per_step))], init,
                     debug_outputs=[TensorSpec("logits", "f32",
                                               (1, frames_per_step, model.cfg.vocab))])


def vocoder_contract(model, ckpt, frames_per_step):
    """The vocoder's contract from its stored config (licence: the
    checkpoint's `license` key if present)."""
    license = ckpt.get("license") if isinstance(ckpt, dict) else None
    try:
        c = model.contract(frames_per_step, license=license)
        c.validate()
    except (TypeError, ValueError) as e:     # ContractError is a ValueError
        raise ExportError(f"invalid vocoder contract: {e}") from None
    if c.role != "vocoder":
        raise ExportError(f"vocoder contract has role {c.role!r}")
    return c


def vocoder_graph(model, c):
    """Step graph of a UnitVocoder for contract `c`."""
    if c.overlap_samples:
        raise ExportError(f"overlap_samples is {c.overlap_samples}: Stage 06 exports stateful "
                          "causal vocoders, which need no OLA tail")
    init = model.init_state()
    if list(init) != list(c.states):
        raise ExportError("the contract's states are not the model's state layout")

    def step(ins, state):
        audio, new = model.step(ins[c.content_name], state=state,
                                **{n: ins[n] for n in c.cond_inputs})
        return {"audio": audio}, new

    return StepGraph(model, step, c.expected_inputs(), c.expected_outputs(), init)


def make_contract(fields, frames_per_step, states):
    try:
        c = fc.ModelContract(**fields, frames_per_step=frames_per_step, states=tuple(states))
        c.validate()
    except (TypeError, fc.ContractError) as e:
        raise ExportError(f"invalid contract fields: {e}") from None
    return c


# -- export --------------------------------------------------------------------
def export_graph(graph, path, exporter="auto"):
    """torch.onnx.export `graph` to `path`. "auto" tries the dynamo exporter
    first and falls back to the TorchScript one. Returns a record: exporter
    used, why dynamo was skipped, warnings raised while tracing, seconds."""
    if exporter not in EXPORTERS:
        raise ValueError(f"exporter must be one of {EXPORTERS}")
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    graph.eval()
    kw = {"input_names": graph.input_names(), "output_names": graph.output_names()}
    record = {"exporter": None, "fallback_reason": None, "warnings": []}
    t0 = time.perf_counter()
    if exporter in ("auto", "dynamo"):
        try:
            with _capture(record):
                torch.onnx.export(graph, graph.example(), str(path), dynamo=True,
                                  external_data=False, optimize=True, verbose=False, **kw)
            record["exporter"] = "dynamo"
        except Exception as e:      # noqa: BLE001 - any capture failure falls back
            if exporter == "dynamo":
                raise ExportError(f"dynamo export failed: {_one_line(e)}") from e
            record["fallback_reason"] = _one_line(e)
            record["warnings"] = []
            path.unlink(missing_ok=True)
    if record["exporter"] is None:
        with _capture(record):
            torch.onnx.export(graph, graph.example(), str(path), dynamo=False,
                              opset_version=TORCHSCRIPT_OPSET, do_constant_folding=True, **kw)
        record["exporter"] = "torchscript"
    record["seconds"] = round(time.perf_counter() - t0, 2)
    return record


def finalize(path, contract, drop_outputs=(), metadata=None):
    """Make an exported file shippable, in place: drop the debug outputs, cap
    the IR version, embed the contract (plus `metadata` props), run strict
    shape inference and the full checker, then reload it the way the C++
    loader checks it. Returns (opset, ir_version)."""
    import onnx
    path = pathlib.Path(path)
    m = onnx.load(str(path))
    opset = default_opset(m)
    if opset > MAX_OPSET:
        raise ExportError(f"{path.name}: opset {opset} > {MAX_OPSET}, which ONNX Runtime 1.31 "
                          "cannot load")
    m.ir_version = min(m.ir_version, MAX_IR_VERSION)
    keep = [o for o in m.graph.output if o.name not in set(drop_outputs)]
    del m.graph.output[:]
    m.graph.output.extend(keep)
    for key, value in sorted((metadata or {}).items()):
        _set_meta(m, key, value)
    fc.embed(m, contract.to_json())
    try:
        m = onnx.shape_inference.infer_shapes(m, check_type=True, strict_mode=True)
        onnx.checker.check_model(m, full_check=True)
    except Exception as e:      # noqa: BLE001 - onnx raises several types
        raise ExportError(f"{path.name}: onnx check failed: {_one_line(e)}") from e
    onnx.save(m, str(path))
    try:
        loaded, _ = fc.load_onnx(path)
    except fc.ContractError as e:
        raise ExportError(f"{path.name}: {e}") from None
    if loaded != contract:
        raise ExportError(f"{path.name}: embedded contract does not round-trip")
    return opset, m.ir_version


def default_opset(model):
    return next((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), 0)


def with_outputs(model, specs):
    """A copy of an onnx.ModelProto that also outputs the named intermediate
    tensors (e.g. the encoder's `logits`)."""
    import onnx
    from onnx import TensorProto, helper
    elem = {"f32": TensorProto.FLOAT, "i64": TensorProto.INT64, "f16": TensorProto.FLOAT16}
    m = onnx.ModelProto()
    m.CopyFrom(model)
    produced = {o for n in m.graph.node for o in n.output}
    have = {o.name for o in m.graph.output}
    for t in specs:
        if t.name in have:
            continue
        if t.name not in produced:
            raise ExportError(f"graph has no tensor named '{t.name}'")
        m.graph.output.append(helper.make_tensor_value_info(t.name, elem[t.dtype], list(t.shape)))
    return m


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# -- ONNX Runtime ----------------------------------------------------------------
def ort_session(model):
    """CPU session configured like Stage 07's OrtModel: one intra-op thread,
    sequential execution, no spinning, denormals flushed to zero. `model` is a
    path or an onnx.ModelProto."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    so.add_session_config_entry("session.inter_op.allow_spinning", "0")
    so.add_session_config_entry("session.set_denormal_as_zero", "1")
    src = str(model) if isinstance(model, (str, pathlib.Path)) else model.SerializeToString()
    return ort.InferenceSession(src, so, providers=["CPUExecutionProvider"])


class OrtStream:
    """A contract-checked step graph run hop by hop in ONNX Runtime, every
    state_out.X fed back as state_in.X (what Stage 07's KVStateCache does).
    `extra_outputs` (TensorSpec) are intermediate tensors to expose too."""

    def __init__(self, path, extra_outputs=()):
        self.path = pathlib.Path(path)
        try:
            self.contract, proto = fc.load_onnx(self.path)
        except fc.ContractError as e:
            raise ExportError(f"{self.path.name}: {e}") from None
        if extra_outputs:
            proto = with_outputs(proto, extra_outputs)
        inputs, _ = fc.graph_io(proto)
        by_name = {t.name: t for t in inputs}
        self.state_specs = [by_name[fc.STATE_IN + s] for s in self.contract.states]
        self.session = ort_session(proto)
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.reset()

    def reset(self):
        self.state = {t.name: np.zeros(t.shape, NP_DTYPES[t.dtype]) for t in self.state_specs}

    def step(self, feeds):
        """One hop: {input name: array} -> {output name: array} (states kept)."""
        values = self.session.run(self.output_names, {**feeds, **self.state})
        out = dict(zip(self.output_names, values))
        for s in self.contract.states:
            self.state[fc.STATE_IN + s] = out.pop(fc.STATE_OUT + s)
        return out


def read_contract(path):
    try:
        return fc.load_onnx(path)[0]
    except fc.ContractError as e:
        raise ExportError(f"{pathlib.Path(path).name}: {e}") from None


def logits_spec(c):
    return TensorSpec("logits", "f32", (1, c.frames_per_step, c.vocab))


def whole_hops(x, hop):
    return np.ascontiguousarray(x[: len(x) // hop * hop], dtype=np.float32)


# -- encoder -------------------------------------------------------------------
@torch.no_grad()
def encoder_offline(model, head, wav):
    """Logits of head `head` through forward() on the whole signal: (frames, vocab)."""
    return model(torch.from_numpy(np.ascontiguousarray(wav, np.float32))[None])["logits"][head][0].numpy()


def encoder_chunked(stream, wav):
    """(units (frames,), logits (frames, vocab) or None) from an encoder
    OrtStream fed one hop at a time from a fresh state."""
    hop = stream.contract.hop
    stream.reset()
    units, logits = [], []
    for i in range(len(wav) // hop):
        out = stream.step({"audio": np.ascontiguousarray(wav[None, i * hop:(i + 1) * hop],
                                                         np.float32)})
        units.append(out["units"][0])
        if "logits" in out:
            logits.append(out["logits"][0])
    return np.concatenate(units), (np.concatenate(logits) if logits else None)


def encoder_parity(model, head, path, wav):
    """ORT chunked (fresh state, states fed back) vs. PyTorch forward() offline.
    Gate: shipped `units` agree with the offline argmax on >= 99.9% of frames;
    max |Δlogit| catches real drift."""
    stream = OrtStream(path, extra_outputs=[logits_spec(read_contract(path))])
    wav = whole_hops(wav, stream.contract.hop)
    ref = encoder_offline(model, head, wav)
    units, logits = encoder_chunked(stream, wav)
    r = fp.compare(ref, logits, atol=LOGIT_ATOL, argmax_axis=-1, steps=False)
    agreement = float(np.mean(units == ref.argmax(-1)))
    return {"seconds": len(wav) / stream.contract.sample_rate, "frames": int(len(units)),
            "unit_agreement": agreement, "max_abs_logit": r.max_abs,
            "mean_abs_logit": r.mean_abs, "logit_snr_db": _finite(r.snr_db),
            "first_divergence_frame": None if r.first_divergence is None else r.first_divergence[0],
            "pass": bool(agreement >= MIN_UNIT_AGREEMENT)}


# -- vocoder -------------------------------------------------------------------
def conditioning(wav, voice, enc, voc):
    """Vocoder conditioning as the C++ engine computes it (Stage 08), for the
    encoder/vocoder contracts `enc`/`voc`: Stage 03's causal tracker pushed one
    cond frame at a time, its lookahead set to the encoder's L, so push t
    yields the features of frame t - L, the frame the encoder's id at t
    describes; then mapped into `voice`'s domain (fvml.features.map_to_target,
    scale 1, shift 0). Returns {cond name: (cond frames,) f32}."""
    from . import features as ft
    ratio = voc.frame_samples // voc.cond_frame_samples
    cfg = ft.PitchConfig(lookahead_frames=enc.lookahead_frames * ratio)
    if cfg.frame_samples != voc.cond_frame_samples:
        raise ExportError(f"conditioning every {voc.cond_frame_samples} samples needs a tracker "
                          f"at that hop; Stage 03's runs every {cfg.frame_samples}")
    tracker = ft.PitchTracker(cfg)
    hop = cfg.frame_samples
    rows = [tracker.push(wav[i * hop:(i + 1) * hop]).as_tuple() for i in range(len(wav) // hop)]
    f = np.asarray(rows, dtype=np.float64).reshape(-1, len(ft.FEATURES))
    col = {name: f[:, i] for i, name in enumerate(ft.FEATURES)}
    logf0, energy = ft.map_to_target(col["z_logf0"], col["z_energy"], voice)
    return {"logf0": logf0.astype(np.float32), "voicing": col["voicing"].astype(np.float32),
            "aperiodicity": col["aperiodicity"].astype(np.float32),
            "energy": energy.astype(np.float32)}


def vocoder_inputs(c, content, cond, spk_emb):
    """Whole-stream vocoder inputs keyed by contract name, batch dim dropped:
    `content` (frames,) ids or (frames, dim), `cond` at the cond rate, cut to
    whole hops."""
    ratio = c.frame_samples // c.cond_frame_samples
    frames = len(content) // c.frames_per_step * c.frames_per_step
    full = {c.content_name: np.ascontiguousarray(
        content[:frames], np.int64 if c.content_kind == "discrete" else np.float32)}
    for name in c.cond_inputs:
        if name == "spk_emb":
            full[name] = np.asarray(spk_emb, np.float32).reshape(c.spk_emb_dim)
        else:
            if len(cond[name]) < frames * ratio:
                raise ExportError(f"{name}: {len(cond[name])} cond frames for {frames} frames")
            full[name] = np.ascontiguousarray(cond[name][:frames * ratio], np.float32)
    return full


@torch.no_grad()
def vocoder_offline(model, c, full):
    """Audio through the vocoder's forward() on whole-stream inputs."""
    ins = {k: torch.from_numpy(np.ascontiguousarray(v))[None] for k, v in full.items()}
    return model(ins.pop(c.content_name), **ins)[0].float().numpy()


def vocoder_chunked(stream, full):
    """Audio from a vocoder OrtStream fed one hop of `full` at a time."""
    c = stream.contract
    steps = len(full[c.content_name]) // c.frames_per_step
    stream.reset()
    audio = []
    for i in range(steps):
        audio.append(stream.step(vocoder_feeds(c, full, i))["audio"][0])
    return np.concatenate(audio) if audio else np.zeros(0, np.float32)


def vocoder_feeds(c, full, i):
    feeds = {}
    for t in c.expected_inputs():
        v = full[t.name]
        if t.name == "spk_emb":
            feeds[t.name] = v[None]
        else:
            k = t.shape[1]           # frames_per_step or cond_frames per hop
            feeds[t.name] = np.ascontiguousarray(v[None, i * k:(i + 1) * k])
    return feeds


def vocoder_parity(model, path, full):
    """ORT chunked vs. PyTorch forward() offline on identical inputs.
    Gate: SNR >= 50 dB."""
    stream = OrtStream(path)
    test = vocoder_chunked(stream, full)
    ref = vocoder_offline(model, stream.contract, full)[: len(test)]
    r = fp.compare(ref, test)
    return {"seconds": len(test) / stream.contract.sample_rate, "samples": int(len(test)),
            "snr_db": _finite(r.snr_db), "max_abs": r.max_abs, "mean_abs": r.mean_abs,
            "max_step_ref": r.max_step_ref, "max_step_test": r.max_step_test,
            "pass": bool(r.snr_db >= MIN_VOCODER_SNR_DB)}


# -- chain ---------------------------------------------------------------------
def chain_parity(enc_model, head, voc_model, enc_path, voc_path, wav, voice):
    """audio -> encoder (ORT) -> vocoder (ORT) vs. the same chain in PyTorch
    (forward() offline), on the same conditioning. Reported, not gated: a
    near-tie argmax flip changes the audio legitimately."""
    enc, voc = OrtStream(enc_path), OrtStream(voc_path)
    wav = whole_hops(wav, enc.contract.hop)
    cond = conditioning(wav, voice, enc.contract, voc.contract)
    emb = np.asarray(voice["embedding"], np.float32)
    ort_units, _ = encoder_chunked(enc, wav)
    ref_units = encoder_offline(enc_model, head, wav).argmax(-1)
    ort_audio = vocoder_chunked(voc, vocoder_inputs(voc.contract, ort_units, cond, emb))
    ref_audio = vocoder_offline(voc_model, voc.contract,
                                vocoder_inputs(voc.contract, ref_units, cond, emb))
    r = fp.compare(ref_audio[: len(ort_audio)], ort_audio)
    return {"seconds": len(wav) / enc.contract.sample_rate,
            "unit_agreement": float(np.mean(ort_units == ref_units)),
            "snr_db": _finite(r.snr_db), "max_abs": r.max_abs}


def chain_outputs(enc_path, voc_path, wav, voice):
    """The ORT chunked chain on `wav`, as golden outputs: (units, cond, audio).
    Without a vocoder, cond and audio are None."""
    enc = OrtStream(enc_path)
    wav = whole_hops(wav, enc.contract.hop)
    units, _ = encoder_chunked(enc, wav)
    if voc_path is None:
        return units, None, None
    voc = OrtStream(voc_path)
    cond = conditioning(wav, voice, enc.contract, voc.contract)
    full = vocoder_inputs(voc.contract, units, cond, np.asarray(voice["embedding"], np.float32))
    return units, cond, vocoder_chunked(voc, full)


# -- cost ----------------------------------------------------------------------
def step_cost(streams, feeds, warmup=20):
    """Per-hop wall time of running `streams` back to back (encoder, or
    vocoder, or encoder then vocoder) on one thread. `feeds` yields one list of
    feed dicts (one per stream) per hop. Times include the Python call."""
    for s in streams:
        s.reset()
    times = []
    for i, hop_feeds in enumerate(feeds):
        t0 = time.perf_counter_ns()
        for s, f in zip(streams, hop_feeds):
            s.step(f)
        if i >= warmup:
            times.append(time.perf_counter_ns() - t0)
    if not times:
        raise ExportError(f"cost needs more than {warmup} warm-up hops")
    ms = np.asarray(times, np.float64) / 1e6
    c = streams[0].contract
    hop_ms = 1e3 * c.hop / c.sample_rate
    p95 = float(np.percentile(ms, 95))
    return {"steps": int(len(ms)), "p50_ms": float(np.percentile(ms, 50)), "p95_ms": p95,
            "p99_ms": float(np.percentile(ms, 99)), "max_ms": float(ms.max()),
            "hop_ms": hop_ms, "p95_fraction_of_hop": p95 / hop_ms,
            "within_budget": bool(p95 <= COMPUTE_BUDGET * hop_ms)}


def encoder_feeds(c, wav):
    hop = c.hop
    for i in range(len(wav) // hop):
        yield {"audio": np.ascontiguousarray(wav[None, i * hop:(i + 1) * hop], np.float32)}


# -- voices.json -------------------------------------------------------------------
def voices_json(model, ckpt):
    """voices.json (schema 1) from the vocoder's speaker table and the
    speaker names and per-speaker stats stored in its checkpoint."""
    get = ckpt.get if isinstance(ckpt, dict) else (lambda key: None)
    speakers, stats = _json_value(get("speakers")), _json_value(get("speaker_stats"))
    if not isinstance(speakers, list) or not isinstance(stats, dict):
        raise ExportError("vocoder checkpoint needs 'speakers' (speaker-table row names, in "
                          "row order) and 'speaker_stats' ({name: logf0/energy mean and std})")
    try:
        return model.voices_json(speakers, stats, _json_value(get("speaker_metadata")))
    except (KeyError, ValueError) as e:      # ContractError is a ValueError
        raise ExportError(f"voices.json: {e}") from None


def _json_value(v):
    return json.loads(v) if isinstance(v, str) else v


# -- internals -------------------------------------------------------------------
@contextlib.contextmanager
def _capture(record):
    """Collect warnings (kept in the record) and swallow exporter chatter: the
    dynamo exporter prints progress lines with emoji that a cp1252 console
    cannot encode."""
    with warnings.catch_warnings(record=True) as caught, \
            contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("always")
        try:
            yield
        finally:
            seen = []
            for w in caught:
                text = str(w.message).strip().splitlines()
                msg = f"{w.category.__name__}: {text[0][:300] if text else ''}"
                if msg not in seen:
                    seen.append(msg)
            record["warnings"] = seen


def _set_meta(model, key, value):
    for p in model.metadata_props:
        if p.key == key:
            p.value = value
            return
    p = model.metadata_props.add()
    p.key, p.value = key, value


def _one_line(e):
    text = str(e).strip().splitlines()
    first = text[0] if text else type(e).__name__
    if isinstance(e, BaseException):
        first = f"{type(e).__name__}: {first}"
    return first[:300]


def _finite(x):
    """JSON has no inf: identical signals report snr_db None."""
    return None if not np.isfinite(x) else float(x)
