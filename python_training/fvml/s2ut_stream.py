"""Mode B streaming driver: WaitKPolicy + the translator, one 80 ms chunk at a
time (Milestone 3 / Stage 05).

StreamingTranslator runs the canonical per-chunk order of fvml/waitk_policy.py
(the Python twin of src/pipeline/WaitKPolicy.hpp) around a *runtime*: the object
that owns the translator state and runs one translator_enc chunk or one
translator_dec step. Two runtimes share that interface:

  TorchRuntime   CausalTranslatorEncoder.step / CausalTranslatorDecoder.step
  OrtRuntime     the exported translator_enc.onnx / translator_dec.onnx in ONNX
                 Runtime, configured like Stage 07's OrtModel (one thread)

so the equivalence tests, the SimulEval runs of gate_s2ut.py and the Stage 06
parity checks all step the policy the C++ worker runs. Decision gates must time
OrtRuntime, not TorchRuntime (MEMORY.md rule 11).

When the policy completes a tail flush (reset to IDLE) the driver zeroes the
whole translator state and goes back to <BOS>, as the C++ worker does. A
mid-flush speech resumption keeps the state.

Writes. One policy write is one 80 ms target chunk:
  stacked_4    one decoder step, chunk_frames units.
  reduced_dur  decoder steps until the emitted durations reach the frame budget
               (chunk_frames more frames per write; any surplus carries over).
Unit frames from the first special unit (<BOS>/<EOS>/<PAD>) onwards are dropped,
like UnitProsodyFifo::push_chunk. <EOS> is the unit itself or eos_prob >
eos_threshold.
"""
from dataclasses import dataclass, field
import math
import pathlib
import time

import numpy as np
import torch

from . import translator as tr
from .waitk_policy import Decision, WaitKConfig, WaitKPolicy


# -- runtimes -------------------------------------------------------------------
class TorchRuntime:
    """The PyTorch step() functions behind the runtime interface (batch 1)."""

    clock = "torch"

    def __init__(self, model, src_lang, tgt_lang):
        self.cfg = model.cfg
        self.encoder, self.decoder = model.encoder, model.decoder
        p = self.encoder.in_proj.weight
        self.device, self.dtype = p.device, p.dtype
        ids = self.cfg.lang_ids
        self.src_lang = torch.tensor([ids[src_lang]], dtype=torch.int64, device=self.device)
        self.tgt_lang = torch.tensor([ids[tgt_lang]], dtype=torch.int64, device=self.device)
        self.reset()

    def reset(self):
        self.enc_state = self.encoder.init_state(1)
        self.dec_state = self.decoder.init_state(1)

    @property
    def src_pos(self):
        return int(self.enc_state["src_pos"][0])

    @torch.no_grad()
    def encode(self, latents):
        """latents [chunk_frames, D] -> readiness_logit (float)."""
        x = torch.as_tensor(np.asarray(latents), dtype=self.dtype, device=self.device)[None]
        out, self.enc_state = self.encoder.step(x, self.src_lang, self.enc_state)
        return float(out["readiness_logit"][0, 0])

    @torch.no_grad()
    def decode(self, prev_units, prev_dur=None):
        """prev_units [U] (and prev_dur for reduced_dur) -> {units [U], eos_prob,
        duration (reduced_dur), logits [U, vocab + 3], dur_logits}."""
        prev = torch.as_tensor(np.asarray(prev_units), dtype=torch.int64, device=self.device)[None]
        dur = None if prev_dur is None else torch.tensor([[int(prev_dur)]], dtype=torch.int64,
                                                         device=self.device)
        out, self.dec_state = self.decoder.step(
            prev, self.tgt_lang, self.enc_state["src_pos"],
            {n: self.enc_state[n] for n in self.decoder.shared_mem_names()}, self.dec_state,
            prev_dur=dur)
        res = {"units": out["units"][0].cpu().numpy(), "eos_prob": float(out["eos_prob"][0, 0]),
               "logits": out["logits"][0].cpu().numpy()}
        if "duration" in out:
            res["duration"] = int(out["duration"][0, 0])
            res["dur_logits"] = out["dur_logits"][0].cpu().numpy()
        return res


class OrtRuntime:
    """translator_enc.onnx + translator_dec.onnx stepped in ONNX Runtime: every
    state_out.X fed back as state_in.X, and translator_enc's memory rings and
    src_pos handed to translator_dec (what Stage 07's KVStateCache does).
    `debug_outputs` (fvml.contract.TensorSpec) re-exposes intermediate tensors,
    e.g. the decoder's `logits`, that the shipped file dropped."""

    clock = "ort"
    _DTYPES = {"tensor(float)": np.float32, "tensor(int64)": np.int64,
               "tensor(float16)": np.float16}

    def __init__(self, enc_path, dec_path, cfg, src_lang, tgt_lang, debug_outputs=()):
        from . import export as fx
        self.cfg = cfg
        self.enc = fx.ort_session(enc_path)
        dec = dec_path
        if debug_outputs:
            import onnx
            dec = fx.with_outputs(onnx.load(str(dec_path)), list(debug_outputs))
        self.dec = fx.ort_session(dec)
        self.enc_outputs = [o.name for o in self.enc.get_outputs()]
        self.dec_outputs = [o.name for o in self.dec.get_outputs()]
        self._enc_state = self._state_specs(self.enc)
        self._dec_state = self._state_specs(self.dec)
        dec_inputs = {i.name for i in self.dec.get_inputs()}
        self.mem_names = [n for n in dec_inputs if n.startswith(("mem_k_", "mem_v_"))]
        self.has_prosody_ctx = "prosody_ctx" in dec_inputs
        ids = cfg.lang_ids
        self.src_lang = np.asarray([ids[src_lang]], np.int64)
        self.tgt_lang = np.asarray([ids[tgt_lang]], np.int64)
        self.reset()

    @classmethod
    def _state_specs(cls, session):
        return {i.name: (tuple(i.shape), cls._DTYPES[i.type]) for i in session.get_inputs()
                if i.name.startswith("state_in.")}

    def reset(self):
        self.enc_state = {n: np.zeros(s, d) for n, (s, d) in self._enc_state.items()}
        self.dec_state = {n: np.zeros(s, d) for n, (s, d) in self._dec_state.items()}

    @property
    def src_pos(self):
        return int(self.enc_state["state_in.src_pos"][0])

    @staticmethod
    def _carry(names, values, state):
        out = {}
        for name, value in zip(names, values):
            if name.startswith("state_out."):
                state["state_in." + name[len("state_out."):]] = value
            else:
                out[name] = value
        return out

    def encode(self, latents):
        feeds = {"latents": np.asarray(latents, np.float32)[None], "src_lang": self.src_lang,
                 **self.enc_state}
        out = self._carry(self.enc_outputs, self.enc.run(self.enc_outputs, feeds), self.enc_state)
        return float(out["readiness_logit"][0, 0])

    def decode(self, prev_units, prev_dur=None):
        feeds = {"prev_units": np.asarray(prev_units, np.int64)[None], "tgt_lang": self.tgt_lang,
                 "src_pos": self.enc_state["state_in.src_pos"], **self.dec_state,
                 **{n: self.enc_state["state_in." + n] for n in self.mem_names}}
        if self.has_prosody_ctx:
            feeds["prosody_ctx"] = np.zeros((1, 2), np.float32)
        if prev_dur is not None:
            feeds["prev_dur"] = np.asarray([[int(prev_dur)]], np.int64)
        out = self._carry(self.dec_outputs, self.dec.run(self.dec_outputs, feeds), self.dec_state)
        res = {"units": out["units"][0], "eos_prob": float(out["eos_prob"][0, 0])}
        if "logits" in out:
            res["logits"] = out["logits"][0]
        if "duration" in out:
            res["duration"] = int(out["duration"][0, 0])
        return res


# -- export ---------------------------------------------------------------------
def export_step_graphs(model, out_dir, exporter="auto", license=tr.LICENSE):
    """Write translator_enc.onnx and translator_dec.onnx (fixed-shape, batch 1,
    stateful) into `out_dir` and return a record: the two paths, the exporter
    each graph went through and any dynamo fallback reason (BUG-016), and the
    contracts. stacked_4 files are contract v2 (embedded, checked by
    fvml.export.finalize). reduced_dur files carry no contract: their decoder
    signature (prev_dur in, duration out) is outside contract v2."""
    from . import contract as fc
    from . import export as fx
    from .contract import TensorSpec
    out_dir = pathlib.Path(out_dir)
    cfg = model.cfg
    model = model.eval()
    enc_path, dec_path = out_dir / "translator_enc.onnx", out_dir / "translator_dec.onnx"
    enc_graph, enc_c = tr.translator_enc_graph(model.encoder, license)
    enc_rec = fx.export_graph(enc_graph, enc_path, exporter=exporter)
    fx.finalize(enc_path, enc_c, metadata={"fluxvoice.exporter": enc_rec["exporter"]})
    record = {"encoder": str(enc_path), "decoder": str(dec_path), "unit_mode": cfg.unit_mode,
              "encoder_exporter": enc_rec["exporter"], "encoder_contract": enc_c,
              "fallback_reason": enc_rec["fallback_reason"]}
    if cfg.unit_mode == "stacked_4":
        dec_graph, dec_c = tr.translator_dec_graph(model.decoder, model.encoder, license)
        dec_rec = fx.export_graph(dec_graph, dec_path, exporter=exporter)
        fx.finalize(dec_path, dec_c, drop_outputs=dec_graph.debug_outputs,
                    metadata={"fluxvoice.exporter": dec_rec["exporter"]})
        _, enc_proto = fc.load_onnx(enc_path)
        _, dec_proto = fc.load_onnx(dec_path)
        fc.validate_s2ut_shared_io(enc_c, fc.graph_io(enc_proto)[1], dec_c,
                                   fc.graph_io(dec_proto)[0])
        record["decoder_contract"] = dec_c
    else:
        mem_init = model.encoder.init_state(1)
        mem_names = model.decoder.shared_mem_names()
        decoder = model.decoder

        def step(ins, state):
            out, new = decoder.step(ins["prev_units"], ins["tgt_lang"], ins["src_pos"],
                                    {n: ins[n] for n in mem_names}, state,
                                    prev_dur=ins["prev_dur"])
            return {"units": out["units"], "duration": out["duration"],
                    "eos_prob": out["eos_prob"]}, new

        dec_graph = fx.StepGraph(
            decoder, step,
            [TensorSpec("prev_units", "i64", (1, 1)), TensorSpec("prev_dur", "i64", (1, 1)),
             TensorSpec("tgt_lang", "i64", (1,)), TensorSpec("src_pos", "i64", (1,))]
            + [TensorSpec(n, "f32", tuple(mem_init[n].shape)) for n in mem_names],
            [TensorSpec("units", "i64", (1, 1)), TensorSpec("duration", "i64", (1, 1)),
             TensorSpec("eos_prob", "f32", (1, 1))], decoder.init_state(1))
        dec_rec = fx.export_graph(dec_graph, dec_path, exporter=exporter)
        _check_plain(dec_path)
        record["decoder_contract"] = None
    record["decoder_exporter"] = dec_rec["exporter"]
    record["fallback_reason"] = record["fallback_reason"] or dec_rec["fallback_reason"]
    return record


def _check_plain(path):
    """IR cap + strict shape inference + full checker for a graph without a
    contract (what fvml.export.finalize does for contract graphs)."""
    import onnx
    from . import export as fx
    m = onnx.load(str(path))
    if fx.default_opset(m) > fx.MAX_OPSET:
        raise fx.ExportError(f"{pathlib.Path(path).name}: opset {fx.default_opset(m)} > "
                             f"{fx.MAX_OPSET}")
    m.ir_version = min(m.ir_version, fx.MAX_IR_VERSION)
    m = onnx.shape_inference.infer_shapes(m, check_type=True, strict_mode=True)
    onnx.checker.check_model(m, full_check=True)
    onnx.save(m, str(path))


# -- the per-chunk driver ---------------------------------------------------------
@dataclass
class ChunkResult:
    decision: Decision
    encoded: bool                   # translator_enc ran on this chunk
    writes: int = 0                 # policy writes (target chunks) this chunk
    decoder_steps: int = 0
    frames: list = field(default_factory=list)   # unit ids emitted (specials dropped)
    eos: bool = False
    flushed: bool = False           # the utterance drained; translator state was reset
    t_src: int = 0                  # policy counters after the chunk
    t_tgt: int = 0
    first_write_t_src: int = 0      # policy t_src at this utterance's first write (0 = none yet)
    enc_ms: float = 0.0
    dec_ms: float = 0.0


class StreamingTranslator:
    """WaitKPolicy + a translator runtime, stepped one chunk at a time.

    `teacher(step_index)`, if given, returns the units ([U] ints; for
    reduced_dur a (unit, duration) pair) the decoder is *forced* to have emitted
    at that decoder step: they replace its own argmax as the emission and as the
    next step's input, and eos_prob is ignored (teacher forcing, used by the
    equivalence tests).
    With record=True every decoder step is logged in `dec_log` (segment,
    src_visible, prev_units, prev_dur, logits, ...) and every chunk in
    `chunk_log`, so a test can replay the same inputs through forward()."""

    def __init__(self, runtime, policy=None, eos_threshold=0.5, teacher=None, record=False):
        self.runtime = runtime
        self.cfg = runtime.cfg
        self.policy = WaitKPolicy(policy if policy is not None else WaitKConfig())
        self.eos_threshold = float(eos_threshold)
        self.teacher = teacher
        self.record = record
        self.dec_log, self.chunk_log = [], []
        self.segment = 0
        self.dec_steps_total = 0
        self.reset()

    def reset(self):
        """New stream: policy to IDLE, translator state zeroed."""
        self.policy.reset()
        self._reset_translator()

    def _reset_translator(self):
        cfg = self.cfg
        self.runtime.reset()
        self.prev_units = np.full(cfg.units_per_step, cfg.bos, np.int64)
        self.prev_dur = 0
        self.frames_emitted = 0         # reduced_dur frame accounting
        self.frame_budget = 0
        self.first_write_t_src = 0

    # -- one decoder step ---------------------------------------------------
    def _decode(self):
        cfg = self.cfg
        reduced = cfg.unit_mode == "reduced_dur"
        src_visible = self.runtime.src_pos
        t0 = time.perf_counter()
        out = self.runtime.decode(self.prev_units, self.prev_dur if reduced else None)
        ms = 1e3 * (time.perf_counter() - t0)
        units = np.asarray(out["units"], np.int64).reshape(-1)
        duration = int(out.get("duration", 0))
        if self.teacher is not None:
            forced = self.teacher(self.dec_steps_total)
            if reduced:
                units, duration = np.asarray([forced[0]], np.int64), int(forced[1])
            else:
                units = np.asarray(forced, np.int64).reshape(-1)
        if self.record:
            self.dec_log.append({
                "segment": self.segment, "src_visible": src_visible,
                "prev_units": self.prev_units.copy(), "prev_dur": self.prev_dur,
                "units": units.copy(), "duration": duration, "eos_prob": out["eos_prob"],
                "logits": out.get("logits"), "dur_logits": out.get("dur_logits")})
        self.dec_steps_total += 1
        self.prev_units, self.prev_dur = units, duration
        special = np.flatnonzero((units < 0) | (units >= cfg.vocab))
        valid = units[:special[0]] if special.size else units
        eos = bool(np.any(units == cfg.eos))
        if self.teacher is None:        # a forced stream ends only where the teacher says
            eos = eos or out["eos_prob"] > self.eos_threshold
        return valid, duration, eos, ms

    def _write(self):
        """One policy write. Returns (unit frames, eos, decoder steps, ms)."""
        cfg = self.cfg
        if cfg.unit_mode == "stacked_4":
            valid, _, eos, ms = self._decode()
            return valid.tolist(), eos, 1, ms
        frames, eos, steps, total_ms = [], False, 0, 0.0
        self.frame_budget += cfg.chunk_frames
        while self.frames_emitted < self.frame_budget and not eos and steps < cfg.chunk_frames:
            valid, duration, eos, ms = self._decode()
            duration = min(max(duration, 1), cfg.max_duration)
            frames += [int(valid[0])] * duration if valid.size else []
            self.frames_emitted += duration
            steps += 1
            total_ms += ms
        return frames, eos, steps, total_ms

    # -- one chunk ------------------------------------------------------------
    def push_chunk(self, latents, speech=None):
        """Advance one 80 ms chunk. `latents` [chunk_frames, D] is Stage 1's
        output for the chunk; `speech` is its VAD verdict (None = no VAD: the
        policy then never flushes and the stream is one endless utterance)."""
        policy = self.policy
        flushes = policy.flushes_completed
        if speech is not None:
            policy.on_chunk_energy(bool(speech), int(round(self.cfg.chunk_ms)))
        res = ChunkResult(Decision.READ, False)
        if policy.source_wanted():
            t0 = time.perf_counter()
            readiness = self.runtime.encode(latents)
            res.enc_ms = 1e3 * (time.perf_counter() - t0)
            res.encoded = True
            policy.notify_source_unit()
            if policy.config.adaptive_mode:
                policy.on_source_readiness(1.0 / (1.0 + math.exp(-readiness)))
        res.decision = policy.decide()
        for _ in range(policy.max_writes_this_step()):
            if not self.first_write_t_src:
                self.first_write_t_src = policy.t_src
            frames, eos, steps, ms = self._write()
            res.frames += frames
            res.eos = res.eos or eos
            res.writes += 1
            res.decoder_steps += steps
            res.dec_ms += ms
            policy.notify_target_unit(eos)
            if policy.flushes_completed != flushes:
                break
        res.first_write_t_src = self.first_write_t_src
        res.t_src, res.t_tgt = policy.t_src, policy.t_tgt
        if self.record:
            self.chunk_log.append({"segment": self.segment, "encoded": res.encoded,
                                   "decision": res.decision, "writes": res.writes})
        if policy.flushes_completed != flushes:
            res.flushed = True
            self._reset_translator()
            self.segment += 1
        return res
