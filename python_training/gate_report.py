#!/usr/bin/env python3
"""Stage 05 decision gate, ONNX export smoke test, and controls readiness checks.

Spec: Documents/Milestone 2/05_causal_vocoder_decision_gate.md.

Responsibilities:
  1. Convert the frozen `test_clean_200.tsv` list into target voices using the
     full *streaming* chain (ContentEncoder.step + PitchTracker.push +
     map_to_target + UnitVocoder.step) for each candidate (student-L0/L1/L2 at
     s1/s2/s4, plus mimi-q0 when provided).
  2. Score each candidate with `fvml.evaluate.evaluate` (WER, CER, tgt_cos,
     src_cos, UTMOS, F0 PCC, RTF).
  3. Measure single-core CPU per-step compute p95 (encoder + tracker + vocoder)
     and compute projected T-B latency from the master latency budget formula.
  4. Run the ONNX export smoke test: export a per-step stateful ONNX graph for
     both encoder and vocoder, validate against `fvml.contract` (including
     `validate_pair`), and run one step in ONNX Runtime CPU.
  5. Apply the Stage 05 eligibility and latency-first selection rule and format
     the Markdown gate table.
  6. Run the Stage 05 controls readiness checks (pitch shift -12/-6/0/+6/+12 st,
     intonation scale 0/1/2, 50/50 speaker-embedding blend).
"""
import argparse
import csv
import dataclasses
import json
import math
import pathlib
import sys
import tempfile
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from fvml import audio, contract, evaluate, evallist, f0_ref, features, paths, vocoder
from fvml import content_encoder as ce
import probe_units

IO_FLOOR_MS = 25.0
GOVERNOR_MARGIN_50HZ_MS = 5.0
GOVERNOR_MARGIN_MIMI_MS = 10.0

ELIGIBILITY = {
    "max_wer_abs": 0.10,
    "max_wer_delta": 0.03,
    "max_tgt_cos_drop": 0.05,
    "min_utmos": 3.0,
    "max_compute_hop_ratio": 0.50,
}


@dataclasses.dataclass(frozen=True)
class CandidateSpec:
    name: str               # e.g. "student-L1 s1"
    tokenizer: str          # "student-L0" | "student-L1" | "student-L2" | "mimi-q0"
    frames_per_step: int    # 1, 2, or 4
    lookahead_frames: int   # 0, 1, or 2
    frame_samples: int = 320
    sample_rate: int = 16000

    @property
    def slug(self):
        return self.name.replace(" ", "_")

    @property
    def hop_samples(self):
        return self.frames_per_step * self.frame_samples

    @property
    def hop_ms(self):
        return 1000.0 * self.hop_samples / self.sample_rate

    @property
    def frame_ms(self):
        return 1000.0 * self.frame_samples / self.sample_rate

    @property
    def alg_latency_ms(self):
        return self.hop_ms + self.lookahead_frames * self.frame_ms

    @property
    def governor_margin_ms(self):
        return GOVERNOR_MARGIN_MIMI_MS if self.tokenizer == "mimi-q0" else GOVERNOR_MARGIN_50HZ_MS


DEFAULT_CANDIDATES = (
    CandidateSpec("student-L0 s1", "student-L0", 1, 0),
    CandidateSpec("student-L1 s1", "student-L1", 1, 1),
    CandidateSpec("student-L2 s1", "student-L2", 1, 2),
    CandidateSpec("student-L1 s2", "student-L1", 2, 1),
)


def parse_candidate_spec(spec_str):
    """Parse 'student-L1:s2' or 'student-L1 s2' or 'mimi-q0:s1' into a CandidateSpec."""
    parts = spec_str.replace(":", " ").split()
    tok = parts[0]
    fps = int(parts[1].lstrip("s")) if len(parts) > 1 else 1
    if tok == "mimi-q0":
        return CandidateSpec(f"mimi-q0 s{fps}" if fps > 1 else "mimi-q0",
                             "mimi-q0", fps, 0, frame_samples=1280, sample_rate=16000)
    L = ce.head_lookahead(tok)
    return CandidateSpec(f"{tok} s{fps}", tok, fps, L, frame_samples=ce.FRAME_SAMPLES,
                         sample_rate=ce.SAMPLE_RATE)


def projected_t_minus_b_ms(cand, compute_p95_ms, io_floor_ms=IO_FLOOR_MS):
    """Master latency budget formula: T - B = IO_floor + hop + lookahead + compute_p95 + margin."""
    return float(io_floor_ms + cand.alg_latency_ms + compute_p95_ms + cand.governor_margin_ms)


# -- ONNX export smoke test ---------------------------------------------------
class _EncoderStepWrapper(nn.Module):
    """Wraps ContentEncoder.step for a single head as flat tensor I/O."""

    def __init__(self, model, head):
        super().__init__()
        self.model = model
        self.head = head
        self.state_names = [n for n, _, _ in model.state_layout()]

    def forward(self, audio_in, *states_in):
        state = dict(zip(self.state_names, states_in))
        out, new_state = self.model.step(audio_in, state)
        logits = out["logits"][self.head]
        content = logits.argmax(dim=-1).to(torch.int64)
        return (content, *[new_state[n] for n in self.state_names])


class _VocoderStepWrapper(nn.Module):
    """Wraps UnitVocoder.step as flat tensor I/O matching ModelContract v1."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.state_names = [n for n, _, _ in model.state_layout()]

    def forward(self, units_in, logf0, voicing, aperiodicity, energy, spk_emb, *states_in):
        state = dict(zip(self.state_names, states_in))
        audio_out, new_state = self.model.step(
            units_in, logf0, voicing, aperiodicity, energy, spk_emb, state
        )
        return (audio_out, *[new_state[n] for n in self.state_names])


def _export_module_to_onnx(wrapper, args_tuple, input_names, output_names, model_contract, out_path):
    """Export `wrapper` to `out_path`, embed `model_contract`, and validate with ONNX + contract."""
    wrapper.eval()
    exporter_used = None
    last_err = None
    for use_dynamo in (True, False):
        try:
            if use_dynamo:
                torch.onnx.export(wrapper, args_tuple, str(out_path), dynamo=True,
                                  opset_version=18, external_data=False,
                                  input_names=input_names, output_names=output_names)
                exporter_used = "dynamo"
            else:
                torch.onnx.export(wrapper, args_tuple, str(out_path), dynamo=False,
                                  opset_version=17,
                                  input_names=input_names, output_names=output_names)
                exporter_used = "torchscript"
            break
        except Exception as e:
            last_err = e
    if exporter_used is None:
        raise RuntimeError(f"ONNX export failed: {last_err}")

    proto = onnx.load(str(out_path), load_external_data=True)
    if proto.ir_version > 10:
        proto.ir_version = 10
    for imp in proto.opset_import:
        if imp.domain in ("", "ai.onnx") and imp.version > 18:
            imp.version = 18
    proto = onnx.shape_inference.infer_shapes(proto)
    contract.embed(proto, model_contract.to_json())
    onnx.checker.check_model(proto, full_check=True)
    onnx.save_model(proto, str(out_path), save_as_external_data=False)
    contract.load_onnx(str(out_path))
    return exporter_used


@torch.inference_mode()
def export_smoke_test(cand, enc=None, voc=None):
    """Export a per-step ONNX graph of the candidate encoder + vocoder and run
    one step in ONNX Runtime CPU. Returns a report dict with `ok: bool`."""
    if cand.tokenizer == "mimi-q0":
        return {"ok": False, "reason": "mimi-q0 stateful ONNX ring-KV export not enabled before kill-test pass"}
    try:
        fps = cand.frames_per_step
        head = ce.LOOKAHEAD.index(cand.lookahead_frames)
        enc_m = (enc or ce.ContentEncoder()).eval().float().cpu()
        voc_m = (voc or vocoder.UnitVocoder(
            vocoder.VocoderConfig(vocab=enc_m.cfg.vocab, frame_samples=cand.frame_samples)
        )).eval().float().cpu()

        enc_states = [n for n, _, _ in enc_m.state_layout()]
        voc_states = [n for n, _, _ in voc_m.state_layout()]
        enc_contract = contract.ModelContract(
            role="encoder",
            sample_rate=enc_m.cfg.vocab and ce.SAMPLE_RATE,
            frame_samples=ce.FRAME_SAMPLES,
            frames_per_step=fps,
            lookahead_frames=cand.lookahead_frames,
            content_kind="discrete",
            vocab=enc_m.cfg.vocab,
            states=tuple(enc_states),
            license="MIT",
        )
        voc_contract = voc_m.contract(frames_per_step=fps)
        contract.validate_pair(enc_contract, voc_contract)

        hop = enc_contract.hop
        cond_frames = voc_contract.cond_frames
        enc_init = enc_m.init_state(1)
        voc_init = voc_m.init_state(1)

        audio_in = torch.randn(1, hop, dtype=torch.float32)
        units_in = torch.zeros(1, fps, dtype=torch.int64)
        logf0 = torch.full((1, cond_frames), 5.0, dtype=torch.float32)
        voicing = torch.ones(1, cond_frames, dtype=torch.float32)
        aperiodicity = torch.full((1, cond_frames), 0.1, dtype=torch.float32)
        energy = torch.full((1, cond_frames), -5.0, dtype=torch.float32)
        spk_emb = voc_m.speaker_table.weight[:1].detach().clone()

        enc_wrap = _EncoderStepWrapper(enc_m, head)
        voc_wrap = _VocoderStepWrapper(voc_m)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = pathlib.Path(tmp)
            enc_path = tmp_dir / f"encoder_s{fps}.onnx"
            voc_path = tmp_dir / f"vocoder_s{fps}.onnx"

            enc_exp = _export_module_to_onnx(
                enc_wrap,
                (audio_in, *[enc_init[n] for n in enc_states]),
                ["audio", *[f"state_in.{n}" for n in enc_states]],
                ["units", *[f"state_out.{n}" for n in enc_states]],
                enc_contract,
                enc_path,
            )
            voc_exp = _export_module_to_onnx(
                voc_wrap,
                (units_in, logf0, voicing, aperiodicity, energy, spk_emb,
                 *[voc_init[n] for n in voc_states]),
                ["units", "logf0", "voicing", "aperiodicity", "energy", "spk_emb",
                 *[f"state_in.{n}" for n in voc_states]],
                ["audio", *[f"state_out.{n}" for n in voc_states]],
                voc_contract,
                voc_path,
            )

            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = 1
            sess_opts.inter_op_num_threads = 1
            enc_sess = ort.InferenceSession(str(enc_path), sess_opts,
                                            providers=["CPUExecutionProvider"])
            voc_sess = ort.InferenceSession(str(voc_path), sess_opts,
                                            providers=["CPUExecutionProvider"])

            enc_feed = {"audio": audio_in.numpy(),
                        **{f"state_in.{n}": enc_init[n].numpy() for n in enc_states}}
            enc_ort_outs = enc_sess.run(None, enc_feed)
            pt_enc_outs = enc_wrap(audio_in, *[enc_init[n] for n in enc_states])
            enc_units_match = bool(np.array_equal(enc_ort_outs[0], pt_enc_outs[0].numpy()))

            voc_feed = {
                "units": enc_ort_outs[0],
                "logf0": logf0.numpy(),
                "voicing": voicing.numpy(),
                "aperiodicity": aperiodicity.numpy(),
                "energy": energy.numpy(),
                "spk_emb": spk_emb.numpy(),
                **{f"state_in.{n}": voc_init[n].numpy() for n in voc_states},
            }
            voc_ort_outs = voc_sess.run(None, voc_feed)
            pt_voc_outs = voc_wrap(
                torch.from_numpy(enc_ort_outs[0]), logf0, voicing, aperiodicity,
                energy, spk_emb, *[voc_init[n] for n in voc_states]
            )
            voc_max_abs = float(np.max(np.abs(voc_ort_outs[0] - pt_voc_outs[0].numpy())))

            # Also measure single-core ORT step latency (encoder + vocoder) over 50 steps.
            ort_enc_times, ort_voc_times = [], []
            for i in range(60):
                t0 = time.perf_counter()
                e_out = enc_sess.run(None, enc_feed)
                t1 = time.perf_counter()
                voc_feed["units"] = e_out[0]
                voc_sess.run(None, voc_feed)
                t2 = time.perf_counter()
                if i >= 10:
                    ort_enc_times.append(t1 - t0)
                    ort_voc_times.append(t2 - t1)
            ort_enc_p = probe_units.percentiles(ort_enc_times)
            ort_voc_p = probe_units.percentiles(ort_voc_times)
            ort_tot_p = probe_units.percentiles(np.asarray(ort_enc_times) + np.asarray(ort_voc_times))

        ok = enc_units_match and voc_max_abs <= 1e-4
        return {
            "ok": ok,
            "encoder_exporter": enc_exp,
            "vocoder_exporter": voc_exp,
            "encoder_units_match": enc_units_match,
            "vocoder_max_abs": voc_max_abs,
            "ort_encoder_p50_ms": ort_enc_p["p50_ms"],
            "ort_encoder_p95_ms": ort_enc_p["p95_ms"],
            "ort_vocoder_p50_ms": ort_voc_p["p50_ms"],
            "ort_vocoder_p95_ms": ort_voc_p["p95_ms"],
            "ort_total_p50_ms": ort_tot_p["p50_ms"],
            "ort_total_p95_ms": ort_tot_p["p95_ms"],
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


# -- single-core compute cost -------------------------------------------------
@torch.inference_mode()
def measure_candidate_cost(cand, enc=None, voc=None, steps=200, warmup=30):
    """Single-core CPU per-step latency (encoder + YIN tracker + vocoder) in ms."""
    probe_units.single_core()
    fps = cand.frames_per_step
    hop = cand.hop_samples
    cond_frames = hop // vocoder.COND_FRAME_SAMPLES
    rng = np.random.default_rng(0)

    voc_m = (voc or vocoder.UnitVocoder(
        vocoder.VocoderConfig(vocab=2048 if cand.tokenizer == "mimi-q0" else 100,
                              frame_samples=cand.frame_samples)
    )).eval().float().cpu()
    voc_state = voc_m.init_state(1)
    spk_emb = voc_m.speaker_table.weight[:1].detach().clone()
    voice_dummy = {"logf0_mean": 5.0, "logf0_std": 0.3, "energy_mean": -6.0, "energy_std": 2.5}
    tracker = features.PitchTracker(features.PitchConfig(lookahead_frames=cand.lookahead_frames))

    if cand.tokenizer == "mimi-q0":
        mimi_c = probe_units.mimi_cost([fps], max(10, steps // 4), max(5, warmup // 4))[f"fps{fps}"]
        voc_c = probe_units.vocoder_cost(None, [fps], steps, warmup, cfg=voc_m.cfg)[f"fps{fps}"]
        p95 = mimi_c["p95_ms"] + voc_c["p95_ms"]
        return {
            "encoder_p95_ms": mimi_c["p95_ms"],
            "vocoder_p95_ms": voc_c["p95_ms"],
            "total_p50_ms": mimi_c["p50_ms"] + voc_c["p50_ms"],
            "total_p95_ms": p95,
            "total_p99_ms": mimi_c["p99_ms"] + voc_c["p99_ms"],
            "hop_ms": cand.hop_ms,
            "rtf_p95": p95 / cand.hop_ms,
        }

    head = ce.LOOKAHEAD.index(cand.lookahead_frames)
    enc_m = (enc or ce.ContentEncoder()).eval().float().cpu()
    enc_state = enc_m.init_state(1)

    wav_chunks = (0.1 * rng.standard_normal((warmup + steps, hop))).astype(np.float32)
    enc_times, voc_times, total_times = [], [], []
    for i in range(warmup + steps):
        chunk_np = wav_chunks[i]
        t0 = time.perf_counter()
        chunk_t = torch.from_numpy(chunk_np)[None]
        enc_out, enc_state = enc_m.step(chunk_t, enc_state)
        u_hop = enc_out["logits"][head].argmax(dim=-1).to(torch.int64)
        t_enc = time.perf_counter()

        pfs = [tracker.push(chunk_np[f * 320:(f + 1) * 320]) for f in range(cond_frames)]
        z_lf = np.array([p.z_logf0 for p in pfs], dtype=np.float32)
        z_en = np.array([p.z_energy for p in pfs], dtype=np.float32)
        lf_tgt, en_tgt = features.map_to_target(z_lf, z_en, voice_dummy)
        lf_t = torch.from_numpy(np.ascontiguousarray(lf_tgt, np.float32))[None]
        en_t = torch.from_numpy(np.ascontiguousarray(en_tgt, np.float32))[None]
        vo_t = torch.tensor([[p.voicing for p in pfs]], dtype=torch.float32)
        ap_t = torch.tensor([[p.aperiodicity for p in pfs]], dtype=torch.float32)

        _, voc_state = voc_m.step(u_hop, lf_t, vo_t, ap_t, en_t, spk_emb, voc_state)
        t_end = time.perf_counter()

        enc_times.append(t_enc - t0)
        voc_times.append(t_end - t_enc)
        total_times.append(t_end - t0)

    tot = probe_units.percentiles(total_times[warmup:])
    enc_p = probe_units.percentiles(enc_times[warmup:])
    voc_p = probe_units.percentiles(voc_times[warmup:])
    return {
        "encoder_p95_ms": enc_p["p95_ms"],
        "vocoder_p95_ms": voc_p["p95_ms"],
        "total_p50_ms": tot["p50_ms"],
        "total_p95_ms": tot["p95_ms"],
        "total_p99_ms": tot["p99_ms"],
        "hop_ms": cand.hop_ms,
        "rtf_p95": tot["p95_ms"] / cand.hop_ms,
    }


# -- streaming conversion chain -----------------------------------------------
@torch.inference_mode()
def convert_utterance_streaming(wav, cand, enc, voc, voice_entry, *,
                                scale=1.0, shift_st=0.0, spk_emb_override=None):
    """Convert one 16 kHz utterance through the full streaming chain at `cand.frames_per_step`.
    Returns (converted_wav_float32, compute_seconds)."""
    enc_m = enc.eval().float().cpu()
    voc_m = voc.eval().float().cpu()
    fps = cand.frames_per_step
    hop = cand.hop_samples
    cond_frames = hop // vocoder.COND_FRAME_SAMPLES
    head = ce.LOOKAHEAD.index(cand.lookahead_frames)

    wav_np = np.asarray(wav, dtype=np.float32)
    n_hops = max(1, -(-len(wav_np) // hop))
    padded = np.zeros(n_hops * hop, dtype=np.float32)
    padded[:len(wav_np)] = wav_np

    enc_state = enc_m.init_state(1)
    voc_state = voc_m.init_state(1)
    tracker = features.PitchTracker(features.PitchConfig(lookahead_frames=cand.lookahead_frames))
    if spk_emb_override is not None:
        spk_emb = torch.as_tensor(spk_emb_override, dtype=torch.float32).reshape(1, -1)
    else:
        spk_emb = torch.tensor([voice_entry["embedding"]], dtype=torch.float32)

    out_chunks = []
    t0 = time.perf_counter()
    for h_idx in range(n_hops):
        seg = padded[h_idx * hop:(h_idx + 1) * hop]
        enc_out, enc_state = enc_m.step(torch.from_numpy(seg)[None], enc_state)
        u_hop = enc_out["logits"][head].argmax(dim=-1).to(torch.int64)

        pfs = [tracker.push(seg[f * 320:(f + 1) * 320]) for f in range(cond_frames)]
        z_lf = np.array([p.z_logf0 for p in pfs], dtype=np.float32)
        z_en = np.array([p.z_energy for p in pfs], dtype=np.float32)
        lf_tgt, en_tgt = features.map_to_target(z_lf, z_en, voice_entry,
                                                scale=scale, shift_st=shift_st)
        lf_t = torch.from_numpy(np.ascontiguousarray(lf_tgt, np.float32))[None]
        en_t = torch.from_numpy(np.ascontiguousarray(en_tgt, np.float32))[None]
        vo_t = torch.tensor([[p.voicing for p in pfs]], dtype=torch.float32)
        ap_t = torch.tensor([[p.aperiodicity for p in pfs]], dtype=torch.float32)

        y_hop, voc_state = voc_m.step(u_hop, lf_t, vo_t, ap_t, en_t, spk_emb, voc_state)
        out_chunks.append(y_hop[0].numpy())
    elapsed = time.perf_counter() - t0
    return np.concatenate(out_chunks)[:len(padded)], elapsed


def convert_eval_list(sources_tsv, cand, enc, voc, voice_entry, out_dir, limit=None):
    """Convert all rows in `sources_tsv` (or first `limit`) and write WAVs + timing.csv."""
    rows = evallist.read_sources(sources_tsv)[:limit or None]
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timings = []
    for r in rows:
        src_wav = audio.load(paths.resolve(r["path"]))
        conv_wav, secs = convert_utterance_streaming(src_wav, cand, enc, voc, voice_entry)
        audio.save(out_dir / f"{r['utt_id']}.wav", conv_wav)
        timings.append({"utt_id": r["utt_id"], "seconds": f"{secs:.6f}"})
    with open(out_dir / "timing.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["utt_id", "seconds"], lineterminator="\n")
        w.writeheader()
        w.writerows(timings)
    return len(rows)


# -- decision gate rules ------------------------------------------------------
def apply_gate_rules(rows, rules=ELIGIBILITY):
    """Annotate candidate dicts with `eligible: bool`, `ineligible_reasons: list`,
    and choose the winner (lowest projected_t_minus_b_ms among eligible, tie-break
    on highest utmos, then lowest wer). Returns (annotated_rows, decision_dict)."""
    valid_wers = [r["wer"] for r in rows if r.get("wer") is not None and not math.isnan(r["wer"])]
    valid_cos = [r["tgt_cos"] for r in rows if r.get("tgt_cos") is not None and not math.isnan(r["tgt_cos"])]
    best_wer = min(valid_wers) if valid_wers else None
    best_tgt_cos = max(valid_cos) if valid_cos else None

    annotated = []
    for r in rows:
        r = dict(r)
        reasons = []
        if not r.get("export_ok", False):
            reasons.append("export smoke test failed")
        hop_ms = r["hop_ms"]
        p95 = r.get("compute_p95_ms")
        if p95 is None or p95 > rules["max_compute_hop_ratio"] * hop_ms:
            reasons.append(f"compute p95 {p95} ms > {rules['max_compute_hop_ratio'] * hop_ms:.1f} ms (50% of hop)")
        wer = r.get("wer")
        if wer is None or math.isnan(wer):
            reasons.append("missing WER")
        else:
            if wer > rules["max_wer_abs"]:
                reasons.append(f"WER {100 * wer:.2f}% > {100 * rules['max_wer_abs']:.1f}%")
            if best_wer is not None and wer > best_wer + rules["max_wer_delta"] + 1e-9:
                reasons.append(f"WER {100 * wer:.2f}% > best ({100 * best_wer:.2f}%) + 3.0 pts")
        tcos = r.get("tgt_cos")
        if tcos is None or math.isnan(tcos):
            reasons.append("missing tgt_cos")
        elif best_tgt_cos is not None and tcos < best_tgt_cos - rules["max_tgt_cos_drop"] - 1e-9:
            reasons.append(f"tgt_cos {tcos:.3f} < best ({best_tgt_cos:.3f}) - 0.05")
        utmos = r.get("utmos")
        if utmos is None or math.isnan(utmos):
            reasons.append("missing UTMOS")
        elif utmos < rules["min_utmos"]:
            reasons.append(f"UTMOS {utmos:.2f} < {rules['min_utmos']:.1f}")

        r["eligible"] = len(reasons) == 0
        r["ineligible_reasons"] = reasons
        annotated.append(r)

    eligible = [r for r in annotated if r["eligible"]]
    if eligible:
        winner = min(
            eligible,
            key=lambda r: (
                round(r["projected_t_minus_b_ms"], 3),
                -(r.get("utmos") or 0.0),
                r.get("wer") or 1.0,
            ),
        )
        decision = {
            "winner": winner["candidate"],
            "hop_ms": winner["hop_ms"],
            "lookahead_frames": winner["lookahead_frames"],
            "projected_t_minus_b_ms": winner["projected_t_minus_b_ms"],
            "fallback": None,
            "reason": (f"Selected {winner['candidate']} with lowest projected T-B "
                       f"({winner['projected_t_minus_b_ms']:.1f} ms) among {len(eligible)} "
                       f"eligible candidate(s)."),
        }
    else:
        decision = {
            "winner": None,
            "hop_ms": None,
            "lookahead_frames": None,
            "projected_t_minus_b_ms": None,
            "fallback": [
                "1. soft units (open decision 2)",
                "2. larger generator (V1-class, --upsample-initial-channel 256/512) if compute allows",
                "3. Mimi q0 (<= 160 ms, recorded as a deviation) if it is the sole quality passer",
            ],
            "reason": "No candidate satisfied all eligibility thresholds; follow fallback order.",
        }
    return annotated, decision


def format_gate_markdown(rows, decision):
    """Render the Stage 05 Markdown gate table."""
    def fmt(v, digits=2, pct=False):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "—"
        return f"{100.0 * v:.{digits}f}%" if pct else f"{v:.{digits}f}"

    lines = [
        "| Candidate | Hop (ms) | L | Alg. latency (ms) | Compute p95 (ms) | Projected T−B (ms) | WER | CER | Tgt cos | Src cos | UTMOS | F0 PCC | Export OK | Eligible |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['candidate']} | {r['hop_ms']:.0f} | {r['lookahead_frames']} "
            f"| {r['alg_latency_ms']:.0f} | {fmt(r.get('compute_p95_ms'), 1)} "
            f"| {fmt(r.get('projected_t_minus_b_ms'), 1)} | {fmt(r.get('wer'), 2, pct=True)} "
            f"| {fmt(r.get('cer'), 2, pct=True)} | {fmt(r.get('tgt_cos'), 3)} "
            f"| {fmt(r.get('src_cos'), 3)} | {fmt(r.get('utmos'), 2)} "
            f"| {fmt(r.get('f0_pcc'), 3)} | {'✓' if r.get('export_ok') else '✗'} "
            f"| {'✓' if r.get('eligible') else '✗'} |"
        )
    lines.append("")
    lines.append(f"**Decision:** {decision['reason']}")
    return "\n".join(lines) + "\n"


# -- controls readiness checks (Stage 05 -> Stage 09) -------------------------
@torch.inference_mode()
def controls_readiness_check(wavs, cand, enc, voc, voices_dict,
                             voice_a=None, voice_b=None,
                             shifts_st=(-12.0, -6.0, 0.0, 6.0, 12.0),
                             scales=(0.0, 1.0, 2.0)):
    """Run the Stage 05 controls readiness checks across `wavs`:
      - pitch_shift in {-12, -6, 0, +6, +12} st: median output F0 and ratio vs 0 st
      - intonation scale in {0, 1, 2}: voiced log-F0 std
      - 50/50 speaker-embedding blend between voice_a and voice_b
    """
    voices_by_name = {v["name"]: v for v in voices_dict["voices"]}
    names = list(voices_by_name)
    va = voices_by_name[voice_a or names[0]]
    vb = voices_by_name[voice_b or (names[1] if len(names) > 1 else names[0])]

    shift_results = {}
    for st in shifts_st:
        f0s = []
        for w in wavs:
            y, _ = convert_utterance_streaming(w, cand, enc, voc, va, shift_st=float(st))
            f0s.append(f0_ref.track(y))
        med, lf_std = f0_ref.stats(np.concatenate(f0s))
        shift_results[str(st)] = {"shift_st": float(st), "median_f0_hz": med, "logf0_std": lf_std}

    base_f0 = shift_results.get("0.0", shift_results.get("0", {})).get("median_f0_hz")
    for entry in shift_results.values():
        if base_f0 and base_f0 > 0 and not math.isnan(base_f0) and not math.isnan(entry["median_f0_hz"]):
            entry["ratio_vs_0st"] = entry["median_f0_hz"] / base_f0
        else:
            entry["ratio_vs_0st"] = float("nan")

    plus12_ratio = shift_results.get("12.0", {}).get("ratio_vs_0st", float("nan"))
    plus12_dod_pass = bool(not math.isnan(plus12_ratio) and abs(plus12_ratio - 2.0) <= 0.06)

    scale_results = {}
    for sc in scales:
        f0s = []
        for w in wavs:
            y, _ = convert_utterance_streaming(w, cand, enc, voc, va, scale=float(sc))
            f0s.append(f0_ref.track(y))
        med, lf_std = f0_ref.stats(np.concatenate(f0s))
        scale_results[str(sc)] = {"scale": float(sc), "median_f0_hz": med, "logf0_std": lf_std}

    emb_a = np.asarray(va["embedding"], dtype=np.float32)
    emb_b = np.asarray(vb["embedding"], dtype=np.float32)
    emb_blend = 0.5 * emb_a + 0.5 * emb_b
    blend_voice = {
        **va,
        "embedding": emb_blend.tolist(),
        "logf0_mean": 0.5 * (va["logf0_mean"] + vb["logf0_mean"]),
        "logf0_std": 0.5 * (va["logf0_std"] + vb["logf0_std"]),
        "energy_mean": 0.5 * (va["energy_mean"] + vb["energy_mean"]),
        "energy_std": 0.5 * (va["energy_std"] + vb["energy_std"]),
    }
    blend_outs = [
        convert_utterance_streaming(w, cand, enc, voc, blend_voice, spk_emb_override=emb_blend)[0]
        for w in wavs
    ]
    blend_f0 = np.concatenate([f0_ref.track(y) for y in blend_outs])
    blend_med, blend_std = f0_ref.stats(blend_f0)
    blend_finite = all(bool(np.isfinite(y).all()) and float(np.max(np.abs(y))) > 0.0
                       for y in blend_outs)

    return {
        "voice_a": va["name"],
        "voice_b": vb["name"],
        "pitch_shift": shift_results,
        "plus12st_ratio": plus12_ratio,
        "plus12st_dod_pass": plus12_dod_pass,
        "intonation_scale": scale_results,
        "blend_50_50": {
            "voice_a": va["name"],
            "voice_b": vb["name"],
            "finite": blend_finite,
            "median_f0_hz": blend_med,
            "logf0_std": blend_std,
        },
    }


def run_gate(args, log=print):
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cands = ([parse_candidate_spec(s) for s in args.candidates.split(",") if s.strip()]
             if args.candidates else list(DEFAULT_CANDIDATES))
    if args.include_mimi and not any(c.tokenizer == "mimi-q0" for c in cands):
        cands.append(parse_candidate_spec("mimi-q0:s1"))

    enc, _ = (ce.ContentEncoder.from_checkpoint(args.encoder_checkpoint)
              if args.encoder_checkpoint else (ce.ContentEncoder(), {}))
    if args.vocoder_checkpoint:
        voc, voc_ck = vocoder.UnitVocoder.from_checkpoint(args.vocoder_checkpoint)
        voices_dict = voc_ck.get("voices") or voc.voices_json(
            voc_ck["speakers"], voc_ck["speaker_stats"]
        )
    else:
        voc = vocoder.UnitVocoder(vocoder.VocoderConfig(vocab=enc.cfg.vocab))
        dummy_stats = {
            s: {"logf0_mean": features.LOGF0_PRIOR_MEAN, "logf0_std": features.LOGF0_PRIOR_STD,
                "energy_mean": features.ENERGY_PRIOR_MEAN, "energy_std": features.ENERGY_PRIOR_STD}
            for s in vocoder.DEFAULT_SPEAKERS[:voc.cfg.n_speakers]
        }
        voices_dict = voc.voices_json(list(dummy_stats), dummy_stats)

    voices_by_name = {v["name"]: v for v in voices_dict["voices"]}
    target_voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    for tv in target_voices:
        if tv not in voices_by_name:
            raise ValueError(f"target voice {tv!r} not in vocoder voices {list(voices_by_name)}")

    metrics = tuple(m.strip() for m in args.metrics.split(",") if m.strip())
    rows = []
    for cand in cands:
        log(f"[gate] candidate {cand.name} (hop={cand.hop_ms:.0f} ms, L={cand.lookahead_frames})")
        smoke = export_smoke_test(cand, enc=enc, voc=voc)
        cost = measure_candidate_cost(cand, enc=enc, voc=voc,
                                      steps=args.cost_steps, warmup=args.cost_warmup)
        p95_ms = cost["total_p95_ms"]
        proj_tb = projected_t_minus_b_ms(cand, p95_ms, io_floor_ms=args.io_floor_ms)

        row = {
            "candidate": cand.name,
            "tokenizer": cand.tokenizer,
            "frames_per_step": cand.frames_per_step,
            "hop_ms": cand.hop_ms,
            "lookahead_frames": cand.lookahead_frames,
            "alg_latency_ms": cand.alg_latency_ms,
            "compute_p95_ms": p95_ms,
            "projected_t_minus_b_ms": proj_tb,
            "cost": cost,
            "export_smoke": smoke,
            "export_ok": bool(smoke.get("ok", False)),
            "wer": None,
            "cer": None,
            "tgt_cos": None,
            "src_cos": None,
            "utmos": None,
            "f0_pcc": None,
            "rtf": None,
            "per_voice": {},
        }

        if not args.skip_eval and cand.tokenizer != "mimi-q0":
            voice_summaries = []
            for tv in target_voices:
                conv_dir = out_dir / "converted" / cand.slug / tv
                eval_dir = out_dir / "eval" / cand.slug / tv
                convert_eval_list(args.list, cand, enc, voc, voices_by_name[tv],
                                  conv_dir, limit=args.limit)
                ev_cfg = evaluate.EvalConfig(
                    sources=args.list,
                    converted=conv_dir,
                    targets=args.targets,
                    voice=tv,
                    metrics=metrics,
                    device=args.device,
                    batch_size=args.batch_size,
                    limit=args.limit,
                )
                res = evaluate.evaluate(ev_cfg, eval_dir, log=log)
                row["per_voice"][tv] = {k: res.get(k) for k in evaluate.SUMMARY_KEYS}
                voice_summaries.append(res)

            for k in evaluate.SUMMARY_KEYS:
                vals = [s[k] for s in voice_summaries
                        if s.get(k) is not None and not math.isnan(s[k])]
                row[k] = float(np.mean(vals)) if vals else None

        rows.append(row)

    annotated, decision = apply_gate_rules(rows)
    report = {
        "candidates": annotated,
        "decision": decision,
        "encoder_checkpoint": args.encoder_checkpoint,
        "vocoder_checkpoint": args.vocoder_checkpoint,
        "voices": target_voices,
    }

    if args.controls_check:
        src_rows = evallist.read_sources(args.list)[:max(1, args.controls_limit)]
        wavs = [audio.load(paths.resolve(r["path"])) for r in src_rows]
        win_name = decision["winner"] or cands[0].name
        win_cand = next((c for c in cands if c.name == win_name), cands[0])
        report["controls_readiness"] = controls_readiness_check(
            wavs, win_cand, enc, voc, voices_dict
        )

    md = format_gate_markdown(annotated, decision)
    (out_dir / "gate_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out_dir / "gate_report.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--encoder-checkpoint", help="ContentEncoder checkpoint (default random init)")
    p.add_argument("--vocoder-checkpoint", help="UnitVocoder checkpoint (default random init)")
    p.add_argument("--candidates", help="comma-separated specs, e.g. 'student-L0:s1,student-L1:s1'")
    p.add_argument("--include-mimi", action="store_true", help="include mimi-q0 row in the gate")
    p.add_argument("--list", default=str(paths.EVAL_LISTS / "test_clean_200.tsv"))
    p.add_argument("--targets", default=str(paths.EVAL_LISTS / "targets.tsv"))
    p.add_argument("--voices", default="lj,hifitts_6097",
                   help="comma-separated target voices in --targets to convert into")
    p.add_argument("--metrics", default=",".join(evaluate.METRICS))
    p.add_argument("--limit", type=int, help="convert/score only first N utterances")
    p.add_argument("--skip-eval", action="store_true",
                   help="run only cost + projected T-B + ONNX export smoke test")
    p.add_argument("--controls-check", action="store_true",
                   help="run pitch shift / intonation / timbre blend readiness checks")
    p.add_argument("--controls-limit", type=int, default=5)
    p.add_argument("--cost-steps", type=int, default=200)
    p.add_argument("--cost-warmup", type=int, default=30)
    p.add_argument("--io-floor-ms", type=float, default=IO_FLOOR_MS)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", required=True, help="output directory for gate_report.{json,md}")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    try:
        return run_gate(args)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"[gate_report] error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
