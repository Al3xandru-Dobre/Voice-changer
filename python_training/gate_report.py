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
import copy
import csv
import dataclasses
import json
import math
import os
import pathlib
import sys
import tempfile
import time

import numpy as np
import torch

from fvml import (audio, contract, evaluate, evallist, export as fx,
                  f0_ref, features, parity, paths, vocoder)
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


def _eval_cpu_model(model):
    """Return an eval-mode CPU float32 view of `model` without mutating a caller's
    training/CUDA module in place."""
    is_cpu_f32 = all(p.device.type == "cpu" and p.dtype == torch.float32
                     for p in model.parameters())
    if is_cpu_f32 and not model.training:
        return model
    return copy.deepcopy(model).eval().float().cpu()


def _host_load_info():
    load_1m = None
    if hasattr(os, "getloadavg"):
        try:
            load_1m = round(float(os.getloadavg()[0]), 2)
        except OSError:
            load_1m = None
    return {"cpu_count": os.cpu_count(), "load_avg_1m": load_1m}


# -- ONNX export smoke test & stateful ORT timing -----------------------------
@torch.inference_mode()
def export_smoke_test(cand, enc=None, voc=None, *, exporter="dynamo",
                      cost_steps=200, cost_warmup=30, smoke_hops=4):
    """Export a per-step ONNX graph of the candidate encoder + vocoder via
    `fvml.export`, run `smoke_hops` carried-state steps in ONNX Runtime CPU vs
    PyTorch `step()`, and measure stateful single-core ORT step latency."""
    if cand.tokenizer == "mimi-q0":
        return {"ok": False, "reason": "mimi-q0 stateful ONNX ring-KV export not enabled before kill-test pass"}
    try:
        probe_units.single_core()
        fps = cand.frames_per_step
        head = ce.LOOKAHEAD.index(cand.lookahead_frames)
        enc_m = _eval_cpu_model(enc) if enc is not None else ce.ContentEncoder().eval().float().cpu()
        voc_m = (_eval_cpu_model(voc) if voc is not None else vocoder.UnitVocoder(
            vocoder.VocoderConfig(vocab=enc_m.cfg.vocab, frame_samples=cand.frame_samples)
        ).eval().float().cpu())

        enc_contract = fx.encoder_contract(enc_m, head, fps, license="MIT")
        voc_contract = voc_m.contract(frames_per_step=fps)
        contract.validate_pair(enc_contract, voc_contract)

        enc_graph = fx.encoder_graph(enc_m, head, fps)
        voc_graph = fx.vocoder_graph(voc_m, voc_contract)

        hop = enc_contract.hop
        cond_frames = voc_contract.cond_frames
        spk_emb = voc_m.speaker_table.weight[:1].detach().clone()
        spk_np = spk_emb.numpy()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = pathlib.Path(tmp)
            enc_path = tmp_dir / f"encoder_s{fps}.onnx"
            voc_path = tmp_dir / f"vocoder_s{fps}.onnx"

            enc_rec = fx.export_graph(enc_graph, enc_path, exporter=exporter)
            fx.finalize(enc_path, enc_contract, drop_outputs=enc_graph.debug_outputs,
                        metadata={"fluxvoice.exporter": enc_rec["exporter"]})

            voc_rec = fx.export_graph(voc_graph, voc_path, exporter=exporter)
            fx.finalize(voc_path, voc_contract,
                        metadata={"fluxvoice.exporter": voc_rec["exporter"]})

            enc_stream = fx.OrtStream(enc_path)
            voc_stream = fx.OrtStream(voc_path)

            # Multi-hop carried-state equivalence check (state_out fed back as state_in).
            rng = np.random.default_rng(0)
            n_check = max(3, int(smoke_hops))
            enc_state_pt = enc_m.init_state(1)
            voc_state_pt = voc_m.init_state(1)
            enc_stream.reset()
            voc_stream.reset()

            ort_units_list, pt_units_list = [], []
            ort_audio_list, pt_audio_list = [], []
            for h_idx in range(n_check):
                audio_np = (0.2 * np.sin(2.0 * np.pi * (180.0 + 15.0 * h_idx)
                            * np.arange(hop, dtype=np.float32) / cand.sample_rate)
                            + 0.05 * rng.standard_normal(hop).astype(np.float32))[None]
                lf_np = (5.0 + 0.15 * np.sin(np.arange(cond_frames, dtype=np.float32) + h_idx))[None]
                vo_np = np.full((1, cond_frames), 0.9 if h_idx % 2 == 0 else 0.4, dtype=np.float32)
                ap_np = np.full((1, cond_frames), 0.1 + 0.05 * (h_idx % 3), dtype=np.float32)
                en_np = np.full((1, cond_frames), -5.0 + 0.5 * (h_idx % 2), dtype=np.float32)

                e_ort = enc_stream.step({"audio": audio_np})["units"]
                e_pt_out, enc_state_pt = enc_m.step(torch.from_numpy(audio_np), enc_state_pt)
                e_pt = e_pt_out["logits"][head].argmax(dim=-1).to(torch.int64).numpy()
                ort_units_list.append(e_ort)
                pt_units_list.append(e_pt)

                v_ort = voc_stream.step({
                    "units": e_ort,
                    "logf0": lf_np,
                    "voicing": vo_np,
                    "aperiodicity": ap_np,
                    "energy": en_np,
                    "spk_emb": spk_np,
                })["audio"]
                y_pt, voc_state_pt = voc_m.step(
                    torch.from_numpy(e_ort),
                    torch.from_numpy(lf_np),
                    torch.from_numpy(vo_np),
                    torch.from_numpy(ap_np),
                    torch.from_numpy(en_np),
                    spk_emb,
                    voc_state_pt,
                )
                ort_audio_list.append(v_ort[0])
                pt_audio_list.append(y_pt[0].numpy())

            enc_units_match = bool(np.array_equal(np.concatenate(ort_units_list, axis=-1),
                                                  np.concatenate(pt_units_list, axis=-1)))
            pt_audio_all = np.concatenate(pt_audio_list)
            ort_audio_all = np.concatenate(ort_audio_list)
            r_voc = parity.compare(pt_audio_all, ort_audio_all, atol=1e-4, steps=False)
            voc_max_abs = float(r_voc.max_abs)
            voc_snr_db = float(r_voc.snr_db)
            voc_ref_peak = float(np.max(np.abs(pt_audio_all)))
            voc_ref_rms = float(np.sqrt(np.mean(np.square(pt_audio_all, dtype=np.float64))))

            # Carried-state single-core ORT step timing over cost_warmup + cost_steps hops.
            n_timed = max(1, int(cost_steps))
            n_warm = max(1, int(cost_warmup))
            enc_stream.reset()
            voc_stream.reset()
            ort_enc_times, ort_voc_times = [], []
            audio_chunks = (0.1 * rng.standard_normal((n_warm + n_timed, 1, hop))).astype(np.float32)
            lf_chunks = (5.0 + 0.1 * rng.standard_normal((n_warm + n_timed, 1, cond_frames))).astype(np.float32)
            vo_chunks = np.ones((n_warm + n_timed, 1, cond_frames), dtype=np.float32)
            ap_chunks = np.full((n_warm + n_timed, 1, cond_frames), 0.1, dtype=np.float32)
            en_chunks = np.full((n_warm + n_timed, 1, cond_frames), -5.0, dtype=np.float32)
            for i in range(n_warm + n_timed):
                t0 = time.perf_counter()
                e_out = enc_stream.step({"audio": audio_chunks[i]})
                t1 = time.perf_counter()
                voc_stream.step({
                    "units": e_out["units"],
                    "logf0": lf_chunks[i],
                    "voicing": vo_chunks[i],
                    "aperiodicity": ap_chunks[i],
                    "energy": en_chunks[i],
                    "spk_emb": spk_np,
                })
                t2 = time.perf_counter()
                if i >= n_warm:
                    ort_enc_times.append(t1 - t0)
                    ort_voc_times.append(t2 - t1)
            ort_enc_p = probe_units.percentiles(ort_enc_times)
            ort_voc_p = probe_units.percentiles(ort_voc_times)
            ort_tot_p = probe_units.percentiles(np.asarray(ort_enc_times) + np.asarray(ort_voc_times))

        voc_rel_ok = (voc_max_abs <= 1e-4 * max(1.0, voc_ref_rms)) and (voc_max_abs <= 1e-4)
        voc_snr_ok = voc_snr_db >= fx.MIN_VOCODER_SNR_DB
        exporter_ok = (exporter != "dynamo") or (
            enc_rec["exporter"] == "dynamo" and voc_rec["exporter"] == "dynamo"
        )
        ok = bool(enc_units_match and voc_rel_ok and voc_snr_ok and voc_ref_peak > 0.0 and exporter_ok)
        fb_reason = enc_rec["fallback_reason"] or voc_rec["fallback_reason"]
        host_info = _host_load_info()
        jitter_warn = bool(ort_tot_p["p99_ms"] > 3.0 * max(ort_tot_p["p50_ms"], 1e-6))
        return {
            "ok": ok,
            "smoke_hops": n_check,
            "encoder_exporter": enc_rec["exporter"],
            "vocoder_exporter": voc_rec["exporter"],
            "fallback_reason": fb_reason,
            "encoder_units_match": enc_units_match,
            "vocoder_max_abs": voc_max_abs,
            "vocoder_snr_db": None if not np.isfinite(voc_snr_db) else voc_snr_db,
            "vocoder_ref_peak": voc_ref_peak,
            "vocoder_ref_rms": voc_ref_rms,
            "ort_encoder_p50_ms": ort_enc_p["p50_ms"],
            "ort_encoder_p95_ms": ort_enc_p["p95_ms"],
            "ort_vocoder_p50_ms": ort_voc_p["p50_ms"],
            "ort_vocoder_p95_ms": ort_voc_p["p95_ms"],
            "ort_total_p50_ms": ort_tot_p["p50_ms"],
            "ort_total_p95_ms": ort_tot_p["p95_ms"],
            "ort_total_p99_ms": ort_tot_p["p99_ms"],
            "ort_steps": n_timed,
            "jitter_warning": jitter_warn,
            **host_info,
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

    voc_m = (_eval_cpu_model(voc) if voc is not None else vocoder.UnitVocoder(
        vocoder.VocoderConfig(vocab=2048 if cand.tokenizer == "mimi-q0" else 100,
                              frame_samples=cand.frame_samples)
    ).eval().float().cpu())
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
    enc_m = (_eval_cpu_model(enc) if enc is not None else ce.ContentEncoder().eval().float().cpu())
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
    """Convert one 16 kHz utterance through the full streaming chain at
    `cand.frames_per_step`, flushing `cand.lookahead_frames` tail frames and
    dropping the leading `L` delay frames so output aligns with the input.
    Returns (converted_wav_float32, compute_seconds)."""
    enc_m = _eval_cpu_model(enc)
    voc_m = _eval_cpu_model(voc)
    fps = cand.frames_per_step
    hop = cand.hop_samples
    cond_frames = hop // vocoder.COND_FRAME_SAMPLES
    head = ce.LOOKAHEAD.index(cand.lookahead_frames)

    wav_np = np.asarray(wav, dtype=np.float32)
    orig_hops = max(1, -(-len(wav_np) // hop))
    orig_len = orig_hops * hop
    drop_samples = cand.lookahead_frames * cand.frame_samples
    flush_hops = -(-drop_samples // hop) if drop_samples > 0 else 0
    n_hops = orig_hops + flush_hops
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
    full_out = np.concatenate(out_chunks)
    return full_out[drop_samples:drop_samples + orig_len], elapsed


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
    on highest utmos, then lowest wer). Computes eligibility on the shipping ORT
    clock (`compute_ort_p95_ms`) whenever present, falling back to `compute_p95_ms`.
    Returns (annotated_rows, decision_dict)."""
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
        ort_p95 = r.get("compute_ort_p95_ms")
        if ort_p95 is None and isinstance(r.get("export_smoke"), dict):
            ort_p95 = r["export_smoke"].get("ort_total_p95_ms")
        if ort_p95 is not None:
            p95 = ort_p95
            r["compute_ort_p95_ms"] = ort_p95
            r["compute_p95_ms"] = ort_p95
            r["compute_clock"] = "ort"
            if r.get("projected_t_minus_b_ort_ms") is not None:
                r["projected_t_minus_b_ms"] = r["projected_t_minus_b_ort_ms"]
        else:
            p95 = r.get("compute_p95_ms")
            r.setdefault("compute_clock", "ort" if "compute_pytorch_p95_ms" not in r else "pytorch")
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
        clock = winner.get("compute_clock", "ort").upper()
        decision = {
            "winner": winner["candidate"],
            "frames_per_step": winner.get("frames_per_step"),
            "hop_ms": winner["hop_ms"],
            "lookahead_frames": winner["lookahead_frames"],
            "compute_clock": winner.get("compute_clock", "ort"),
            "projected_t_minus_b_ms": winner["projected_t_minus_b_ms"],
            "fallback": None,
            "reason": (f"Selected {winner['candidate']} with lowest projected T-B "
                       f"({winner['projected_t_minus_b_ms']:.1f} ms, {clock} clock) among "
                       f"{len(eligible)} eligible candidate(s)."),
        }
    else:
        decision = {
            "winner": None,
            "frames_per_step": None,
            "hop_ms": None,
            "lookahead_frames": None,
            "compute_clock": "ort",
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
    """Render the Stage 05 Markdown gate table (showing both PyTorch and ORT
    clocks when available, and marking the exporter / fallback state)."""
    def fmt(v, digits=2, pct=False):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "—"
        return f"{100.0 * v:.{digits}f}%" if pct else f"{v:.{digits}f}"

    def fmt_pair(r, pt_key, ort_key, single_key):
        pt = r.get(pt_key)
        ort_v = r.get(ort_key)
        if pt is not None or ort_v is not None:
            ort_str = f"**{fmt(ort_v, 1)}**" if ort_v is not None else "—"
            return f"{fmt(pt, 1)} / {ort_str}"
        return fmt(r.get(single_key), 1)

    def fmt_export(r):
        smoke = r.get("export_smoke") if isinstance(r.get("export_smoke"), dict) else {}
        enc_e = r.get("encoder_exporter") or smoke.get("encoder_exporter")
        voc_e = r.get("vocoder_exporter") or smoke.get("vocoder_exporter")
        fb = r.get("fallback_reason") or smoke.get("fallback_reason")
        if not r.get("export_ok"):
            return f"✗ ({fb})" if fb else "✗"
        if fb or (enc_e and enc_e != "dynamo") or (voc_e and voc_e != "dynamo"):
            exp_tag = voc_e or enc_e or "fallback"
            return f"✓ ({exp_tag}: {fb})" if fb else f"✓ ({exp_tag})"
        return "✓"

    lines = [
        "| Candidate | Hop (ms) | L | Alg. latency (ms) | Compute p95 PyTorch / ORT (ms) | Projected T−B PyTorch / ORT (ms) | WER | CER | Tgt cos | Src cos | UTMOS | F0 PCC | Export OK | Eligible |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        comp_cell = fmt_pair(r, "compute_pytorch_p95_ms", "compute_ort_p95_ms", "compute_p95_ms")
        proj_cell = fmt_pair(r, "projected_t_minus_b_pytorch_ms",
                             "projected_t_minus_b_ort_ms", "projected_t_minus_b_ms")
        lines.append(
            f"| {r['candidate']} | {r['hop_ms']:.0f} | {r['lookahead_frames']} "
            f"| {r['alg_latency_ms']:.0f} | {comp_cell} "
            f"| {proj_cell} | {fmt(r.get('wer'), 2, pct=True)} "
            f"| {fmt(r.get('cer'), 2, pct=True)} | {fmt(r.get('tgt_cos'), 3)} "
            f"| {fmt(r.get('src_cos'), 3)} | {fmt(r.get('utmos'), 2)} "
            f"| {fmt(r.get('f0_pcc'), 3)} | {fmt_export(r)} "
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
    exporter = getattr(args, "exporter", "dynamo")
    rows = []
    for cand in cands:
        log(f"[gate] candidate {cand.name} (hop={cand.hop_ms:.0f} ms, L={cand.lookahead_frames})")
        smoke = export_smoke_test(cand, enc=enc, voc=voc, exporter=exporter,
                                  cost_steps=args.cost_steps, cost_warmup=args.cost_warmup)
        cost = measure_candidate_cost(cand, enc=enc, voc=voc,
                                      steps=args.cost_steps, warmup=args.cost_warmup)
        pt_p95_ms = cost["total_p95_ms"]
        ort_p95_ms = smoke.get("ort_total_p95_ms")
        p95_ms = ort_p95_ms if ort_p95_ms is not None else pt_p95_ms
        proj_tb = projected_t_minus_b_ms(cand, p95_ms, io_floor_ms=args.io_floor_ms)
        proj_tb_pt = (projected_t_minus_b_ms(cand, pt_p95_ms, io_floor_ms=args.io_floor_ms)
                      if pt_p95_ms is not None else None)
        proj_tb_ort = (projected_t_minus_b_ms(cand, ort_p95_ms, io_floor_ms=args.io_floor_ms)
                       if ort_p95_ms is not None else None)
        if smoke.get("jitter_warning"):
            log(f"[gate] warning: {cand.name} ORT p99 ({smoke.get('ort_total_p99_ms'):.2f} ms) "
                f"> 3x p50 ({smoke.get('ort_total_p50_ms'):.2f} ms) — host CPU may be loaded")

        row = {
            "candidate": cand.name,
            "tokenizer": cand.tokenizer,
            "frames_per_step": cand.frames_per_step,
            "hop_ms": cand.hop_ms,
            "lookahead_frames": cand.lookahead_frames,
            "alg_latency_ms": cand.alg_latency_ms,
            "compute_clock": "ort" if ort_p95_ms is not None else "pytorch",
            "compute_p95_ms": p95_ms,
            "compute_ort_p95_ms": ort_p95_ms,
            "compute_pytorch_p95_ms": pt_p95_ms,
            "projected_t_minus_b_ms": proj_tb,
            "projected_t_minus_b_ort_ms": proj_tb_ort,
            "projected_t_minus_b_pytorch_ms": proj_tb_pt,
            "encoder_exporter": smoke.get("encoder_exporter"),
            "vocoder_exporter": smoke.get("vocoder_exporter"),
            "fallback_reason": smoke.get("fallback_reason"),
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
        "frames_per_step": decision.get("frames_per_step"),
        "encoder_checkpoint": args.encoder_checkpoint,
        "vocoder_checkpoint": args.vocoder_checkpoint,
        "exporter": exporter,
        "host": _host_load_info(),
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
    p.add_argument("--exporter", choices=fx.EXPORTERS, default="dynamo",
                   help="ONNX exporter for the smoke test (default: dynamo; 'auto' allows fallback)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", required=True, help="output directory for gate_report.{json,md}")
    return p.parse_args(argv)


def main(argv=None):
    for stream_io in (sys.stdout, sys.stderr):
        if hasattr(stream_io, "reconfigure"):
            stream_io.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    args = parse_args(argv)
    try:
        return run_gate(args)
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as e:
        print(f"[gate_report] error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
