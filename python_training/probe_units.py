#!/usr/bin/env python3
"""Probe unit streams: phone purity / cluster purity / PNMI, speaker-ID leakage, cost.

Milestone 2 / Stage 04. The math lives in fvml/unit_probes.py.

  --units DIR          a units directory from extract_units.py (repeatable; with
                       several, each PNMI is also given as a ratio of the first's,
                       e.g. the teacher's for the Mimi kill test)
  --alignments DIR     Montreal Forced Aligner TextGrids (<utt_id>.TextGrid,
                       searched recursively; tier --tier, default "phones").
                       Frame labels are majority phones at each stream's own
                       frame rate (units_meta.json), frames centred half a frame in
                       unless --centre-offset-ms says otherwise
  --speaker-probe      logistic regression on per-utterance unigram+bigram unit
                       histograms; --speakers N (default 100) chosen among the
                       manifest's speakers, --test-fraction of each speaker's
                       utterances held out; accuracy vs. the 1/N chance level
  --cost               single-core CPU (torch.set_num_threads(1), denormals
                       flushed) per-step latency p50/p95/p99 of ContentEncoder.step
                       for --fps 1,2,4 (the three heads share one trunk, so one
                       number covers every student head); --checkpoint selects the
                       weights (random init otherwise: cost does not depend on
                       them). --cost-teacher / --cost-mimi add HuBERT's offline
                       forward (non-causal: ms per second of audio) and Mimi's
                       streaming encode per 80 ms frame step.

Output: the report as JSON on stdout and in --out (if given).
Exit status: 0 = done, 1 = missing inputs.
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

from fvml import unit_probes, units


def phone_probe(units_dir, align_index, tier, centre_offset_ms, log):
    meta, rows = units.read_units_dir(units_dir)
    fr = meta["frame_rate_hz"]
    offset = None if centre_offset_ms is None else centre_offset_ms / 1000.0
    ph, un, used = [], [], 0
    for r in rows:
        tg = align_index.get(r["utt_id"])
        if tg is None:
            continue
        ids = units.load_ids(units_dir, r)
        intervals = unit_probes.phone_tier(unit_probes.read_textgrid(tg), tier)
        ph += unit_probes.frame_labels(intervals, len(ids), fr, offset)
        un.append(ids.astype(np.int64))
        used += 1
    if not used:
        raise FileNotFoundError(f"no TextGrid matches any utterance in {units_dir}")
    log(f"[probe] {meta['tokenizer']}: {used}/{len(rows)} utterances aligned")
    out = unit_probes.probe_scores(np.array(ph), np.concatenate(un))
    out["utterances"] = used
    return out


def speaker_probe(units_dir, args):
    meta, rows = units.read_units_dir(units_dir)
    items = [(r["speaker"], units.load_ids(units_dir, r)) for r in rows if r.get("speaker")]
    if not items:
        raise FileNotFoundError(f"{units_dir}: manifest rows carry no speaker field")
    return unit_probes.speaker_probe(items, meta["vocab"], n_speakers=args.speakers,
                                     test_fraction=args.test_fraction, min_utts=args.min_utts,
                                     seed=args.seed)


def percentiles(seconds):
    ms = np.asarray(seconds) * 1e3
    return {"p50_ms": float(np.percentile(ms, 50)), "p95_ms": float(np.percentile(ms, 95)),
            "p99_ms": float(np.percentile(ms, 99)), "mean_ms": float(ms.mean()), "n": int(len(ms))}


def single_core():
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass                       # already set by an earlier call in this process
    torch.set_flush_denormal(True)  # the C++ runtime runs with denormals-as-zero
    return torch


def student_cost(checkpoint, fps_list, steps, warmup):
    torch = single_core()
    from fvml import content_encoder as ce
    model = ce.ContentEncoder.from_checkpoint(checkpoint)[0] if checkpoint else ce.ContentEncoder()
    model.eval()
    rng = np.random.default_rng(0)
    out = {"params": model.num_params(), "macs_per_second": model.macs_per_second(),
           "state_floats": model.state_numel()}
    for fps in fps_list:
        hop = fps * ce.FRAME_SAMPLES
        chunks = torch.from_numpy((0.1 * rng.standard_normal((warmup + steps, 1, hop))).astype(np.float32))
        state, times = model.init_state(1), []
        with torch.inference_mode():
            for i in range(warmup + steps):
                t0 = time.perf_counter()
                _, state = model.step(chunks[i], state)
                times.append(time.perf_counter() - t0)
        p = percentiles(times[warmup:])
        p["hop_ms"] = 1000.0 * hop / ce.SAMPLE_RATE
        p["rtf_p95"] = p["p95_ms"] / p["hop_ms"]
        out[f"fps{fps}"] = p
    return out


def vocoder_cost(checkpoint, fps_list, steps, warmup, cfg=None):
    """Single-core CPU latency p50/p95/p99 of UnitVocoder.step (Stage 05)."""
    torch = single_core()
    from fvml import vocoder as voc
    if checkpoint:
        model = voc.UnitVocoder.from_checkpoint(checkpoint)[0]
    else:
        model = voc.UnitVocoder(cfg or voc.VocoderConfig())
    model.eval()
    rng = np.random.default_rng(0)
    out = {"params": model.num_params(), "macs_per_second": model.macs_per_second(),
           "state_floats": model.state_numel(), "frame_samples": model.cfg.frame_samples}
    rep = model.cfg.unit_repeat
    spk = model.speaker_table.weight[:1].detach().clone()
    for fps in fps_list:
        hop = fps * model.cfg.frame_samples
        cond_step = fps * rep
        if model.cfg.content_kind == "discrete":
            u = torch.from_numpy(rng.integers(0, model.cfg.vocab,
                                              size=(warmup + steps, 1, fps), dtype=np.int64))
        else:
            u = torch.from_numpy(rng.standard_normal(
                (warmup + steps, 1, fps, model.cfg.content_dim)).astype(np.float32))
        lf = torch.from_numpy(rng.standard_normal((warmup + steps, 1, cond_step)).astype(np.float32))
        vo = torch.from_numpy(rng.uniform(0.0, 1.0, (warmup + steps, 1, cond_step)).astype(np.float32))
        ap = torch.from_numpy(rng.uniform(0.0, 1.0, (warmup + steps, 1, cond_step)).astype(np.float32))
        en = torch.from_numpy(rng.standard_normal((warmup + steps, 1, cond_step)).astype(np.float32))
        state, times = model.init_state(1), []
        with torch.inference_mode():
            for i in range(warmup + steps):
                t0 = time.perf_counter()
                _, state = model.step(u[i], lf[i], vo[i], ap[i], en[i], spk, state)
                times.append(time.perf_counter() - t0)
        p = percentiles(times[warmup:])
        p["hop_ms"] = 1000.0 * hop / model.cfg.sample_rate
        p["rtf_p95"] = p["p95_ms"] / p["hop_ms"]
        out[f"fps{fps}"] = p
    return out


def teacher_cost(seconds=10.0, repeats=3):
    torch = single_core()
    from fvml import teacher
    t = teacher.HubertTeacher("cpu")
    x = (0.1 * np.random.default_rng(0).standard_normal(int(seconds * 16000) // 320 * 320)).astype(np.float32)
    t.features(x)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        t.features(x)
        times.append(time.perf_counter() - t0)
    return {"kind": "offline (non-causal; no streaming step exists)", "seconds": seconds,
            "ms_per_second_of_audio": 1e3 * float(np.median(times)) / seconds,
            "threads": torch.get_num_threads()}


def mimi_cost(fps_list, steps, warmup):
    """Mimi's own streaming encode (padding cache + transformer KV cache), one
    call per fps x 80 ms of 24 kHz audio."""
    torch = single_core()
    tok = units.MimiTokenizer("cpu")
    m, hop0 = tok.model, tok.frame_24k
    rng = np.random.default_rng(0)
    out = {"frame_ms": 1000.0 * hop0 / tok.sr}
    for fps in fps_list:
        hop = fps * hop0
        x = torch.from_numpy((0.1 * rng.standard_normal((warmup + steps, 1, 1, hop))).astype(np.float32))
        pkv, pad_cache, times = None, None, []
        with torch.inference_mode():
            for i in range(warmup + steps):
                t0 = time.perf_counter()
                enc = m.encode(x[i], num_quantizers=1, encoder_past_key_values=pkv,
                               padding_cache=pad_cache, use_streaming=True, return_dict=True)
                times.append(time.perf_counter() - t0)
                pkv, pad_cache = enc.encoder_past_key_values, enc.padding_cache
        p = percentiles(times[warmup:])
        p["hop_ms"] = 1000.0 * hop / tok.sr
        p["rtf_p95"] = p["p95_ms"] / p["hop_ms"]
        out[f"fps{fps}"] = p
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.splitlines()[2:]))
    p.add_argument("--units", action="append", default=[], help="units directory (repeatable)")
    p.add_argument("--alignments", help="directory of MFA TextGrids")
    p.add_argument("--tier", default="phones")
    p.add_argument("--centre-offset-ms", type=float,
                   help="centre of frame 0 in ms (default half a frame)")
    p.add_argument("--speaker-probe", action="store_true")
    p.add_argument("--speakers", type=int, default=100)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--min-utts", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cost", action="store_true")
    p.add_argument("--checkpoint", help="--cost: student weights (default random init)")
    p.add_argument("--fps", default="1,2,4")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--cost-teacher", action="store_true")
    p.add_argument("--cost-mimi", action="store_true")
    p.add_argument("--cost-vocoder", action="store_true")
    p.add_argument("--vocoder-checkpoint", help="--cost-vocoder: vocoder weights (default random init)")
    p.add_argument("--out", help="write the JSON report here")
    args = p.parse_args(argv)
    if not (args.units or args.cost):
        p.error("nothing to do: give --units (with --alignments and/or --speaker-probe) or --cost")
    if args.units and not (args.alignments or args.speaker_probe):
        p.error("--units needs --alignments and/or --speaker-probe")
    log = lambda s: print(s, file=sys.stderr)     # noqa: E731 - stdout carries the JSON
    report = {"units": {}}
    try:
        align_index = unit_probes.index_alignments(args.alignments) if args.alignments else {}
        if args.alignments and not align_index:
            raise FileNotFoundError(f"no .TextGrid files under {args.alignments}")
        for d in args.units:
            meta = units.read_meta(d)
            entry = {k: meta.get(k) for k in ("tokenizer", "frame_rate_hz", "lookahead_frames", "vocab")}
            if args.alignments:
                entry["phones"] = phone_probe(d, align_index, args.tier, args.centre_offset_ms, log)
            if args.speaker_probe:
                entry["speaker_id"] = speaker_probe(d, args)
            report["units"][str(d)] = entry
        first = next(iter(report["units"].values()), {}).get("phones")
        if first and len(report["units"]) > 1:
            for e in report["units"].values():
                e["phones"]["pnmi_ratio_vs_first"] = e["phones"]["pnmi"] / first["pnmi"]
                e["phones"]["phone_purity_ratio_vs_first"] = e["phones"]["phone_purity"] / first["phone_purity"]
        fps = [int(f) for f in args.fps.split(",") if f.strip()]
        if args.cost:
            report["cost"] = {"student": student_cost(args.checkpoint, fps, args.steps, args.warmup)}
            if args.cost_vocoder:
                report["cost"]["vocoder"] = vocoder_cost(args.vocoder_checkpoint, fps,
                                                         args.steps, args.warmup)
            if args.cost_teacher:
                report["cost"]["teacher"] = teacher_cost()
            if args.cost_mimi:
                report["cost"]["mimi"] = mimi_cost(fps, max(20, args.steps // 4), 10)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"[probe_units] error: {e}", file=sys.stderr)
        return 1
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
