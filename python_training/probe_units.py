#!/usr/bin/env python3
"""Probe unit streams: phone purity / cluster purity / PNMI, speaker-ID leakage, cost.

Milestone 2 / Stage 04; multilingual teacher probe in Milestone 3 / Stage 03.
The math lives in fvml/unit_probes.py.

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

Multilingual teacher probe (Milestone 3 / Stage 03): give one --units directory
per codebook to compare (the balanced mhubert-vp km1000 first, then the stock
km1000, then mHuBERT-147 with --include-noncommercial), labelled on held-out
{en, es, fr, de} dev clips. Rows that carry a language are also scored per
language (`by_lang`).
  --lang-reference LANG=DIR   the units of a codebook fitted on LANG alone; each
                       --units directory then reports its phone purity on LANG as
                       a ratio of that reference's (definition of done: >= 0.90
                       in every language)
  --max-speaker-accuracy      speaker-ID leakage ceiling (default 0.05): the
                       speaker probe reports pass/fail against it
  --markdown FILE      write the comparison table for the stage document
  --include-noncommercial     accept units from the CC-BY-NC-SA mHuBERT-147 teacher
  --multilingual       --cost / --cost-vocoder: time the M3 configuration (student
                       D=128 / K=1000, vocoder vocab 1000) when no checkpoint is given

Output: the report as JSON on stdout and in --out (if given).
Exit status: 0 = done, 1 = missing inputs or a licence refusal.
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

from fvml import unit_probes, units


def find_alignment(align_index, row):
    """The TextGrid of a unit row: named after the utterance id, or after the
    audio file (paired S2UT ids differ from the clip names MFA saw)."""
    tg = align_index.get(row["utt_id"])
    if tg is None and row.get("path"):
        tg = align_index.get(pathlib.PurePosixPath(str(row["path"]).replace("\\", "/")).stem)
    return tg


def phone_probe(units_dir, align_index, tier, centre_offset_ms, log, only_lang=None):
    """Purity / PNMI of one units directory against the aligned phones, overall
    and (when rows carry `lang`) per language. `only_lang` restricts it to one
    language's rows."""
    meta, rows = units.read_units_dir(units_dir)
    fr = meta["frame_rate_hz"]
    offset = None if centre_offset_ms is None else centre_offset_ms / 1000.0
    ph, un, used, per = [], [], 0, {}
    for r in rows:
        lang = r.get("lang")
        if only_lang and lang != only_lang:
            continue
        tg = find_alignment(align_index, r)
        if tg is None:
            continue
        ids = units.load_ids(units_dir, r)
        intervals = unit_probes.phone_tier(unit_probes.read_textgrid(tg), tier)
        labels = unit_probes.frame_labels(intervals, len(ids), fr, offset)
        ph += labels
        un.append(ids.astype(np.int64))
        used += 1
        if lang:
            bucket = per.setdefault(lang, {"ph": [], "un": [], "utts": 0})
            bucket["ph"] += labels
            bucket["un"].append(ids.astype(np.int64))
            bucket["utts"] += 1
    if not used:
        raise FileNotFoundError(f"no TextGrid matches any utterance in {units_dir}"
                                + (f" for language {only_lang}" if only_lang else ""))
    log(f"[probe] {meta['tokenizer']}: {used}/{len(rows)} utterances aligned")
    out = unit_probes.probe_scores(np.array(ph), np.concatenate(un))
    out["utterances"] = used
    if per:
        out["by_lang"] = {}
        for lang, b in sorted(per.items()):
            out["by_lang"][lang] = unit_probes.probe_scores(np.array(b["ph"]), np.concatenate(b["un"]))
            out["by_lang"][lang]["utterances"] = b["utts"]
    return out


def speaker_probe(units_dir, args):
    meta, rows = units.read_units_dir(units_dir)
    items = [(r["speaker"], units.load_ids(units_dir, r)) for r in rows if r.get("speaker")]
    if not items:
        raise FileNotFoundError(f"{units_dir}: manifest rows carry no speaker field")
    out = unit_probes.speaker_probe(items, meta["vocab"], n_speakers=args.speakers,
                                    test_fraction=args.test_fraction, min_utts=args.min_utts,
                                    seed=args.seed)
    out["max_accuracy"] = args.max_speaker_accuracy
    out["pass"] = bool(out["accuracy"] <= args.max_speaker_accuracy)
    return out


MIN_PURITY_RATIO = 0.90        # M3 Stage 03: vs. the language-specific codebook


def lang_reference_scores(specs, align_index, args, log):
    """--lang-reference LANG=DIR values -> {lang: phone probe of DIR on LANG}."""
    refs = {}
    for spec in specs:
        lang, sep, d = spec.partition("=")
        if not sep or not lang.strip() or not d.strip():
            raise ValueError(f"--lang-reference wants LANG=DIR, got {spec!r}")
        units.check_license(units.read_meta(d.strip()), args.include_noncommercial)
        refs[lang.strip()] = phone_probe(d.strip(), align_index, args.tier,
                                         args.centre_offset_ms, log, only_lang=lang.strip())
    return refs


def add_purity_ratios(entry, refs):
    """Per language, this codebook's phone purity over the language-specific
    codebook's; `dod` is true when every referenced language reaches 0.90."""
    by_lang = entry["phones"].get("by_lang") or {}
    ratios = {}
    for lang, ref in refs.items():
        if lang in by_lang and ref["phone_purity"] > 0:
            ratios[lang] = by_lang[lang]["phone_purity"] / ref["phone_purity"]
            by_lang[lang]["phone_purity_ratio_vs_lang_specific"] = ratios[lang]
            by_lang[lang]["pnmi_ratio_vs_lang_specific"] = \
                by_lang[lang]["pnmi"] / ref["pnmi"] if ref["pnmi"] > 0 else None
    missing = sorted(set(refs) - set(ratios))
    entry["phones"]["purity_vs_lang_specific"] = {
        "min_ratio": min(ratios.values()) if ratios else None, "threshold": MIN_PURITY_RATIO,
        "missing_languages": missing,
        "pass": bool(ratios) and not missing and min(ratios.values()) >= MIN_PURITY_RATIO}


def probe_markdown(report):
    """The Stage 03 probe table: one row per units directory."""
    langs = sorted({lang for e in report["units"].values()
                    for lang in (e.get("phones") or {}).get("by_lang", {})})
    head = ["Codebook", "Teacher", "K", "Licence"]
    head += [f"{lang} purity / PNMI" for lang in langs]
    head += ["Purity vs. language-specific (min)", "Speaker-ID acc. (chance)"]
    lines = ["| " + " | ".join(head) + " |", "|---" * len(head) + "|"]
    for path, e in report["units"].items():
        phones = e.get("phones") or {}
        cells = [f"`{pathlib.Path(path).name}`", str(e.get("teacher") or e.get("tokenizer")),
                 str(e.get("vocab")), str(e.get("license") or "—")
                 + (" (NC)" if e.get("noncommercial") else "")]
        for lang in langs:
            s = phones.get("by_lang", {}).get(lang)
            cells.append(f"{s['phone_purity']:.3f} / {s['pnmi']:.3f}" if s else "—")
        vs = phones.get("purity_vs_lang_specific") or {}
        cells.append("—" if vs.get("min_ratio") is None else
                     f"{vs['min_ratio']:.3f} ({'pass' if vs['pass'] else 'FAIL'})")
        spk = e.get("speaker_id")
        cells.append("—" if not spk else f"{100 * spk['accuracy']:.1f}% ({100 * spk['chance']:.1f}%)"
                     f" {'pass' if spk['pass'] else 'FAIL'}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


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


def student_cost(checkpoint, fps_list, steps, warmup, cfg=None):
    torch = single_core()
    from fvml import content_encoder as ce
    model = ce.ContentEncoder.from_checkpoint(checkpoint)[0] if checkpoint \
        else ce.ContentEncoder(cfg or ce.EncoderConfig())
    model.eval()
    rng = np.random.default_rng(0)
    out = {"params": model.num_params(), "macs_per_second": model.macs_per_second(),
           "state_floats": model.state_numel(), "latent_dim": model.cfg.latent_dim,
           "vocab": model.cfg.vocab}
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
    p.add_argument("--lang-reference", action="append", default=[], metavar="LANG=DIR",
                   help="units of the codebook fitted on LANG alone (repeatable)")
    p.add_argument("--max-speaker-accuracy", type=float, default=0.05,
                   help="speaker-ID leakage ceiling for the pass/fail flag (default 0.05)")
    p.add_argument("--include-noncommercial", action="store_true",
                   help="accept units from the CC-BY-NC-SA mHuBERT-147 teacher (research only)")
    p.add_argument("--multilingual", action="store_true",
                   help="--cost: time the random-init M3 student / vocoder configuration")
    p.add_argument("--markdown", help="write the probe comparison table (Markdown) here")
    p.add_argument("--out", help="write the JSON report here")
    args = p.parse_args(argv)
    for stream_io in (sys.stdout, sys.stderr):
        if hasattr(stream_io, "reconfigure"):
            stream_io.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
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
        if args.lang_reference and not args.alignments:
            raise ValueError("--lang-reference needs --alignments")
        refs = lang_reference_scores(args.lang_reference, align_index, args, log)
        if refs:
            report["lang_references"] = {
                lang: {k: r[k] for k in ("phone_purity", "pnmi", "frames", "utterances")}
                for lang, r in refs.items()}
        for d in args.units:
            meta = units.read_meta(d)
            units.check_license(meta, args.include_noncommercial)
            entry = {k: meta.get(k) for k in ("tokenizer", "frame_rate_hz", "lookahead_frames", "vocab")}
            if meta.get("license") or meta.get("noncommercial"):     # M3 streams say what they are
                entry.update(license=meta.get("license"),
                             noncommercial=bool(meta.get("noncommercial")),
                             teacher=(meta.get("source") or {}).get("teacher"),
                             label_space=meta.get("label_space"))
            if args.alignments:
                entry["phones"] = phone_probe(d, align_index, args.tier, args.centre_offset_ms, log)
                if refs:
                    add_purity_ratios(entry, refs)
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
            enc_cfg = voc_cfg = None
            if args.multilingual:
                from fvml import content_encoder as ce
                from fvml import vocoder as voc
                enc_cfg = ce.MULTILINGUAL
                voc_cfg = voc.VocoderConfig(vocab=voc.MULTILINGUAL_VOCAB)
            report["cost"] = {"student": student_cost(args.checkpoint, fps, args.steps,
                                                      args.warmup, enc_cfg)}
            if args.cost_vocoder:
                report["cost"]["vocoder"] = vocoder_cost(args.vocoder_checkpoint, fps,
                                                         args.steps, args.warmup, voc_cfg)
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
    if args.markdown:
        pathlib.Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.markdown).write_text(probe_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
