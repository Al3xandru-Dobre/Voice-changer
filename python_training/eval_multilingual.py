#!/usr/bin/env python3
"""Mode A non-regression and per-language resynthesis scoring for the
multilingual encoder + vocoder.

Milestone 3 / Stage 03 (Documents/Milestone 3/03_multilingual_encoder_vocoder_upgrade.md).
Conversion is the full *streaming* chain of the Stage 05 gate
(gate_report.convert_eval_list: ContentEncoder.step + PitchTracker.push +
map_to_target + UnitVocoder.step) and scoring is eval_vc.py's engine
(fvml.evaluate), so the numbers are comparable with Milestone 2's.

  lists     Build the held-out per-language source lists from manifests
            (eval_lists/resynth/resynth_<lang>_<n>.tsv, same columns as
            test_clean_200.tsv) and, with --voice-refs, the target reference
            rows of the multilingual voices appended to a copy of targets.tsv.
            Selection is deterministic: rows of the language in --subsets, with
            a transcript, 3-12 s, taken round-robin over speakers.
  resynth   Student `units` + a target speaker embedding -> vocoder on each
            language's list, scored with Whisper WER/CER in that language (forced
            language, diacritic-preserving normaliser), UTMOS and ECAPA target
            cosine. Definition of done: WER <= 8% (en, es), <= 10% (fr, de) and
            UTMOS >= 3.3 in all four languages.
  mode-a    English Mode A VC on test_clean_200 into the given voices, compared
            with the Milestone 2 number: WER <= 10% and within +1.0 pt of it
            (--baseline-summary: an eval_vc.py summary.json or a gate_report.py
            JSON; or --baseline-wer).

--converted KEY=DIR (KEY = language for resynth, voice for mode-a) scores WAVs
that already exist, e.g. written by the C++ fv_vc_file from the exported bundle,
instead of converting in PyTorch.

Usage:
  eval_multilingual.py lists --manifest css10-es.jsonl,cml-tts-es.jsonl --lang es \\
      --voice-refs css10_es --out-dir eval_lists/resynth
  eval_multilingual.py resynth --encoder-checkpoint checkpoints/ce_m3/best.pt \\
      --vocoder-checkpoint checkpoints/voc_m3/best.pt --candidate student-L1:s1 \\
      --list en=eval_lists/test_clean_200.tsv --list es=eval_lists/resynth/resynth_es_100.tsv \\
      --list fr=... --list de=... --voice en=lj --voice es=css10_es --voice fr=css10_fr \\
      --voice de=css10_de --targets eval_lists/resynth/targets_m3.tsv --out runs/eval/m3_resynth
  eval_multilingual.py mode-a --encoder-checkpoint ... --vocoder-checkpoint ... \\
      --voices lj,hifitts_92 --baseline-summary runs/gate/m2/gate_report.json \\
      --out runs/eval/m3_mode_a

Exit status: 0 = every gate passed, 1 = a gate failed, is incomplete (a language
or the M2 baseline is missing) or inputs are missing.
"""
import argparse
import json
import math
import pathlib
import sys

import numpy as np

from fvml import evaluate, evallist, paths, sampling, units

RESYNTH_MAX_WER = {"en": 0.08, "es": 0.08, "fr": 0.10, "de": 0.10}
RESYNTH_MIN_UTMOS = 3.3
MODE_A_MAX_WER = 0.10
MODE_A_MAX_WER_DELTA = 0.01          # +1.0 pt over the M2 English-only vocoder
RESYNTH_METRICS = ("wer", "spk", "utmos")
LIST_SECONDS = (3.0, 12.0)


def _bad(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


# -- lists ---------------------------------------------------------------------
def round_robin(rows):
    """Rows interleaved over speakers (speakers and utterances in sorted
    order): the first n are as speaker-diverse as the pool allows."""
    by_spk = {}
    for r in sorted(rows, key=lambda r: (str(r.get("speaker")), r["utt_id"])):
        by_spk.setdefault(str(r.get("speaker")), []).append(r)
    out, depth = [], 0
    while len(out) < len(rows):
        for utts in by_spk.values():
            if depth < len(utts):
                out.append(utts[depth])
        depth += 1
    return out


def build_source_list(rows, lang, n=100, subsets=("test",), seconds=LIST_SECONDS):
    """n held-out source rows (evallist.SOURCE_COLUMNS) of one language."""
    lo, hi = seconds
    pool = [r for r in units.filter_subsets(rows, subsets)
            if r.get("lang") == lang and str(r.get("text") or "").strip()
            and lo <= float(r["duration"]) <= hi]
    if len(pool) < n:
        raise ValueError(f"{lang}: only {len(pool)} held-out clips with a transcript of "
                         f"{lo:g}-{hi:g} s in subsets {list(subsets)}, need {n}")
    return [{"utt_id": r["utt_id"], "speaker": r.get("speaker") or "unknown",
             "gender": r.get("gender") or "?", "duration": float(r["duration"]),
             "path": r["path"], "text": " ".join(str(r["text"]).split())}
            for r in round_robin(pool)[:n]]


def build_target_rows(rows, voice, n=20, subsets=("test",), exclude=(), seconds=LIST_SECONDS):
    """n held-out reference clips (evallist.TARGET_COLUMNS) of one voice,
    spread evenly over its held-out clips of 3-12 s; `exclude` are utt_ids
    already used as resynthesis sources."""
    lo, hi = seconds
    pool = sorted((r for r in units.filter_subsets(rows, subsets)
                   if str(r.get("speaker") or "").lower() == voice and r["utt_id"] not in exclude
                   and lo <= float(r["duration"]) <= hi),
                  key=lambda r: r["utt_id"])
    if len(pool) < n:
        raise ValueError(f"voice '{voice}': only {len(pool)} held-out clips of {lo:g}-{hi:g} s "
                         f"in subsets {list(subsets)}, need {n}")
    return [{"voice": voice, "clip_id": r["utt_id"], "speaker": r.get("speaker"),
             "gender": r.get("gender") or "?", "duration": float(r["duration"]),
             "path": r["path"], "dataset": r.get("dataset") or "unknown"}
            for r in evallist.evenly_spaced(pool, n)]


def run_lists(args, log=print):
    rows, _ = units.read_manifests(args.manifest)
    rows = units.audio_rows(rows, args.side, args.default_lang)
    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = build_source_list(rows, args.lang, args.n, subsets, (args.min_seconds, args.max_seconds))
    path = out_dir / f"resynth_{args.lang}_{args.n}.tsv"
    evallist.write(path, src, evallist.SOURCE_COLUMNS)
    log(f"[lists] {path}: {len(src)} clips, {len({r['speaker'] for r in src})} speaker(s)")
    voices = [v.strip() for v in (args.voice_refs or "").split(",") if v.strip()]
    if voices:
        base = pathlib.Path(args.base_targets) if args.base_targets else None
        tpath = out_dir / args.targets_name
        targets = evallist.read_targets(tpath if tpath.is_file() else base) \
            if (tpath.is_file() or (base and base.is_file())) else []
        used = {r["utt_id"] for r in src}
        for v in voices:
            targets = [t for t in targets if t["voice"] != v]
            targets += build_target_rows(rows, v, args.refs, subsets, used,
                                         (args.min_seconds, args.max_seconds))
        evallist.write(tpath, targets, evallist.TARGET_COLUMNS)
        log(f"[lists] {tpath}: {len(targets)} reference clips, voices "
            f"{sorted({t['voice'] for t in targets})}")
    return 0


# -- verdicts ------------------------------------------------------------------
def resynth_verdict(results, languages=sampling.LANGS):
    """Per-language pass/fail against Stage 03's resynthesis floors. Complete
    only when every language in `languages` was scored."""
    rows, ok = {}, True
    for lang, r in results.items():
        max_wer = RESYNTH_MAX_WER.get(lang, max(RESYNTH_MAX_WER.values()))
        wer_ok = not _bad(r.get("wer")) and r["wer"] <= max_wer
        mos_ok = not _bad(r.get("utmos")) and r["utmos"] >= RESYNTH_MIN_UTMOS
        rows[lang] = {"max_wer": max_wer, "wer_pass": bool(wer_ok),
                      "min_utmos": RESYNTH_MIN_UTMOS, "utmos_pass": bool(mos_ok),
                      "pass": bool(wer_ok and mos_ok)}
        ok = ok and wer_ok and mos_ok
    missing = [lang for lang in languages if lang not in results]
    return {"languages": rows, "missing_languages": missing,
            "pass": bool(ok and not missing and results)}


def mode_a_verdict(wer, baseline_wer=None):
    """English Mode A: WER <= 10% absolute, and no more than +1.0 pt over the
    M2 number. Without a baseline the verdict is incomplete, never a pass."""
    abs_ok = not _bad(wer) and wer <= MODE_A_MAX_WER
    delta = None if _bad(wer) or _bad(baseline_wer) else wer - baseline_wer
    delta_ok = None if delta is None else delta <= MODE_A_MAX_WER_DELTA + 1e-12
    return {"wer": wer, "baseline_wer": baseline_wer, "max_wer": MODE_A_MAX_WER,
            "max_wer_delta": MODE_A_MAX_WER_DELTA, "wer_delta": delta,
            "abs_pass": bool(abs_ok), "delta_pass": delta_ok,
            "complete": delta_ok is not None, "pass": bool(abs_ok and delta_ok)}


def baseline_wer(path):
    """The M2 WER from an eval_vc.py summary.json ({"wer": ...}) or a
    gate_report.py JSON (the decision's winner row)."""
    data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and not _bad(data.get("wer")):
        return float(data["wer"])
    if isinstance(data, dict) and isinstance(data.get("candidates"), list):
        winner = (data.get("decision") or {}).get("winner")
        for row in data["candidates"]:
            if row.get("candidate") == winner and not _bad(row.get("wer")):
                return float(row["wer"])
    raise ValueError(f"{path}: no WER found (need an eval_vc summary.json or a gate report "
                     "with a scored winner)")


# -- conversion + scoring -------------------------------------------------------
def load_chain(args):
    """(encoder, vocoder, {voice name: voices.json entry}) from the checkpoints."""
    from fvml import content_encoder as ce
    from fvml import vocoder
    enc, _ = ce.ContentEncoder.from_checkpoint(args.encoder_checkpoint)
    voc, ck = vocoder.UnitVocoder.from_checkpoint(args.vocoder_checkpoint)
    if enc.cfg.vocab != voc.cfg.vocab:
        raise ValueError(f"encoder vocab {enc.cfg.vocab} != vocoder vocab {voc.cfg.vocab}")
    voices = ck.get("voices") or voc.voices_json(ck["speakers"], ck["speaker_stats"])
    return enc, voc, {v["name"]: v for v in voices["voices"]}


def key_values(specs, what):
    out = {}
    for spec in specs or []:
        key, sep, value = spec.partition("=")
        if not sep or not key.strip() or not value.strip():
            raise ValueError(f"{what} wants KEY=VALUE, got {spec!r}")
        out[key.strip()] = value.strip()
    return out


def convert_and_score(args, chain, list_path, voice, out_dir, language, metrics, converted, log):
    """One list -> one voice: convert (unless `converted` names existing WAVs),
    then fvml.evaluate. Returns the evaluation summary."""
    import gate_report
    if converted is None:
        enc, voc, voices = chain()
        if voice not in voices:
            raise ValueError(f"voice '{voice}' is not in the vocoder ({', '.join(voices)})")
        converted = out_dir / "wav"
        n = gate_report.convert_eval_list(list_path, gate_report.parse_candidate_spec(args.candidate),
                                          enc, voc, voices[voice], converted, limit=args.limit)
        log(f"[convert] {language} -> {voice}: {n} utterances")
    cfg = evaluate.EvalConfig(sources=list_path, converted=converted,
                              targets=args.targets if "spk" in metrics else None, voice=voice,
                              metrics=metrics, device=args.device, batch_size=args.batch_size,
                              limit=args.limit, language=language)
    return evaluate.evaluate(cfg, out_dir / "eval", log=log)


def lazy_chain(args):
    cache = {}

    def get():
        if "chain" not in cache:
            if not (args.encoder_checkpoint and args.vocoder_checkpoint):
                raise ValueError("conversion needs --encoder-checkpoint and --vocoder-checkpoint "
                                 "(or pass --converted for every key)")
            cache["chain"] = load_chain(args)
        return cache["chain"]
    return get


def _pct(x):
    return "—" if _bad(x) else f"{100 * x:.2f}"


def _num(x, digits=3):
    return "—" if _bad(x) else f"{x:.{digits}f}"


def run_resynth(args, log=print):
    lists = key_values(args.list, "--list")
    voice_map = key_values(args.voice, "--voice")
    converted = key_values(args.converted, "--converted")
    if not lists:
        raise ValueError("resynth needs at least one --list LANG=TSV")
    out = pathlib.Path(args.out)
    chain, results = lazy_chain(args), {}
    for lang, list_path in lists.items():
        voice = voice_map.get(lang) or voice_map.get("*")
        if not voice:
            raise ValueError(f"no target voice for {lang}: pass --voice {lang}=NAME (or *=NAME)")
        res = convert_and_score(args, chain, list_path, voice, out / lang, lang, RESYNTH_METRICS,
                                converted.get(lang), log)
        results[lang] = {"voice": voice, "list": str(list_path),
                         **{k: res.get(k) for k in ("n", "wer", "cer", "tgt_cos", "src_cos", "utmos")}}
    verdict = resynth_verdict(results)
    report = {"kind": "resynthesis", "candidate": args.candidate, "results": results,
              "verdict": verdict, "encoder_checkpoint": args.encoder_checkpoint,
              "vocoder_checkpoint": args.vocoder_checkpoint}
    lines = ["| Language | Voice | N | WER % (max) | CER % | UTMOS (min) | Tgt cos | Src cos | Pass |",
             "|---|---|---|---|---|---|---|---|---|"]
    for lang, r in results.items():
        v = verdict["languages"][lang]
        lines.append(f"| {lang} | `{r['voice']}` | {r['n']} | {_pct(r['wer'])} ({100 * v['max_wer']:.0f}) "
                     f"| {_pct(r['cer'])} | {_num(r['utmos'], 2)} ({v['min_utmos']}) | "
                     f"{_num(r['tgt_cos'])} | {_num(r['src_cos'])} | {'yes' if v['pass'] else 'NO'} |")
    if verdict["missing_languages"]:
        lines.append(f"\nNot scored: {', '.join(verdict['missing_languages'])}.")
    write_report(out, "resynth", report, lines)
    log("\n".join(lines))
    return 0 if verdict["pass"] else 1


def run_mode_a(args, log=print):
    voices = [v.strip() for v in args.voices.split(",") if v.strip()]
    converted = key_values(args.converted, "--converted")
    if not voices:
        raise ValueError("mode-a needs --voices")
    metrics = tuple(m.strip() for m in args.metrics.split(",") if m.strip())
    out = pathlib.Path(args.out)
    chain, per_voice = lazy_chain(args), {}
    for voice in voices:
        res = convert_and_score(args, chain, args.list_path, voice, out / voice, "en", metrics,
                                converted.get(voice), log)
        per_voice[voice] = {k: res.get(k) for k in (*evaluate.SUMMARY_KEYS, "n")}
    summary = {}
    for k in evaluate.SUMMARY_KEYS:               # mean over voices, as the Stage 05 gate does
        vals = [v[k] for v in per_voice.values() if not _bad(v.get(k))]
        summary[k] = float(np.mean(vals)) if vals else None
    base = args.baseline_wer if args.baseline_wer is not None else (
        baseline_wer(args.baseline_summary) if args.baseline_summary else None)
    verdict = mode_a_verdict(summary["wer"], base)
    report = {"kind": "mode-a", "candidate": args.candidate, "list": str(args.list_path),
              "per_voice": per_voice, "summary": summary, "verdict": verdict,
              "encoder_checkpoint": args.encoder_checkpoint,
              "vocoder_checkpoint": args.vocoder_checkpoint}
    lines = ["| Voice | WER % | CER % | Tgt cos | Src cos | UTMOS | F0 PCC |", "|---|---|---|---|---|---|---|"]
    for voice, r in per_voice.items():
        lines.append(f"| `{voice}` | {_pct(r['wer'])} | {_pct(r['cer'])} | {_num(r['tgt_cos'])} | "
                     f"{_num(r['src_cos'])} | {_num(r['utmos'], 2)} | {_num(r['f0_pcc'])} |")
    lines.append(f"\nMean WER {_pct(summary['wer'])}% (max {100 * MODE_A_MAX_WER:.0f}%): "
                 f"{'pass' if verdict['abs_pass'] else 'FAIL'}. M2 baseline "
                 + ("not given: the +1.0 pt check is NOT RUN." if not verdict["complete"] else
                    f"{_pct(base)}%, delta {100 * verdict['wer_delta']:+.2f} pt (max +1.00): "
                    f"{'pass' if verdict['delta_pass'] else 'FAIL'}."))
    write_report(out, "mode_a", report, lines)
    log("\n".join(lines))
    return 0 if verdict["pass"] else 1


def write_report(out, name, report, lines):
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(report, indent=2, default=float) + "\n",
                                      encoding="utf-8")
    (out / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# -- CLI -----------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.splitlines()[2:]))
    sub = p.add_subparsers(dest="command", required=True)

    li = sub.add_parser("lists", help="build held-out per-language lists")
    li.add_argument("--manifest", action="append", required=True)
    li.add_argument("--lang", required=True, choices=sampling.LANGS)
    li.add_argument("--n", type=int, default=100)
    li.add_argument("--subsets", default="test", help="held-out subsets (default test)")
    li.add_argument("--min-seconds", type=float, default=LIST_SECONDS[0])
    li.add_argument("--max-seconds", type=float, default=LIST_SECONDS[1])
    li.add_argument("--side", choices=units.SIDES, help="paired S2UT manifests: src or tgt")
    li.add_argument("--default-lang", help="language of rows without a `lang` field")
    li.add_argument("--voice-refs", help="voices (speaker names) to add reference clips for")
    li.add_argument("--refs", type=int, default=20, help="reference clips per voice")
    li.add_argument("--base-targets", default=str(paths.EVAL_LISTS / "targets.tsv"),
                    help="targets list the new voices are added to ('' = none)")
    li.add_argument("--targets-name", default="targets_m3.tsv")
    li.add_argument("--out-dir", default=str(paths.EVAL_LISTS / "resynth"))

    def common(q):
        q.add_argument("--encoder-checkpoint")
        q.add_argument("--vocoder-checkpoint")
        q.add_argument("--candidate", default="student-L1:s1",
                       help="encoder head and frames per step (default student-L1:s1)")
        q.add_argument("--targets", default=str(paths.EVAL_LISTS / "targets.tsv"))
        q.add_argument("--converted", action="append", metavar="KEY=DIR",
                       help="score existing WAVs for this language/voice instead of converting")
        q.add_argument("--limit", type=int, help="first N rows of each list")
        q.add_argument("--device", default="cuda")
        q.add_argument("--batch-size", type=int, default=8)
        q.add_argument("--out", required=True)

    rs = sub.add_parser("resynth", help="per-language resynthesis scores")
    common(rs)
    rs.add_argument("--list", action="append", metavar="LANG=TSV", help="source list per language")
    rs.add_argument("--voice", action="append", metavar="LANG=NAME",
                    help="target voice per language (*=NAME for all)")
    rs.set_defaults(limit=100)

    ma = sub.add_parser("mode-a", help="English Mode A non-regression vs. Milestone 2")
    common(ma)
    ma.add_argument("--list", dest="list_path", default=str(paths.EVAL_LISTS / "test_clean_200.tsv"))
    ma.add_argument("--voices", default="lj,hifitts_92")
    ma.add_argument("--metrics", default=",".join(evaluate.METRICS))
    ma.add_argument("--baseline-summary", help="M2 eval_vc summary.json or gate_report JSON")
    ma.add_argument("--baseline-wer", type=float, help="M2 WER as a fraction (e.g. 0.071)")
    return p.parse_args(argv)


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    args = parse_args(argv)
    try:
        return {"lists": run_lists, "resynth": run_resynth, "mode-a": run_mode_a}[args.command](args)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"[eval_multilingual] error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
