#!/usr/bin/env python3
"""Stage 10 DoD: does the fine-tuned "me" voice beat the zero-shot baseline?

Milestone 2 / Stage 10 (Documents/Milestone 2/10_target_voice_finetune.md).

Gates measure the shipped runtime (MEMORY.md rule 11): every conversion goes
through fv_vc_file (fvml/vcfile.py). The script converts the eval list:

  after/<voice me>          the fine-tuned bundle, target voice "me"
  before/<baseline-voice>   the pre-fine-tune bundle, the row "me" was
                            initialised from (zero-shot baseline); both are
                            scored against the held-out "me" references
                            (--me-targets, <speaker>_targets.tsv)
  before|after/<public>     every public voice on both bundles (regression)

Plus own-voice resynthesis: the held-out clips themselves converted through
after/me -> UTMOS (reported, not a DoD gate).

Scoring goes through fvml.evaluate (WER + tgt_cos; UTMOS where needed).
evaluate_fn and convert_list are injectable so tests run without fv_vc_file
or the heavy models.

DoD (target_voice_dod):
  - tgt_cos(me, after) - tgt_cos(baseline, before) >= 0.05
  - per public voice: tgt_cos(before) - tgt_cos(after) <= 0.02 and
    WER(after) - WER(before) <= 0.02 (absolute fraction, i.e. 2 pts)

Writes report.json + report.md under --out; exit 0 iff every DoD line passes.
"""
import argparse
import json
import pathlib
import sys

from fvml import evallist, paths, vcfile

COS_IMPROVEMENT_MIN = 0.05
COS_REGRESSION_MAX = 0.02
WER_REGRESSION_MAX = 0.02


def target_voice_dod(results):
    """[(check, ok, detail)] from a results dict:

    {"me": {"tgt_cos":..,"wer":..}, "baseline": {...},
     "public": {name: {"before": {...}, "after": {...}}}}"""
    lines = []
    delta = results["me"]["tgt_cos"] - results["baseline"]["tgt_cos"]
    lines.append(("me_tgt_cos_delta", delta >= COS_IMPROVEMENT_MIN,
                  f"{delta:+.3f} (>= +{COS_IMPROVEMENT_MIN})"))
    for name, r in sorted(results.get("public", {}).items()):
        dc = r["before"]["tgt_cos"] - r["after"]["tgt_cos"]
        dw = r["after"]["wer"] - r["before"]["wer"]
        lines.append((f"{name}_cos_regression", dc <= COS_REGRESSION_MAX,
                      f"{dc:+.3f} (<= {COS_REGRESSION_MAX})"))
        lines.append((f"{name}_wer_regression", dw <= WER_REGRESSION_MAX,
                      f"{dw:+.4f} (<= {WER_REGRESSION_MAX})"))
    return lines


def _eval(evaluate_fn, sources, converted, targets, voice, metrics, device):
    cfg = evaluate_fn(sources=sources, converted=converted, targets=targets,
                      voice=voice, metrics=metrics, device=device)
    return cfg


def default_evaluate(sources, converted, targets, voice, metrics, device):
    from fvml import evaluate
    cfg = evaluate.EvalConfig(sources=sources, converted=converted, targets=targets,
                              voice=voice, metrics=metrics, device=device)
    return evaluate.evaluate(cfg, pathlib.Path(converted) / "_eval",
                             log=lambda *a: None)


def run(args, convert_list=None, evaluate_fn=None, log=print):
    """-> (results, dod). Results feed target_voice_dod and the report."""
    convert_list = convert_list or vcfile.convert_list
    evaluate_fn = evaluate_fn or default_evaluate
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    vc_file = vcfile.find_vc_file()
    if vc_file is None:
        raise FileNotFoundError("no fv_vc_file: set FLUXVOICE_VC_FILE or build an ORT tree")

    rows = evallist.read_sources(paths.resolve(args.list))[: args.limit or None]
    me_rows = None
    if args.heldout_manifest:
        from fvml.datasets import read_jsonl
        me_rows = read_jsonl(paths.resolve(args.heldout_manifest))
    baseline_voice = args.baseline_voice
    if not baseline_voice and args.checkpoint:
        import torch
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        baseline_voice = (ck.get("speaker_init") or {}).get(args.voice, {}).get("from")
    if not baseline_voice:
        raise ValueError("--baseline-voice is required (or pass --checkpoint with "
                         "speaker_init[<voice>].from)")
    public = list(args.public_voices.split(",")) if args.public_voices else []

    results = {"public": {}}
    def conv(bundle, voice_name, tag, rows_=rows):
        dest = out_dir / tag
        convert_list(vc_file, bundle, rows_, dest, voice=voice_name)
        return dest

    # The zero-shot baseline and the fine-tuned "me", scored vs the "me" refs.
    before_dir = conv(args.bundle_before, baseline_voice, "before_baseline")
    results["baseline"] = _eval(evaluate_fn, paths.resolve(args.list), before_dir,
                                args.me_targets, args.voice, ("wer", "spk"), args.device)
    after_me_dir = conv(args.bundle_after, args.voice, "after_me")
    results["me"] = _eval(evaluate_fn, paths.resolve(args.list), after_me_dir,
                          args.me_targets, args.voice, ("wer", "spk"), args.device)

    for name in public:
        b = conv(args.bundle_before, name, f"before_{name}")
        a = conv(args.bundle_after, name, f"after_{name}")
        results["public"][name] = {
            "before": _eval(evaluate_fn, paths.resolve(args.list), b,
                            args.targets, name, ("wer", "spk"), args.device),
            "after": _eval(evaluate_fn, paths.resolve(args.list), a,
                           args.targets, name, ("wer", "spk"), args.device),
        }

    # Own-voice resynthesis on the held-out clips -> UTMOS (reported). The
    # eval engine reads a sources TSV, so write one for the held-out clips.
    if me_rows:
        held_tsv = out_dir / "heldout_sources.tsv"
        evallist.write(held_tsv,
                       [{"utt_id": r["utt_id"], "speaker": args.voice, "gender": "U",
                         "duration": r.get("duration", 0), "path": r["path"], "text": ""}
                        for r in me_rows], evallist.SOURCE_COLUMNS)
        resyn = conv(args.bundle_after, args.voice, "me_resynthesis", rows_=me_rows)
        res = _eval(evaluate_fn, held_tsv, resyn, None, None, ("utmos",), args.device)
        results["me_resynthesis_utmos"] = res.get("utmos")

    dod = target_voice_dod(results)
    write_report(results, dod, out_dir, args)
    for name, ok, detail in dod:
        print(f"[target_voice] {name}: {'PASS' if ok else 'FAIL'} — {detail}")
    return results, dod


def write_report(results, dod, out_dir, args):
    lines = ["# Stage 10 DoD — target-voice fine-tune", "",
             f"before `{args.bundle_before}`, after `{args.bundle_after}`, "
             f"voice `{args.voice}`.", ""]
    if results.get("me_resynthesis_utmos") is not None:
        lines.append(f"Own-voice resynthesis UTMOS: "
                     f"{results['me_resynthesis_utmos']:.2f}")
        lines.append("")
    lines += ["| check | result | detail |", "|---|---|---|"]
    for name, ok, detail in dod:
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "report.json").write_text(
        json.dumps({"results": results,
                    "dod": [{"check": n, "pass": bool(ok), "detail": d}
                            for n, ok, d in dod],
                    "verdict": "PASS" if all(ok for _, ok, _ in dod) else "FAIL"},
                   indent=2, default=float) + "\n", encoding="utf-8")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.splitlines()[2:]))
    p.add_argument("--bundle-before", required=True, help="pre-fine-tune bundle")
    p.add_argument("--bundle-after", required=True, help="fine-tuned bundle (with the new voice)")
    p.add_argument("--voice", required=True, help="new voice name (e.g. me)")
    p.add_argument("--baseline-voice",
                   help="row the new voice was initialised from (zero-shot baseline); "
                        "default: speaker_init[<voice>].from read from --checkpoint")
    p.add_argument("--checkpoint", help="fine-tune checkpoint, for the baseline voice")
    p.add_argument("--public-voices",
                   help="comma-separated public voices (default: all voices of the before-bundle)")
    p.add_argument("--list", default=str(paths.EVAL_LISTS / "test_clean_200.tsv"))
    p.add_argument("--targets", default=str(paths.EVAL_LISTS / "targets.tsv"),
                   help="public targets TSV")
    p.add_argument("--me-targets", required=True,
                   help="<speaker>_targets.tsv from prepare_target_voice.py "
                        "(held-out 'me' references the baseline and fine-tuned "
                        "voice are scored against)")
    p.add_argument("--heldout-manifest", help="heldout.jsonl (resynthesis source list)")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


def _bundle_voices(bundle):
    v = json.loads((pathlib.Path(bundle) / "voices.json").read_text(encoding="utf-8"))
    return [x["name"] for x in v["voices"]]


def main(argv=None):
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args(argv)
    if not args.public_voices:
        args.public_voices = ",".join(
            v for v in _bundle_voices(args.bundle_before) if v != args.voice)
    try:
        _, dod = run(args)
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as e:
        print(f"[eval_target_voice] error: {e}", file=sys.stderr)
        return 1
    return 0 if all(ok for _, ok, _ in dod) else 1


if __name__ == "__main__":
    sys.exit(main())
