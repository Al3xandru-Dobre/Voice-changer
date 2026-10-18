#!/usr/bin/env python3
"""Score converted speech: WER/CER, target and source similarity, UTMOS, F0 PCC, RTF.

Input is a directory of converted WAVs named <utt_id>.wav (or .flac), matched to
an eval list. Output is per_utt.csv, summary.json (keys wer, cer, tgt_cos,
src_cos, utmos, f0_pcc, rtf) and summary.md. The engine is fvml/evaluate.py;
call fvml.evaluate.evaluate() from Python (Stage 05's gate does).

Usage:
    eval_vc.py --baseline --out runs/eval/baseline                 # score the sources
    eval_vc.py --converted out/lj --voice lj --out runs/eval/lj    # a conversion
    eval_vc.py --converted live/ --ref-text source --out ...       # no transcripts:
                                                                   # Whisper on the source
    eval_vc.py ... --targets private/me_targets.tsv --voice me     # a private target list

Defaults: --list eval_lists/test_clean_200.tsv, --targets eval_lists/targets.tsv.
Model weights are cached under $FLUXVOICE_DATA/cache unless HF_HOME is set.

Exit status: 0 = scored, 1 = missing inputs.
"""
import argparse
import sys

from fvml import evaluate, paths


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--list", default=str(paths.EVAL_LISTS / "test_clean_200.tsv"),
                   help="source eval list (default eval_lists/test_clean_200.tsv)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--converted", help="directory of converted <utt_id>.wav files")
    src.add_argument("--baseline", action="store_true", help="score the unconverted sources")
    p.add_argument("--targets", default=str(paths.EVAL_LISTS / "targets.tsv"),
                   help="target reference list (default eval_lists/targets.tsv)")
    p.add_argument("--voice", help="target voice name in --targets (enables tgt_cos)")
    p.add_argument("--ref-text", choices=("list", "source"), default="list",
                   help="WER reference: the list transcript, or Whisper on the source (default list)")
    p.add_argument("--metrics", default=",".join(evaluate.METRICS),
                   help=f"subset of {','.join(evaluate.METRICS)} (default all)")
    p.add_argument("--timing", help="CSV utt_id,seconds of compute (default <converted>/timing.csv)")
    p.add_argument("--max-lag-ms", type=float, default=200.0,
                   help="search range for the output delay before F0 PCC (default 200; 0 = off)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8, help="GPU batch (default 8 fits an 8 GB card)")
    p.add_argument("--limit", type=int, help="score only the first N utterances (smoke runs)")
    p.add_argument("--out", required=True, help="output directory")
    args = p.parse_args()

    metrics = tuple(m.strip() for m in args.metrics.split(",") if m.strip())
    unknown = set(metrics) - set(evaluate.METRICS)
    if unknown:
        p.error(f"unknown metric(s): {', '.join(sorted(unknown))}")
    cfg = evaluate.EvalConfig(
        sources=args.list, converted=None if args.baseline else args.converted,
        targets=args.targets if args.voice else None, voice=args.voice, ref_text=args.ref_text,
        metrics=metrics, device=args.device, batch_size=args.batch_size,
        max_lag_ms=args.max_lag_ms, timing=args.timing, limit=args.limit)
    sys.stdout.reconfigure(line_buffering=True)
    try:
        result = evaluate.evaluate(cfg, args.out)
    except (FileNotFoundError, ValueError) as e:
        print(f"eval_vc: {e}", file=sys.stderr)
        return 1
    print((paths.resolve(args.out) / "summary.md").read_text(encoding="utf-8"))
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
