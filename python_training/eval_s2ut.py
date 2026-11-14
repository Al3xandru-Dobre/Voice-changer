#!/usr/bin/env python3
"""Score simultaneous S2UT outputs (Milestone 3 / Stage 01).

Evaluates any directory of translated WAV/FLAC files (or --oracle ground-truth
target audio) against a frozen S2UT evaluation TSV on:
  * ASR-BLEU (SacreBLEU 13a) & ASR-chrF++ (word_order=2) via forced-language
    Whisper large-v3
  * First-audio onset delay (ms) & SimulEval Average Lagging (AL and AL_CA, ms)
  * ECAPA-TDNN target similarity (tgt_cos) & source leakage (src_cos), UTMOS
  * Tail-flush non-silent duration & silence/noise hallucination word count

Usage:
    eval_s2ut.py --list eval_lists/s2ut/test_es_en_200.tsv --oracle --out runs/s2ut/oracle_es_en
    eval_s2ut.py --list eval_lists/s2ut/test_es_en_200.tsv --converted out/es_en \\
                 --trace out/es_en/trace.jsonl --voice lj --out runs/s2ut/es_en
    eval_s2ut.py --list eval_lists/s2ut/silence_hallucination_20.tsv --converted out/sil \\
                 --metrics flush --out runs/s2ut/silence

Exit status: 0 = scored, 1 = missing or invalid inputs.
"""
import argparse
import sys

from fvml import paths, s2ut_eval


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--list", required=True,
                   help="S2UT evaluation TSV (e.g. eval_lists/s2ut/test_es_en_200.tsv)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--converted", help="directory of translated <utt_id>.wav|.flac files")
    src.add_argument("--oracle", action="store_true",
                     help="score the ground-truth tgt_path audio from --list")
    p.add_argument("--trace", help="emission timestamp JSONL file or directory (for AL / AL_CA)")
    p.add_argument("--targets", default=str(paths.EVAL_LISTS / "targets.tsv"),
                   help="target voice reference TSV (default eval_lists/targets.tsv)")
    p.add_argument("--voice", help="target voice name in --targets (enables tgt_cos)")
    p.add_argument("--metrics", default=",".join(s2ut_eval.S2UT_METRICS),
                   help=f"subset of {','.join(s2ut_eval.S2UT_METRICS)} (default all)")
    p.add_argument("--silence-dbfs", type=float, default=-55.0,
                   help="20 ms frame RMS threshold in dBFS for onset/tail/silence detection")
    p.add_argument("--tail-allowance-ms", type=float, default=s2ut_eval.DEFAULT_TAIL_ALLOWANCE_MS,
                   help="expected post-speech flush tail allowance in ms (default 320)")
    p.add_argument("--timing", help="CSV utt_id,seconds of compute (default <converted>/timing.csv)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, help="score only the first N utterances (smoke runs)")
    p.add_argument("--out", required=True, help="output directory")
    args = p.parse_args(argv)

    metrics = tuple(m.strip() for m in args.metrics.split(",") if m.strip())
    unknown = set(metrics) - set(s2ut_eval.S2UT_METRICS)
    if unknown:
        p.error(f"unknown metric(s): {', '.join(sorted(unknown))}")

    cfg = s2ut_eval.S2utEvalConfig(
        eval_list=args.list,
        converted=None if args.oracle else args.converted,
        oracle=args.oracle,
        trace=args.trace,
        targets=args.targets if args.voice else None,
        voice=args.voice,
        metrics=metrics,
        silence_dbfs=args.silence_dbfs,
        tail_allowance_ms=args.tail_allowance_ms,
        device=args.device,
        batch_size=args.batch_size,
        timing=args.timing,
        limit=args.limit,
    )
    try:
        result = s2ut_eval.evaluate_s2ut(cfg, args.out)
    except (FileNotFoundError, ValueError) as e:
        print(f"eval_s2ut: {e}", file=sys.stderr)
        return 1
    print((paths.resolve(args.out) / "summary.md").read_text(encoding="utf-8"))
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
