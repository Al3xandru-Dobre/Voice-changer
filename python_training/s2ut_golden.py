#!/usr/bin/env python3
"""Milestone 3 / Stage 07 golden parity: Python ORT reference for fv_s2ut_file.

Spec: Documents/Milestone 3/07_cpp_translator_runtime_offline_runner.md. The
logic (and the exact Mode B semantics it reproduces) is in fvml/s2ut_golden.py.

make   runs the C++ Mode B pipeline (InferencePipeline, raw output) in Python
       ONNX Runtime and writes <out_dir>/input.wav, output.wav (32-bit float
       mono), trace.jsonl (one object per 80 ms chunk step) and meta.json
       (options, bundle SHA-256s, stats, the matching fv_s2ut_file command).
check  compares a `fv_s2ut_file --raw --no-prosody-ctx --dump-trace` run with
       a golden: identical chunk steps (decision, VAD, writes, t_src/t_tgt,
       raw units, <EOS>, utterance ends, emitted frames), identical output
       length and max |delta| <= --atol on a non-silent golden.

  s2ut_golden.py make --model-dir tests/data/fixtures/s2ut --src-lang es --tgt-lang de \\
      --wait-k 3 --frames-per-step 4 --no-prosody-ctx in.wav golden/
  fv_s2ut_file --model-dir tests/data/fixtures/s2ut --src-lang es --tgt-lang de \\
      --wait-k 3 --frames-per-step 4 --raw --no-prosody-ctx \\
      --dump-trace cpp.jsonl golden/input.wav cpp.wav
  s2ut_golden.py check --golden golden/ --trace cpp.jsonl --wav cpp.wav
  s2ut_golden.py --self-test        # offline, standard library only

--no-prosody-ctx is required by make: fv_s2ut_file's default prosody_ctx is the
C++ pitch tracker's utterance summary, which Python cannot reproduce bit for
bit (fvml/features.py is a tolerance-level port).

Exit status: make 0 = golden written, 1 = invalid input/bundle/option;
check 0 = PASS, 1 = FAIL (or unreadable inputs); 2 = usage error.
"""
import argparse
import pathlib
import sys

# Standard library only up here and in fvml.s2ut_golden's import, so `check`
# and --self-test run under `python -I` without the training environment.
if str(pathlib.Path(__file__).resolve().parent) not in sys.path:    # `python -I s2ut_golden.py`
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from fvml import s2ut_golden as G  # noqa: E402


def change_arg(text):
    try:
        return G.parse_change(text)
    except G.GoldenError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.splitlines()[2:]))
    p.add_argument("--self-test", action="store_true", help="offline checks (standard library)")
    sub = p.add_subparsers(dest="cmd", metavar="{make,check}")

    m = sub.add_parser("make", help="write a golden directory from Python ORT")
    m.add_argument("--model-dir", required=True, help="Mode B bundle (encoder_s<n>, "
                   "translator_enc/dec, vocoder_s<n>, voices.json, langs.json)")
    m.add_argument("--src-lang", required=True, help="langs.json code or id")
    m.add_argument("--tgt-lang", required=True, help="langs.json code or id")
    m.add_argument("--voice", help="voices.json name (default: the first voice)")
    m.add_argument("--wait-k", type=int, help="1..10 (default: langs.json default_wait_k for "
                                              f"the pair, else {G.DEFAULT_WAIT_K})")
    m.add_argument("--frames-per-step", type=int, choices=(1, 2, 4), default=4,
                   help="encoder/vocoder variant (default %(default)s, like fv_s2ut_file)")
    m.add_argument("--adaptive", action="store_true", help="CTC-adaptive readiness gating")
    m.add_argument("--no-prosody-ctx", action="store_true",
                   help="prosody_ctx = [0, 0] (required; see above)")
    m.add_argument("--max-tail-hops", type=int,
                   help="zero hops allowed after the input (default: ceil((tail_flush_ms + "
                        "(max_flush_steps + k + 2) x 80 ms) / hop), fv_s2ut_file's default "
                        "--max-tail-ms; the C++ equivalent is --max-tail-ms <hops x hop_ms>)")
    m.add_argument("--change", type=change_arg, action="append", default=[],
                   metavar="SECONDS:NAME=VALUE",
                   help="scheduled policy control, as fv_s2ut_file --change (repeatable; "
                        f"{' or '.join(G.CHANGE_PARAMS)} only)")
    m.add_argument("in_wav", help="mono 16 kHz WAV (16-bit PCM or 32-bit float)")
    m.add_argument("out_dir", help="golden directory to write")

    c = sub.add_parser("check", help="compare a fv_s2ut_file run with a golden")
    c.add_argument("--golden", required=True, help="directory written by `make`")
    c.add_argument("--trace", required=True, help="fv_s2ut_file --dump-trace JSONL")
    c.add_argument("--wav", required=True, help="fv_s2ut_file output WAV (--raw)")
    c.add_argument("--atol", type=float, default=G.DEFAULT_ATOL,
                   help="waveform max |delta| (default %(default)g)")
    c.add_argument("--allow-silent", action="store_true",
                   help="accept a golden whose output peaks below 100 x atol")
    c.add_argument("--max-report", type=int, default=12, help="mismatching fields to list")

    args = p.parse_args(argv)
    if not args.self_test and args.cmd is None:
        p.error("give a command (make / check) or --self-test")
    if args.cmd == "make" and not args.no_prosody_ctx:
        p.error("make: --no-prosody-ctx is required: fv_s2ut_file's default prosody_ctx is the "
                "C++ pitch tracker's utterance summary, which this golden cannot reproduce bit "
                "for bit; make the golden and the C++ run with --no-prosody-ctx")
    if args.cmd == "check" and not args.atol >= 0.0:
        p.error("--atol must be >= 0")
    return args


def main(argv=None):
    # BUG-015/016 family: the exit code must not depend on the console codepage.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    args = parse_args(argv)
    if args.self_test:
        return G.self_test()
    if args.cmd == "check":
        passed, _ = G.check(args.golden, args.trace, args.wav, atol=args.atol,
                            allow_silent=args.allow_silent, max_report=args.max_report)
        return 0 if passed else 1
    options = G.GoldenOptions(src_lang=args.src_lang, tgt_lang=args.tgt_lang, voice=args.voice,
                              wait_k=args.wait_k, frames_per_step=args.frames_per_step,
                              adaptive=args.adaptive, max_tail_hops=args.max_tail_hops,
                              changes=tuple(args.change))
    try:
        meta = G.make(args.model_dir, args.in_wav, args.out_dir, options,
                      log=lambda s: print(s, file=sys.stderr))
    except (G.GoldenError, OSError) as e:
        print(f"s2ut_golden make: {e}", file=sys.stderr)
        return 1
    o, s, out = meta["options"], meta["stats"], meta["output"]
    print(f"golden {args.out_dir}: {o['src_lang']}->{o['tgt_lang']} k={o['wait_k']} "
          f"s{o['frames_per_step']}{' adaptive' if o['adaptive'] else ''}, voice {o['voice']}")
    print(f"  {meta['chunk_steps']} chunk steps: {s['reads']} READ, {s['writes']} WRITE, "
          f"{s['flushes']} FLUSH; {s['utterances']} utterance(s), {s['eos_ends']} on <EOS>, "
          f"{s['cap_ends']} capped; {s['pad_frames']} pad frame(s), {s['fifo_stalls']} stall(s)")
    print(f"  output {out['samples']} samples ({out['input_hops']} input + {out['tail_hops']} "
          f"tail hops{', CAPPED' if out['tail_capped'] else ''}), peak {out['peak']:.4g}")
    print("  C++: " + " ".join(meta["cpp_command"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
