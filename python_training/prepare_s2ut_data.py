#!/usr/bin/env python3
"""Prepare bidirectional En<->{Es,Fr,De} S2UT & multilingual vocoder datasets
(Milestone 3 / Stage 01).

Downloads/verifies archives against SHA256 pins in datasets.lock.json, resamples
all audio to 16 kHz mono 24-bit FLAC, synthesizes canonical target audio for
`En->{Es,Fr,De}` pairs with round-trip Whisper CER <= 5% filtering, writes
paired JSONL manifests under `$FLUXVOICE_DATA/manifests/s2ut/` and monolingual
vocoder manifests under `$FLUXVOICE_DATA/manifests/`, and fits shared 6,000-token
SentencePiece unigram models (`$FLUXVOICE_DATA/spm/spm_{en,es,fr,de}.model`).

VoxPopuli (CC BY-NC 4.0) is excluded by default and isolated under
`$FLUXVOICE_DATA/nc/` when `--include-noncommercial` is given.

Usage:
    prepare_s2ut_data.py --list
    prepare_s2ut_data.py --datasets paired --common-voice-root D:/Corpora/cv-corpus
    prepare_s2ut_data.py --datasets vocoder
    prepare_s2ut_data.py --train-spm
    prepare_s2ut_data.py --summary
"""
import argparse
import sys

from fvml import datasets as fd
from fvml import paths, s2ut_data as sd


def list_datasets(sctx):
    bctx = sctx.base_ctx(noncommercial=False)
    print(f"data root: {bctx.root}  ({fd.disk_free_gb(bctx.root):.0f} GB free)")
    for d in sd.S2UT_DATASETS.values():
        if d.noncommercial and not sctx.include_noncommercial:
            state = "nc-locked"
        else:
            mf = sd.manifest_path_for(sctx, d)
            state = "manifest" if mf.is_file() else "-"
        tag = f"{d.group}" + (" [NC]" if d.noncommercial else "")
        print(f"  {d.key:24s} {tag:13s} {state:10s}  {d.licence}")


def print_summary(summary):
    for key, s in summary.items():
        print(
            f"  {key:24s} src {s['src_hours']:7.2f} h  tgt {s['tgt_hours']:7.2f} h  "
            f"{s['rows']:7d} rows  {s['speakers']:5d} spk  ({s['license']})"
        )


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--datasets", help="paired | vocoder | all | comma-separated keys (see --list)")
    p.add_argument("--list", action="store_true", help="show S2UT dataset registry and status")
    p.add_argument("--train-spm", action="store_true",
                   help="fit shared 6,000-token SentencePiece models for {en,es,fr,de}")
    p.add_argument("--spm-vocab-size", type=int, default=sd.SPM_VOCAB_SIZE)
    p.add_argument("--summary", action="store_true", help="print summary of prepared S2UT manifests")
    p.add_argument("--include-noncommercial", action="store_true",
                   help="opt in to non-commercial VoxPopuli prep (isolated in $FLUXVOICE_DATA/nc/)")
    p.add_argument("--common-voice-root",
                   help="directory containing Common Voice {en,es,fr,de} clips/")
    p.add_argument("--max-synth-cer", type=float, default=sd.MAX_SYNTH_CER,
                   help="maximum Whisper round-trip CER for synthetic En->X target clips (default 0.05)")
    p.add_argument("--data-root", help="default $FLUXVOICE_DATA or <repo>/data")
    p.add_argument("--jobs", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--delete-archives", action="store_true")
    p.add_argument("--force-manifests", action="store_true")
    args = p.parse_args(argv)

    root = paths.resolve(args.data_root) if args.data_root else paths.data_root()
    cv_root = paths.resolve(args.common_voice_root) if args.common_voice_root else None
    sctx = sd.S2utContext(
        root=root,
        common_voice_root=cv_root,
        include_noncommercial=args.include_noncommercial,
        max_synth_cer=args.max_synth_cer,
        delete_archives=args.delete_archives,
        force_manifests=args.force_manifests,
        device=args.device,
        batch_size=args.batch_size,
    )
    if args.jobs:
        sctx.jobs = args.jobs
    sctx.root.mkdir(parents=True, exist_ok=True)

    if args.list:
        list_datasets(sctx)
        return 0
    if not (args.datasets or args.train_spm or args.summary):
        p.error("nothing to do: give --datasets, --train-spm, --list or --summary")

    try:
        if args.datasets:
            for ds in sd.select_s2ut(args.datasets, args.include_noncommercial):
                sd.prepare_s2ut(sctx, ds)
        if args.train_spm:
            written = sd.train_spm_models(sctx, vocab_size=args.spm_vocab_size)
            if not written:
                print("prepare_s2ut_data: no manifests with transcripts found for --train-spm",
                      file=sys.stderr)
                return 1
        summary = sd.summarize_s2ut(sctx)
        if summary:
            print("S2UT paired summary:")
            print_summary(summary)
    except fd.PrepareError as e:
        print(f"prepare_s2ut_data: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
