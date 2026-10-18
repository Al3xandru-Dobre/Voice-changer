#!/usr/bin/env python3
"""Download, verify, extract and resample the Milestone 2 datasets.

Every archive is checked against its published MD5 and the SHA256 pinned in
datasets.lock.json. Audio that is not 16 kHz is resampled to 16 kHz mono, and
one JSONL manifest per dataset (utt_id, path, duration, speaker, dataset,
subset, plus gender/text where the corpus has them) lands in
$FLUXVOICE_DATA/manifests/. Reruns skip verified downloads, finished
extractions and up-to-date manifests. See fvml/datasets.py for the layout and
DATASETS.md for sources and licences.

Usage:
    prepare_data.py --list                       # registry, sizes, what is prepared
    prepare_data.py --datasets eval              # test-clean + dev-clean (~0.7 GB)
    prepare_data.py --datasets core              # + train-clean-100/360, LJSpeech,
                                                 #   Hi-Fi TTS, MUSAN, RIRs (~85 GB download)
    prepare_data.py --datasets ljspeech,vctk --jobs 8 --delete-archives
    prepare_data.py --metadata-only hifitts      # stream the Hi-Fi TTS manifests only
    prepare_data.py --summary                    # hours per dataset and speaker

Exit status: 0 = done, 1 = a download, checksum or extraction failed.
"""
import argparse
import sys

from fvml import datasets as fd
from fvml import paths


def list_datasets(ctx):
    print(f"data root: {ctx.root}  ({fd.disk_free_gb(ctx.root):.0f} GB free)")
    for d in fd.DATASETS.values():
        size = sum(a.size for a in d.archives) / 1e9
        state = "manifest" if (ctx.root / "manifests" / f"{d.key}.jsonl").is_file() else \
            "extracted" if (ctx.root / ".state" / f"{d.key}.extracted.json").is_file() else "-"
        print(f"  {d.key:28s} {d.group:8s} {size:7.2f} GB  {state:9s}  {d.licence}")


def print_summary(summary):
    total = 0.0
    for key, s in summary.items():
        total += s["hours"]
        subsets = ", ".join(f"{k} {v:.2f} h" for k, v in s["subsets"].items())
        print(f"  {key:28s} {s['hours']:8.2f} h  {s['rows']:7d} rows  {s['speakers']:4d} speakers"
              f"  [{subsets}]")
        if 0 < s["speakers"] <= 12:
            for spk, h in s["speaker_hours"].items():
                print(f"      {spk:24s} {h:7.2f} h")
    print(f"  {'total':28s} {total:8.2f} h")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--datasets", help="eval | core | all | comma-separated keys (see --list)")
    p.add_argument("--list", action="store_true", help="show the registry and what is prepared")
    p.add_argument("--summary", action="store_true", help="recompute manifests/summary.json")
    p.add_argument("--metadata-only", metavar="hifitts",
                   help="stream only Hi-Fi TTS's leading manifests (for eval_lists/targets.tsv)")
    p.add_argument("--data-root", help="default $FLUXVOICE_DATA or <repo>/data")
    p.add_argument("--jobs", type=int, default=None, help="resampling processes (default: half the cores)")
    p.add_argument("--hifitts-speakers", default=",".join(fd.HIFITTS_DEFAULT),
                   help=f"Hi-Fi TTS reader ids to extract (default {','.join(fd.HIFITTS_DEFAULT)})")
    p.add_argument("--delete-archives", action="store_true", help="remove archives after extraction")
    p.add_argument("--force-manifests", action="store_true", help="rebuild manifests even if up to date")
    args = p.parse_args()

    ctx = fd.Context(root=paths.resolve(args.data_root) if args.data_root else paths.data_root(),
                     hifitts_speakers=tuple(s.strip() for s in args.hifitts_speakers.split(",") if s.strip()),
                     delete_archives=args.delete_archives, force_manifests=args.force_manifests)
    if args.jobs:
        ctx.jobs = args.jobs
    ctx.root.mkdir(parents=True, exist_ok=True)
    sys.stdout.reconfigure(line_buffering=True)

    if args.list:
        list_datasets(ctx)
        return 0
    try:
        if args.metadata_only:
            if args.metadata_only != "hifitts":
                p.error("--metadata-only supports hifitts")
            archive = fd.DATASETS["hifitts"].archives[0]
            fd.stream_tar_metadata(ctx, archive, "hi_fi_tts_v0/audio/", ctx.raw)
            return 0
        if args.datasets:
            todo = fd.select(args.datasets)
            need = sum(a.size for d in todo for a in d.archives) / 1e9
            print(f"data root: {ctx.root}  ({fd.disk_free_gb(ctx.root):.0f} GB free; "
                  f"these archives total {need:.1f} GB)")
            for d in todo:
                fd.prepare(ctx, d)
        elif not args.summary:
            p.error("nothing to do: give --datasets, --list, --summary or --metadata-only")
    except fd.PrepareError as e:
        print(f"prepare_data: {e}", file=sys.stderr)
        return 1
    print("summary (all manifests):")
    print_summary(fd.summarize(ctx))
    return 0


if __name__ == "__main__":
    sys.exit(main())
