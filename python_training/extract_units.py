#!/usr/bin/env python3
"""Waveform -> discrete unit ids for every Stage 04 content-encoder candidate.

Milestone 2 / Stage 04 (Documents/Milestone 2/04_content_encoder_candidates.md).

Tokenizers (all at 16 kHz input; see fvml/units.py for the on-disk format):
  teacher-km100   HuBERT-Base layer 7 + k-means(100); offline, non-causal; 50 Hz
  student-L0/1/2  causal content encoder head with lookahead L frames; 50 Hz
  mimi-q0         Kyutai Mimi semantic codebook 0; causal; 12.5 Hz (kill test)

Output: <out-dir>/<utt_id>.npy (int16 ids, frame-aligned), manifest.jsonl and
units_meta.json (frame rate, lookahead, vocab, label space and sources, so Stage
05 cannot mix incompatible streams: fvml.units.check_compatible). Students can
also write <utt_id>.logits.npy (float16, pre-softmax) for the soft-unit fallback.
Rerunning into the same directory skips utterances already done and refuses a
directory written by an incompatible tokenizer.

Pitch is not extracted here: causal pitch/energy belongs to fvml/features.py
(Stage 03). The old 12.5 Hz / Mimi-default / RMVPE-SwiftF0 stub is gone; those
choices are superseded or non-causal (MEMORY.md §1).

Usage:
  # 1. fit the teacher's k-means (writes km100_centroids.npy + km100_meta.json)
  extract_units.py --fit-kmeans --manifest librispeech-train-clean-100.jsonl \\
      --out-dir units/ [--hours 100] [--max-frames 500000] [--seed 0]
  # 2. label corpora with the teacher
  extract_units.py --tokenizer teacher-km100 --centroids units/ \\
      --manifest librispeech-train-clean-100.jsonl --out-dir data/units/teacher-km100/tc100
  # 3. a student head or Mimi
  extract_units.py --tokenizer student-L1 --checkpoint checkpoints/<run>/final.pt ...
  extract_units.py --tokenizer mimi-q0 --manifest librispeech-dev-clean.jsonl ...

Manifest paths may be absolute or relative to the data root (data/manifests/...
is tried for bare names). Exit status: 0 = done, 1 = missing inputs or a
mismatched output directory.
"""
import argparse
import json
import pathlib
import random
import sys
import time

import numpy as np

from fvml import audio, paths, units


def fit_kmeans(args, rows, manifests, log=print):
    from fvml import teacher
    from fvml.content_encoder import pad_to_frames
    rng = random.Random(args.seed)
    order = rows[:]
    rng.shuffle(order)
    if args.hours:
        budget, picked = args.hours * 3600, []
        for r in order:
            if budget <= 0:
                break
            picked.append(r)
            budget -= r.get("duration", 0)
        order = picked
    if args.limit:
        order = order[:args.limit]
    model = teacher.HubertTeacher(args.device)
    t0 = time.monotonic()

    def feats():
        for i, r in enumerate(order):
            yield model.features(pad_to_frames(audio.load(paths.resolve(r["path"]))))
            if (i + 1) % 200 == 0:
                log(f"  {i + 1}/{len(order)} utterances, {time.monotonic() - t0:.0f} s")

    centroids, info = teacher.fit_kmeans(feats(), k=args.k, max_frames=args.max_frames,
                                         seed=args.seed, log=log)
    info["hours_requested"] = args.hours
    info["hours_used"] = round(sum(r.get("duration", 0) for r in order) / 3600, 3)
    meta = teacher.save_kmeans(args.out_dir, centroids, info, model.info(), manifests)
    log(f"[kmeans] inertia/frame {meta['inertia_per_frame']:.2f}; wrote "
        f"{pathlib.Path(args.out_dir) / teacher.CENTROIDS_FILE} (sha256 {meta['centroids_sha256'][:12]})")
    return 0


def extract(args, rows, manifests, log=print):
    out_dir = pathlib.Path(args.out_dir)
    tok = units.make_tokenizer(args.tokenizer, centroids=args.centroids,
                               checkpoint=args.checkpoint, device=args.device)
    meta = {**tok.meta, "manifests": [str(m) for m in manifests], "created": units.now()}
    done = {}
    if (out_dir / units.META_FILE).is_file():
        old, old_rows = units.read_units_dir(out_dir)
        units.check_compatible(old, meta)

        def identity(m):
            s = m.get("source", {})
            return m.get("tokenizer"), s.get("revision"), s.get("checkpoint_sha256")
        if identity(old) != identity(meta):
            raise units.UnitsMismatch(f"{out_dir} holds {old.get('tokenizer')} units from another "
                                      "model/checkpoint; use a new --out-dir")
        done = {r["utt_id"]: r for r in old_rows if (out_dir / r["units"]).is_file()}
        meta["created"] = old.get("created", meta["created"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / units.META_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    todo = rows[:args.limit] if args.limit else rows
    out_rows, t0, frames, skipped = [], time.monotonic(), 0, 0
    for i, r in enumerate(todo):
        if r["utt_id"] in done:
            out_rows.append(done[r["utt_id"]])
            skipped += 1
            continue
        wav = audio.load(paths.resolve(r["path"]))
        ids, logits = tok.encode(wav)
        name = units.safe_name(r["utt_id"])
        np.save(out_dir / f"{name}.npy", ids)
        if args.save_logits and logits is not None:
            np.save(out_dir / f"{name}.logits.npy", logits)
        frames += len(ids)
        out_rows.append({"utt_id": r["utt_id"], "units": f"{name}.npy", "n_frames": int(len(ids)),
                         "path": r["path"], **{k: r[k] for k in units.COPY_FIELDS if k in r}})
        if (i + 1) % 200 == 0:
            el = time.monotonic() - t0
            log(f"  {i + 1}/{len(todo)} utterances, {frames} frames, {el:.0f} s")
            write_manifest(out_dir, out_rows, done)
    write_manifest(out_dir, out_rows, done)
    log(f"[extract_units] {args.tokenizer}: {len(out_rows) - skipped} new + {skipped} existing "
        f"utterances -> {out_dir} ({meta['frame_rate_hz']} Hz, vocab {meta['vocab']}, "
        f"lookahead {meta['lookahead_frames']}) in {time.monotonic() - t0:.0f} s")
    return 0


def write_manifest(out_dir, new_rows, done):
    from fvml.datasets import write_jsonl
    seen = {r["utt_id"] for r in new_rows}
    rows = new_rows + [r for k, r in done.items() if k not in seen]   # keep rows from earlier runs
    write_jsonl(out_dir / units.MANIFEST_FILE, rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.splitlines()[2:]))
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tokenizer", choices=units.TOKENIZERS)
    mode.add_argument("--fit-kmeans", action="store_true",
                      help="fit the teacher's k-means instead of extracting units")
    p.add_argument("--manifest", action="append", required=True,
                   help="JSONL manifest (repeatable or comma-separated)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--centroids", help="teacher-km100: km100_centroids.npy or its directory")
    p.add_argument("--checkpoint", help="student-L*: a train_content_encoder.py checkpoint")
    p.add_argument("--save-logits", action="store_true", help="students: also write float16 logits")
    p.add_argument("--limit", type=int, help="first N utterances only (smoke runs)")
    p.add_argument("--device", default=None, help="default cuda if available, else cpu")
    g = p.add_argument_group("--fit-kmeans")
    g.add_argument("--k", type=int, default=100)
    g.add_argument("--hours", type=float, help="audio to run HuBERT on (random utterances); default all")
    g.add_argument("--max-frames", type=int, default=500_000,
                   help="reservoir size: frames the k-means sees (500k x 768 floats = 1.5 GB)")
    g.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    sys.stdout.reconfigure(line_buffering=True)
    try:
        rows, manifests = units.read_manifests(args.manifest)
        if args.fit_kmeans:
            return fit_kmeans(args, rows, manifests)
        return extract(args, rows, manifests)
    except (FileNotFoundError, ValueError) as e:
        print(f"[extract_units] error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
