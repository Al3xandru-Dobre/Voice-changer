#!/usr/bin/env python3
"""Streaming equivalence of the content encoder: step() vs forward(), per head.

Milestone 2 / Stage 04 definition of done: on 60 s of concatenated dev-clean,
running step() frames_per_step frames at a time (1, 2 and 4) must match one
forward() over the whole signal with max |delta logit| <= 1e-5 and argmax
agreement >= 99.9 % for every head (latents are reported too). float32 on CPU,
compared with fvml.parity.compare.

  stream_check.py                                  # random-init model
  stream_check.py --checkpoint checkpoints/<run>/final.pt --seconds 60

Exit status: 0 = every head within tolerance, 1 = a head failed or inputs missing.
"""
import argparse
import json
import sys

import numpy as np
import torch

from fvml import audio, parity, paths, units
from fvml import content_encoder as ce

MAX_ABS = 1e-5
MIN_AGREEMENT = 0.999


def stream(model, wav, fps):
    """Run step() over `wav` (1-D tensor, multiple of fps*320) -> outputs like forward()."""
    hop = fps * ce.FRAME_SAMPLES
    state = model.init_state(1)
    logits = [[] for _ in model.heads]
    latents = []
    for i in range(0, wav.shape[-1], hop):
        out, state = model.step(wav[None, i:i + hop], state)
        for h, lg in enumerate(out["logits"]):
            logits[h].append(lg)
        latents.append(out["latents"])
    return {"logits": [torch.cat(lg, 1) for lg in logits], "latents": torch.cat(latents, 1)}


@torch.inference_mode()
def check(model, wav, fps_list=(1, 2, 4)):
    """Per fps and head: max |delta|, argmax agreement, pass/fail."""
    model = model.eval().float().cpu()
    wav = torch.as_tensor(np.asarray(wav, np.float32))
    hop = max(fps_list) * ce.FRAME_SAMPLES
    wav = wav[:wav.shape[-1] // hop * hop]
    ref = model(wav[None])
    report = {"seconds": wav.shape[-1] / ce.SAMPLE_RATE, "frames": int(wav.shape[-1] // ce.FRAME_SAMPLES),
              "fps": {}, "pass": True}
    for fps in fps_list:
        got = stream(model, wav, fps)
        entry = {}
        for h, L in enumerate(model.cfg.lookahead):
            r = parity.compare(ref["logits"][h][0].numpy(), got["logits"][h][0].numpy(),
                               atol=MAX_ABS, argmax_axis=-1, steps=False)
            ok = r.max_abs <= MAX_ABS and r.argmax_agreement >= MIN_AGREEMENT
            entry[f"L{L}"] = {"max_abs": r.max_abs, "argmax_agreement": r.argmax_agreement,
                              "snr_db": r.snr_db, "pass": ok}
            report["pass"] &= ok
        r = parity.compare(ref["latents"][0].numpy(), got["latents"][0].numpy(), steps=False)
        entry["latents"] = {"max_abs": r.max_abs, "snr_db": r.snr_db}
        report["fps"][fps] = entry
    return report


def dev_clean_paths(manifest):
    rows, _ = units.read_manifests([manifest])
    return [paths.resolve(r["path"]) for r in rows]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.splitlines()[2:]))
    p.add_argument("--checkpoint", help="train_content_encoder.py checkpoint (default random init)")
    p.add_argument("--manifest", default=str(paths.data_root() / "manifests" / "librispeech-dev-clean.jsonl"))
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--fps", default="1,2,4")
    p.add_argument("--seed", type=int, default=0, help="random-init seed")
    p.add_argument("--threads", type=int, default=0, help="torch threads (0 = default)")
    p.add_argument("--out", help="write the JSON report here")
    args = p.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    try:
        wav = audio.concat(dev_clean_paths(args.manifest), args.seconds)
    except (FileNotFoundError, ValueError) as e:
        print(f"[stream_check] error: {e}", file=sys.stderr)
        return 1
    if args.checkpoint:
        model = ce.ContentEncoder.from_checkpoint(args.checkpoint)[0]
    else:
        torch.manual_seed(args.seed)
        model = ce.ContentEncoder()
    report = check(model, wav, [int(f) for f in args.fps.split(",") if f.strip()])
    report["checkpoint"] = args.checkpoint
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    for fps, entry in report["fps"].items():
        for head, r in entry.items():
            if head != "latents":
                print(f"fps {fps} {head}: max|d| {r['max_abs']:.2e}  argmax {r['argmax_agreement']:.5f}"
                      f"  {'ok' if r['pass'] else 'FAIL'}", file=sys.stderr)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
