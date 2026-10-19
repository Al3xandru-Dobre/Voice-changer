#!/usr/bin/env python3
"""Streaming equivalence of the content encoder or unit vocoder: step() vs forward().

Milestone 2 / Stage 04 (encoder): on 60 s of concatenated dev-clean, running
step() frames_per_step frames at a time (1, 2 and 4) must match one forward()
over the whole signal with max |delta logit| <= 1e-5 and argmax agreement
>= 99.9 % for every head.

Milestone 2 / Stage 05 (vocoder): on >= 30 s of speech, running UnitVocoder.step()
frames_per_step frames at a time (1, 2 and 4) must match offline forward() with
max |delta| <= 1e-4 and no chunk-boundary step larger than the reference signal's
own maximum sample-to-sample slope.

  stream_check.py                                         # random-init encoder
  stream_check.py --role vocoder --seconds 30             # random-init vocoder
  stream_check.py --checkpoint checkpoints/<run>/final.pt # auto-detects role

Exit status: 0 = within tolerance, 1 = failed or inputs missing.
"""
import argparse
import json
import sys

import numpy as np
import torch

from fvml import audio, features, parity, paths, units, vocoder
from fvml import content_encoder as ce

MAX_ABS = 1e-5
MIN_AGREEMENT = 0.999
VOCODER_MAX_ABS = 1e-4
VOCODER_MIN_SNR_DB = 90.0


def stream(model, wav, fps):
    """Run ContentEncoder.step() over `wav` (1-D tensor, multiple of fps*320) -> outputs like forward()."""
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


def stream_vocoder(model, u, logf0, voicing, aperiodicity, energy, spk_emb, fps):
    """Run UnitVocoder.step() over (u, cond) `fps` content frames at a time."""
    rep = model.cfg.unit_repeat
    cond_step = fps * rep
    state = model.init_state(1)
    chunks = []
    for i in range(0, u.shape[1], fps):
        j = i * rep
        y, state = model.step(
            u[:, i:i + fps],
            logf0[:, j:j + cond_step],
            voicing[:, j:j + cond_step],
            aperiodicity[:, j:j + cond_step],
            energy[:, j:j + cond_step],
            spk_emb,
            state,
        )
        chunks.append(y)
    return torch.cat(chunks, dim=-1)


def max_boundary_step(signal, hop):
    """Largest |x[k*hop] - x[k*hop - 1]| across chunk boundaries."""
    x = np.asarray(signal, dtype=np.float64)
    if x.shape[-1] <= hop:
        return 0.0
    idx = np.arange(hop, x.shape[-1], hop)
    return float(np.max(np.abs(x[..., idx] - x[..., idx - 1])))


@torch.inference_mode()
def check(model, wav, fps_list=(1, 2, 4)):
    """Encoder check: per fps and head, max |delta|, argmax agreement, pass/fail."""
    model = model.eval().float().cpu()
    wav = torch.as_tensor(np.asarray(wav, np.float32))
    hop = max(fps_list) * ce.FRAME_SAMPLES
    wav = wav[:wav.shape[-1] // hop * hop]
    ref = model(wav[None])
    report = {"role": "encoder", "seconds": wav.shape[-1] / ce.SAMPLE_RATE,
              "frames": int(wav.shape[-1] // ce.FRAME_SAMPLES),
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


@torch.inference_mode()
def check_vocoder(model, wav, fps_list=(1, 2, 4), seed=0):
    """Vocoder check (Stage 05 DoD): step() vs forward() over `wav`, checking
    max |delta| <= 1e-4, SNR >= 90 dB, non-silent output, and boundary slope."""
    model = model.eval().float().cpu()
    wav_np = np.asarray(wav, dtype=np.float32)
    fs = model.cfg.frame_samples
    rep = model.cfg.unit_repeat
    hop_max = max(fps_list) * fs
    n_samples = len(wav_np) // hop_max * hop_max
    if n_samples == 0:
        raise ValueError(f"audio shorter than one max hop ({hop_max} samples)")
    wav_np = wav_np[:n_samples]
    n_u = n_samples // fs
    n_c = n_u * rep

    feats = features.extract(wav_np)[:n_c]
    logf0 = torch.from_numpy(np.ascontiguousarray(feats[:, 4], np.float32))[None]
    voicing = torch.from_numpy(np.ascontiguousarray(feats[:, 2], np.float32))[None]
    aperiodicity = torch.from_numpy(np.ascontiguousarray(feats[:, 1], np.float32))[None]
    energy = torch.from_numpy(np.ascontiguousarray(feats[:, 3], np.float32))[None]

    rng = np.random.default_rng(seed)
    if model.cfg.content_kind == "discrete":
        u = torch.from_numpy(rng.integers(0, model.cfg.vocab, size=(1, n_u), dtype=np.int64))
    else:
        u = torch.from_numpy(rng.standard_normal((1, n_u, model.cfg.content_dim)).astype(np.float32))
    spk_emb = model.speaker_table.weight[:1].detach().clone()

    ref = model(u, logf0, voicing, aperiodicity, energy, spk_emb)[0].numpy()
    ref_rms = float(np.sqrt(np.mean(np.square(ref, dtype=np.float64))))
    ref_peak = float(np.max(np.abs(ref)))
    report = {
        "role": "vocoder",
        "seconds": n_samples / model.cfg.sample_rate,
        "frames": int(n_u),
        "cond_frames": int(n_c),
        "ref_rms": ref_rms,
        "ref_peak": ref_peak,
        "fps": {},
        "pass": True,
    }
    for fps in fps_list:
        got = stream_vocoder(model, u, logf0, voicing, aperiodicity, energy, spk_emb, fps)[0].numpy()
        r = parity.compare(ref, got, atol=VOCODER_MAX_ABS, steps=True)
        b_step = max_boundary_step(got, fps * fs)
        ref_step = float(r.max_step_ref or 0.0)
        # Scale-aware slack so a quiet random-init signal cannot pass with a boundary jump.
        slope_slack = max(1e-3 * ref_step, 1e-6 * max(ref_peak, 1e-3))
        slope_ok = b_step <= ref_step + slope_slack
        snr_ok = bool(r.snr_db >= VOCODER_MIN_SNR_DB)
        abs_ok = bool(r.max_abs <= VOCODER_MAX_ABS * max(1.0, ref_rms)) and bool(r.max_abs <= VOCODER_MAX_ABS)
        ok = bool(abs_ok and snr_ok and slope_ok and ref_peak > 0.0)
        report["fps"][fps] = {
            "max_abs": r.max_abs,
            "mean_abs": r.mean_abs,
            "snr_db": r.snr_db,
            "ref_rms": ref_rms,
            "ref_peak": ref_peak,
            "max_step_ref": r.max_step_ref,
            "max_step_test": r.max_step_test,
            "max_boundary_step": b_step,
            "pass": ok,
        }
        report["pass"] &= ok
    return report


def dev_clean_paths(manifest):
    rows, _ = units.read_manifests([manifest])
    return [paths.resolve(r["path"]) for r in rows]


def _detect_role(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = ckpt.get("config", "{}")
    if isinstance(cfg, str):
        cfg = json.loads(cfg)
    return "vocoder" if "upsample_rates" in cfg else "encoder"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.splitlines()[2:]))
    p.add_argument("--role", choices=("encoder", "vocoder"), default=None,
                   help="model role (auto-detected from --checkpoint; default encoder)")
    p.add_argument("--checkpoint", help="encoder or vocoder checkpoint (default random init)")
    p.add_argument("--manifest", default=str(paths.data_root() / "manifests" / "librispeech-dev-clean.jsonl"))
    p.add_argument("--seconds", type=float, default=None,
                   help="stream duration in seconds (default 60 for encoder, 30 for vocoder)")
    p.add_argument("--fps", default="1,2,4")
    p.add_argument("--seed", type=int, default=0, help="random-init seed")
    p.add_argument("--threads", type=int, default=0, help="torch threads (0 = default)")
    p.add_argument("--out", help="write the JSON report here")
    args = p.parse_args(argv)
    for stream_io in (sys.stdout, sys.stderr):
        if hasattr(stream_io, "reconfigure"):
            stream_io.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    if args.threads:
        torch.set_num_threads(args.threads)
    role = args.role or (_detect_role(args.checkpoint) if args.checkpoint else "encoder")
    seconds = args.seconds if args.seconds is not None else (30.0 if role == "vocoder" else 60.0)
    try:
        wav = audio.concat(dev_clean_paths(args.manifest), seconds)
    except (FileNotFoundError, ValueError) as e:
        print(f"[stream_check] error: {e}", file=sys.stderr)
        return 1
    fps_list = [int(f) for f in args.fps.split(",") if f.strip()]
    if role == "vocoder":
        if args.checkpoint:
            model = vocoder.UnitVocoder.from_checkpoint(args.checkpoint)[0]
        else:
            torch.manual_seed(args.seed)
            model = vocoder.UnitVocoder()
        report = check_vocoder(model, wav, fps_list, seed=args.seed)
    else:
        if args.checkpoint:
            model = ce.ContentEncoder.from_checkpoint(args.checkpoint)[0]
        else:
            torch.manual_seed(args.seed)
            model = ce.ContentEncoder()
        report = check(model, wav, fps_list)
    report["checkpoint"] = args.checkpoint
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    if role == "vocoder":
        for fps, r in report["fps"].items():
            print(f"fps {fps} vocoder: max|d| {r['max_abs']:.2e}  SNR {r['snr_db']:.1f} dB  "
                  f"b_step {r['max_boundary_step']:.4f} <= ref_step {r['max_step_ref']:.4f}  "
                  f"{'ok' if r['pass'] else 'FAIL'}", file=sys.stderr)
    else:
        for fps, entry in report["fps"].items():
            for head, r in entry.items():
                if head != "latents":
                    print(f"fps {fps} {head}: max|d| {r['max_abs']:.2e}  argmax {r['argmax_agreement']:.5f}"
                          f"  {'ok' if r['pass'] else 'FAIL'}", file=sys.stderr)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
