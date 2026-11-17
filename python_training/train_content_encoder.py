#!/usr/bin/env python3
"""Distil a unit teacher into the causal 50 Hz content encoder.

Milestone 2 / Stage 04. One trunk, three delay heads (L = 0/1/2 frames), trained
with cross-entropy summed over heads (fvml/content_encoder.py). Labels come from
a teacher units directory written by extract_units.py --tokenizer teacher-km100,
always from the clean audio; only the student's input is augmented
(fvml/augment.py: MUSAN noise, RIR reverb, EQ, gain; each optional).

Milestone 3 / Stage 03 (the dual-output multilingual student): the same network
with --latent-dim 128 over K = 1000 mhubert-vp labels (the vocabulary always
comes from the labels), trained on balanced {en, es, fr, de} speech:
  --train-units       repeatable / comma-separated: one units directory per corpus
  --side src          read the source audio of paired S2UT manifests
  --default-lang en   tag for rows without a `lang` field (LibriSpeech)
  --lang-ratios balanced   every language is drawn equally often, whatever its hours
  --train-subsets / --val-subsets   keep only these manifest subsets
Validation then reports top-1 and top-5 unit accuracy per head and per language
(Stage 03's definition of done: >= 65% / >= 90% in every language).
Labels from the non-commercial mHuBERT-147 teacher are refused without
--include-noncommercial, and the checkpoint stays marked non-commercial.

Data: --train-manifest (JSONL audio rows) joined by utt_id with --train-units.
Each example is a random 2 s crop on a 320-sample frame boundary (labels cropped
to the same frames; short utterances are zero-padded with ignored labels).
Utterances are drawn with probability proportional to their length, so every
crop position is about equally likely.

Training: AdamW, linear warmup + cosine decay, bf16 autocast on CUDA, gradient
clipping. TensorBoard logs in <repo>/runs/<run>/, checkpoints in
<repo>/checkpoints/<run>/: last.pt every --ckpt-minutes (and --ckpt-steps),
best.pt by mean validation accuracy, final.pt at the end. --resume auto|PATH
restores model, optimizer, scheduler, step and RNG states; the sampler is
reseeded from the step, so the data order after a resume is new but reproducible.

  train_content_encoder.py --train-manifest librispeech-train-clean-100.jsonl \\
      --train-units data/units/teacher-km100/tc100 \\
      --val-manifest librispeech-dev-clean.jsonl --val-units data/units/teacher-km100/dev \\
      --musan-manifest musan.jsonl --rir-manifest rirs.jsonl --run-name ce_v1
  train_content_encoder.py ... --run-name ce_v1 --resume auto
  # Milestone 3: multilingual dual-output student (D = 128, K = 1000)
  train_content_encoder.py --latent-dim 128 --lang-ratios balanced --default-lang en \\
      --side src --train-subsets train \\
      --train-manifest librispeech-train-clean-360.jsonl,s2ut/cvss-c-es_en.jsonl,... \\
      --train-units data/units/mhubert/ls360,data/units/mhubert/cvss-c-es_en.src,... \\
      --val-manifest ... --val-units ... --musan-manifest musan.jsonl \\
      --rir-manifest rirs.jsonl --run-name ce_m3

Exit status: 0 = trained, 1 = missing or inconsistent inputs.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")   # inherited by DataLoader workers

import argparse          # noqa: E402
import datetime          # noqa: E402
import json              # noqa: E402
import math              # noqa: E402
import pathlib           # noqa: E402
import random            # noqa: E402
import sys               # noqa: E402
import time              # noqa: E402

import numpy as np       # noqa: E402
import soundfile as sf   # noqa: E402
import torch             # noqa: E402

from fvml import audio, paths, sampling, units         # noqa: E402
from fvml import content_encoder as ce                 # noqa: E402
from fvml.augment import AugmentConfig, Augmenter      # noqa: E402

FS = ce.FRAME_SAMPLES
DOD_TOP1, DOD_TOP5 = 0.65, 0.90      # M3 Stage 03: held-out accuracy in every language


# -- data ---------------------------------------------------------------------
def join_items(manifest_spec, units_dir, log=print, side=None, default_lang=None, subsets=None):
    """Manifest rows x teacher units -> [(audio path, units path, n_frames, lang)],
    labels meta. `units_dir` is one directory or several that share a label
    space; `side` picks the audio of paired S2UT rows; `lang` is None for rows
    that carry no language (and no `default_lang` was given)."""
    rows, _ = units.read_manifests(manifest_spec)
    rows = units.filter_subsets(units.audio_rows(rows, side, default_lang), subsets)
    meta, by_id = units.read_units_dirs(units_dir)
    items, missing, bad = [], 0, 0
    for r in rows:
        found = by_id.get(r["utt_id"])
        if found is None:
            missing += 1
            continue
        udir, u = found
        expect = math.ceil(round(r["duration"] * audio.SAMPLE_RATE) / FS)
        if abs(expect - u["n_frames"]) > 1:      # duration is rounded to 1 ms in manifests
            bad += 1
            continue
        items.append((str(paths.resolve(r["path"])), str(udir / u["units"]),
                      int(u["n_frames"]), r.get("lang")))
    where = ", ".join(units.split_specs(units_dir))
    if missing or bad:
        log(f"[data] {where}: {len(items)} utterances; {missing} without units, "
            f"{bad} with a frame-count mismatch (skipped)")
    if not items:
        raise ValueError(f"no manifest row has units in {where}")
    return items, meta


def read_frames(path, start_frame, n_frames):
    """Samples of frames [start, start + n) of a 16 kHz file, zero-padded at the end."""
    want = n_frames * FS
    info = sf.info(path)
    if info.samplerate == audio.SAMPLE_RATE:
        x, _ = sf.read(path, start=start_frame * FS, frames=want, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
    else:                                         # not expected: manifests point at 16 kHz files
        x = audio.load(path)[start_frame * FS:start_frame * FS + want]
    return np.pad(x, (0, want - len(x))) if len(x) < want else x


class CropDataset(torch.utils.data.Dataset):
    def __init__(self, items, crop_frames, augmenter=None):
        self.items, self.crop, self.aug = items, crop_frames, augmenter
        self.rng = None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        if self.rng is None:                      # per worker; worker seeds follow torch's RNG
            self.rng = np.random.default_rng(torch.initial_seed() % (1 << 63))
        apath, upath, n = self.items[i][:3]
        start = int(self.rng.integers(0, n - self.crop + 1)) if n > self.crop else 0
        labels = np.full(self.crop, ce.IGNORE, np.int64)
        lab = np.load(upath, mmap_mode="r")[start:start + self.crop]
        labels[:len(lab)] = lab
        x = read_frames(apath, start, self.crop)
        if self.aug is not None:
            x = self.aug(x, self.rng)
        return torch.from_numpy(np.ascontiguousarray(x, np.float32)), torch.from_numpy(labels)


class WeightedStream(torch.utils.data.Sampler):
    """Endless indices drawn proportionally to `weights`, seeded per run+step."""

    def __init__(self, weights, seed, chunk=4096):
        self.w = torch.as_tensor(weights, dtype=torch.double)
        self.seed, self.chunk = seed, chunk

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed)
        while True:
            yield from torch.multinomial(self.w, self.chunk, replacement=True, generator=g).tolist()


def worker_init(_):
    torch.set_num_threads(1)


def manifest_paths(spec, subsets=None):
    if not spec:
        return []
    rows, _ = units.read_manifests(spec)
    keep = [r for r in rows if not subsets or r.get("subset") in subsets]
    if not keep:
        raise ValueError(f"{spec}: no rows (subsets {subsets})")
    return [str(paths.resolve(r["path"])) for r in keep]


# -- training -----------------------------------------------------------------
def lr_lambda(warmup, total, floor):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        t = min(1.0, (step - warmup) / max(1, total - warmup))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t))
    return f


def select_val(items, limit):
    """The first `limit` utterances; with several languages, `limit` split
    evenly between them (in manifest order), so no language goes unscored."""
    langs = list(dict.fromkeys(it[3] for it in items))
    if len(langs) < 2 or not limit:
        return items[:limit] if limit else list(items)
    per = max(1, limit // len(langs))
    return [it for lang in langs for it in [x for x in items if x[3] == lang][:per]]


@torch.no_grad()
def validate(model, items, device, limit):
    """Loss, top-1 (`acc`) and top-5 accuracy per head over the validation
    utterances, plus the same per language (`by_lang`) and, per head, whether
    every language meets Stage 03's 65% / 90% floor (`dod`)."""
    model.eval()
    heads = len(model.heads)
    loss_sum, n_utts = np.zeros(heads), 0
    by_lang = {}
    for apath, upath, n, lang in select_val(items, limit):
        x = torch.from_numpy(read_frames(apath, 0, n))[None].to(device)
        lab = torch.from_numpy(np.load(upath).astype(np.int64))[None].to(device)
        out = model(x)
        _, per = ce.distill_loss(out["logits"], lab, model.cfg.lookahead)
        loss_sum += [float(p) for p in per]
        n_utts += 1
        tally = by_lang.setdefault(lang, {"top1": np.zeros(heads), "top5": np.zeros(heads),
                                          "frames": np.zeros(heads), "utts": 0})
        tally["utts"] += 1
        top1 = ce.head_accuracy(out["logits"], lab, model.cfg.lookahead)
        top5 = ce.head_topk_accuracy(out["logits"], lab, model.cfg.lookahead, k=5)
        for h in range(heads):
            tally["top1"][h] += top1[h][0]
            tally["top5"][h] += top5[h][0]
            tally["frames"][h] += top1[h][1]
    model.train()
    total = {k: sum(t[k] for t in by_lang.values()) if by_lang else np.zeros(heads)
             for k in ("top1", "top5", "frames")}
    acc = (total["top1"] / np.maximum(total["frames"], 1)).tolist()
    result = {"loss": (loss_sum / max(n_utts, 1)).tolist(), "acc": acc,
              "acc_mean": float(np.mean(acc)),
              "top5": (total["top5"] / np.maximum(total["frames"], 1)).tolist(),
              "by_lang": {}}
    for lang, t in by_lang.items():
        if lang is not None:
            result["by_lang"][lang] = {
                "acc": (t["top1"] / np.maximum(t["frames"], 1)).tolist(),
                "top5": (t["top5"] / np.maximum(t["frames"], 1)).tolist(),
                "frames": int(t["frames"].max()), "utts": int(t["utts"])}
    result["dod"] = {
        f"L{L}": bool(result["by_lang"]) and all(
            v["acc"][h] >= DOD_TOP1 and v["top5"][h] >= DOD_TOP5
            for v in result["by_lang"].values())
        for h, L in enumerate(model.cfg.lookahead)}
    return result


def rng_state():
    np_state = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "numpy": [np_state[0], torch.from_numpy(np_state[1].astype(np.int64)), *np_state[2:]],
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def set_rng_state(s):
    random.setstate(s["python"])
    torch.set_rng_state(s["torch"])
    n = s["numpy"]
    np.random.set_state((n[0], n[1].numpy().astype(np.uint32), *n[2:]))
    if s["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def save_checkpoint(path, model, opt, sched, step, args, labels_meta, extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(), "step": step,
                "config": json.dumps(model.cfg.to_dict()), "labels_meta": json.dumps(labels_meta),
                "args": json.dumps(vars(args)), "rng": rng_state(), **extra}, tmp)
    tmp.replace(path)


def train(args, log=print):
    device = torch.device(args.device)
    run = args.run_name
    ckpt_dir = pathlib.Path(args.ckpt_dir) / run
    resume = None
    if args.resume:
        resume = ckpt_dir / "last.pt" if args.resume == "auto" else pathlib.Path(args.resume)
        if not resume.is_file():
            if args.resume != "auto":
                raise FileNotFoundError(f"{resume}: no checkpoint to resume")
            resume = None

    train_items, labels_meta = join_items(args.train_manifest, args.train_units, log,
                                          side=args.side, default_lang=args.default_lang,
                                          subsets=args.train_subsets)
    units.check_license(labels_meta, args.include_noncommercial)
    if not str(labels_meta.get("tokenizer", "")).startswith("teacher-"):
        log(f"[data] warning: training on {labels_meta.get('tokenizer')} labels, not a teacher's")
    val_items = []
    if args.val_manifest:
        val_items, val_meta = join_items(args.val_manifest, args.val_units, log, side=args.side,
                                         default_lang=args.default_lang,
                                         subsets=args.val_subsets)
        units.check_compatible(labels_meta, val_meta)
    keep = ["tokenizer", "label_space", "vocab", "frame_rate_hz", "source"]
    if labels_meta.get("noncommercial"):         # the taint follows the weights; M2 metadata is unchanged
        keep += ["license", "noncommercial"]
    labels_meta = {k: labels_meta.get(k) for k in keep}

    cfg = ce.EncoderConfig(vocab=int(labels_meta["vocab"]),
                           **({"latent_dim": args.latent_dim} if args.latent_dim else {}),
                           **({"channels": tuple(int(c) for c in args.channels.split(","))}
                              if args.channels else {}))
    model = ce.ContentEncoder(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98),
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(args.warmup, args.steps, args.lr_floor))
    step, best, elapsed0 = 0, -1.0, 0.0
    if resume:
        ck = torch.load(resume, map_location="cpu", weights_only=True)
        if json.loads(ck["config"]) != cfg.to_dict():
            raise ValueError(f"{resume}: model config differs from this run's")
        if json.loads(ck["labels_meta"]).get("label_space") != labels_meta["label_space"]:
            raise ValueError(f"{resume}: trained on another label space")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        set_rng_state(ck["rng"])
        step, best, elapsed0 = int(ck["step"]), float(ck.get("best_val", -1.0)), float(ck.get("elapsed", 0))
        log(f"[train] resumed {resume} at step {step}")
    else:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    summary = model.summary()
    log(f"[train] {run}: {summary['params'] / 1e6:.2f}M params, {summary['macs_per_second'] / 1e9:.2f} "
        f"GMAC/s, D={cfg.latent_dim}, vocab {cfg.vocab}, {len(train_items)} train / "
        f"{len(val_items)} val utterances, device {device}")
    if cfg.vocab >= ce.MULTILINGUAL_VOCAB and summary["params"] > ce.MAX_PARAMS_MULTILINGUAL:
        log(f"[train] warning: {summary['params'] / 1e6:.2f}M params exceeds Stage 03's "
            f"{ce.MAX_PARAMS_MULTILINGUAL / 1e6:.0f}M budget")

    augmenter = None
    noise = manifest_paths(args.musan_manifest, set(args.musan_subsets.split(",")))
    rirs = manifest_paths(args.rir_manifest, set(args.rir_subsets.split(",")))
    if not args.no_augment:
        acfg = AugmentConfig(noise_p=args.noise_p, rir_p=args.rir_p, eq_p=args.eq_p,
                             gain_p=args.gain_p)
        augmenter = Augmenter(acfg, noise, rirs)
        log(f"[aug] noise files {len(noise)} (p {acfg.noise_p if noise else 0}), RIRs {len(rirs)} "
            f"(p {acfg.rir_p if rirs else 0}), EQ p {acfg.eq_p}, gain p {acfg.gain_p}")
    crop = int(round(args.crop_seconds * ce.FRAME_RATE_HZ))
    ds = CropDataset(train_items, crop, augmenter)
    weights = [max(it[2], crop) for it in train_items]
    ratios = sampling.parse_ratios(args.lang_ratios)
    if ratios:
        weights, shares = sampling.language_weights(weights, [it[3] for it in train_items], ratios)
        log("[data] language sampling: " + ", ".join(
            f"{lang} {100 * s['share']:.0f}% of draws ({s['items']} utterances)"
            for lang, s in shares.items()))
    sampler = WeightedStream(weights, seed=args.seed * 1_000_003 + step)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
        pin_memory=device.type == "cuda", drop_last=True, worker_init_fn=worker_init,
        persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers > 0 else None)

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(pathlib.Path(args.runs_dir) / run), purge_step=step if resume else None)
    if not resume:
        writer.add_text("args", json.dumps(vars(args), indent=2))
        writer.add_text("model", json.dumps(summary, indent=2))
    use_bf16 = device.type == "cuda" and not args.fp32
    t_start, t_ckpt, t_log = time.monotonic(), time.monotonic(), time.monotonic()
    acc_c = np.zeros(len(cfg.lookahead))
    acc_n = np.zeros(len(cfg.lookahead))
    loss_sum, loss_n = np.zeros(len(cfg.lookahead)), 0
    last_val = {}

    def checkpoint(name):
        save_checkpoint(ckpt_dir / name, model, opt, sched, step, args, labels_meta,
                        {"best_val": best, "run_name": run, "val": json.dumps(last_val),
                         "elapsed": elapsed0 + time.monotonic() - t_start})

    model.train()
    it = iter(loader)
    while step < args.steps:
        wav, lab = next(it)
        wav = wav.to(device, non_blocking=True)
        lab = lab.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            out = model(wav)
        loss, per = ce.distill_loss(out["logits"], lab, cfg.lookahead)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip))
        opt.step()
        sched.step()
        step += 1
        loss_sum += [float(p.detach()) for p in per]
        loss_n += 1
        for h, (c, k) in enumerate(ce.head_accuracy(out["logits"], lab, cfg.lookahead)):
            acc_c[h] += c
            acc_n[h] += k
        if step % args.log_every == 0 or step == args.steps:
            dt = time.monotonic() - t_log
            t_log = time.monotonic()
            for h, L in enumerate(cfg.lookahead):
                writer.add_scalar(f"train/loss_L{L}", loss_sum[h] / loss_n, step)
                writer.add_scalar(f"train/acc_L{L}", acc_c[h] / max(acc_n[h], 1), step)
            writer.add_scalar("train/lr", sched.get_last_lr()[0], step)
            writer.add_scalar("train/grad_norm", gnorm, step)
            writer.add_scalar("train/steps_per_s", loss_n / max(dt, 1e-9), step)
            log(f"step {step:7d}  loss " + " ".join(f"{v:.3f}" for v in loss_sum / loss_n)
                + "  acc " + " ".join(f"{c / max(n, 1):.3f}" for c, n in zip(acc_c, acc_n))
                + f"  lr {sched.get_last_lr()[0]:.2e}  gn {gnorm:.2f}  {loss_n / max(dt, 1e-9):.2f} it/s")
            acc_c[:] = 0
            acc_n[:] = 0
            loss_sum[:] = 0
            loss_n = 0
        if val_items and (step % args.val_every == 0 or step == args.steps):
            v = validate(model, val_items, device, args.val_limit)
            for L, a, lo in zip(cfg.lookahead, v["acc"], v["loss"]):
                writer.add_scalar(f"val/acc_L{L}", a, step)
                writer.add_scalar(f"val/loss_L{L}", lo, step)
            log(f"[val] step {step}: loss " + " ".join(f"{lo:.3f}" for lo in v["loss"])
                + "  acc " + " ".join(f"{a:.3f}" for a in v["acc"]))
            for L, t5 in zip(cfg.lookahead, v["top5"]):
                writer.add_scalar(f"val/top5_L{L}", t5, step)
            for lang, s in v["by_lang"].items():
                for L, a, t5 in zip(cfg.lookahead, s["acc"], s["top5"]):
                    writer.add_scalar(f"val_{lang}/acc_L{L}", a, step)
                    writer.add_scalar(f"val_{lang}/top5_L{L}", t5, step)
                log(f"[val]   {lang}: top-1 " + " ".join(f"{a:.3f}" for a in s["acc"])
                    + "  top-5 " + " ".join(f"{a:.3f}" for a in s["top5"])
                    + f"  ({s['utts']} utterances)")
            last_val = {"step": step, **v}
            if v["acc_mean"] > best:
                best = v["acc_mean"]
                checkpoint("best.pt")
        due = time.monotonic() - t_ckpt >= args.ckpt_minutes * 60 or \
            (args.ckpt_steps and step % args.ckpt_steps == 0)
        if due and step < args.steps:
            checkpoint("last.pt")
            t_ckpt = time.monotonic()
            log(f"[ckpt] step {step} -> {ckpt_dir / 'last.pt'}")
    checkpoint("last.pt")
    checkpoint("final.pt")
    writer.close()
    peak = f", peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"         if device.type == "cuda" else ""
    log(f"[train] done at step {step} -> {ckpt_dir / 'final.pt'}{peak}")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.splitlines()[2:]))
    p.add_argument("--train-manifest", action="append", required=True)
    p.add_argument("--train-units", action="append", required=True,
                   help="teacher units directory (repeatable / comma-separated: one per corpus)")
    p.add_argument("--val-manifest", action="append")
    p.add_argument("--val-units", action="append")
    p.add_argument("--side", choices=units.SIDES,
                   help="paired S2UT manifests: which audio to train on (src for the student)")
    p.add_argument("--default-lang", help="language of rows without a `lang` field (e.g. en)")
    p.add_argument("--train-subsets", help="keep only these manifest subsets (comma-separated)")
    p.add_argument("--val-subsets", help="keep only these manifest subsets for validation")
    p.add_argument("--lang-ratios", help="share of draws per language: 'balanced' (25%% each of "
                   "en/es/fr/de), 'en,es,fr,de' or 'en=0.4,es=0.2,...' (default: by length)")
    p.add_argument("--latent-dim", type=int,
                   help=f"bottleneck / `latents` width D (default {ce.EncoderConfig().latent_dim}; "
                   f"{ce.MULTILINGUAL_LATENT_DIM} for the M3 dual-output student)")
    p.add_argument("--include-noncommercial", action="store_true",
                   help="allow labels from the CC-BY-NC-SA mHuBERT-147 teacher (research only)")
    p.add_argument("--val-limit", type=int, default=200, help="validation utterances")
    p.add_argument("--val-every", type=int, default=2000)
    p.add_argument("--musan-manifest", action="append", help="MUSAN manifest (noise augmentation)")
    p.add_argument("--musan-subsets", default="noise,music")
    p.add_argument("--rir-manifest", action="append", help="RIRS_NOISES manifest (reverb augmentation)")
    p.add_argument("--rir-subsets", default="real_rir,simulated_rir")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--noise-p", type=float, default=0.6)
    p.add_argument("--rir-p", type=float, default=0.4)
    p.add_argument("--eq-p", type=float, default=0.5)
    p.add_argument("--gain-p", type=float, default=1.0)
    p.add_argument("--channels", help="trunk widths, stem then one per block (default "
                   f"{','.join(map(str, ce.EncoderConfig().channels))}; e.g. 16,32,64,128,256 is ~4x cheaper)")
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--crop-seconds", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lr-floor", type=float, default=0.02, help="final LR as a fraction of --lr")
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--fp32", action="store_true", help="disable bf16 autocast")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--run-name", default=None, help="default ce_<date>_<time>")
    p.add_argument("--runs-dir", default=str(paths.REPO_ROOT / "runs"))
    p.add_argument("--ckpt-dir", default=str(paths.REPO_ROOT / "checkpoints"))
    p.add_argument("--ckpt-minutes", type=float, default=30.0)
    p.add_argument("--ckpt-steps", type=int, default=0, help="also checkpoint every N steps (0 = off)")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--resume", help="'auto' (<ckpt-dir>/<run>/last.pt if present) or a checkpoint path")
    args = p.parse_args(argv)
    if bool(args.val_manifest) != bool(args.val_units):
        p.error("--val-manifest and --val-units go together")
    if args.resume and args.resume != "auto" and not args.run_name:
        ck = torch.load(args.resume, map_location="cpu", weights_only=True)
        args.run_name = ck.get("run_name")
    if not args.run_name:
        if args.resume == "auto":
            p.error("--resume auto needs --run-name")
        args.run_name = "ce_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sys.stdout.reconfigure(line_buffering=True)
    try:
        return train(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"[train] error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
