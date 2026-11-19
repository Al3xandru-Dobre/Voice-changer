#!/usr/bin/env python3
"""Train the bidirectional En<->{Es,Fr,De} causal S2UT translator.

Milestone 3 / Stage 05 (Documents/Milestone 3/05_causal_s2ut_translator_policy_gate.md).
Model, target layouts and the joint loss live in fvml/translator.py:

    L = L_unit_CE + 0.3 L_ASR_CTC + 0.3 L_S2TT_CTC + 0.1 L_eos_BCE + 0.1 L_readiness

Data (all written by earlier stages; nothing is decoded from audio here):
  --manifest      paired S2UT manifests (prepare_s2ut_data.py): id, src_lang,
                  tgt_lang, src_text, tgt_text, split
  --src-latents   units directories written by
                  `extract_units.py --tokenizer student-L<n> --save-latents --side src`:
                  <id>.latents.npy, the Stage 1 student's [T, 128] latents at 50 Hz
  --tgt-units     units directories written by
                  `extract_units.py --tokenizer teacher-mhubert-km1000 --side tgt`:
                  <id>.npy, the target audio's unit ids at 50 Hz
  --spm-dir       spm_{en,es,fr,de}.model (prepare_s2ut_data.py --train-spm); the
                  CTC blank is each model's <pad> id, so a head is 4 * V_spm wide
  --silence-latents  [T, 128] .npy of the student's latents on silence / room
                  tone: the bank the silence tails are cut from

Curriculum (one checkpoint for every k the Wait-k slider can ask for): the first
30 % of the updates see the whole source (full-context cross-attention); then,
over a short ramp, batches switch to simultaneous masks with k ~ U{1..8} drawn
per batch.

Silence tails: 30 % of the utterances get 400-800 ms of silence latents
appended. The policy keeps writing through that tail, so the target layout
makes the decoder finish its units, emit <EOS> and then stay on <PAD>.

Sampling: a direction is drawn with probability (n_d / N)^(1/T) (T = 5 flattens
the 6 directions without ignoring their sizes), then an utterance uniformly
inside it. Batches are length-bucketed and capped by padded source chunks
(--max-chunks), so memory is bounded by the cap, not by the longest utterance.

Training: AdamW, linear warm-up + cosine decay, bf16 autocast on CUDA, gradient
accumulation (--accum micro-batches per update), gradient clipping. TensorBoard
logs in <repo>/runs/<run>/, checkpoints in <repo>/checkpoints/<run>/: last.pt
every --ckpt-minutes (and --ckpt-steps), best.pt by validation unit accuracy at
k = 3, final.pt at the end. --resume auto|PATH restores model, optimizer,
scheduler, step and RNG states; samplers and the curriculum are seeded from the
step, so the data order after a resume is new but reproducible.

  train_s2ut.py --manifest s2ut/cvss-c-es_en.jsonl,... --src-latents data/units/student/... \\
      --tgt-units data/units/mhubert/... --spm-dir data/spm \\
      --silence-latents data/units/student/silence.latents.npy --run-name s2ut_v1
  train_s2ut.py ... --run-name s2ut_v1 --resume auto
  train_s2ut.py ... --unit-mode reduced_dur --run-name s2ut_reduced_v1
  train_s2ut.py ... --ctc-weight 0 --steps 20000 --run-name s2ut_de_en_noctc   # ablation
  train_s2ut.py --smoke <dir> --steps 60      # synthetic data, tiny model, CPU

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
import torch             # noqa: E402

from fvml import asr, paths                    # noqa: E402
from fvml import translator as tr              # noqa: E402

WARMUP_FRACTION = 0.30          # full-context share of the updates
RAMP_FRACTION = 0.05            # then this share ramps full-context -> Wait-k
K_RANGE = (1, 8)
SILENCE_TAIL_P = 0.30
SILENCE_TAIL_MS = (400, 800)
SAMPLING_TEMPERATURE = 5.0


# -- curriculum & sampling ------------------------------------------------------
def simultaneous_probability(step, total, warmup=WARMUP_FRACTION, ramp=RAMP_FRACTION):
    """Probability that update `step` (0-based) trains with a Wait-k mask: 0 for
    the first `warmup` share of the updates, then a linear ramp to 1."""
    start = warmup * total
    if step < start:
        return 0.0
    width = ramp * total
    return 1.0 if width <= 0 else min(1.0, (step - start) / width)


def curriculum_k(step, total, rng, k_range=K_RANGE, warmup=WARMUP_FRACTION, ramp=RAMP_FRACTION):
    """Wait-k lag for one micro-batch: None (full context) during the warm-up,
    else k ~ U{k_min..k_max}."""
    if rng.random() >= simultaneous_probability(step, total, warmup, ramp):
        return None
    return rng.randint(*k_range)


def temperature_weights(counts, temperature=SAMPLING_TEMPERATURE):
    """{direction: sampling probability} with p_d proportional to
    (n_d / N)^(1 / T). T = 1 is proportional sampling, T -> inf is uniform."""
    total = float(sum(counts.values()))
    raw = {d: (n / total) ** (1.0 / temperature) for d, n in counts.items() if n > 0}
    norm = sum(raw.values())
    return {d: v / norm for d, v in raw.items()}


class DirectionBatches(torch.utils.data.Sampler):
    """Endless length-bucketed batches of item indices. Each draw picks a
    direction by `weights`, then an item uniformly inside it; a pool of draws is
    sorted by source length and cut into batches whose padded size (longest
    item x batch size) stays within `max_chunks`. Seeded per run + step."""

    def __init__(self, items, weights, max_chunks, max_batch, seed, pool=1024):
        self.lengths = [it["src_chunks_max"] for it in items]
        self.by_dir = {}
        for i, it in enumerate(items):
            self.by_dir.setdefault(it["direction"], []).append(i)
        self.dirs = sorted(weights)
        self.p = [weights[d] for d in self.dirs]
        self.max_chunks, self.max_batch, self.seed, self.pool = max_chunks, max_batch, seed, pool

    def __iter__(self):
        rng = random.Random(self.seed)
        while True:
            dirs = rng.choices(self.dirs, self.p, k=self.pool)
            pool = sorted((rng.choice(self.by_dir[d]) for d in dirs),
                          key=lambda i: self.lengths[i])
            batches, cur, longest = [], [], 0
            for i in pool:
                longest_next = max(longest, self.lengths[i])
                if cur and (longest_next * (len(cur) + 1) > self.max_chunks
                            or len(cur) >= self.max_batch):
                    batches.append(cur)
                    cur, longest_next = [], self.lengths[i]
                cur.append(i)
                longest = longest_next
            batches.append(cur)
            rng.shuffle(batches)
            yield from batches


# -- data -----------------------------------------------------------------------
class CtcTokenizer:
    """The per-language SentencePiece models behind the CTC heads. Text is
    normalised like the SPM training text (asr.normalize_multilingual); the CTC
    blank is the model's <pad> id, which encode() never emits."""

    def __init__(self, spm_dir, langs):
        import sentencepiece as spm
        self.sp, self.vocab, self.blank = {}, {}, {}
        for lang in langs:
            path = pathlib.Path(spm_dir) / f"spm_{lang}.model"
            if not path.is_file():
                raise FileNotFoundError(f"{path}: no SentencePiece model for '{lang}' "
                                        "(prepare_s2ut_data.py --train-spm)")
            sp = spm.SentencePieceProcessor(model_file=str(path))
            if sp.pad_id() < 0:
                raise ValueError(f"{path}: the model has no <pad> id to use as the CTC blank")
            self.sp[lang], self.vocab[lang], self.blank[lang] = sp, sp.get_piece_size(), sp.pad_id()

    def encode(self, text, lang):
        return self.sp[lang].encode(asr.normalize_multilingual(text))


def _read_units_dirs(dirs, key):
    """{utt_id: path of its `key` file} over several units directories, and the
    directories' metas."""
    from fvml import units
    files, metas = {}, []
    for d in dirs:
        meta, rows = units.read_units_dir(d)
        metas.append(meta)
        for r in rows:
            if r.get(key):
                files[r["utt_id"]] = str(pathlib.Path(d) / r[key])
    return files, metas


def join_items(args, cfg, split, log=print):
    """Paired manifest rows of `split` x source latents x target units ->
    [{id, direction, src_lang, tgt_lang, latents, units, src_text, tgt_text,
    src_chunks_max}], plus the label space of the target units."""
    from fvml import units
    rows, _ = units.read_manifests(args.manifest)
    latents, lat_metas = _read_units_dirs(args.src_latents, "latents")
    unit_files, unit_metas = _read_units_dirs(args.tgt_units, "units")
    for meta in lat_metas:
        dim = (meta.get("latents") or {}).get("dim")
        if dim != cfg.latent_dim:
            raise ValueError(f"source latents have dim {dim}, the translator expects "
                             f"{cfg.latent_dim} (extract with --save-latents)")
    spaces = {(m.get("vocab"), m.get("label_space")) for m in unit_metas}
    if len(spaces) != 1:
        raise ValueError(f"target units directories disagree on (vocab, label_space): {spaces}")
    vocab, label_space = next(iter(spaces))
    if vocab != cfg.vocab:
        raise ValueError(f"target units have vocab {vocab}, the translator expects {cfg.vocab}")
    # Never ship what an NC teacher labelled - nor a translator fed by a student
    # that was distilled from one (the latents directory carries that flag too),
    # nor one trained on NC corpus audio (VoxPopuli) - unless opted in.
    for meta in (*unit_metas, *lat_metas):
        units.check_license(meta, args.include_noncommercial)
    corpus_nc = units.check_rows_license(rows, args.include_noncommercial)
    langs = cfg.lang_ids
    max_frames = int(round(args.max_seconds * 50))
    items, skipped = [], {"split": 0, "language": 0, "features": 0, "length": 0, "text": 0}
    for r in rows:
        if r.get("split", "train") != split:
            skipped["split"] += 1
        elif r["src_lang"] not in langs or r["tgt_lang"] not in langs:
            skipped["language"] += 1
        elif r["id"] not in latents or r["id"] not in unit_files:
            skipped["features"] += 1
        elif not r.get("src_text") or not r.get("tgt_text"):
            skipped["text"] += 1
        elif max(r["src_duration_s"], r["tgt_duration_s"]) * 50 > max_frames:
            skipped["length"] += 1
        else:
            tail = math.ceil(SILENCE_TAIL_MS[1] / cfg.chunk_ms)
            items.append({
                "id": r["id"], "direction": f"{r['src_lang']}_{r['tgt_lang']}",
                "src_lang": r["src_lang"], "tgt_lang": r["tgt_lang"],
                "latents": latents[r["id"]], "units": unit_files[r["id"]],
                "src_text": r["src_text"], "tgt_text": r["tgt_text"],
                "src_chunks_max": math.ceil(r["src_duration_s"] * 50 / cfg.chunk_frames) + tail})
    dropped = ", ".join(f"{n} skipped: {why}" for why, n in skipped.items()
                        if n and why != "split")
    log(f"[data] {split}: {len(items)} utterances" + (f" ({dropped})" if dropped else ""))
    if not items:
        raise ValueError(f"no '{split}' utterance has both source latents and target units")
    labels_meta = {"vocab": vocab, "label_space": label_space}
    nc_licenses = [str(m.get("license") or "non-commercial")
                   for m in (*unit_metas, *lat_metas) if m.get("noncommercial")]
    if corpus_nc:
        nc_licenses.append(corpus_nc)
    if nc_licenses:                   # BUG-031: the taint follows the weights into the ckpt
        labels_meta.update(noncommercial=True,
                           license=" + ".join(dict.fromkeys(nc_licenses)))
    return items, labels_meta


class S2utDataset(torch.utils.data.Dataset):
    """One utterance: latents padded to whole chunks (+ an optional silence
    tail and Gaussian latent noise), target units and CTC token ids."""

    def __init__(self, items, cfg, tokenizer, silence=None, tail_p=0.0, noise=0.0, seed=0):
        self.items, self.cfg, self.tok = items, cfg, tokenizer
        self.silence = None if silence is None else np.asarray(silence, np.float32)
        self.tail_p, self.noise, self.seed = tail_p, noise, seed
        self.rng = None

    def __len__(self):
        return len(self.items)

    def _silence(self, frames, rng):
        """`frames` consecutive silence latents from the bank (wrapping)."""
        start = int(rng.integers(0, len(self.silence)))
        return self.silence[(start + np.arange(frames)) % len(self.silence)]

    def __getitem__(self, i):
        if self.rng is None:
            info = torch.utils.data.get_worker_info()
            self.rng = np.random.default_rng([self.seed, info.id if info else 0])
        rng, cfg, it = self.rng, self.cfg, self.items[i]
        lat = np.load(it["latents"]).astype(np.float32)
        units = np.load(it["units"]).astype(np.int64)
        pad = (-len(lat)) % cfg.chunk_frames
        if pad:                         # what follows the last partial chunk at run time
            fill = self._silence(pad, rng) if self.silence is not None else np.repeat(lat[-1:], pad, 0)
            lat = np.concatenate([lat, fill])
        tail = 0
        if self.silence is not None and rng.random() < self.tail_p:
            lo, hi = (math.ceil(ms / cfg.chunk_ms) for ms in SILENCE_TAIL_MS)
            tail = int(rng.integers(lo, hi + 1))
            lat = np.concatenate([lat, self._silence(tail * cfg.chunk_frames, rng)])
        if self.noise > 0:
            lat = lat + self.noise * rng.standard_normal(lat.shape).astype(np.float32)
        return {"latents": lat, "units": units, "tail_chunks": tail,
                "src_lang": it["src_lang"], "tgt_lang": it["tgt_lang"],
                "src_tokens": self.tok.encode(it["src_text"], it["src_lang"]),
                "tgt_tokens": self.tok.encode(it["tgt_text"], it["tgt_lang"])}


def collate(samples):
    """Pad latents to the longest utterance; keep the ragged parts as lists."""
    chunks = [len(s["latents"]) for s in samples]
    lat = torch.zeros(len(samples), max(chunks), samples[0]["latents"].shape[1])
    for i, s in enumerate(samples):
        lat[i, :chunks[i]] = torch.from_numpy(s["latents"])
    return {"latents": lat, "src_frames": chunks,
            **{k: [s[k] for s in samples]
               for k in ("units", "src_lang", "tgt_lang", "src_tokens", "tgt_tokens", "tail_chunks")}}


def make_batch(raw, k, cfg, blank, device):
    """Lay the targets out for Wait-k lag `k` (None = full context) and move the
    batch to `device`: everything S2utTranslator.forward() and s2ut_loss() read."""
    src_chunks = [n // cfg.chunk_frames for n in raw["src_frames"]]
    batch = tr.collate_targets([tr.build_targets(u, s, k, cfg)
                                for u, s in zip(raw["units"], src_chunks)], cfg)
    ids = cfg.lang_ids
    batch.update(latents=raw["latents"], src_chunks=torch.tensor(src_chunks),
                 src_lang=torch.tensor([ids[x] for x in raw["src_lang"]]),
                 tgt_lang=torch.tensor([ids[x] for x in raw["tgt_lang"]]))
    batch = {n: t.to(device, non_blocking=True) for n, t in batch.items()}
    batch.update(src_tokens=raw["src_tokens"], tgt_tokens=raw["tgt_tokens"], ctc_blank=blank)
    return batch


def forward_loss(model, batch, use_bf16, label_smoothing=0.0, weights=None):
    device = batch["latents"].device
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
        out = model(batch["latents"], batch["src_lang"], batch["prev_units"], batch["tgt_lang"],
                    batch["src_visible"], batch.get("prev_dur"))
    return tr.s2ut_loss(model, out, batch, weights, label_smoothing)


# -- training -------------------------------------------------------------------
TERMS = ("total", "unit", "asr_ctc", "s2tt_ctc", "eos", "readiness", "unit_acc")


def lr_lambda(warmup, total, floor):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        t = min(1.0, (step - warmup) / max(1, total - warmup))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * t))
    return f


@torch.no_grad()
def validate(model, dataset, cfg, blank, device, limit, batch_size=8):
    """Teacher-forced loss terms on the first `limit` validation utterances at
    k = 3 and with full context (no silence tails, no noise)."""
    model.eval()
    result = {}
    for name, k in (("k3", 3), ("full", None)):
        sums, n = np.zeros(len(TERMS)), 0
        for start in range(0, min(limit, len(dataset)), batch_size):
            raw = collate([dataset[i] for i in range(start, min(start + batch_size, limit,
                                                               len(dataset)))])
            terms = forward_loss(model, make_batch(raw, k, cfg, blank, device), False)
            sums += [float(terms[t]) for t in TERMS]
            n += 1
        result[name] = dict(zip(TERMS, (sums / max(n, 1)).tolist()))
    model.train()
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


def save_checkpoint(path, model, opt, sched, step, args, tokenizer, labels_meta, extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(), "step": step,
                "config": json.dumps(model.cfg.to_dict()),
                "ctc_vocab": json.dumps(tokenizer.vocab), "ctc_blank": json.dumps(tokenizer.blank),
                "labels_meta": json.dumps(labels_meta), "args": json.dumps(vars(args)),
                "rng": rng_state(), **extra}, tmp)
    tmp.replace(path)


def build_config(args):
    fields = {"unit_mode": args.unit_mode}
    if args.tiny:
        fields = {**tr.tiny_config().to_dict(), **fields, "dropout": 0.0}
    for item in args.set or ():
        key, _, value = item.partition("=")
        if key not in tr.TranslatorConfig.__dataclass_fields__ or key in ("langs", "unit_mode"):
            raise ValueError(f"--set: unknown config field '{key}'")
        fields[key] = type(getattr(tr.TranslatorConfig, key))(value)
    return tr.TranslatorConfig(**fields)


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

    cfg = build_config(args)
    tokenizer = CtcTokenizer(args.spm_dir, [code for code, _ in cfg.langs])
    train_items, labels_meta = join_items(args, cfg, args.split, log)
    val_items = join_items(args, cfg, args.val_split, log)[0] if args.val_split else []
    silence = None
    if args.silence_tail_p > 0:
        if not args.silence_latents:
            raise ValueError("silence tails need --silence-latents (the student's latents on "
                             "silence); pass --silence-tail-p 0 to train without them")
        silence = np.load(args.silence_latents).astype(np.float32)
        if silence.ndim != 2 or silence.shape[1] != cfg.latent_dim or len(silence) < 4:
            raise ValueError(f"{args.silence_latents}: expected [T >= 4, {cfg.latent_dim}] latents, "
                             f"got {silence.shape}")

    model = tr.S2utTranslator(cfg, tokenizer.vocab).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98),
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(args.warmup, args.steps, args.lr_floor))
    step, best, elapsed0 = 0, -1.0, 0.0
    if resume:
        ck = torch.load(resume, map_location="cpu", weights_only=True)
        if json.loads(ck["config"]) != cfg.to_dict():
            raise ValueError(f"{resume}: model config differs from this run's")
        if json.loads(ck["ctc_vocab"]) != tokenizer.vocab:
            raise ValueError(f"{resume}: trained with other SentencePiece models")
        if json.loads(ck["labels_meta"]) != labels_meta:
            raise ValueError(f"{resume}: trained on another target unit label space")
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

    counts = {}
    for it in train_items:
        counts[it["direction"]] = counts.get(it["direction"], 0) + 1
    weights = temperature_weights(counts, args.temperature)
    params = model.param_counts()
    log(f"[train] {run}: {cfg.unit_mode}, {params['runtime'] / 1e6:.2f}M run-time parameters "
        f"(+ {params['training_only'] / 1e6:.2f}M training-only CTC), device {device}")
    log("[data] sampling: " + ", ".join(f"{d} {counts[d]} utts p={weights[d]:.3f}"
                                         for d in sorted(counts)))

    data_seed = args.seed * 1_000_003 + step
    ds = S2utDataset(train_items, cfg, tokenizer, silence, args.silence_tail_p, args.latent_noise,
                     seed=data_seed)
    sampler = DirectionBatches(train_items, weights, args.max_chunks, args.max_batch, data_seed)
    loader = torch.utils.data.DataLoader(
        ds, batch_sampler=sampler, collate_fn=collate, num_workers=args.workers,
        pin_memory=device.type == "cuda", persistent_workers=args.workers > 0,
        worker_init_fn=lambda _: torch.set_num_threads(1))
    val_ds = S2utDataset(val_items, cfg, tokenizer) if val_items else None

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(pathlib.Path(args.runs_dir) / run), purge_step=step if resume else None)
    if not resume:
        writer.add_text("args", json.dumps(vars(args), indent=2))
        writer.add_text("model", json.dumps({**cfg.to_dict(), **params}, indent=2))
    use_bf16 = device.type == "cuda" and not args.fp32
    t_start, t_ckpt, t_log = time.monotonic(), time.monotonic(), time.monotonic()
    sums, n_micro, n_sim, n_tail, n_utts = np.zeros(len(TERMS)), 0, 0, 0, 0
    history = []

    def checkpoint(name):
        save_checkpoint(ckpt_dir / name, model, opt, sched, step, args, tokenizer, labels_meta,
                        {"best_val": best, "run_name": run,
                         "elapsed": elapsed0 + time.monotonic() - t_start})

    model.train()
    it = iter(loader)
    while step < args.steps:
        opt.zero_grad(set_to_none=True)
        for micro in range(args.accum):
            k = curriculum_k(step, args.steps, random.Random(args.seed * 7919 + step * 131 + micro),
                             (args.k_min, args.k_max), args.full_context_fraction,
                             args.ramp_fraction)
            raw = next(it)
            terms = forward_loss(model, make_batch(raw, k, cfg, tokenizer.blank, device),
                                 use_bf16, args.label_smoothing,
                                 {"asr_ctc": args.ctc_weight, "s2tt_ctc": args.ctc_weight})
            (terms["total"] / args.accum).backward()
            sums += [float(terms[t].detach()) for t in TERMS]
            n_micro += 1
            n_sim += k is not None
            n_tail += sum(t > 0 for t in raw["tail_chunks"])
            n_utts += len(raw["units"])
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip))
        opt.step()
        sched.step()
        step += 1
        if step % args.log_every == 0 or step == args.steps:
            dt = time.monotonic() - t_log
            t_log = time.monotonic()
            mean = dict(zip(TERMS, sums / n_micro))
            for name, v in mean.items():
                writer.add_scalar(f"train/{name}", v, step)
            writer.add_scalar("train/lr", sched.get_last_lr()[0], step)
            writer.add_scalar("train/grad_norm", gnorm, step)
            writer.add_scalar("train/simultaneous_share", n_sim / n_micro, step)
            mem = (f"  gpu {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
                   if device.type == "cuda" else "")
            log(f"step {step:7d}  loss {mean['total']:.3f}  unit {mean['unit']:.3f} "
                f"(acc {mean['unit_acc']:.3f})  asr {mean['asr_ctc']:.3f}  s2tt {mean['s2tt_ctc']:.3f}  "
                f"eos {mean['eos']:.3f}  ready {mean['readiness']:.3f}  "
                f"wait-k {n_sim / n_micro:.2f}  tails {n_tail / max(n_utts, 1):.2f}  "
                f"lr {sched.get_last_lr()[0]:.2e}  gn {gnorm:.2f}  "
                f"{args.log_every / max(dt, 1e-9):.2f} it/s{mem}")
            history.append({"step": step, **mean, "simultaneous_share": n_sim / n_micro,
                            "silence_tail_share": n_tail / max(n_utts, 1)})
            sums[:] = 0
            n_micro = n_sim = n_tail = n_utts = 0
        if val_ds is not None and (step % args.val_every == 0 or step == args.steps):
            v = validate(model, val_ds, cfg, tokenizer.blank, device, args.val_limit)
            for name, terms in v.items():
                for t, value in terms.items():
                    writer.add_scalar(f"val_{name}/{t}", value, step)
            log(f"[val] step {step}: " + "  ".join(
                f"{name} unit {t['unit']:.3f} acc {t['unit_acc']:.3f}" for name, t in v.items()))
            if v["k3"]["unit_acc"] > best:
                best = v["k3"]["unit_acc"]
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
    (ckpt_dir / "history.json").write_text(json.dumps(history, indent=1) + "\n", encoding="utf-8")
    peak = (f", peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
            if device.type == "cuda" else "")
    log(f"[train] done at step {step} -> {ckpt_dir / 'final.pt'}{peak}")
    return 0


# -- synthetic data (smoke runs and tests) ----------------------------------------
def make_synthetic(root, cfg, n_per_direction=16, seed=0, spm_vocab=48):
    """A tiny learnable corpus in the on-disk layout the trainer reads: every
    source "phone" is a fixed latent vector held 2-5 frames, the target unit
    stream is a per-language relabelling of the same phones (same durations),
    and both transcripts spell the phones as syllables. Returns the argument
    dict (manifest, src_latents, tgt_units, spm_dir, silence_latents)."""
    import sentencepiece as spm
    from fvml import datasets as fd
    from fvml import s2ut_data as sd
    root = pathlib.Path(root)
    rng = np.random.default_rng(seed)
    phones = min(cfg.vocab, 20)
    codebook = rng.standard_normal((phones, cfg.latent_dim)).astype(np.float32)
    relabel = {lang: rng.permutation(cfg.vocab)[:phones] for lang, _ in cfg.langs}
    syllables = {lang: [c + v + lang[0] for c in "bdfgklmnprst" for v in "aeiou"][:phones]
                 for lang, _ in cfg.langs}
    lat_dir, unit_dir = root / "units" / "student.src", root / "units" / "teacher.tgt"
    for d in (lat_dir, unit_dir, root / "manifests", root / "spm"):
        d.mkdir(parents=True, exist_ok=True)
    rows, lat_rows, unit_rows, texts = [], [], [], {lang: [] for lang, _ in cfg.langs}
    for direction in sd.DIRECTIONS:
        src, tgt = direction.split("_")
        for i in range(n_per_direction):
            uid = f"syn_{direction}_{i:04d}"
            seq = rng.integers(0, phones, int(rng.integers(6, 15)))
            durs = rng.integers(2, 6, len(seq))
            frames = np.repeat(seq, durs)
            lat = codebook[frames] + 0.1 * rng.standard_normal((len(frames), cfg.latent_dim))
            np.save(lat_dir / f"{uid}.latents.npy", lat.astype(np.float16))
            np.save(unit_dir / f"{uid}.npy", relabel[tgt][frames].astype(np.int16))
            src_text = " ".join(syllables[src][p] for p in seq)
            tgt_text = " ".join(syllables[tgt][p] for p in seq)
            texts[src].append(src_text)
            texts[tgt].append(tgt_text)
            dur = round(len(frames) / 50.0, 3)
            rows.append({"id": uid, "src_lang": src, "tgt_lang": tgt,
                         "src_audio_path": f"syn/{uid}.src.flac", "src_duration_s": dur,
                         "src_text": src_text, "tgt_audio_path": f"syn/{uid}.tgt.flac",
                         "tgt_duration_s": dur, "tgt_text": tgt_text, "speaker_id": "syn",
                         "dataset": "synthetic", "license": "CC0-1.0",
                         "split": "dev" if i % 8 == 7 else "train"})
            lat_rows.append({"utt_id": uid, "units": f"{uid}.npy", "latents": f"{uid}.latents.npy",
                             "n_frames": len(frames)})
            unit_rows.append({"utt_id": uid, "units": f"{uid}.npy", "n_frames": len(frames)})
    fd.write_jsonl(root / "manifests" / "synthetic.jsonl", rows)
    meta = {"tokenizer": "synthetic", "frame_rate_hz": 50.0, "vocab": cfg.vocab,
            "label_space": "synthetic", "lookahead_frames": 0}
    fd.write_jsonl(lat_dir / "manifest.jsonl", lat_rows)
    (lat_dir / "units_meta.json").write_text(json.dumps(
        {**meta, "latents": {"dim": cfg.latent_dim, "dtype": "float16"}}), encoding="utf-8")
    fd.write_jsonl(unit_dir / "manifest.jsonl", unit_rows)
    (unit_dir / "units_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    silence = 0.05 * rng.standard_normal((64, cfg.latent_dim)).astype(np.float32)
    np.save(root / "silence.latents.npy", silence)
    for lang, lines in texts.items():
        corpus = root / "spm" / f"{lang}.txt"
        corpus.write_text("\n".join(lines) + "\n", encoding="utf-8")
        spm.SentencePieceTrainer.train(
            input=str(corpus), model_prefix=str(root / "spm" / f"spm_{lang}"),
            vocab_size=spm_vocab, model_type="unigram", character_coverage=1.0,
            unk_id=0, bos_id=1, eos_id=2, pad_id=3, hard_vocab_limit=False, minloglevel=2)
        corpus.unlink()
    return {"manifest": [str(root / "manifests" / "synthetic.jsonl")],
            "src_latents": [str(lat_dir)], "tgt_units": [str(unit_dir)],
            "spm_dir": str(root / "spm"), "silence_latents": str(root / "silence.latents.npy")}


def apply_smoke(args):
    """--smoke DIR: synthetic corpus + tiny model on CPU, everything under DIR."""
    root = pathlib.Path(args.smoke)
    args.tiny, args.device, args.workers = True, "cpu", 0
    cfg = build_config(args)
    data = root / "data"
    if not (data / "spm" / "spm_en.model").is_file():
        make_synthetic(data, cfg, seed=args.seed)
    args.manifest = [str(data / "manifests" / "synthetic.jsonl")]
    args.src_latents, args.tgt_units = [str(data / "units" / "student.src")], \
        [str(data / "units" / "teacher.tgt")]
    args.spm_dir, args.silence_latents = str(data / "spm"), str(data / "silence.latents.npy")
    args.runs_dir, args.ckpt_dir = str(root / "runs"), str(root / "checkpoints")
    args.val_split = args.val_split or "dev"
    defaults = {"lr": 3e-3, "warmup": 10, "max_chunks": 400, "max_batch": 16, "accum": 1,
                "log_every": 10, "val_every": 20, "label_smoothing": 0.0}
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    args.run_name = args.run_name or "smoke"


FULL_DEFAULTS = {"lr": 5e-4, "warmup": 4000, "max_chunks": 2400, "max_batch": 48, "accum": 4,
                 "log_every": 100, "val_every": 2000, "label_smoothing": 0.1}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.splitlines()[2:]))
    p.add_argument("--manifest", action="append", help="paired S2UT manifest(s)")
    p.add_argument("--src-latents", action="append", help="units dir(s) with <id>.latents.npy")
    p.add_argument("--tgt-units", action="append", help="units dir(s) with the target <id>.npy")
    p.add_argument("--spm-dir", help="directory of spm_<lang>.model")
    p.add_argument("--silence-latents", help="[T, D] .npy bank of silence latents")
    p.add_argument("--smoke", metavar="DIR", help="synthetic data + tiny model on CPU under DIR")
    p.add_argument("--split", default="train")
    p.add_argument("--val-split", help="manifest split to validate on (e.g. dev)")
    p.add_argument("--val-limit", type=int, default=200)
    p.add_argument("--val-every", type=int)
    p.add_argument("--include-noncommercial", action="store_true",
                   help="allow target units from a non-commercial teacher (research only)")
    p.add_argument("--unit-mode", choices=tr.UNIT_MODES, default="stacked_4")
    p.add_argument("--tiny", action="store_true", help="fvml.translator.tiny_config (smoke runs)")
    p.add_argument("--set", action="append", metavar="FIELD=VALUE",
                   help="override a TranslatorConfig field, e.g. --set d_model=256")
    p.add_argument("--steps", type=int, default=200_000, help="optimizer updates")
    p.add_argument("--accum", type=int, help="micro-batches per update (default 4)")
    p.add_argument("--max-chunks", type=int,
                   help="padded source chunks per micro-batch (default 2400 = 192 s; sized for "
                        "8 GB with bf16, see the Stage 05 file)")
    p.add_argument("--max-batch", type=int, help="utterances per micro-batch (default 48)")
    p.add_argument("--max-seconds", type=float, default=20.0, help="skip longer utterances")
    p.add_argument("--temperature", type=float, default=SAMPLING_TEMPERATURE,
                   help="direction sampling temperature (1 = proportional)")
    p.add_argument("--full-context-fraction", type=float, default=WARMUP_FRACTION)
    p.add_argument("--ramp-fraction", type=float, default=RAMP_FRACTION)
    p.add_argument("--k-min", type=int, default=K_RANGE[0])
    p.add_argument("--k-max", type=int, default=K_RANGE[1])
    p.add_argument("--silence-tail-p", type=float, default=SILENCE_TAIL_P)
    p.add_argument("--latent-noise", type=float, default=0.0,
                   help="std of Gaussian noise added to the source latents")
    p.add_argument("--label-smoothing", type=float)
    p.add_argument("--ctc-weight", type=float, default=tr.LOSS_WEIGHTS["asr_ctc"],
                   help="weight of each auxiliary CTC loss (0 = the no-CTC ablation)")
    p.add_argument("--lr", type=float)
    p.add_argument("--lr-floor", type=float, default=0.02, help="final LR as a fraction of --lr")
    p.add_argument("--warmup", type=int, help="LR warm-up updates (default 4000)")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--fp32", action="store_true", help="disable bf16 autocast")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--run-name", default=None, help="default s2ut_<date>_<time>")
    p.add_argument("--runs-dir", default=str(paths.REPO_ROOT / "runs"))
    p.add_argument("--ckpt-dir", default=str(paths.REPO_ROOT / "checkpoints"))
    p.add_argument("--ckpt-minutes", type=float, default=30.0)
    p.add_argument("--ckpt-steps", type=int, default=0, help="also checkpoint every N steps (0 = off)")
    p.add_argument("--log-every", type=int)
    p.add_argument("--resume", help="'auto' (<ckpt-dir>/<run>/last.pt if present) or a checkpoint path")
    args = p.parse_args(argv)
    if args.smoke:
        apply_smoke(args)
    else:
        for name in ("manifest", "src_latents", "tgt_units", "spm_dir"):
            if not getattr(args, name):
                p.error(f"--{name.replace('_', '-')} is required (or use --smoke DIR)")
    for key, value in FULL_DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if not 1 <= args.k_min <= args.k_max <= 10:
        p.error("need 1 <= --k-min <= --k-max <= 10")
    if not args.run_name:
        if args.resume == "auto":
            p.error("--resume auto needs --run-name")
        args.run_name = "s2ut_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return args


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    try:
        return train(parse_args(argv))
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"[train] error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
