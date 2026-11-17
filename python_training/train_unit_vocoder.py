#!/usr/bin/env python3
"""Train the causal, stateful, multi-speaker Unit-HiFi-GAN vocoder.

Milestone 2 / Stage 05 (Documents/Milestone 2/05_causal_vocoder_decision_gate.md).
Maps (discrete unit ids, target-domain log-f0, voicing, aperiodicity, energy,
speaker embedding) -> 16 kHz waveform using strictly causal convolutions
(fvml/vocoder.py).

Data:
  - Repeatable --train-manifest (JSONL audio rows) joined by utt_id with
    repeatable --train-units (teacher-km100 or student/Mimi units directories).
  - Clips listed in --exclude-targets (default eval_lists/targets.tsv;
    comma-separated TSVs, e.g. public targets + <speaker>_targets.tsv so
    held-out own-voice clips never train) are excluded from training.
  - Conditioning features [logf0_held, voicing, aperiodicity, log_energy] are
    extracted from each speaker's *own* audio with the Stage 03 causal YIN
    tracker (fvml/features.py). Per-speaker F0/energy mean & std are computed
    and stored in the checkpoint (and written to voices.json).

Losses & discriminators:
  - Multi-Period Discriminator (MPD) + Multi-Resolution STFT Discriminator (MRD)
  - LSGAN adversarial loss + feature-matching loss (weight 2.0) +
    multi-resolution log-mel L1 loss (weight 45.0)
  - AdamW lr 2e-4, betas (0.8, 0.99), exponential decay 0.999/epoch, bf16
    autocast on CUDA, resumable checkpoints every --ckpt-minutes.

Multilingual vocoder (Milestone 3 / Stage 03,
Documents/Milestone 3/03_multilingual_encoder_vocoder_upgrade.md): the vocabulary
follows the labels (K = 1000 mhubert-vp units), the unit embedding stays 128 wide,
and no language token enters the generator. What changes is the data:
  --lang-ratios m3-vocoder   40% en / 20% es / 20% fr / 20% de of all draws, so
                             English Mode A quality does not regress
  --default-lang en          tag for rows without a `lang` field (LJSpeech, Hi-Fi TTS)
  --train-subsets train      hold out the corpora's dev/test clips (CSS10, CML-TTS)
Each voice's language (one per speaker) and the licence/source its manifest rows
carry are stored in the checkpoint and written to voices.json (`lang`).
Units from the non-commercial mHuBERT-147 teacher need --include-noncommercial.

Usage:
  train_unit_vocoder.py --train-manifest ljspeech.jsonl --train-manifest hifitts.jsonl \\
      --train-units data/units/teacher-km100/lj --train-units data/units/teacher-km100/hifitts \\
      --run-name voc_a
  train_unit_vocoder.py ... --run-name voc_a --resume auto
  train_unit_vocoder.py ... --finetune-from checkpoints/voc_a/final.pt --steps 50000 --run-name voc_ft
  # Milestone 3: {en, es, fr, de} studio voices over K = 1000 units
  train_unit_vocoder.py --lang-ratios m3-vocoder --default-lang en --train-subsets train \\
      --train-manifest ljspeech.jsonl,hifitts.jsonl,css10-es.jsonl,css10-fr.jsonl,css10-de.jsonl \\
      --train-manifest cml-tts-es.jsonl,cml-tts-fr.jsonl,cml-tts-de.jsonl \\
      --train-units data/units/mhubert/lj,data/units/mhubert/hifitts,... --run-name voc_m3

Stage 10 target-voice fine-tune recipe (Documents/Milestone 2/10_*.md):
  # prepare_target_voice.py -> <data>/private/me/{train.jsonl, me_targets.tsv,
  #   speaker_metadata.json}; extract_units.py --perturb-fraction 0.5 on it
  train_unit_vocoder.py \\
      --train-manifest <public manifests> --train-manifest <data>/private/me/train.jsonl \\
      --train-units <public units> --train-units <data>/units/<winner>/me \\
      --exclude-targets eval_lists/targets.tsv,<data>/private/me/me_targets.tsv \\
      --finetune-from checkpoints/<base>/final.pt \\
      --new-speaker-init nearest --mix-speakers me --mix-ratio 0.5 \\
      --speaker-metadata <data>/private/me/speaker_metadata.json \\
      --early-stop-patience 3 --val-utmos --steps 50000 --run-name voc_me

Exit status: 0 = trained, 1 = missing or inconsistent inputs.
"""
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse          # noqa: E402
import concurrent.futures  # noqa: E402
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

from fvml import audio, evallist, features, paths, sampling, units, vocoder  # noqa: E402

COND_FS = vocoder.COND_FRAME_SAMPLES
PREROLL_COND_FRAMES = 20     # 400 ms causal warmup for HPF, YIN window and logf0_held


def validate_speaker_name(name):
    """Enforce the voices.json speaker-name rule ([a-z0-9_-]+) before training."""
    if not isinstance(name, str) or not name or \
            not all(ch.isascii() and (ch.isalnum() or ch in "-_") for ch in name) or \
            name != name.lower():
        raise ValueError(f"invalid speaker name {name!r}: must match lowercase [a-z0-9_-]+")
    return name


# Injectable for tests: heavy models are built lazily on first use.
UTMOS_FN = None       # defaults to mos.Utmos(device).score
EMBED_FN = None       # defaults to speaker.Ecapa(device).embed


def _lazy_utmos(device):
    holder = {}

    def fn(waves):
        if "u" not in holder:
            from fvml import mos
            holder["u"] = mos.Utmos(device)
        return holder["u"].score(waves)

    return fn


def _lazy_embed(device):
    holder = {}

    def fn(waves):
        if "e" not in holder:
            from fvml import speaker
            holder["e"] = speaker.Ecapa(device)
        return holder["e"].embed(waves)

    return fn


def nearest_speaker(new_items, existing_items_by_speaker, embed_fn,
                    max_utts=20, max_seconds=10, speaker_name="(new)",
                    log=print):
    """ECAPA centroid of the new speaker's items vs each existing speaker's ->
    (nearest_name, {speaker: cosine}). Items are (audio_path, ...) tuples;
    embed_fn([wave]) -> L2-normalised [N, D] embeddings. Existing speakers
    with no training items are skipped (logged); an empty new speaker or no
    candidates at all is a ValueError naming it."""
    from fvml import speaker
    if not new_items:
        raise ValueError(f"new speaker '{speaker_name}' has no training items")
    existing = {s: its for s, its in existing_items_by_speaker.items() if its}
    for s in existing_items_by_speaker.keys() - existing.keys():
        log(f"[train] nearest speaker init: skipping '{s}' (no training items)")
    if not existing:
        raise ValueError(f"nearest speaker init for '{speaker_name}': no existing "
                         "speaker has training items")

    def centroid_of(items):
        picked = items if len(items) <= max_utts \
            else evallist.evenly_spaced(items, max_utts)
        waves = [audio.load(it[0])[: int(max_seconds * audio.SAMPLE_RATE)]
                 for it in picked]
        return speaker.centroid(embed_fn(waves))

    c_new = centroid_of(new_items)
    cosines = {s: float(c_new @ centroid_of(its)) for s, its in existing.items()}
    return max(cosines, key=cosines.get), cosines


class EarlyStopper:
    """A validation counts as improved when val mel improves, or (with UTMOS
    validation) mean val UTMOS improves by > 0.01. `stop` after `patience`
    validations without improvement. best.pt stays chosen by mel alone."""

    def __init__(self, patience):
        self.patience = int(patience)
        self.since = 0
        self.best_val = float("inf")
        self.best_utmos = float("-inf")

    def update(self, val_mel, utmos=None):
        improved = val_mel < self.best_val
        if improved:
            self.best_val = val_mel
        if utmos is not None and utmos > self.best_utmos + 0.01:
            improved = True
            self.best_utmos = float(utmos)
        self.since = 0 if improved else self.since + 1
        return improved

    @property
    def stop(self):
        return self.patience > 0 and self.since >= self.patience


# -- data & speaker stats -----------------------------------------------------
def load_excluded_ids(targets_path):
    """Set of clip_ids and relative paths in the targets TSV(s) (comma-separated)
    that must stay held out."""
    ids, pths = set(), set()
    for spec in (targets_path or "").split(","):
        spec = spec.strip()
        p = pathlib.Path(spec) if spec else None
        if not p or not p.is_file():
            continue
        rows = evallist.read_targets(p)
        ids.update(r["clip_id"] for r in rows)
        pths.update(r["path"] for r in rows)
    return ids, pths


def read_all_units_dirs(specs):
    """Read one or more units directories and verify they share a compatible space."""
    dirs = [s.strip() for v in ([specs] if isinstance(specs, str) else specs)
            for s in v.split(",") if s.strip()]
    if not dirs:
        raise ValueError("no units directory given")
    meta0 = None
    by_id = {}
    for d in dirs:
        meta, urows = units.read_units_dir(d)
        if meta0 is None:
            meta0 = meta
        else:
            units.check_compatible(meta0, meta)
            # M3: the licence check reads the first directory's meta, so a later one may
            # not smuggle in non-commercial units (fvml.units.read_units_dirs does the same).
            if bool(meta.get("noncommercial")) != bool(meta0.get("noncommercial")):
                raise units.UnitsMismatch(f"{d}: mixes commercial and non-commercial unit streams")
        for r in urows:
            by_id[r["utt_id"]] = (pathlib.Path(d) / r["units"], int(r["n_frames"]), r)
    return meta0, by_id


def join_vocoder_items(manifest_spec, units_spec, exclude_targets=None, speakers_order=None, log=print,
                       *, side=None, default_lang=None, subsets=None, info=None):
    """Join manifest rows with unit files, excluding held-out target clips.
    Returns (items, labels_meta, speakers), where each item is
    (audio_path, units_path, n_unit_frames, spk_idx, speaker_name).

    Milestone 3: `side` reads one audio of paired S2UT rows, `default_lang`
    tags rows without a language, `subsets` keeps only those manifest subsets.
    A dict passed as `info` receives "speaker_lang" ({speaker: lang or None})
    and "speaker_meta" ({speaker: lang / license / source found in its rows})."""
    rows, _ = units.read_manifests(manifest_spec)
    rows = units.filter_subsets(units.audio_rows(rows, side, default_lang), subsets)
    meta, by_id = read_all_units_dirs(units_spec)
    excl_ids, excl_paths = load_excluded_ids(exclude_targets)
    fs = int(meta["frame_samples"])

    raw_items = []
    missing, bad, excluded = 0, 0, 0
    found_speakers = []
    seen_speakers = set()
    spk_langs, spk_meta = {}, {}
    for r in rows:
        uid = r["utt_id"]
        if uid in excl_ids or r.get("path") in excl_paths:
            excluded += 1
            continue
        u = by_id.get(uid)
        if u is None:
            missing += 1
            continue
        upath, n_frames, _ = u
        expect = math.ceil(round(r["duration"] * audio.SAMPLE_RATE) / fs)
        if abs(expect - n_frames) > 1:
            bad += 1
            continue
        spk = validate_speaker_name(str(r.get("speaker") or "default").lower())
        if spk not in seen_speakers:
            seen_speakers.add(spk)
            found_speakers.append(spk)
        raw_items.append((str(paths.resolve(r["path"])), str(upath), n_frames, spk))
        spk_langs.setdefault(spk, set()).add(r.get("lang"))
        derived = spk_meta.setdefault(spk, {})
        if r.get("lang"):
            derived["lang"] = r["lang"]
        if r.get("license"):                 # the M3 corpora say where each row comes from
            derived.setdefault("license", r["license"])
            if r.get("dataset"):
                derived.setdefault("source", r["dataset"])

    if not raw_items:
        raise ValueError(f"no usable training utterances found for {units_spec}")
    mixed = sorted(s for s, langs in spk_langs.items() if len(langs) > 1)
    if mixed:
        raise ValueError(f"speaker(s) {', '.join(mixed)} have rows in several languages (or some "
                         "rows untagged): a voice has one `lang`; check --default-lang")
    if isinstance(info, dict):
        info["speaker_lang"] = {s: next(iter(langs)) for s, langs in spk_langs.items()}
        info["speaker_meta"] = {s: m for s, m in spk_meta.items() if m}

    if speakers_order:
        speakers = [validate_speaker_name(str(s)) for s in speakers_order]
        for s in found_speakers:
            if s not in speakers:
                speakers.append(s)
    else:
        # Keep DEFAULT_SPEAKERS order first when present, then alphabetical extras.
        pref = [s for s in vocoder.DEFAULT_SPEAKERS if s in seen_speakers]
        rest = sorted(s for s in found_speakers if s not in pref)
        speakers = pref + rest

    spk_to_idx = {s: i for i, s in enumerate(speakers)}
    items = [(ap, up, nf, spk_to_idx[s], s) for ap, up, nf, s in raw_items]
    if missing or bad or excluded:
        log(f"[data] {len(items)} utterances across {len(speakers)} speaker(s); "
            f"{excluded} excluded (targets.tsv), {missing} without units, {bad} frame-mismatch")
    return items, meta, speakers


def read_audio_samples(path, start_sample, n_samples):
    """Read [start_sample, start_sample + n_samples) from a 16 kHz file, zero-padded."""
    info = sf.info(path)
    if info.samplerate == audio.SAMPLE_RATE:
        avail = max(0, info.frames - start_sample)
        take = min(n_samples, avail)
        if take > 0:
            x, _ = sf.read(path, start=start_sample, frames=take, dtype="float32", always_2d=True)
            x = x.mean(axis=1)
        else:
            x = np.zeros(0, dtype=np.float32)
    else:
        full = audio.load(path)
        x = full[start_sample:start_sample + n_samples]
    return np.pad(x, (0, n_samples - len(x))) if len(x) < n_samples else x


def extract_cond_crop(apath, start_cond_frame, n_cond_frames, preroll=PREROLL_COND_FRAMES,
                      lookahead_frames=0):
    """Extract causal [logf0_held, voicing, aperiodicity, log_energy] for conditioning
    frames [start, start + n), warming the tracker up on `preroll` preceding frames."""
    pre = min(start_cond_frame, preroll)
    la = max(0, int(lookahead_frames))
    s0 = (start_cond_frame - pre) * COND_FS
    total_cond = pre + n_cond_frames
    wav = read_audio_samples(apath, s0, (total_cond + la) * COND_FS)
    cfg = features.PitchConfig(lookahead_frames=la) if la else None
    feats = features.extract(wav[:total_cond * COND_FS] if la == 0 else wav, cfg=cfg)
    crop = feats[pre:pre + n_cond_frames]
    # Columns in features.FEATURES:
    # 0: f0_hz, 1: aperiodicity, 2: voicing, 3: log_energy, 4: logf0_held
    logf0 = np.ascontiguousarray(crop[:, 4], dtype=np.float32)
    voicing = np.ascontiguousarray(crop[:, 2], dtype=np.float32)
    aperiodicity = np.ascontiguousarray(crop[:, 1], dtype=np.float32)
    energy = np.ascontiguousarray(crop[:, 3], dtype=np.float32)
    target_wav = np.ascontiguousarray(wav[pre * COND_FS:(pre + n_cond_frames) * COND_FS],
                                      dtype=np.float32)
    return target_wav, logf0, voicing, aperiodicity, energy


def _stats_worker(apath):
    wav = audio.load(apath)
    o = features.extract(wav)
    voiced = o[:, 0] > 0.0
    lf = np.log(o[voiced, 0]) if voiced.any() else np.zeros(0, dtype=np.float64)
    e = o[:, 3]
    return int(len(lf)), float(lf.sum()), float((lf ** 2).sum()), int(len(e)), float(e.sum()), float((e ** 2).sum())


def compute_speaker_stats(items, speakers, max_utts_per_speaker=500, jobs=1, existing=None):
    """Compute per-speaker logf0_mean/std (voiced frames) and energy_mean/std (all frames)."""
    existing = dict(existing or {})
    by_spk = {s: [] for s in speakers}
    for apath, _, _, _, spk in items:
        by_spk.setdefault(spk, []).append(apath)

    stats = {}
    for spk in speakers:
        paths_list = by_spk.get(spk, [])
        if not paths_list:
            if spk in existing:
                stats[spk] = existing[spk]
                continue
            stats[spk] = {
                "logf0_mean": features.LOGF0_PRIOR_MEAN,
                "logf0_std": features.LOGF0_PRIOR_STD,
                "energy_mean": features.ENERGY_PRIOR_MEAN,
                "energy_std": features.ENERGY_PRIOR_STD,
                "voiced_frames": 0,
                "frames": 0,
            }
            continue
        if max_utts_per_speaker and len(paths_list) > max_utts_per_speaker:
            paths_list = evallist.evenly_spaced(paths_list, max_utts_per_speaker)
        if jobs > 1 and len(paths_list) >= 8:
            with concurrent.futures.ProcessPoolExecutor(jobs) as ex:
                parts = list(ex.map(_stats_worker, paths_list, chunksize=8))
        else:
            parts = [_stats_worker(p) for p in paths_list]
        arr = np.sum(np.asarray(parts, dtype=np.float64), axis=0)
        n_lf, s1_lf, s2_lf, n_e, s1_e, s2_e = arr
        if n_lf > 0:
            lf_mean = float(s1_lf / n_lf)
            lf_std = max(math.sqrt(max(float(s2_lf / n_lf) - lf_mean * lf_mean, 0.0)),
                         features.PitchConfig().logf0_std_floor)
        else:
            lf_mean, lf_std = features.LOGF0_PRIOR_MEAN, features.LOGF0_PRIOR_STD
        if n_e > 0:
            e_mean = float(s1_e / n_e)
            e_std = max(math.sqrt(max(float(s2_e / n_e) - e_mean * e_mean, 0.0)),
                        features.PitchConfig().energy_std_floor)
        else:
            e_mean, e_std = features.ENERGY_PRIOR_MEAN, features.ENERGY_PRIOR_STD
        stats[spk] = {
            "logf0_mean": round(lf_mean, 6),
            "logf0_std": round(lf_std, 6),
            "energy_mean": round(e_mean, 6),
            "energy_std": round(e_std, 6),
            "voiced_frames": int(n_lf),
            "frames": int(n_e),
        }
    return stats


class VocoderCropDataset(torch.utils.data.Dataset):
    """Random frame-aligned segment loader yielding
    (wav, units, logf0, voicing, aperiodicity, energy, spk_id)."""

    def __init__(self, items, crop_unit_frames, unit_repeat=1, lookahead_frames=0):
        self.items = items
        self.crop_u = crop_unit_frames
        self.unit_repeat = unit_repeat
        self.crop_c = crop_unit_frames * unit_repeat
        self.lookahead_frames = max(0, int(lookahead_frames))
        self.rng = None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        if self.rng is None:
            self.rng = np.random.default_rng(torch.initial_seed() % (1 << 63))
        apath, upath, n_u, spk_idx, _ = self.items[i]
        start_u = int(self.rng.integers(0, n_u - self.crop_u + 1)) if n_u > self.crop_u else 0
        raw_u = np.load(upath, mmap_mode="r")[start_u:start_u + self.crop_u]
        u = np.zeros(self.crop_u, dtype=np.int64)
        u[:len(raw_u)] = raw_u
        if len(raw_u) < self.crop_u and len(raw_u) > 0:
            u[len(raw_u):] = raw_u[-1]

        start_c = start_u * self.unit_repeat
        wav, logf0, voicing, aperiodicity, energy = extract_cond_crop(
            apath, start_c, self.crop_c, lookahead_frames=self.lookahead_frames
        )
        return (
            torch.from_numpy(wav),
            torch.from_numpy(u),
            torch.from_numpy(logf0),
            torch.from_numpy(voicing),
            torch.from_numpy(aperiodicity),
            torch.from_numpy(energy),
            torch.tensor(spk_idx, dtype=torch.int64),
        )


class WeightedStream(torch.utils.data.Sampler):
    """Endless indices drawn proportionally to `weights`, seeded per run+step."""

    def __init__(self, weights, seed, chunk=4096):
        self.w = torch.as_tensor(weights, dtype=torch.double)
        self.seed, self.chunk = seed, chunk

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed)
        while True:
            yield from torch.multinomial(self.w, self.chunk, replacement=True, generator=g).tolist()


class MixedBatchSampler(torch.utils.data.Sampler):
    """Endless batches for the Stage 10 fine-tune mix: each batch is drawn
    entirely from the mix group (`in_group_mask`) with probability `ratio`,
    else entirely from the rest. Within a group, items are drawn
    proportionally to `weights` like WeightedStream (length-weighted)."""

    def __init__(self, weights, in_group_mask, batch_size, ratio, seed):
        w = torch.as_tensor(weights, dtype=torch.double)
        m = torch.as_tensor(in_group_mask, dtype=torch.bool)
        if w.numel() != m.numel():
            raise ValueError("weights and in_group_mask must have the same length")
        if not bool(m.any()):
            raise ValueError("--mix-speakers selects no training items")
        if bool(m.all()):
            raise ValueError("--mix-speakers selects every training item")
        if not 0.0 < ratio < 1.0:
            raise ValueError(f"--mix-ratio must be in (0, 1), got {ratio}")
        zero = torch.zeros_like(w)
        self.w_in = torch.where(m, w, zero)
        self.w_out = torch.where(m, zero, w)
        self.batch_size = int(batch_size)
        self.ratio = float(ratio)
        self.seed = int(seed)

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed)
        while True:
            pick_in = bool(torch.rand((), generator=g) < self.ratio)
            w = self.w_in if pick_in else self.w_out
            yield torch.multinomial(w, self.batch_size, replacement=True,
                                    generator=g).tolist()


def worker_init(_):
    torch.set_num_threads(1)


# -- validation & checkpointing -----------------------------------------------
def select_val_items(items, limit):
    """Speaker-stratified validation subset using `evallist.evenly_spaced` so
    multi-speaker manifests are not dominated by whichever speaker comes first."""
    if not items or limit is None or limit <= 0 or len(items) <= limit:
        return list(items)
    by_spk = {}
    spk_order = []
    for it in items:
        spk = it[4]
        if spk not in by_spk:
            by_spk[spk] = []
            spk_order.append(spk)
        by_spk[spk].append(it)
    n_spk = len(spk_order)
    if n_spk <= 1:
        return evallist.evenly_spaced(items, limit)
    # Interleave evenly spaced rows per speaker, then cap to `limit`.
    per_spk_cap = max(1, math.ceil(limit / n_spk))
    spaced = {s: evallist.evenly_spaced(by_spk[s], min(len(by_spk[s]), per_spk_cap))
              for s in spk_order}
    interleaved = []
    depth = 0
    while len(interleaved) < limit:
        added = False
        for s in spk_order:
            if depth < len(spaced[s]):
                interleaved.append(spaced[s][depth])
                added = True
                if len(interleaved) >= limit:
                    break
        if not added:
            break
        depth += 1
    return interleaved


@torch.no_grad()
def validate(gen, mel_loss_fn, items, unit_repeat, device, limit,
             lookahead_frames=0, per_speaker_out=None, collect_preds=False):
    gen.eval()
    subset = select_val_items(items, limit)
    losses = []
    preds = []
    by_spk = {}
    for apath, upath, n_u, spk_idx, spk in subset:
        take_u = min(n_u, max(1, 250 // unit_repeat))    # cap at ~5 s per val utterance
        raw_u = np.load(upath, mmap_mode="r")[:take_u].astype(np.int64)
        wav, logf0, voicing, aperiodicity, energy = extract_cond_crop(
            apath, 0, take_u * unit_repeat, preroll=0, lookahead_frames=lookahead_frames
        )
        u_t = torch.from_numpy(raw_u)[None].to(device)
        lf_t = torch.from_numpy(logf0)[None].to(device)
        v_t = torch.from_numpy(voicing)[None].to(device)
        ap_t = torch.from_numpy(aperiodicity)[None].to(device)
        en_t = torch.from_numpy(energy)[None].to(device)
        spk_t = torch.tensor([spk_idx], dtype=torch.int64, device=device)
        wav_t = torch.from_numpy(wav)[None].to(device)
        pred = gen(u_t, lf_t, v_t, ap_t, en_t, spk_t)
        mel_l1, _ = mel_loss_fn(wav_t, pred)
        val = float(mel_l1)
        losses.append(val)
        by_spk.setdefault(spk, []).append(val)
        if collect_preds:
            preds.append(pred.squeeze(0).float().cpu().numpy())
    if isinstance(per_speaker_out, dict):
        per_speaker_out.clear()
        for s, vals in by_spk.items():
            per_speaker_out[s] = float(np.mean(vals))
    gen.train()
    val_mel = float(np.mean(losses)) if losses else 0.0
    return (val_mel, preds) if collect_preds else val_mel


def rng_state():
    np_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "numpy": [np_state[0], torch.from_numpy(np_state[1].astype(np.int64)), *np_state[2:]],
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def set_rng_state(s):
    random.setstate(s["python"])
    torch.set_rng_state(s["torch"])
    n = s["numpy"]
    np.random.set_state((n[0], n[1].numpy().astype(np.uint32), *n[2:]))
    if s["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def save_checkpoint(path, gen, disc, opt_g, opt_d, sched_g, sched_d, step, args,
                    labels_meta, speakers, speaker_stats, extra,
                    extra_metadata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    voices_dict = gen.voices_json(speakers, speaker_stats, extra_metadata)
    contract_json = gen.contract(1).to_json()
    tmp = path.with_name(path.name + ".tmp")
    torch.save({
        "model": gen.state_dict(),
        "discriminator": disc.state_dict(),
        "optimizer_g": opt_g.state_dict(),
        "optimizer_d": opt_d.state_dict(),
        "scheduler_g": sched_g.state_dict(),
        "scheduler_d": sched_d.state_dict(),
        "step": step,
        "config": json.dumps(gen.cfg.to_dict()),
        "contract": contract_json,
        "labels_meta": json.dumps(labels_meta),
        "speakers": list(speakers),
        "speaker_stats": speaker_stats,
        "voices": voices_dict,
        "args": json.dumps(vars(args)),
        "rng": rng_state(),
        **extra,
    }, tmp)
    tmp.replace(path)
    voices_path = path.parent / "voices.json"
    voices_tmp = path.parent / "voices.json.tmp"
    voices_tmp.write_text(json.dumps(voices_dict, indent=2) + "\n", encoding="utf-8")
    voices_tmp.replace(voices_path)


# -- training loop ------------------------------------------------------------
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

    prior_ck = None
    if resume:
        prior_ck = torch.load(resume, map_location="cpu", weights_only=True)
    elif args.finetune_from:
        ft_path = pathlib.Path(args.finetune_from)
        if not ft_path.is_file():
            raise FileNotFoundError(f"{ft_path}: no checkpoint to fine-tune from")
        prior_ck = torch.load(ft_path, map_location="cpu", weights_only=True)

    prior_speakers = prior_ck.get("speakers") if prior_ck else None
    data_info = {}
    train_items, labels_meta, speakers = join_vocoder_items(
        args.train_manifest, args.train_units, args.exclude_targets,
        speakers_order=prior_speakers, log=log, side=args.side,
        default_lang=args.default_lang, subsets=args.train_subsets, info=data_info
    )
    units.check_license(labels_meta, args.include_noncommercial)
    val_items = []
    if args.val_manifest:
        val_items, val_meta, _ = join_vocoder_items(
            args.val_manifest, args.val_units, exclude_targets=None,
            speakers_order=speakers, log=log, side=args.side,
            default_lang=args.default_lang, subsets=args.val_subsets
        )
        units.check_compatible(labels_meta, val_meta)

    keep = ["tokenizer", "label_space", "vocab", "frame_samples", "frame_rate_hz",
            "lookahead_frames", "source"]
    if labels_meta.get("noncommercial"):     # the taint follows the weights; M2 metadata is unchanged
        keep += ["license", "noncommercial"]
    labels_meta_clean = {k: labels_meta.get(k) for k in keep}
    speaker_lang = data_info.get("speaker_lang", {})
    lang_ratios = sampling.parse_ratios(args.lang_ratios)

    cfg = vocoder.VocoderConfig(
        sample_rate=audio.SAMPLE_RATE,
        frame_samples=int(labels_meta["frame_samples"]),
        cond_frame_samples=COND_FS,
        vocab=int(labels_meta["vocab"]),
        n_speakers=len(speakers),
        upsample_initial_channel=args.upsample_initial_channel,
    )
    gen = vocoder.UnitVocoder(cfg).to(device)
    disc = vocoder.VocoderDiscriminator().to(device)
    mel_loss_fn = vocoder.MultiResolutionMelLoss(audio.SAMPLE_RATE).to(device)

    step, best_val, elapsed0 = 0, float("inf"), 0.0
    best_utmos = None
    stopped_early = False
    speaker_stats = {}
    # Resuming keeps the prior checkpoint's speaker_init records; a fresh
    # fine-tune replaces/extends them below when rows are added.
    speaker_init = dict(prior_ck.get("speaker_init") or {}) if prior_ck is not None else {}
    early_stop_since = 0
    # Stage 10: private voice display/licence. CLI metadata wins over whatever
    # the prior checkpoint stored; both end up in the new checkpoint and in
    # voices.json via gen.voices_json's extra_metadata.
    speaker_metadata = {}
    if prior_ck is not None and prior_ck.get("speaker_metadata"):
        pm = prior_ck["speaker_metadata"]
        speaker_metadata.update(json.loads(pm) if isinstance(pm, str) else pm)
    if args.speaker_metadata:
        mp = pathlib.Path(args.speaker_metadata)
        if not mp.is_file():
            raise FileNotFoundError(f"--speaker-metadata {mp} not found")
        speaker_metadata.update(json.loads(mp.read_text(encoding="utf-8")))
    # Milestone 3: what the manifest rows say about a voice (its language, and the
    # licence/source of the multilingual corpora) fills the gaps left by the built-in
    # table, the prior checkpoint and --speaker-metadata. It never overrides them.
    for spk, found in data_info.get("speaker_meta", {}).items():
        derived = {k: val for k, val in found.items()
                   if k not in vocoder.SPEAKER_METADATA.get(spk, {})}
        if derived:
            speaker_metadata[spk] = {**derived, **speaker_metadata.get(spk, {})}
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if prior_ck is not None:
        old_meta = json.loads(prior_ck["labels_meta"])
        # Post-gate fine-tuning drives a teacher-trained vocoder with student ids,
        # which purposefully differ only in lookahead_frames.
        ignore = ("lookahead_frames",) if args.finetune_from and not resume else ()
        units.check_compatible(old_meta, labels_meta_clean, ignore=ignore)
        old_cfg = vocoder.VocoderConfig.from_dict(json.loads(prior_ck["config"]))
        old_gen = vocoder.UnitVocoder(old_cfg)
        old_gen.load_state_dict(prior_ck["model"])
        new_speakers = speakers[old_cfg.n_speakers:] if len(speakers) > old_cfg.n_speakers else []
        if new_speakers:
            # Stage 10: each new row is initialised per --new-speaker-init:
            # "first" copies row 0 (the pre-Stage-10 behaviour), "nearest"
            # picks the ECAPA-closest existing row, or a named existing row.
            mode = str(args.new_speaker_init).strip()
            if mode == "first":
                old_gen.expand_speakers(len(speakers), init_from_row=0)
                speaker_init.update({s: {"from": speakers[0], "cosines": {}}
                                     for s in new_speakers})
            else:
                old_gen.expand_speakers(len(speakers), init_from_row=None)
                existing = speakers[: old_cfg.n_speakers]
                if mode != "nearest" and mode not in existing:
                    raise ValueError(
                        f"--new-speaker-init {mode!r}: expected 'first', 'nearest' "
                        f"or one of the existing speakers {existing}")
                by_spk = {s: [it for it in train_items if it[4] == s] for s in existing}
                embed_fn = EMBED_FN or _lazy_embed(device)
                for i, s in enumerate(new_speakers):
                    if mode == "nearest":
                        src, cosines = nearest_speaker(
                            [it for it in train_items if it[4] == s], by_spk,
                            embed_fn, speaker_name=s)
                    else:
                        src, cosines = mode, {}
                    with torch.no_grad():
                        old_gen.speaker_table.weight[old_cfg.n_speakers + i].copy_(
                            old_gen.speaker_table.weight[speakers.index(src)])
                    speaker_init[s] = {"from": src, "cosines": cosines}
            log(f"[train] new speaker row(s): "
                + ", ".join(f"{s} <- {speaker_init[s]['from']}" for s in new_speakers))
        if old_gen.cfg != cfg:
            raise ValueError("checkpoint vocoder config does not match this run's")
        gen.load_state_dict(old_gen.state_dict())
        if "discriminator" in prior_ck:
            disc.load_state_dict(prior_ck["discriminator"])
        speaker_stats = compute_speaker_stats(
            train_items, speakers, max_utts_per_speaker=args.stats_max_utts,
            jobs=max(1, args.workers), existing=prior_ck.get("speaker_stats")
        )
    else:
        speaker_stats = compute_speaker_stats(
            train_items, speakers, max_utts_per_speaker=args.stats_max_utts,
            jobs=max(1, args.workers)
        )

    # Validate voices.json schema upfront before any training step runs (F-4).
    gen.voices_json(speakers, speaker_stats, speaker_metadata)

    opt_g = torch.optim.AdamW(gen.parameters(), lr=args.lr, betas=(0.8, 0.99),
                              weight_decay=args.weight_decay)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=args.lr, betas=(0.8, 0.99),
                              weight_decay=args.weight_decay)
    steps_per_epoch = max(1, args.steps_per_epoch or (len(train_items) // max(1, args.batch_size)))
    lr_fn = lambda s: args.lr_decay ** (s / steps_per_epoch)  # noqa: E731
    sched_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_fn)
    sched_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lr_fn)

    if resume and prior_ck is not None:
        opt_g.load_state_dict(prior_ck["optimizer_g"])
        opt_d.load_state_dict(prior_ck["optimizer_d"])
        sched_g.load_state_dict(prior_ck["scheduler_g"])
        sched_d.load_state_dict(prior_ck["scheduler_d"])
        set_rng_state(prior_ck["rng"])
        step = int(prior_ck["step"])
        best_val = float(prior_ck.get("best_val", float("inf")))
        best_utmos = prior_ck.get("best_utmos")
        early_stop_since = int(prior_ck.get("early_stop_since", 0))
        elapsed0 = float(prior_ck.get("elapsed", 0.0))
        log(f"[train] resumed {resume} at step {step}")
    elif args.finetune_from:
        log(f"[train] initialized weights from {args.finetune_from}")

    summary = gen.summary()
    d_params = sum(p.numel() for p in disc.parameters())
    log(f"[train] {run}: gen {summary['params'] / 1e6:.2f}M params ({summary['macs_per_second'] / 1e9:.2f} "
        f"GMAC/s), disc {d_params / 1e6:.2f}M params, {len(speakers)} speaker(s) {speakers}, "
        f"{len(train_items)} train / {len(val_items)} val utts, device {device}")

    cond_lookahead = (args.lookahead_frames if getattr(args, "lookahead_frames", None) is not None
                      else int(labels_meta.get("lookahead_frames") or 0))
    crop_u = max(1, round(args.segment_samples / cfg.frame_samples))
    ds = VocoderCropDataset(train_items, crop_u, unit_repeat=cfg.unit_repeat,
                            lookahead_frames=cond_lookahead)
    weights = [max(n, crop_u) for _, _, n, _, _ in train_items]
    if lang_ratios:
        weights, shares = sampling.language_weights(
            weights, [speaker_lang.get(it[4]) for it in train_items], lang_ratios)
        log("[train] language sampling: " + ", ".join(
            f"{lang} {100 * s['share']:.0f}% of draws ({s['items']} utterances, "
            f"{sum(1 for sp in speakers if speaker_lang.get(sp) == lang)} voice(s))"
            for lang, s in shares.items()))
    mix_names = {s.strip() for s in (args.mix_speakers or "").split(",") if s.strip()}
    if mix_names:
        unknown = mix_names - set(speakers)
        if unknown:
            raise ValueError(f"--mix-speakers: unknown speaker(s) {sorted(unknown)}")
        in_group = [it[4] in mix_names for it in train_items]
        batch_sampler = MixedBatchSampler(weights, in_group, args.batch_size,
                                          args.mix_ratio,
                                          seed=args.seed * 1_000_003 + step)
        log(f"[train] mix: {sum(in_group)} of {len(train_items)} utterances in the "
            f"mix group {sorted(mix_names)}, batch-level ratio {args.mix_ratio}")
        loader = torch.utils.data.DataLoader(
            ds, batch_sampler=batch_sampler, num_workers=args.workers,
            pin_memory=device.type == "cuda", worker_init_fn=worker_init,
            persistent_workers=args.workers > 0,
            prefetch_factor=4 if args.workers > 0 else None
        )
    else:
        sampler = WeightedStream(weights, seed=args.seed * 1_000_003 + step)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
            pin_memory=device.type == "cuda", drop_last=True, worker_init_fn=worker_init,
            persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers > 0 else None
        )

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(pathlib.Path(args.runs_dir) / run), purge_step=step if resume else None)
    if not resume:
        writer.add_text("args", json.dumps(vars(args), indent=2))
        writer.add_text("model", json.dumps(summary, indent=2))
        writer.add_text("speaker_stats", json.dumps(speaker_stats, indent=2))

    use_bf16 = device.type == "cuda" and not args.fp32
    t_start, t_ckpt, t_log = time.monotonic(), time.monotonic(), time.monotonic()
    acc_losses = np.zeros(4, dtype=np.float64)   # [d_loss, g_adv, fm, mel]
    acc_n = 0
    nonfinite_streak = 0
    max_nonfinite = 5

    def checkpoint(name):
        save_checkpoint(
            ckpt_dir / name, gen, disc, opt_g, opt_d, sched_g, sched_d, step, args,
            labels_meta_clean, speakers, speaker_stats,
            {"best_val": best_val, "best_utmos": best_utmos,
             "stopped_early": stopped_early, "speaker_init": speaker_init,
             "early_stop_since": stopper.since,
             "speaker_metadata": speaker_metadata, "run_name": run,
             "lang_ratios": json.dumps(lang_ratios) if lang_ratios else None,
             "elapsed": elapsed0 + time.monotonic() - t_start},
            extra_metadata=speaker_metadata
        )

    stopper = EarlyStopper(args.early_stop_patience)
    stopper.since = early_stop_since            # restored on --resume
    stopper.best_val = best_val
    if best_utmos is not None:
        stopper.best_utmos = float(best_utmos)
    utmos_fn = UTMOS_FN or _lazy_utmos(device)

    gen.train()
    disc.train()
    it = iter(loader)
    while step < args.steps:
        wav, u, logf0, voicing, aperiodicity, energy, spk_id = next(it)
        wav = wav.to(device, non_blocking=True)
        u = u.to(device, non_blocking=True)
        logf0 = logf0.to(device, non_blocking=True)
        voicing = voicing.to(device, non_blocking=True)
        aperiodicity = aperiodicity.to(device, non_blocking=True)
        energy = energy.to(device, non_blocking=True)
        spk_id = spk_id.to(device, non_blocking=True)

        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            fake = gen(u, logf0, voicing, aperiodicity, energy, spk_id)
            real_s, _ = disc(wav)
            fake_s_det, _ = disc(fake.detach())
            loss_d = vocoder.discriminator_loss(real_s, fake_s_det)

        if not torch.isfinite(loss_d):
            nonfinite_streak += 1
            log(f"[train] warning: non-finite loss_d at step {step + 1} "
                f"({nonfinite_streak}/{max_nonfinite})")
            if nonfinite_streak >= max_nonfinite:
                writer.close()
                raise RuntimeError(f"aborted after {nonfinite_streak} consecutive non-finite steps")
            continue

        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        dnorm = float(torch.nn.utils.clip_grad_norm_(disc.parameters(), args.clip))
        if math.isfinite(dnorm):
            opt_d.step()

        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            real_s, real_fmap = disc(wav)
            fake_s, fake_fmap = disc(fake)
            loss_adv = vocoder.generator_adv_loss(fake_s)
            loss_fm = vocoder.feature_matching_loss(real_fmap, fake_fmap)
            loss_mel, _ = mel_loss_fn(wav, fake)
            loss_g = loss_adv + args.fm_weight * loss_fm + args.mel_weight * loss_mel

        if not torch.isfinite(loss_g):
            nonfinite_streak += 1
            opt_g.zero_grad(set_to_none=True)
            log(f"[train] warning: non-finite loss_g at step {step + 1} "
                f"({nonfinite_streak}/{max_nonfinite})")
            if nonfinite_streak >= max_nonfinite:
                writer.close()
                raise RuntimeError(f"aborted after {nonfinite_streak} consecutive non-finite steps")
            continue

        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(gen.parameters(), args.clip))
        if not math.isfinite(gnorm):
            nonfinite_streak += 1
            opt_g.zero_grad(set_to_none=True)
            log(f"[train] warning: non-finite grad_norm_g at step {step + 1} "
                f"({nonfinite_streak}/{max_nonfinite})")
            if nonfinite_streak >= max_nonfinite:
                writer.close()
                raise RuntimeError(f"aborted after {nonfinite_streak} consecutive non-finite steps")
            continue

        nonfinite_streak = 0
        opt_g.step()
        sched_g.step()
        sched_d.step()

        step += 1
        acc_losses += [float(loss_d.detach()), float(loss_adv.detach()),
                       float(loss_fm.detach()), float(loss_mel.detach())]
        acc_n += 1

        if step % args.log_every == 0 or step == args.steps:
            dt = time.monotonic() - t_log
            t_log = time.monotonic()
            d_l, adv_l, fm_l, mel_l = acc_losses / max(acc_n, 1)
            writer.add_scalar("train/loss_d", d_l, step)
            writer.add_scalar("train/loss_adv", adv_l, step)
            writer.add_scalar("train/loss_fm", fm_l, step)
            writer.add_scalar("train/loss_mel", mel_l, step)
            writer.add_scalar("train/lr", sched_g.get_last_lr()[0], step)
            writer.add_scalar("train/grad_norm_g", gnorm, step)
            writer.add_scalar("train/steps_per_s", acc_n / max(dt, 1e-9), step)
            log(f"step {step:7d}  mel {mel_l:.3f}  adv {adv_l:.3f}  fm {fm_l:.3f}  "
                f"disc {d_l:.3f}  lr {sched_g.get_last_lr()[0]:.2e}  "
                f"{acc_n / max(dt, 1e-9):.2f} it/s")
            acc_losses[:] = 0.0
            acc_n = 0

        if val_items and (step % args.val_every == 0 or step == args.steps):
            per_spk_val = {}
            val_utmos = None
            if args.val_utmos:
                val_mel, preds = validate(
                    gen, mel_loss_fn, val_items, cfg.unit_repeat, device,
                    args.val_limit, lookahead_frames=cond_lookahead,
                    per_speaker_out=per_spk_val, collect_preds=True)
                val_utmos = float(np.mean(utmos_fn(preds))) if preds else None
                if val_utmos is not None:
                    writer.add_scalar("val/utmos", val_utmos, step)
            else:
                val_mel = validate(gen, mel_loss_fn, val_items, cfg.unit_repeat, device,
                                   args.val_limit, lookahead_frames=cond_lookahead,
                                   per_speaker_out=per_spk_val)
            writer.add_scalar("val/loss_mel", val_mel, step)
            for spk_name, spk_mel in per_spk_val.items():
                writer.add_scalar(f"val/loss_mel_{spk_name}", spk_mel, step)
            per_lang_val = {}
            for spk_name, spk_mel in per_spk_val.items():
                if speaker_lang.get(spk_name):
                    per_lang_val.setdefault(speaker_lang[spk_name], []).append(spk_mel)
            for lang, vals in sorted(per_lang_val.items()):   # English must not regress
                writer.add_scalar(f"val/loss_mel_lang_{lang}", float(np.mean(vals)), step)
            spk_str = (" (" + ", ".join(f"{k}={v:.3f}" for k, v in per_spk_val.items()) + ")"
                       if len(per_spk_val) > 1 else "")
            utmos_str = f"  utmos {val_utmos:.3f}" if val_utmos is not None else ""
            log(f"[val] step {step}: mel_l1 {val_mel:.4f}{spk_str}{utmos_str}")
            if val_mel < best_val:
                best_val = val_mel
                checkpoint("best.pt")
            if val_utmos is not None and (best_utmos is None or val_utmos > best_utmos):
                best_utmos = val_utmos
            stopper.update(val_mel, val_utmos)
            if stopper.stop:
                stopped_early = True
                log(f"[train] early stop at step {step}: no validation improvement "
                    f"for {stopper.patience} validation(s)")
                break

        due = (time.monotonic() - t_ckpt >= args.ckpt_minutes * 60
               or (args.ckpt_steps and step % args.ckpt_steps == 0))
        if due and step < args.steps:
            checkpoint("last.pt")
            t_ckpt = time.monotonic()
            log(f"[ckpt] step {step} -> {ckpt_dir / 'last.pt'}")

    checkpoint("last.pt")
    if not (ckpt_dir / "best.pt").is_file():
        checkpoint("best.pt")
    checkpoint("final.pt")
    writer.close()
    peak = (f", peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"
            if device.type == "cuda" else "")
    log(f"[train] done at step {step} -> {ckpt_dir / 'final.pt'}{peak}")
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[2:])
    )
    p.add_argument("--train-manifest", action="append", required=True,
                   help="JSONL manifest(s) of target-speaker audio (repeatable, comma-separated)")
    p.add_argument("--train-units", action="append", required=True,
                   help="units directory from extract_units.py (repeatable, comma-separated)")
    p.add_argument("--val-manifest", action="append", help="optional validation manifest(s)")
    p.add_argument("--val-units", action="append", help="optional validation units directory")
    p.add_argument("--val-limit", type=int, default=50)
    p.add_argument("--val-every", type=int, default=2000)
    p.add_argument("--lang-ratios", help="share of draws per language: 'm3-vocoder' (40%% en, "
                   "20%% es/fr/de), 'balanced', 'en,es' or 'en=0.4,es=0.2,...' (default: by length)")
    p.add_argument("--default-lang", help="language of rows without a `lang` field (e.g. en)")
    p.add_argument("--train-subsets", help="keep only these manifest subsets (comma-separated)")
    p.add_argument("--val-subsets", help="keep only these manifest subsets for validation")
    p.add_argument("--side", choices=units.SIDES,
                   help="paired S2UT manifests: which audio of each pair to train on")
    p.add_argument("--include-noncommercial", action="store_true",
                   help="allow units from the CC-BY-NC-SA mHuBERT-147 teacher (research only)")
    p.add_argument("--exclude-targets", default=str(paths.EVAL_LISTS / "targets.tsv"),
                   help="held-out target reference TSV(s), comma-separated, to exclude "
                        "from training ('' = none)")
    p.add_argument("--upsample-initial-channel", type=int, default=128,
                   help="generator initial width (128 = V2-class ~1M params; 256/512 = V1-class)")
    p.add_argument("--segment-samples", type=int, default=12800,
                   help="training segment length in samples (default 12800 = 800 ms > 606 ms RF)")
    p.add_argument("--lookahead-frames", type=int, default=None,
                   help="pitch tracker lookahead frames (default: from units_meta.json or 0)")
    p.add_argument("--steps", type=int, default=150_000,
                   help="training steps (150k for gate vocoder; 50k for post-gate fine-tune)")
    p.add_argument("--steps-per-epoch", type=int, default=1000,
                   help="steps per exponential LR decay epoch (0 = dataset length / batch_size)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lr-decay", type=float, default=0.999, help="LR decay factor per epoch")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--mel-weight", type=float, default=45.0)
    p.add_argument("--fm-weight", type=float, default=2.0)
    p.add_argument("--clip", type=float, default=5.0)
    p.add_argument("--stats-max-utts", type=int, default=500,
                   help="max utterances per speaker used to estimate F0/energy stats")
    p.add_argument("--fp32", action="store_true", help="disable bf16 autocast")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--run-name", default=None, help="default voc_<date>_<time>")
    p.add_argument("--runs-dir", default=str(paths.REPO_ROOT / "runs"))
    p.add_argument("--ckpt-dir", default=str(paths.REPO_ROOT / "checkpoints"))
    p.add_argument("--ckpt-minutes", type=float, default=30.0)
    p.add_argument("--ckpt-steps", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--resume", help="'auto' (<ckpt-dir>/<run>/last.pt if present) or a checkpoint path")
    p.add_argument("--finetune-from", help="initialize generator/discriminator from a trained checkpoint")
    p.add_argument("--new-speaker-init", default="first", metavar="{first,nearest,NAME}",
                   help="init for speaker rows added by --finetune-from: 'first' = copy "
                        "row 0 (default), 'nearest' = ECAPA-closest existing row, or an "
                        "existing speaker name")
    p.add_argument("--mix-speakers", default=None,
                   help="comma-separated speakers forming the fine-tune mix group (e.g. 'me')")
    p.add_argument("--mix-ratio", type=float, default=0.5,
                   help="probability a batch is drawn entirely from the mix group")
    p.add_argument("--early-stop-patience", type=int, default=0,
                   help="stop after N validations without improvement (0 = off)")
    p.add_argument("--val-utmos", action="store_true",
                   help="score validation resynthesis with UTMOS (early stopping, tensorboard)")
    p.add_argument("--speaker-metadata", default=None,
                   help="JSON {speaker: {display, license, source}} merged into the "
                        "checkpoint's speaker_metadata and voices.json")
    return p.parse_args(argv)


def main(argv=None):
    for stream_io in (sys.stdout, sys.stderr):
        if hasattr(stream_io, "reconfigure"):
            stream_io.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    args = parse_args(argv)
    if bool(args.val_manifest) != bool(args.val_units):
        print("[train_unit_vocoder] error: --val-manifest and --val-units go together",
              file=sys.stderr)
        return 1
    if args.resume and args.resume != "auto" and not args.run_name:
        try:
            ck = torch.load(args.resume, map_location="cpu", weights_only=True)
            args.run_name = ck.get("run_name") if isinstance(ck, dict) else None
        except Exception as e:
            print(f"[train_unit_vocoder] error: {e}", file=sys.stderr)
            return 1
    if not args.run_name:
        if args.resume == "auto":
            print("[train_unit_vocoder] error: --resume auto needs --run-name", file=sys.stderr)
            return 1
        args.run_name = "voc_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        return train(args)
    except (FileNotFoundError, ValueError, KeyError, TypeError, RuntimeError, units.UnitsMismatch) as e:
        print(f"[train_unit_vocoder] error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
