#!/usr/bin/env python3
"""Waveform -> discrete unit ids (plus causal prosody and student latents) for
every content-encoder candidate.

Milestone 2 / Stage 04 (Documents/Milestone 2/04_content_encoder_candidates.md)
and Milestone 3 / Stage 03 (Documents/Milestone 3/03_multilingual_encoder_vocoder_upgrade.md).

Tokenizers (all at 16 kHz input; see fvml/units.py for the on-disk format):
  teacher-km100   HuBERT-Base layer 7 + k-means(100); offline, non-causal; 50 Hz
  teacher-mhubert-km1000     mhubert_base_vp_en_es_fr_it3 layer 11 + k-means(1000)
                             fitted on balanced {en,es,fr,de} (M3 default, MIT)
  teacher-mhubert147-km1000  mHuBERT-147 (CC-BY-NC-SA): research probe only,
                             refused without --include-noncommercial and written
                             under a .../nc/... directory
  student-L0/1/2  causal content encoder head with lookahead L frames; 50 Hz
  mimi-q0         Kyutai Mimi semantic codebook 0; causal; 12.5 Hz (kill test)

Output: <out-dir>/<utt_id>.npy (int16 ids, frame-aligned), manifest.jsonl and
units_meta.json (frame rate, lookahead, vocab, label space, licence and sources,
so Stage 05 cannot mix incompatible streams: fvml.units.check_compatible).
Students can also write <utt_id>.logits.npy (float16, pre-softmax) for the
soft-unit fallback. Rerunning into the same directory skips utterances already
done and refuses a directory written by an incompatible tokenizer.

Milestone 3 additions:
  --prosody        also write <utt_id>.prosody.npy, float32 (frames, 4) =
                   [logf0 (held), voicing, aperiodicity, log_energy] from the
                   Stage 03 causal YIN tracker on the clean audio, and
                   prosody_stats.json (per-speaker log-f0 / energy mean and std:
                   the z-normalisation Stage 06's prosody targets and voices.json
                   need). A directory labelled earlier can gain prosody on a rerun
                   without running the tokenizer again.
  --save-latents   students: also write <utt_id>.latents.npy, the D-dim `latents`
                   port per frame (float16 by default), so Stage 05 trains the
                   translator on what the C++ encoder emits at run time.
  --side src|tgt   which audio of a paired S2UT manifest row to read (target
                   units + prosody: --side tgt; source latents: --side src).
  --default-lang   language tag for rows that carry none (the M2 English corpora).
  --subsets        keep only rows of these subsets (e.g. train).
  --silence-latents OUT.npy   instead of labelling: the student's `latents` on
                   room tone ([T, D] float32; synthetic noise floors at
                   --silence-dbfs levels, plus any --manifest recordings), the
                   bank Stage 05's train_s2ut.py cuts its silence tails from.

Pitch for the *vocoder trainer* is still computed on the fly by
train_unit_vocoder.py (fvml/features.py, Stage 03), with the lookahead of the
chosen encoder head. The old 12.5 Hz / Mimi-default / RMVPE-SwiftF0 stub is gone;
those choices are superseded or non-causal (MEMORY.md §1).

Speaker perturbation (Stage 10): --perturb-fraction encodes a deterministic
subset of clips from a formant/pitch/EQ-perturbed copy (fvml/augment.py), so
the target-voice vocoder learns to ignore residual speaker information in the
units. The manifest row's `path` still names the clean audio, so the trainer's
conditioning features always come from the clean clip; perturbed rows gain
"perturbed": true and a "perturb" parameter record, and units_meta.json
records the config ("perturbation", null when off). Rerunning into a
directory extracted with a different perturbation config raises UnitsMismatch.

Usage:
  # 1. fit the teacher's k-means (writes km100_centroids.npy + km100_meta.json)
  extract_units.py --fit-kmeans --manifest librispeech-train-clean-100.jsonl \\
      --out-dir units/ [--hours 100] [--max-frames 500000] [--seed 0]
  # 1b. M3: the balanced multilingual K=1000 codebook (25 h per language)
  extract_units.py --fit-kmeans --teacher mhubert-vp --balance-langs en,es,fr,de \\
      --hours 100 --default-lang en --side src --manifest <en>,<es>,<fr>,<de> \\
      --out-dir units/      # -> units/mhubert_vp_en_es_fr_de_km1000.npy
  # 1c. M3: import the release's own km1000 for the probe comparison
  extract_units.py --import-kmeans mhubert_base_vp_en_es_fr_it3_L11_km1000.bin \\
      --teacher mhubert-vp --out-dir units/stock/
  # 2. label corpora with the teacher
  extract_units.py --tokenizer teacher-km100 --centroids units/ \\
      --manifest librispeech-train-clean-100.jsonl --out-dir data/units/teacher-km100/tc100
  extract_units.py --tokenizer teacher-mhubert-km1000 --centroids units/ --prosody \\
      --side tgt --manifest s2ut/cvss-c-es_en.jsonl --out-dir data/units/mhubert/cvss-c-es_en.tgt
  # 3. a student head or Mimi
  extract_units.py --tokenizer student-L1 --checkpoint checkpoints/<run>/final.pt ...
  extract_units.py --tokenizer student-L1 --checkpoint ... --save-latents --side src \\
      --manifest s2ut/cvss-c-es_en.jsonl --out-dir data/units/student/cvss-c-es_en.src
  extract_units.py --tokenizer mimi-q0 --manifest librispeech-dev-clean.jsonl ...
  # 4. M3: the silence-latent bank train_s2ut.py --silence-latents reads
  extract_units.py --silence-latents data/units/student/silence.latents.npy \\
      --checkpoint checkpoints/<run>/final.pt [--silence-seconds 60] [--manifest room_tone.jsonl]

Manifest paths may be absolute or relative to the data root (data/manifests/...
is tried for bare names). Exit status: 0 = done, 1 = missing inputs, a licence
refusal or a mismatched output directory.
"""
import argparse
import dataclasses
import json
import math
import pathlib
import random
import sys
import time

import numpy as np

from fvml import audio, paths, units

LATENT_DTYPES = {"f16": np.float16, "f32": np.float32}


def pick_rows(rows, hours, limit, rng):
    """A random subset: shuffled, cut at `hours` of audio, then at `limit` rows."""
    order = rows[:]
    rng.shuffle(order)
    if hours:
        budget, picked = hours * 3600, []
        for r in order:
            if budget <= 0:
                break
            picked.append(r)
            budget -= r.get("duration", 0)
        order = picked
    if limit:
        order = order[:limit]
    return order


def require_nc_dir(out_dir, noncommercial):
    """Non-commercial outputs live under a directory named `nc` (Stage 01's
    isolation rule), so a default training run cannot pick them up by path."""
    if noncommercial and "nc" not in pathlib.Path(out_dir).resolve().parts:
        raise ValueError(f"{out_dir}: non-commercial outputs must be written under an 'nc' "
                         "directory, e.g. <data root>/nc/units/...")


def teacher_checkpoint(args, spec, log=print):
    """The fairseq file for `spec`: --teacher-checkpoint, or the archive path,
    downloaded first when --download-teacher is given."""
    if spec.source != "fairseq":
        return None
    if args.download_teacher and not args.teacher_checkpoint:
        from fvml import teacher
        return teacher.fetch_checkpoint(spec, log)
    return args.teacher_checkpoint


def fit_kmeans(args, rows, manifests, log=print):
    from fvml import teacher
    from fvml.content_encoder import pad_to_frames
    spec = teacher.get_teacher(args.teacher or teacher.DEFAULT_TEACHER, args.include_noncommercial)
    corpus_nc = units.check_rows_license(rows, args.include_noncommercial)
    require_nc_dir(args.out_dir, spec.noncommercial or corpus_nc is not None)
    k = args.k or spec.k
    rng = random.Random(args.seed)
    langs = [s.strip() for s in (args.balance_langs or "").split(",") if s.strip()]
    if langs:
        by_lang = {lang: [r for r in rows if r.get("lang") == lang] for lang in langs}
        missing = [lang for lang in langs if not by_lang[lang]]
        if missing:
            raise ValueError(f"--balance-langs: no manifest rows tagged {', '.join(missing)} "
                             "(rows without a `lang` field need --default-lang)")
        per_hours = args.hours / len(langs) if args.hours else None
        per_limit = max(1, args.limit // len(langs)) if args.limit else None
        picked = {lang: pick_rows(by_lang[lang], per_hours, per_limit, rng) for lang in langs}
        order = [(lang, r) for lang in langs for r in picked[lang]]
    else:
        order = [(None, r) for r in pick_rows(rows, args.hours, args.limit, rng)]
    model = teacher.HubertTeacher(args.device, spec=spec,
                                  checkpoint=teacher_checkpoint(args, spec, log))
    t0 = time.monotonic()

    def feats():
        for i, (lang, r) in enumerate(order):
            f = model.features(pad_to_frames(audio.load(paths.resolve(r["path"]))))
            yield (lang, f) if langs else f
            if (i + 1) % 200 == 0:
                log(f"  {i + 1}/{len(order)} utterances, {time.monotonic() - t0:.0f} s")

    if langs:
        centroids, info = teacher.fit_kmeans_balanced(feats(), langs, k=k,
                                                      max_frames=args.max_frames,
                                                      seed=args.seed, log=log)
        info["languages"] = {
            lang: {"utterances": len(picked[lang]),
                   "hours_used": round(sum(r.get("duration", 0) for r in picked[lang]) / 3600, 3)}
            for lang in langs}
    else:
        centroids, info = teacher.fit_kmeans(feats(), k=k, max_frames=args.max_frames,
                                             seed=args.seed, log=log)
    info["hours_requested"] = args.hours
    info["hours_used"] = round(sum(r.get("duration", 0) for _, r in order) / 3600, 3)
    meta = teacher.save_kmeans(args.out_dir, centroids, info, model.info(), manifests, spec)
    log(f"[kmeans] inertia/frame {meta['inertia_per_frame']:.2f}; wrote "
        f"{pathlib.Path(args.out_dir) / spec.centroids_file} (sha256 {meta['centroids_sha256'][:12]})")
    return 0


def import_kmeans(args, log=print):
    """A published scikit-learn k-means (joblib .bin) -> centroids .npy + meta,
    named after the source file so it cannot be mistaken for our own fit."""
    from fvml import teacher
    spec = teacher.get_teacher(args.teacher or teacher.MULTILINGUAL_TEACHER,
                               args.include_noncommercial)
    require_nc_dir(args.out_dir, spec.noncommercial)
    src = pathlib.Path(args.import_kmeans)
    md5 = spec.stock_kmeans_md5 if args.kmeans_md5 is None else args.kmeans_md5
    md5 = None if str(md5).lower() in ("", "none", "skip") else md5
    centroids = teacher.import_sklearn_kmeans(src, md5)
    named = dataclasses.replace(spec, centroids_file=src.stem + ".npy", meta_file=src.stem + ".json")
    info = {"k": int(len(centroids)), "fit": "imported", "source_file": src.name,
            "source_md5": md5, "source_url": spec.stock_kmeans_url or None}
    who = {"teacher": spec.name, "model": spec.url or spec.model_id, "layer": spec.layer,
           "license": spec.license, "noncommercial": spec.noncommercial}
    meta = teacher.save_kmeans(args.out_dir, centroids, info, who, [], named)
    log(f"[kmeans] imported {src.name}: {centroids.shape[0]} x {centroids.shape[1]} centroids -> "
        f"{pathlib.Path(args.out_dir) / named.centroids_file} "
        f"(sha256 {meta['centroids_sha256'][:12]})")
    return 0


def prosody_of(wav, frame_samples, cfg):
    """((frames, 4) float32 [logf0_held, voicing, aperiodicity, log_energy], the
    sums behind the speaker statistics) from the causal tracker; the signal is
    padded to whole unit frames first, so the prosody lines up with the ids."""
    from fvml import features
    from fvml.content_encoder import pad_to_frames
    f = features.extract(pad_to_frames(np.asarray(wav, np.float32), frame_samples), cfg)
    col = {name: f[:, i] for i, name in enumerate(features.FEATURES)}
    pros = np.stack([col["logf0_held"], col["voicing"], col["aperiodicity"], col["log_energy"]],
                    axis=1).astype(np.float32)
    voiced = col["f0_hz"] > 0.0
    lf, e = np.log(col["f0_hz"][voiced]), col["log_energy"]
    sums = [int(voiced.sum()), float(lf.sum()), float((lf ** 2).sum()),
            int(len(e)), float(e.sum()), float((e ** 2).sum())]
    return pros, sums


def prosody_stats(rows):
    """{speaker: logf0 mean/std over voiced frames, energy mean/std over all
    frames} from the per-utterance sums in the manifest rows. Same estimator and
    floors as train_unit_vocoder.compute_speaker_stats."""
    from fvml import features
    cfg = features.PitchConfig()
    acc = {}
    for r in rows:
        if r.get("prosody_sums"):
            a = acc.setdefault(r.get("speaker") or "default", {"sums": np.zeros(6), "lang": set()})
            a["sums"] += np.asarray(r["prosody_sums"], np.float64)
            if r.get("lang"):
                a["lang"].add(r["lang"])
    out = {}
    for spk, a in sorted(acc.items()):
        n_lf, s_lf, q_lf, n_e, s_e, q_e = a["sums"]
        lf_mean = s_lf / n_lf if n_lf else features.LOGF0_PRIOR_MEAN
        lf_std = max(math.sqrt(max(q_lf / n_lf - lf_mean ** 2, 0.0)), cfg.logf0_std_floor) \
            if n_lf else features.LOGF0_PRIOR_STD
        e_mean = s_e / n_e if n_e else features.ENERGY_PRIOR_MEAN
        e_std = max(math.sqrt(max(q_e / n_e - e_mean ** 2, 0.0)), cfg.energy_std_floor) \
            if n_e else features.ENERGY_PRIOR_STD
        out[spk] = {"logf0_mean": round(float(lf_mean), 6), "logf0_std": round(float(lf_std), 6),
                    "energy_mean": round(float(e_mean), 6), "energy_std": round(float(e_std), 6),
                    "voiced_frames": int(n_lf), "frames": int(n_e)}
        if len(a["lang"]) == 1:
            out[spk]["lang"] = next(iter(a["lang"]))
    return out


# Options added after Milestone 2 (Stage 03 of Milestone 3), with the values that
# reproduce the M2 behaviour. extract() fills them in for callers that build their
# own argument namespace and predate them (scripts, tests).
EXTRACT_DEFAULTS = {"include_noncommercial": False, "teacher_checkpoint": None,
                    "prosody": False, "prosody_lookahead": 0, "save_latents": None,
                    "save_logits": False, "limit": None, "perturb_fraction": 0.0,
                    "perturb_seed": 0, "perturb_formant": "0.10,0.15", "perturb_pitch_st": 2.0}


def with_defaults(args):
    """A copy of `args` (argparse or SimpleNamespace) with every option extract()
    reads present; attributes the caller set are kept as they are."""
    return argparse.Namespace(**{**EXTRACT_DEFAULTS, **vars(args)})


def extract(args, rows, manifests, log=print):
    from fvml import augment, features
    args = with_defaults(args)
    out_dir = pathlib.Path(args.out_dir)
    tok = units.make_tokenizer(args.tokenizer, centroids=args.centroids,
                               checkpoint=args.checkpoint, device=args.device,
                               include_noncommercial=args.include_noncommercial,
                               teacher_checkpoint=args.teacher_checkpoint)
    units.check_license(tok.meta, args.include_noncommercial)   # a student of an NC teacher
    corpus_nc = units.check_rows_license(rows, args.include_noncommercial)
    require_nc_dir(out_dir, tok.meta.get("noncommercial") or corpus_nc is not None)
    meta = {**tok.meta, "manifests": [str(m) for m in manifests], "created": units.now()}
    if corpus_nc:                    # BUG-031: the corpus taint survives a clean teacher
        meta["noncommercial"] = True
        meta["license"] = " + ".join(dict.fromkeys(l for l in (
            tok.meta.get("license") if tok.meta.get("noncommercial") else None, corpus_nc) if l))
    perturb_cfg = None
    if args.perturb_fraction > 0.0:
        lo, hi = (float(v) for v in args.perturb_formant.split(","))
        perturb_cfg = augment.PerturbConfig(formant_range=(lo, hi),
                                            pitch_st=args.perturb_pitch_st,
                                            fraction=args.perturb_fraction)
        meta["perturbation"] = {"fraction": args.perturb_fraction,
                                "formant_range": [lo, hi],
                                "pitch_st": args.perturb_pitch_st,
                                "eq_gain_db": perturb_cfg.eq_gain_db,
                                "seed": args.perturb_seed,
                                "conditioning": "clean"}
    else:
        meta["perturbation"] = None
    pitch_cfg = None
    meta["prosody"] = meta["latents"] = None
    if args.prosody:
        pitch_cfg = features.PitchConfig(lookahead_frames=args.prosody_lookahead)
        if tok.frame_samples % pitch_cfg.frame_samples:
            raise ValueError(f"--prosody: the tracker's {pitch_cfg.frame_samples}-sample hop does "
                             f"not divide the {tok.frame_samples}-sample unit frame")
        meta["prosody"] = {"columns": list(units.PROSODY_COLUMNS), "dtype": "float32",
                           "frame_samples": pitch_cfg.frame_samples,
                           "lookahead_frames": args.prosody_lookahead,
                           "tracker": "causal YIN, fvml.features (Stage 03), clean audio",
                           "stats": units.PROSODY_STATS_FILE}
    if args.save_latents:
        if not hasattr(tok, "encode_all"):
            raise ValueError("--save-latents needs a student tokenizer: `latents` is the causal "
                             "encoder's port, a teacher has none")
        meta["latents"] = {"dim": int(tok.latent_dim), "dtype": f"float{args.save_latents[1:]}",
                           "alignment": "frame t = trunk frame t (no lookahead delay)",
                           "port": "LayerNorm output read by the unit heads (contract v2 latents)"}
    have = {}
    if (out_dir / units.META_FILE).is_file():
        old, old_rows = units.read_units_dir(out_dir)
        units.check_compatible(old, meta)

        def identity(m):
            s = m.get("source", {})
            return (m.get("tokenizer"), s.get("revision"), s.get("checkpoint_sha256"),
                    m.get("perturbation"))
        if identity(old) != identity(meta):
            raise units.UnitsMismatch(f"{out_dir} holds {old.get('tokenizer')} units from another "
                                      "model/checkpoint; use a new --out-dir")
        for key, flag in (("prosody", "--prosody"), ("latents", "--save-latents")):
            if old.get(key) and old[key] != meta[key]:      # adding one later is fine
                raise units.UnitsMismatch(
                    f"{out_dir} was extracted with other {flag} settings ({old[key]}); pass "
                    "the same ones or use a new --out-dir")
        have = {r["utt_id"]: r for r in old_rows if (out_dir / r["units"]).is_file()}
        meta["created"] = old.get("created", meta["created"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / units.META_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    def has(row, key):
        return bool(row.get(key)) and (out_dir / row[key]).is_file()

    todo = rows[:args.limit] if args.limit else rows
    out_rows, t0, frames, skipped, added = [], time.monotonic(), 0, 0, 0
    for i, r in enumerate(todo):
        row = have.get(r["utt_id"])
        need_units = row is None or (args.save_latents and not has(row, "latents"))
        need_prosody = pitch_cfg is not None and (need_units or not has(row, "prosody"))
        if not (need_units or need_prosody):
            out_rows.append(row)
            skipped += 1
            continue
        wav = audio.load(paths.resolve(r["path"]))
        name = units.safe_name(r["utt_id"])
        if need_units:
            src = wav
            perturbed = augment.should_perturb(r["utt_id"], args.perturb_fraction,
                                               args.perturb_seed) \
                if perturb_cfg else False
            perturb_info = None
            if perturbed:
                src, perturb_info = augment.perturb_speaker(
                    wav, augment.perturb_rng(r["utt_id"], args.perturb_seed), cfg=perturb_cfg)
            if args.save_latents:
                ids, logits, latents = tok.encode_all(src)
                np.save(out_dir / f"{name}.latents.npy",
                        latents.astype(LATENT_DTYPES[args.save_latents]))
            else:
                ids, logits = tok.encode(src)
            np.save(out_dir / f"{name}.npy", ids)
            if args.save_logits and logits is not None:
                np.save(out_dir / f"{name}.logits.npy", logits)
            frames += len(ids)
            row = {"utt_id": r["utt_id"], "units": f"{name}.npy", "n_frames": int(len(ids)),
                   "path": r["path"], **{k: r[k] for k in units.COPY_FIELDS if k in r}}
            if args.save_latents:
                row["latents"] = f"{name}.latents.npy"
            if perturbed:
                row.update(perturbed=True, perturb=perturb_info)
        else:
            row = dict(row)
            added += 1
        if need_prosody:                    # always from the clean clip
            pros, sums = prosody_of(wav, tok.frame_samples, pitch_cfg)
            want = row["n_frames"] * (tok.frame_samples // pitch_cfg.frame_samples)
            if len(pros) != want:
                raise ValueError(f"{r['utt_id']}: {len(pros)} prosody frames for {want} "
                                 "conditioning frames")
            np.save(out_dir / f"{name}.prosody.npy", pros)
            row.update(prosody=f"{name}.prosody.npy", prosody_sums=sums)
        out_rows.append(row)
        if (i + 1) % 200 == 0:
            el = time.monotonic() - t0
            log(f"  {i + 1}/{len(todo)} utterances, {frames} frames, {el:.0f} s")
            write_manifest(out_dir, out_rows, have)
    all_rows = write_manifest(out_dir, out_rows, have)
    if meta["prosody"]:
        (out_dir / units.PROSODY_STATS_FILE).write_text(
            json.dumps(prosody_stats(all_rows), indent=2) + "\n", encoding="utf-8")
    extras = "".join(f", {what}" for what, on in (("prosody", meta["prosody"]),
                                                  ("latents", meta["latents"])) if on)
    log(f"[extract_units] {args.tokenizer}: {len(out_rows) - skipped - added} new + {skipped} "
        f"existing utterances" + (f" ({added} completed with prosody)" if added else "")
        + f" -> {out_dir} ({meta['frame_rate_hz']} Hz, vocab {meta['vocab']}, "
        f"lookahead {meta['lookahead_frames']}{extras}) in {time.monotonic() - t0:.0f} s")
    return 0


SILENCE_SEGMENT_SECONDS = 2.0


def silence_audio(seconds, dbfs_range, rng, room_tone=()):
    """16 kHz room tone for the silence bank: a digital-silence segment, then
    SILENCE_SEGMENT_SECONDS segments of low-passed Gaussian noise (a mic/room
    floor) at RMS levels drawn uniformly in `dbfs_range`, then each clip of
    `room_tone` (real recordings) scaled to a level drawn the same way."""
    from scipy.signal import lfilter
    lo, hi = dbfs_range
    seg = int(round(SILENCE_SEGMENT_SECONDS * audio.SAMPLE_RATE))
    parts = [np.zeros(seg, np.float32)]
    total = int(round(seconds * audio.SAMPLE_RATE))
    while sum(len(p) for p in parts) < total:
        pole = rng.uniform(0.0, 0.95)                # white (0) to strongly low-passed
        x = lfilter([1.0 - pole], [1.0, -pole], rng.standard_normal(seg))
        level = 10.0 ** (rng.uniform(lo, hi) / 20.0)
        parts.append((x * level / max(float(np.sqrt(np.mean(x ** 2))), 1e-12)).astype(np.float32))
    for clip in room_tone:
        clip = np.asarray(clip, np.float64)
        rms = float(np.sqrt(np.mean(clip ** 2))) if len(clip) else 0.0
        if rms > 0.0:
            parts.append((clip * 10.0 ** (rng.uniform(lo, hi) / 20.0) / rms).astype(np.float32))
    return np.concatenate(parts)


def silence_latents(args, rows, manifests, log=print):
    """The `--silence-latents` bank train_s2ut.py cuts its silence tails from:
    the student's `latents` [T, D] float32 on room tone, written to
    args.silence_latents with a JSON sidecar (<stem>.json) that records how."""
    import torch
    from fvml import content_encoder as ce
    out = pathlib.Path(args.silence_latents)
    if out.suffix != ".npy":
        raise ValueError(f"--silence-latents {out}: give a .npy file name")
    if not args.checkpoint:
        raise ValueError("--silence-latents needs --checkpoint (the student whose latents "
                         "the translator will read)")
    model, ckpt = ce.ContentEncoder.from_checkpoint(args.checkpoint, map_location=args.device)
    model.eval()
    label_meta = json.loads(ckpt.get("labels_meta", "{}"))
    units.check_license(label_meta, args.include_noncommercial)
    require_nc_dir(out.parent, label_meta.get("noncommercial"))
    lo, hi = sorted(float(v) for v in args.silence_dbfs.split(","))
    room = [audio.load(paths.resolve(r["path"])) for r in rows]
    wav = silence_audio(args.silence_seconds, (lo, hi), np.random.default_rng(args.seed), room)
    wav = ce.pad_to_frames(wav)
    # A zero conv state is not "after a long silence": every cache starts at 0, not at
    # the steady activations of silent input. Run the receptive field of digital
    # silence first and drop those frames, so the bank holds steady-state latents only.
    warm = -(-model.past_context_samples() // ce.FRAME_SAMPLES)
    x = np.concatenate([np.zeros(warm * ce.FRAME_SAMPLES, np.float32), wav])
    with torch.inference_mode():
        lat = model(torch.from_numpy(x)[None].to(args.device))["latents"][0, warm:]
    lat = lat.float().cpu().numpy()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, lat.astype(np.float32))
    from fvml.teacher import file_sha256
    info = {"kind": "silence latents (train_s2ut.py --silence-latents)",
            "frames": int(lat.shape[0]), "dim": int(lat.shape[1]), "dtype": "float32",
            "frame_rate_hz": ce.FRAME_RATE_HZ, "seconds": round(len(wav) / audio.SAMPLE_RATE, 3),
            "levels_dbfs": [lo, hi], "segment_seconds": SILENCE_SEGMENT_SECONDS,
            "warmup_frames_dropped": int(warm),
            "synthetic_seconds": args.silence_seconds, "seed": args.seed,
            "room_tone_clips": len(room), "manifests": [str(m) for m in manifests],
            "checkpoint": str(args.checkpoint), "checkpoint_sha256": file_sha256(args.checkpoint),
            "step": ckpt.get("step"), "labels": label_meta.get("tokenizer"),
            "license": label_meta.get("license"),
            "noncommercial": bool(label_meta.get("noncommercial", False)),
            "port": "LayerNorm output read by the unit heads (contract v2 latents)",
            "created": units.now()}
    out.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    log(f"[extract_units] silence bank: {lat.shape[0]} x {lat.shape[1]} latents "
        f"({info['seconds']:.0f} s of room tone at {lo:g}..{hi:g} dBFS"
        + (f", {len(room)} recorded clip(s)" if room else "") + f") -> {out}")
    return 0


def write_manifest(out_dir, new_rows, done):
    from fvml.datasets import write_jsonl
    seen = {r["utt_id"] for r in new_rows}
    rows = new_rows + [r for k, r in done.items() if k not in seen]   # keep rows from earlier runs
    write_jsonl(out_dir / units.MANIFEST_FILE, rows)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.splitlines()[2:]))
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tokenizer", choices=units.TOKENIZERS)
    mode.add_argument("--fit-kmeans", action="store_true",
                      help="fit the teacher's k-means instead of extracting units")
    mode.add_argument("--import-kmeans", metavar="BIN",
                      help="convert a published scikit-learn k-means (joblib) to centroids")
    mode.add_argument("--silence-latents", metavar="NPY",
                      help="write the student's latents on room tone ([T, D] float32): the "
                      "train_s2ut.py --silence-latents bank (needs --checkpoint)")
    p.add_argument("--manifest", action="append",
                   help="JSONL manifest (repeatable or comma-separated)")
    p.add_argument("--out-dir", help="output directory (every mode but --silence-latents)")
    p.add_argument("--centroids", help="teacher-*: the centroids .npy or its directory")
    p.add_argument("--checkpoint", help="student-L*: a train_content_encoder.py checkpoint")
    p.add_argument("--save-logits", action="store_true", help="students: also write float16 logits")
    p.add_argument("--save-latents", nargs="?", const="f16", choices=sorted(LATENT_DTYPES),
                   help="students: also write the per-frame latents (default dtype f16)")
    p.add_argument("--prosody", action="store_true",
                   help="also write causal-YIN prosody [logf0, voicing, aperiodicity, log_energy]")
    p.add_argument("--prosody-lookahead", type=int, default=0,
                   help="pitch tracker lookahead in frames (default 0; >= 1 centres the window)")
    p.add_argument("--side", choices=units.SIDES,
                   help="paired S2UT manifests: which audio of each pair to read")
    p.add_argument("--default-lang", help="language of rows without a `lang` field (e.g. en)")
    p.add_argument("--subsets", help="keep only rows of these subsets (comma-separated)")
    p.add_argument("--limit", type=int, help="first N utterances only (smoke runs)")
    p.add_argument("--device", default=None, help="default cuda if available, else cpu")
    t = p.add_argument_group("teacher")
    t.add_argument("--teacher", help="--fit-kmeans / --import-kmeans: hubert-base (default for "
                   "a fit), mhubert-vp or mhubert-147")
    t.add_argument("--teacher-checkpoint", help="fairseq teachers: the .pt file (default "
                   "<data root>/archives/<file>)")
    t.add_argument("--download-teacher", action="store_true",
                   help="fairseq teachers: download the checkpoint first (1.1 GB for mhubert-vp)")
    t.add_argument("--include-noncommercial", action="store_true",
                   help="allow the CC-BY-NC-SA mHuBERT-147 research teacher")
    g = p.add_argument_group("--fit-kmeans")
    g.add_argument("--k", type=int, help="clusters (default: the teacher's, 100 or 1000)")
    g.add_argument("--hours", type=float, help="audio to run HuBERT on (random utterances); default all")
    g.add_argument("--balance-langs", help="languages to balance, e.g. en,es,fr,de: --hours is "
                   "split evenly and every language contributes the same number of frames")
    g.add_argument("--max-frames", type=int, default=500_000,
                   help="reservoir size: frames the k-means sees (500k x 768 floats = 1.5 GB)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--kmeans-md5", help="--import-kmeans: expected MD5 (default: the teacher's "
                   "published one; 'none' for another trusted file)")
    s = p.add_argument_group("--silence-latents")
    s.add_argument("--silence-seconds", type=float, default=60.0,
                   help="seconds of synthetic room tone (default 60; --manifest clips are added)")
    s.add_argument("--silence-dbfs", default="-90,-45",
                   help="RMS level range of the room tone in dBFS (default -90,-45; write it as --silence-dbfs=-80,-50, the value starts with a minus)")
    h = p.add_argument_group("speaker perturbation (Stage 10)")
    h.add_argument("--perturb-fraction", type=float, default=0.0,
                   help="fraction of clips encoded from a perturbed copy (0 = off)")
    h.add_argument("--perturb-seed", type=int, default=0)
    h.add_argument("--perturb-formant", default="0.10,0.15",
                   help="formant ratio |1 - r| range, e.g. 0.10,0.15")
    h.add_argument("--perturb-pitch-st", type=float, default=2.0,
                   help="pitch shift range in semitones (uniform ±)")
    args = p.parse_args(argv)
    if not (args.import_kmeans or args.silence_latents) and not args.manifest:
        p.error("--manifest is required")
    if not args.silence_latents and not args.out_dir:
        p.error("--out-dir is required")
    if args.silence_latents:
        try:
            lo, hi = (float(v) for v in args.silence_dbfs.split(","))
        except ValueError:
            p.error("--silence-dbfs wants two levels, e.g. -90,-45")
        if max(lo, hi) >= 0.0 or args.silence_seconds <= 0.0:
            p.error("--silence-dbfs levels must be < 0 and --silence-seconds > 0")
    if args.prosody_lookahead < 0:
        p.error("--prosody-lookahead must be >= 0")
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    sys.stdout.reconfigure(line_buffering=True)
    try:
        if args.import_kmeans:
            return import_kmeans(args)
        if args.silence_latents:
            rows, manifests = units.read_manifests(args.manifest) if args.manifest else ([], [])
            return silence_latents(args, units.filter_subsets(rows, args.subsets), manifests)
        rows, manifests = units.read_manifests(args.manifest)
        rows = units.filter_subsets(units.audio_rows(rows, args.side, args.default_lang),
                                    args.subsets)
        if not rows:
            raise ValueError(f"no manifest rows left (subsets {args.subsets})")
        if args.fit_kmeans:
            return fit_kmeans(args, rows, manifests)
        if args.download_teacher and args.tokenizer.startswith("teacher-"):
            from fvml import teacher
            spec = teacher.get_teacher(args.tokenizer, args.include_noncommercial)
            args.teacher_checkpoint = teacher_checkpoint(args, spec)
        return extract(args, rows, manifests)
    except (FileNotFoundError, ValueError) as e:
        print(f"[extract_units] error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
