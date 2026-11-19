"""Discrete unit streams on disk (Milestone 2 / Stage 04; multilingual
labelling, prosody and student latents added in Milestone 3 / Stage 03).

A units directory, written by extract_units.py, holds

  units_meta.json   tokenizer, frame_rate_hz, frame_samples, lookahead_frames,
                    vocab, label_space, padding, model/centroid/checkpoint sources,
                    license / noncommercial, and (when written) the prosody and
                    latents layouts
  manifest.jsonl    one row per utterance: utt_id, units (file name), n_frames,
                    path (the source audio, relative to the data root), plus the
                    speaker/dataset/subset/duration/lang of the source row
  <utt_id>.npy      int16 ids, one per frame
  <utt_id>.prosody.npy   float32 (n_frames, 4): logf0, voicing, aperiodicity,
                         log_energy from the causal YIN tracker (--prosody)
  <utt_id>.latents.npy   float16/float32 (n_frames, D): the student's `latents`
                         port (--save-latents; students only)

Every stream is frame-aligned: index i is frame i of the zero-padded input
(samples [frame_samples*i, frame_samples*(i+1))). Student heads are shifted
back by their lookahead L at extraction (L zero frames appended), so a student
stream lines up with the teacher stream it was trained on; `lookahead_frames`
records the delay the head costs at runtime.

check_compatible() is the gate Stage 05 runs before it mixes streams: frame
rate, vocab, label space (which centroids the ids index) and lookahead must
agree unless a key is explicitly ignored (driving a teacher-trained vocoder with
a student head ignores lookahead_frames on purpose).

Tokenizers (pitch is not extracted here any more: fvml/features.py, Stage 03,
owns causal pitch/energy):

  teacher-km100   HuBERT-Base layer 7 + k-means(100), offline, non-causal, 50 Hz
  teacher-mhubert-km1000     mhubert_base_vp_en_es_fr_it3 layer 11 + k-means(1000)
                             on balanced {en, es, fr, de} (M3 default, MIT)
  teacher-mhubert147-km1000  mHuBERT-147 + k-means(1000): CC-BY-NC-SA research
                             probe, refused without include_noncommercial
  student-L0/1/2  causal content encoder head L (fvml/content_encoder.py), 50 Hz
  mimi-q0         Kyutai Mimi semantic codebook 0, causal, 12.5 Hz (kill test)

Manifests come in two shapes. Monolingual rows (prepare_data.py, and the CSS10 /
CML-TTS rows of prepare_s2ut_data.py) carry utt_id / path / duration / speaker
and, for the multilingual corpora, lang. Paired S2UT rows carry both sides of a
translation pair; audio_rows() turns one side of them into the monolingual
shape, so every tool downstream reads a single row format.
"""
import datetime
import json
import pathlib
import re

import numpy as np

from . import audio, paths

TOKENIZERS = ("teacher-km100", "teacher-mhubert-km1000", "teacher-mhubert147-km1000",
              "student-L0", "student-L1", "student-L2", "mimi-q0")
META_FILE = "units_meta.json"
MANIFEST_FILE = "manifest.jsonl"
PROSODY_STATS_FILE = "prosody_stats.json"
COMPAT_KEYS = ("frame_rate_hz", "vocab", "label_space", "lookahead_frames")
COPY_FIELDS = ("duration", "speaker", "dataset", "subset", "gender", "lang", "pair", "side")
SIDES = ("src", "tgt")
PROSODY_COLUMNS = ("logf0", "voicing", "aperiodicity", "log_energy")

MIMI_ID = "kyutai/mimi"
MIMI_REVISION = "89091b3e466eb6a9d11e537bf26b144f194978f7"   # pinned 2026-10-18 (Stage 04)


class UnitsMismatch(ValueError):
    """Two unit streams that must not be mixed."""


def check_compatible(meta_a, meta_b, ignore=()):
    """Raise UnitsMismatch unless the two streams share frame rate, vocab,
    label space and lookahead (minus the keys in `ignore`)."""
    unknown = set(ignore) - set(COMPAT_KEYS)
    if unknown:
        raise ValueError(f"cannot ignore unknown key(s): {', '.join(sorted(unknown))}")
    bad = [f"{k}: {meta_a.get(k)!r} vs {meta_b.get(k)!r}"
           for k in COMPAT_KEYS if k not in ignore and meta_a.get(k) != meta_b.get(k)]
    if bad:
        raise UnitsMismatch(f"incompatible unit streams ({meta_a.get('tokenizer')} vs "
                            f"{meta_b.get('tokenizer')}): " + "; ".join(bad))


def check_license(meta, include_noncommercial=False):
    """Refuse a unit stream (or anything trained on one) that descends from a
    non-commercial teacher unless the caller opted in."""
    if meta.get("noncommercial") and not include_noncommercial:
        from .teacher import LicenseError
        raise LicenseError(
            f"{meta.get('tokenizer')} units are non-commercial ({meta.get('license')}): pass "
            "--include-noncommercial for research runs; never ship what is trained on them")


def check_rows_license(rows, include_noncommercial=False):
    """Refuse manifest rows from a non-commercial corpus (VoxPopuli CC BY-NC)
    unless the caller opted in — the corpus taint is independent of the
    teacher's licence (BUG-031). Returns the distinct NC licence text(s) joined
    with ' + ', or None when every row is commercial-clean."""
    nc = sorted({str(r.get("license") or "non-commercial")
                 for r in rows if r.get("noncommercial")})
    if nc and not include_noncommercial:
        from .teacher import LicenseError
        raise LicenseError(
            "the manifest mixes in non-commercial corpus audio ("
            + ", ".join(nc) + "): pass --include-noncommercial for research runs; never "
            "ship what is trained on them")
    return " + ".join(nc) if nc else None


def is_paired(row):
    return "src_audio_path" in row and "tgt_audio_path" in row


def audio_rows(rows, side=None, default_lang=None):
    """Manifest rows -> monolingual audio rows (utt_id, path, duration, speaker,
    lang, ...). Monolingual rows pass through; a row without `lang` takes
    `default_lang` when one is given (the M2 corpora are English but do not say
    so). A paired S2UT row yields its `side` ("src" or "tgt"): the source side
    keeps the source speaker, the target side is the corpus's single canonical
    voice for that language, named <dataset>_tgt_<lang>."""
    out = []
    for r in rows:
        if not is_paired(r):
            if default_lang and "lang" not in r:
                r = {**r, "lang": default_lang}
            out.append(r)
            continue
        if side not in SIDES:
            raise ValueError("paired S2UT manifest rows need --side src or --side tgt")
        lang, dataset = r[f"{side}_lang"], r.get("dataset") or "s2ut"
        row = {"utt_id": r["id"], "path": r[f"{side}_audio_path"],
               "duration": r[f"{side}_duration_s"], "dataset": dataset,
               "subset": r.get("split", "train"), "lang": lang,
               "pair": f"{r['src_lang']}_{r['tgt_lang']}", "side": side,
               "speaker": r.get("speaker_id") if side == "src" else f"{dataset}_tgt_{lang}"}
        for src_key, key in ((f"{side}_text", "text"), ("license", "license")):
            if r.get(src_key):
                row[key] = r[src_key]
        if r.get("noncommercial"):            # BUG-031: the corpus taint follows the audio
            row["noncommercial"] = True
        if side == "src" and r.get("gender"):
            row["gender"] = r["gender"]
        out.append({k: v for k, v in row.items() if v is not None})
    return out


def filter_subsets(rows, subsets):
    """Rows whose `subset` is in `subsets` (a comma-separated string or a
    collection); everything when `subsets` is empty."""
    if isinstance(subsets, str):
        subsets = [s.strip() for s in subsets.split(",") if s.strip()]
    if not subsets:
        return list(rows)
    keep = set(subsets)
    return [r for r in rows if r.get("subset") in keep]


def safe_name(utt_id):
    """File stem for an utterance id (ids with path separators stay unique)."""
    return re.sub(r"[^\w.\-]", "_", utt_id.replace("/", "__"))


def read_meta(units_dir):
    p = pathlib.Path(units_dir) / META_FILE
    if not p.is_file():
        raise FileNotFoundError(f"{p}: not a units directory (run extract_units.py first)")
    return json.loads(p.read_text(encoding="utf-8"))


def read_units_dir(units_dir):
    """(meta, rows) of a units directory."""
    from .datasets import read_jsonl
    units_dir = pathlib.Path(units_dir)
    meta = read_meta(units_dir)
    m = units_dir / MANIFEST_FILE
    rows = read_jsonl(m) if m.is_file() else []
    return meta, rows


def split_specs(specs):
    """--train-units style values (a string or a list, each comma-separated) -> [str]."""
    return [s.strip() for v in ([specs] if isinstance(specs, str) else specs or [])
            for s in str(v).split(",") if s.strip()]


def read_units_dirs(specs):
    """Several units directories that share one label space ->
    (meta of the first, {utt_id: (directory, row)}). Raises UnitsMismatch when
    they cannot be mixed (check_compatible)."""
    dirs = split_specs(specs)
    if not dirs:
        raise ValueError("no units directory given")
    meta0, by_id = None, {}
    for d in dirs:
        meta, rows = read_units_dir(d)
        if meta0 is None:
            meta0 = meta
        else:
            check_compatible(meta0, meta)
            if bool(meta.get("noncommercial")) != bool(meta0.get("noncommercial")):
                raise UnitsMismatch(f"{d}: mixes commercial and non-commercial unit streams")
        for r in rows:
            by_id[r["utt_id"]] = (pathlib.Path(d), r)
    return meta0, by_id


def load_ids(units_dir, row):
    return np.load(pathlib.Path(units_dir) / row["units"])


def manifest_path(spec):
    """A manifest given as a path, a path relative to the data root, or a bare
    name in <data root>/manifests/ ('librispeech-dev-clean.jsonl')."""
    p = pathlib.Path(spec)
    if p.is_file() or p.is_absolute():
        return p
    if len(p.parts) == 1:
        return paths.data_root() / "manifests" / p
    return paths.resolve(p)


def read_manifests(spec):
    """--manifest values (repeatable, comma-separated) -> (rows, [paths]).
    Fails with the missing paths listed, so an absent corpus is obvious."""
    from .datasets import read_jsonl
    items = [s.strip() for v in ([spec] if isinstance(spec, str) else spec)
             for s in v.split(",") if s.strip()]
    found = [manifest_path(s) for s in items]
    missing = [str(p) for p in found if not p.is_file()]
    if missing:
        raise FileNotFoundError("manifest(s) not found (prepare the dataset first, e.g. "
                                "prepare_data.py --datasets librispeech-train-clean-100): "
                                + ", ".join(missing))
    rows = []
    for p in found:
        rows.extend(read_jsonl(p))
    return rows, found


def base_meta(tokenizer, frame_samples, lookahead, vocab, label_space, causal, padding, source):
    return {"tokenizer": tokenizer, "sample_rate": audio.SAMPLE_RATE,
            "frame_samples": frame_samples, "frame_rate_hz": audio.SAMPLE_RATE / frame_samples,
            "lookahead_frames": lookahead, "causal": causal, "vocab": vocab,
            "label_space": label_space, "dtype": "int16", "alignment": "frame",
            "padding": padding, "source": source}


# -- tokenizers ---------------------------------------------------------------
class TeacherTokenizer:
    """An offline teacher (fvml.teacher.TEACHERS) plus its k-means codebook.
    `spec` defaults to the M2 HuBERT-Base km100 teacher; `teacher_checkpoint`
    is the fairseq file of a fairseq teacher (default: its archive path)."""

    def __init__(self, centroids, device="cuda", spec=None, teacher_checkpoint=None):
        from . import teacher
        self.spec = spec = spec or teacher.TEACHERS[teacher.DEFAULT_TEACHER]
        self.centroids, sha = teacher.load_centroids(centroids, spec.centroids_file)
        self.model = teacher.HubertTeacher(device, spec=spec, checkpoint=teacher_checkpoint)
        self.frame_samples = teacher.FRAME_SAMPLES
        info = self.model.info()
        k = len(self.centroids)
        if self.centroids.shape[1] != self.model.model.config.hidden_size:
            raise ValueError(f"{centroids}: centroids have dim {self.centroids.shape[1]}, the "
                             f"teacher's features have {self.model.model.config.hidden_size}")
        self.meta = base_meta(
            spec.tokenizer, teacher.FRAME_SAMPLES, None, k, f"km{k}:{sha}", False,
            f"end zero-padded to 320*n; HuBERT input padded {teacher.PAD} samples each side",
            {**info, "centroids": str(centroids), "centroids_sha256": sha})
        self.meta.update(license=spec.license, noncommercial=spec.noncommercial)

    def encode(self, wav):
        from . import teacher
        from .content_encoder import pad_to_frames
        return teacher.assign(self.model.features(pad_to_frames(wav)), self.centroids), None


class StudentTokenizer:
    def __init__(self, checkpoint, head, device="cpu"):
        import torch
        from . import content_encoder as ce
        from .teacher import file_sha256
        self.torch, self.ce = torch, ce
        self.model, ckpt = ce.ContentEncoder.from_checkpoint(checkpoint)
        self.model = self.model.eval().to(device)
        self.device, self.head = torch.device(device), head
        self.frame_samples = ce.FRAME_SAMPLES
        label_meta = json.loads(ckpt.get("labels_meta", "{}"))
        L = self.model.cfg.lookahead[head]
        self.latent_dim = self.model.cfg.latent_dim
        self.meta = base_meta(
            f"student-L{L}", ce.FRAME_SAMPLES, L, self.model.cfg.vocab,
            label_meta.get("label_space"), True,
            f"end zero-padded to 320*n, plus {L} zero frames to flush the head's lookahead",
            {"checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
             "step": ckpt.get("step"), "head": head, "labels": label_meta.get("tokenizer")})
        # A student inherits its teacher's licence: distilled NC labels stay NC.
        self.meta.update(license=label_meta.get("license"),
                         noncommercial=bool(label_meta.get("noncommercial", False)))

    def encode(self, wav):
        ids, logits, _ = self.encode_all(wav)
        return ids, logits

    def encode_all(self, wav):
        """(ids int16 (n,), logits float16 (n, vocab), latents float32 (n, D)),
        all frame-aligned: index t describes frame t. The head's ids are shifted
        back by its lookahead; the latents never had a delay."""
        torch = self.torch
        x = torch.from_numpy(np.asarray(wav, np.float32))[None].to(self.device)
        with torch.inference_mode():
            ids, logits, latents = self.ce.encode_utterance(self.model, x, self.head)
        return (ids[0].cpu().numpy().astype(np.int16),
                logits[0].float().cpu().numpy().astype(np.float16),
                latents[0].float().cpu().numpy())


class MimiTokenizer:
    """Mimi codebook 0 (the semantic codebook) at 12.5 Hz. Input is resampled
    16 -> 24 kHz on the fly; a 16 kHz frame is 1280 samples."""

    def __init__(self, device="cuda", model_id=MIMI_ID, revision=MIMI_REVISION):
        paths.configure_model_caches()
        import torch
        from transformers import MimiModel
        from .teacher import hf_cache_dir
        self.torch = torch
        self.device = torch.device(device)
        self.model = MimiModel.from_pretrained(model_id, revision=revision,
                                               cache_dir=hf_cache_dir()).eval().to(self.device)
        cfg = self.model.config
        self.sr = cfg.sampling_rate
        self.frame_24k = int(round(cfg.sampling_rate / cfg.frame_rate))         # 1920
        self.frame_samples = int(round(audio.SAMPLE_RATE / cfg.frame_rate))     # 1280
        self.meta = base_meta(
            "mimi-q0", self.frame_samples, 0, int(cfg.codebook_size),
            f"mimi-q0:{model_id}@{revision}", True,
            f"end zero-padded to {self.frame_samples}*n at 16 kHz, resampled to {self.sr} Hz",
            {"model": model_id, "revision": revision, "codebook": 0,
             "frame_rate": float(cfg.frame_rate), "model_sample_rate": self.sr})

    def encode(self, wav):
        from .content_encoder import pad_to_frames
        torch = self.torch
        x16 = pad_to_frames(np.asarray(wav, np.float32), self.frame_samples)
        n = len(x16) // self.frame_samples
        x = audio.resample(x16, audio.SAMPLE_RATE, self.sr)
        x = np.pad(x, (0, max(0, n * self.frame_24k - len(x))))[:n * self.frame_24k]
        with torch.inference_mode():
            codes = self.model.encode(torch.from_numpy(x)[None, None].to(self.device),
                                      num_quantizers=1, return_dict=True).audio_codes
        ids = codes[0, 0].cpu().numpy()
        if abs(len(ids) - n) > 1:
            raise ValueError(f"Mimi returned {len(ids)} frames for {n} input frames")
        ids = np.concatenate([ids, ids[-1:]])[:n] if len(ids) < n else ids[:n]
        return ids.astype(np.int16), None


def make_tokenizer(name, *, centroids=None, checkpoint=None, device="cuda",
                   include_noncommercial=False, teacher_checkpoint=None):
    if name.startswith("teacher-"):
        from . import teacher
        spec = teacher.get_teacher(name, include_noncommercial)      # LicenseError without opt-in
        if not centroids:
            raise ValueError(f"{name} needs --centroids (fit them with --fit-kmeans)")
        return TeacherTokenizer(centroids, device, spec, teacher_checkpoint)
    if name.startswith("student-L"):
        if not checkpoint:
            raise ValueError(f"{name} needs --checkpoint")
        from .content_encoder import LOOKAHEAD, head_lookahead
        return StudentTokenizer(checkpoint, LOOKAHEAD.index(head_lookahead(name)), device)
    if name == "mimi-q0":
        return MimiTokenizer(device)
    raise ValueError(f"unknown tokenizer {name!r}; choose from {', '.join(TOKENIZERS)}")


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")
