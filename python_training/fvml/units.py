"""Discrete unit streams on disk (Milestone 2 / Stage 04).

A units directory, written by extract_units.py, holds

  units_meta.json   tokenizer, frame_rate_hz, frame_samples, lookahead_frames,
                    vocab, label_space, padding, model/centroid/checkpoint sources
  manifest.jsonl    one row per utterance: utt_id, units (file name), n_frames,
                    path (the source audio, relative to the data root), plus the
                    speaker/dataset/subset/duration of the source row
  <utt_id>.npy      int16 ids, one per frame

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
  student-L0/1/2  causal content encoder head L (fvml/content_encoder.py), 50 Hz
  mimi-q0         Kyutai Mimi semantic codebook 0, causal, 12.5 Hz (kill test)
"""
import datetime
import json
import pathlib
import re

import numpy as np

from . import audio, paths

TOKENIZERS = ("teacher-km100", "student-L0", "student-L1", "student-L2", "mimi-q0")
META_FILE = "units_meta.json"
MANIFEST_FILE = "manifest.jsonl"
COMPAT_KEYS = ("frame_rate_hz", "vocab", "label_space", "lookahead_frames")
COPY_FIELDS = ("duration", "speaker", "dataset", "subset", "gender")

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
    def __init__(self, centroids, device="cuda"):
        from . import teacher
        self.centroids, sha = teacher.load_centroids(centroids)
        self.model = teacher.HubertTeacher(device)
        self.frame_samples = teacher.FRAME_SAMPLES
        info = self.model.info()
        k = len(self.centroids)
        self.meta = base_meta(
            "teacher-km100", teacher.FRAME_SAMPLES, None, k, f"km{k}:{sha}", False,
            f"end zero-padded to 320*n; HuBERT input padded {teacher.PAD} samples each side",
            {**info, "centroids": str(centroids), "centroids_sha256": sha})

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
        self.meta = base_meta(
            f"student-L{L}", ce.FRAME_SAMPLES, L, self.model.cfg.vocab,
            label_meta.get("label_space"), True,
            f"end zero-padded to 320*n, plus {L} zero frames to flush the head's lookahead",
            {"checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
             "step": ckpt.get("step"), "head": head, "labels": label_meta.get("tokenizer")})

    def encode(self, wav):
        torch = self.torch
        x = torch.from_numpy(np.asarray(wav, np.float32))[None].to(self.device)
        with torch.inference_mode():
            ids, logits = self.ce.aligned_ids(self.model, x, self.head)
        return (ids[0].cpu().numpy().astype(np.int16),
                logits[0].float().cpu().numpy().astype(np.float16))


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


def make_tokenizer(name, *, centroids=None, checkpoint=None, device="cuda"):
    if name == "teacher-km100":
        if not centroids:
            raise ValueError("teacher-km100 needs --centroids (fit them with --fit-kmeans)")
        return TeacherTokenizer(centroids, device)
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
