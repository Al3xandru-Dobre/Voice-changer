"""Offline unit teachers: HuBERT features + k-means (Milestone 2 / Stage 04,
Milestone 3 / Stage 03).

Offline and non-causal: a teacher labels training data and bounds quality from
above. TEACHERS lists the three this project knows:

  hubert-base   facebook/hubert-base-ls960, layer 7, K=100 (M2, English; MIT)
  mhubert-vp    Meta's mhubert_base_vp_en_es_fr_it3 (fairseq release, MIT),
                layer 11, K=1000 refitted on balanced {en, es, fr, de}: the M3
                commercial-clean default
  mhubert-147   utter-project/mHuBERT-147, CC-BY-NC-SA-4.0: research probe only,
                refused unless the caller passes include_noncommercial

The rest of this note describes the M2 teacher; the M3 teachers share its
framing and k-means code and differ only in weights, layer and K.

Features: `facebook/hubert-base-ls960` (transformers HubertModel), eval mode,
hidden_states[7] = output of transformer layer 7 (hidden_states[0] is the input
to layer 1; tests/test_content_encoder.py checks this with a forward hook on
encoder.layers[6]). Layers 8-12 are dropped: they cannot change layer 7.

Framing: HuBERT's CNN has a 400-sample window and a 320-sample hop. The input is
zero-padded by 40 samples each side, so a 320*n-sample input yields exactly n
frames and frame i is centred on sample 320*i + 160, the centre of student frame
i. Raw waveform in, no per-utterance normalisation: fairseq pretrained
HuBERT-Base with `task.normalize: false` and its km-label feature dump follows
that (the HF preprocessor_config.json says do_normalize=true; it is not used
here). One utterance per forward:
the first conv layer's GroupNorm spans the whole utterance, so batching padded
utterances would change the features. fp32 with TF32 off, so labels do not depend
on the GPU's matmul mode.

k-means: MiniBatchKMeans with fairseq's simple_kmeans settings on a reservoir
sample of frames (100 h x 50 Hz x 768 floats is 55 GB; the reservoir keeps a
uniform sample under a fixed frame budget in one pass).
"""
import dataclasses
import datetime
import hashlib
import json
import os
import pathlib
import pickle
import re
import types

import numpy as np

from . import paths

MODEL_ID = "facebook/hubert-base-ls960"
REVISION = "dba3bb02fda4248b6e082697eee756de8fe8aa8a"   # pinned 2026-10-18 (Stage 04)
LAYER = 7
FRAME_SAMPLES = 320
WINDOW_SAMPLES = 400
PAD = (WINDOW_SAMPLES - FRAME_SAMPLES) // 2              # 40
FRAME_RATE_HZ = 50.0
CENTROIDS_FILE = "km100_centroids.npy"
META_FILE = "km100_meta.json"
MAX_INT16_UNITS = 1 << 15                                # unit ids are stored as int16


class LicenseError(ValueError):
    """A non-commercial asset requested without the explicit opt-in flag."""


@dataclasses.dataclass(frozen=True)
class TeacherSpec:
    """One offline teacher: where its weights come from, which layer feeds the
    k-means, the default K, the licence and the codebook file names."""
    name: str
    tokenizer: str                  # unit-stream tokenizer name (fvml.units.TOKENIZERS)
    layer: int
    k: int
    license: str
    centroids_file: str
    meta_file: str
    noncommercial: bool = False
    model_id: str = ""              # Hugging Face repo (transformers HubertModel)
    revision: str | None = None
    url: str = ""                   # fairseq checkpoint; converted in load_fairseq_hubert()
    file: str = ""                  # its file name under <data root>/archives/
    size: int = 0                   # bytes (Content-Length of `url`)
    stock_kmeans_url: str = ""      # the release's own k-means (import_sklearn_kmeans)
    stock_kmeans_md5: str = ""

    @property
    def source(self):
        return "fairseq" if self.url else "hf"

    def default_checkpoint(self):
        """Where fetch_checkpoint() puts a fairseq teacher's weights."""
        return paths.data_root() / "archives" / self.file


DEFAULT_TEACHER = "hubert-base"
MULTILINGUAL_TEACHER = "mhubert-vp"
TEACHERS = {t.name: t for t in (
    TeacherSpec(name="hubert-base", tokenizer="teacher-km100", layer=LAYER, k=100,
                license="MIT (fairseq HuBERT-Base)", centroids_file=CENTROIDS_FILE,
                meta_file=META_FILE, model_id=MODEL_ID, revision=REVISION),
    # `size` and the k-means MD5 are the Content-Length / ETag the server reported on
    # 2026-11-14. The checkpoint's SHA-256 is pinned in datasets.lock.json on the first
    # verified download (fvml.datasets.verify), like every other archive.
    TeacherSpec(name="mhubert-vp", tokenizer="teacher-mhubert-km1000", layer=11, k=1000,
                license="MIT (fairseq mhubert_base_vp_en_es_fr_it3)",
                centroids_file="mhubert_vp_en_es_fr_de_km1000.npy",
                meta_file="mhubert_vp_en_es_fr_de_km1000.json",
                url="https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3.pt",
                file="mhubert_base_vp_en_es_fr_it3.pt", size=1136474383,
                stock_kmeans_url="https://dl.fbaipublicfiles.com/hubert/"
                                 "mhubert_base_vp_en_es_fr_it3_L11_km1000.bin",
                stock_kmeans_md5="8074f6c086f61901602689a889ad4efa"),
    # Research comparison only. Layer 11 mirrors mhubert-vp so the two probes differ in
    # weights alone; the model card names no canonical unit layer (override with layer=).
    TeacherSpec(name="mhubert-147", tokenizer="teacher-mhubert147-km1000", layer=11, k=1000,
                license="CC-BY-NC-SA-4.0 (utter-project/mHuBERT-147), non-commercial",
                centroids_file="mhubert147_en_es_fr_de_km1000.npy",
                meta_file="mhubert147_en_es_fr_de_km1000.json", noncommercial=True,
                model_id="utter-project/mHuBERT-147",
                revision="7ad3fc0bc5106c58c9c13526abccad527150d135"),   # pinned 2026-11-14
)}
TOKENIZER_TEACHERS = {t.tokenizer: t for t in TEACHERS.values()}


def get_teacher(name, include_noncommercial=False):
    """TeacherSpec by teacher name or tokenizer name. A non-commercial teacher
    raises LicenseError unless the caller opted in."""
    spec = TEACHERS.get(name) or TOKENIZER_TEACHERS.get(name)
    if spec is None:
        raise ValueError(f"unknown teacher {name!r}; choose from {', '.join(TEACHERS)}")
    if spec.noncommercial and not include_noncommercial:
        raise LicenseError(
            f"teacher '{spec.name}' is {spec.license}: pass --include-noncommercial to use it "
            "for research probes; nothing distilled from it may ship in the default bundle")
    return spec


class HubertTeacher:
    """Frozen HuBERT-style feature extractor for one TeacherSpec (default: the
    M2 HuBERT-Base layer-7 teacher). `layer`, `model_id` and `revision`
    override the spec; `checkpoint` is the fairseq file of a fairseq teacher."""

    def __init__(self, device="cuda", layer=None, model_id=None, revision=None, truncate=True,
                 spec=None, checkpoint=None):
        paths.configure_model_caches()
        import torch
        self.spec = spec = spec or TEACHERS[DEFAULT_TEACHER]
        self.torch = torch
        self.device = torch.device(device)
        self.layer = spec.layer if layer is None else layer
        self.checkpoint, self.normalize = None, False
        if spec.source == "fairseq" and model_id is None:
            self.checkpoint = pathlib.Path(checkpoint) if checkpoint else spec.default_checkpoint()
            model, fairseq_cfg = load_fairseq_hubert(self.checkpoint, spec)
            task = fairseq_cfg.get("task")
            self.normalize = bool(task.get("normalize", False)) if isinstance(task, dict) else False
            self.model_id, self.revision = spec.url, None
        else:
            from transformers import HubertModel
            self.model_id = model_id or spec.model_id
            self.revision = revision if (revision or model_id) else spec.revision
            model = HubertModel.from_pretrained(self.model_id, revision=self.revision,
                                                cache_dir=hf_cache_dir())
        self.model = model.eval().to(self.device)
        if self.model.config.do_stable_layer_norm:
            raise ValueError(f"{self.model_id}: stable-layer-norm encoders end in a LayerNorm, so "
                             "hidden_states[-1] is not the last layer's output")
        if not 1 <= self.layer <= self.model.config.num_hidden_layers:
            raise ValueError(f"{self.model_id}: no layer {self.layer} (the encoder has "
                             f"{self.model.config.num_hidden_layers})")
        if truncate:
            self.model.encoder.layers = self.model.encoder.layers[:self.layer]

    def features(self, wav):
        """16 kHz mono float array of 320*n samples -> (n, 768) float32 layer features."""
        torch = self.torch
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim != 1 or len(wav) % FRAME_SAMPLES or not len(wav):
            raise ValueError(f"need a non-empty 1-D signal of 320*n samples, got shape {wav.shape}")
        n = len(wav) // FRAME_SAMPLES
        if self.normalize:               # fairseq task.normalize (the large models, not Base)
            wav = (wav - wav.mean()) / np.sqrt(wav.var() + 1e-5)
        x = torch.from_numpy(np.pad(wav, (PAD, PAD))).to(self.device)[None]
        with torch.inference_mode(), _fp32_matmuls(torch):
            h = self.model(x, output_hidden_states=True).hidden_states[self.layer][0]
        return fix_length(h.float().cpu().numpy(), n)

    def info(self):
        out = {"model": self.model_id, "revision": self.revision, "layer": self.layer,
               "frame_rate_hz": FRAME_RATE_HZ, "frame_samples": FRAME_SAMPLES,
               "padding": f"{PAD} zero samples each side (frame i centred at 320*i+160)",
               "normalize": self.normalize, "precision": "fp32, TF32 off"}
        if self.spec.name != DEFAULT_TEACHER:        # the M2 teacher's metadata stays as it was
            out.update(teacher=self.spec.name, license=self.spec.license,
                       noncommercial=self.spec.noncommercial)
        if self.checkpoint is not None:
            out.update(checkpoint=self.checkpoint.name,
                       checkpoint_sha256=file_sha256(self.checkpoint))
        return out


# -- fairseq checkpoints without fairseq ----------------------------------------
# fairseq does not install on Python 3.12, and mhubert_base_vp_en_es_fr_it3 ships only as
# a fairseq checkpoint. Its weights are a plain state dict stored next to pickled fairseq
# objects (the task's Dictionary, config enums). The unpickler below resolves a short
# allowlist of tensor-rebuilding globals and turns every other global into an inert
# placeholder, so the file loads without fairseq and without running any code it names.
_SAFE_GLOBALS = {
    ("collections", "OrderedDict"), ("torch._utils", "_rebuild_tensor_v2"),
    ("torch._utils", "_rebuild_parameter"), ("torch", "FloatStorage"), ("torch", "HalfStorage"),
    ("torch", "DoubleStorage"), ("torch", "LongStorage"), ("torch", "IntStorage"),
    ("torch", "ShortStorage"), ("torch", "CharStorage"), ("torch", "ByteStorage"),
    ("torch", "BoolStorage"),
}


class _Inert:
    """Stands in for any object the checkpoint pickled that is not a tensor."""

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return _Inert()

    def __setstate__(self, state):
        pass

    def __setitem__(self, key, value):
        pass

    def append(self, item):
        pass

    def extend(self, items):
        pass

    def add(self, item):
        pass

    def update(self, *args, **kwargs):
        pass


class _StateDictUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _SAFE_GLOBALS:
            return super().find_class(module, name)
        return _Inert


def _state_dict_load(file, **kwargs):
    return _StateDictUnpickler(file, **kwargs).load()


_STATE_DICT_PICKLE = types.SimpleNamespace(
    __name__="fvml.teacher.state_dict_pickle", Unpickler=_StateDictUnpickler,
    load=_state_dict_load)

_FAIRSEQ_CONV = re.compile(r"feature_extractor\.conv_layers\.(\d+)\.(0|2)\.(weight|bias)")
_FAIRSEQ_LAYER = re.compile(r"encoder\.layers\.(\d+)\.(.+)\.(weight|bias)")
_FAIRSEQ_LAYER_KEYS = {
    "self_attn.k_proj": "attention.k_proj", "self_attn.v_proj": "attention.v_proj",
    "self_attn.q_proj": "attention.q_proj", "self_attn.out_proj": "attention.out_proj",
    "self_attn_layer_norm": "layer_norm", "fc1": "feed_forward.intermediate_dense",
    "fc2": "feed_forward.output_dense", "final_layer_norm": "final_layer_norm"}
_FAIRSEQ_PRETRAIN_ONLY = ("label_embs_concat", "final_proj.", "target_glu.")
_FAIRSEQ_POS_CONV = "encoder.pos_conv.0."


def convert_fairseq_hubert_state(state, target_keys):
    """fairseq HubertModel state dict -> transformers HubertModel names.
    `target_keys` are the receiving model's state-dict keys (how the positional
    conv's weight norm is named depends on the transformers version). The
    pretraining heads are dropped; any other key without a mapping is an error."""
    new_style = "encoder.pos_conv_embed.conv.parametrizations.weight.original0" in target_keys
    g = "parametrizations.weight.original0" if new_style else "weight_g"
    v = "parametrizations.weight.original1" if new_style else "weight_v"
    pos = {"weight_g": g, "weight_v": v, "parametrizations.weight.original0": g,
           "parametrizations.weight.original1": v, "bias": "bias"}
    out, unknown = {}, []
    for key, value in state.items():
        if key.startswith(_FAIRSEQ_PRETRAIN_ONLY):
            continue
        conv, layer = _FAIRSEQ_CONV.fullmatch(key), _FAIRSEQ_LAYER.fullmatch(key)
        if key == "mask_emb":
            new = "masked_spec_embed"
        elif conv:                      # .0 is the conv, .2 is layer 0's GroupNorm
            part = "conv" if conv.group(2) == "0" else "layer_norm"
            new = f"feature_extractor.conv_layers.{conv.group(1)}.{part}.{conv.group(3)}"
        elif key.startswith("post_extract_proj."):
            new = "feature_projection.projection." + key.split(".", 1)[1]
        elif key.startswith("layer_norm."):
            new = "feature_projection.layer_norm." + key.split(".", 1)[1]
        elif key.startswith(_FAIRSEQ_POS_CONV) and key[len(_FAIRSEQ_POS_CONV):] in pos:
            new = "encoder.pos_conv_embed.conv." + pos[key[len(_FAIRSEQ_POS_CONV):]]
        elif key.startswith("encoder.layer_norm."):
            new = key
        elif layer and layer.group(2) in _FAIRSEQ_LAYER_KEYS:
            new = (f"encoder.layers.{layer.group(1)}.{_FAIRSEQ_LAYER_KEYS[layer.group(2)]}"
                   f".{layer.group(3)}")
        else:
            unknown.append(key)
            continue
        out[new] = value
    if unknown:
        raise ValueError("not a fairseq HuBERT state dict: no mapping for "
                         + ", ".join(sorted(unknown)[:8]))
    return out


def load_fairseq_hubert(path, spec=None, config=None):
    """fairseq HuBERT checkpoint -> (transformers HubertModel, fairseq cfg dict),
    without fairseq. `config` is the HubertConfig to build (default: the Base
    architecture every teacher here uses). Loading is strict: a missing,
    unexpected or mis-shaped tensor raises instead of leaving random weights."""
    import torch
    from transformers import HubertConfig, HubertModel
    path = pathlib.Path(path)
    if not path.is_file():
        hint = (f"; download {spec.url} ({spec.size / 1e9:.2f} GB) to that path, or rerun "
                f"extract_units.py with --download-teacher") if spec and spec.url else ""
        raise FileNotFoundError(f"{path}: no teacher checkpoint{hint}")
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False,
                          pickle_module=_STATE_DICT_PICKLE)
    except (pickle.UnpicklingError, EOFError, RuntimeError, AttributeError, TypeError) as e:
        raise ValueError(f"{path.name}: cannot read it as a fairseq checkpoint "
                         f"({type(e).__name__}: {str(e).splitlines()[0][:200]})") from e
    state = ckpt.get("model") if isinstance(ckpt, dict) else None
    if not isinstance(state, dict) or not state:
        raise ValueError(f"{path.name}: no 'model' state dict (not a fairseq checkpoint)")
    cfg = ckpt.get("cfg")
    cfg = cfg if isinstance(cfg, dict) else {}
    arch = cfg.get("model") if isinstance(cfg.get("model"), dict) else {}
    if arch.get("layer_norm_first") or arch.get("extractor_mode", "default") != "default":
        raise ValueError(
            f"{path.name}: only the HuBERT-Base layout is supported (group-norm extractor, "
            f"post-norm encoder); this checkpoint has layer_norm_first="
            f"{arch.get('layer_norm_first')}, extractor_mode={arch.get('extractor_mode')}")
    model = HubertModel(config or HubertConfig())
    target = model.state_dict()
    converted = convert_fairseq_hubert_state(state, target)
    missing = sorted(set(target) - set(converted))
    extra = sorted(set(converted) - set(target))
    bad = sorted(k for k in set(target) & set(converted)
                 if tuple(target[k].shape) != tuple(converted[k].shape))
    if missing or extra or bad:
        raise ValueError(f"{path.name}: does not fit the HuBERT architecture (missing "
                         f"{missing[:4]}, unexpected {extra[:4]}, mis-shaped {bad[:4]})")
    model.load_state_dict(converted, strict=True)
    return model, cfg


def fetch_checkpoint(spec, log=print):
    """Download a fairseq teacher's checkpoint into <data root>/archives/ with
    the dataset machinery (resume, size check, SHA-256 pinned in
    datasets.lock.json on first use). Returns its path."""
    from . import datasets as fd
    if spec.source != "fairseq":
        raise ValueError(f"teacher '{spec.name}' comes from the Hugging Face hub on first use")
    ctx = fd.Context(root=paths.data_root(), log=log)
    archive = fd.Archive(spec.file, (spec.url,), spec.size)
    path = fd.download(ctx, archive)
    fd.verify(ctx, archive, path)
    return path


def hf_cache_dir():
    """The hub cache under HF_HOME, passed explicitly: HF_HOME is read once, when
    huggingface_hub is first imported, which may predate configure_model_caches()."""
    paths.configure_model_caches()
    return str(pathlib.Path(os.environ["HF_HOME"]) / "hub")


def weights_cached(model_id=MODEL_ID, revision=REVISION, filename="config.json"):
    """True when the pinned snapshot is already downloaded (tests skip otherwise)."""
    from huggingface_hub import try_to_load_from_cache
    hit = try_to_load_from_cache(model_id, filename, cache_dir=hf_cache_dir(), revision=revision)
    return isinstance(hit, str)


class _fp32_matmuls:
    """Turn TF32 off for convs and matmuls inside the block, restore afterwards."""

    def __init__(self, torch):
        self.torch = torch

    def __enter__(self):
        b = self.torch.backends
        self.saved = (b.cuda.matmul.allow_tf32, b.cudnn.allow_tf32)
        b.cuda.matmul.allow_tf32 = b.cudnn.allow_tf32 = False

    def __exit__(self, *exc):
        b = self.torch.backends
        b.cuda.matmul.allow_tf32, b.cudnn.allow_tf32 = self.saved


def fix_length(h, n):
    """Trim or edge-pad to n frames. With 40-sample padding HuBERT returns exactly
    n; a model whose CNN rounds differently may return n +- 1, never more."""
    if abs(len(h) - n) > 1:
        raise ValueError(f"teacher returned {len(h)} frames for {n} input frames")
    if len(h) > n:
        return h[:n]
    if len(h) < n:
        return np.concatenate([h, h[-1:]], axis=0)
    return h


# -- k-means ------------------------------------------------------------------
class Reservoir:
    """Uniform sample of at most `capacity` rows from a stream (Algorithm R,
    vectorised per batch)."""

    def __init__(self, capacity, dim, seed=0):
        self.buf = np.empty((capacity, dim), np.float32)
        self.capacity, self.seen, self.filled = capacity, 0, 0
        self.rng = np.random.default_rng(seed)

    def add(self, rows):
        rows = np.asarray(rows, np.float32)
        take = min(len(rows), self.capacity - self.filled)
        if take:
            self.buf[self.filled:self.filled + take] = rows[:take]
            self.filled += take
        rest = rows[take:]
        if len(rest):
            idx = self.rng.integers(0, self.seen + take + np.arange(1, len(rest) + 1))
            keep = np.flatnonzero(idx < self.capacity)
            # A slot drawn twice keeps the later row, as sequential Algorithm R
            # would (numpy does not define the order of repeated fancy writes).
            slots = idx[keep]
            _, first_rev = np.unique(slots[::-1], return_index=True)
            last = len(slots) - 1 - first_rev
            self.buf[slots[last]] = rest[keep[last]]
        self.seen += len(rows)

    def sample(self):
        return self.buf[:self.filled]


def fit_kmeans(feature_iter, k=100, max_frames=500_000, seed=0, log=print):
    """Fit MiniBatchKMeans on a reservoir sample of the frames `feature_iter`
    yields ((n_i, dim) arrays). Returns (centroids (k, dim) float32, info dict)."""
    res, utts = None, 0
    for feats in feature_iter:
        if res is None:
            res = Reservoir(max_frames, feats.shape[1], seed)
        res.add(feats)
        utts += 1
    if res is None or res.filled < k:
        raise ValueError(f"need at least k={k} frames, got {0 if res is None else res.filled}")
    return _run_kmeans(res.sample(), k, seed, log, seen=res.seen, utts=utts, max_frames=max_frames)


def _run_kmeans(x, k, seed, log, seen, utts, max_frames):
    from sklearn.cluster import MiniBatchKMeans
    log(f"[kmeans] fitting k={k} on {len(x)} of {seen} frames from {utts} utterances")
    # fairseq examples/hubert/simple_kmeans/learn_kmeans.py settings.
    km = MiniBatchKMeans(n_clusters=k, init="k-means++", max_iter=100, batch_size=10000, tol=0.0,
                         max_no_improvement=100, n_init=20, reassignment_ratio=0.0,
                         random_state=seed, compute_labels=False)
    km.fit(x)
    inertia = float(-km.score(x)) / len(x)
    return km.cluster_centers_.astype(np.float32), {
        "k": k, "seed": seed, "n_frames_used": int(len(x)), "n_frames_seen": int(seen),
        "n_utterances": utts, "max_frames": max_frames, "inertia_per_frame": inertia}


def fit_kmeans_balanced(labelled_iter, groups, k=1000, max_frames=500_000, seed=0, log=print):
    """fit_kmeans() on a sample that holds the same number of frames from every
    group. `labelled_iter` yields (group, (n_i, dim) array) in one streaming
    pass; `groups` names the groups (languages) it will yield. Each group fills
    its own reservoir of max_frames / len(groups) frames, and every reservoir is
    then cut to the smallest one, so no language outweighs another whatever the
    hours behind it. Returns (centroids, info) like fit_kmeans(); info["groups"]
    records the frames seen and used per group."""
    order = list(dict.fromkeys(groups))
    if not order:
        raise ValueError("balanced k-means: no groups")
    per_group = max_frames // len(order)
    reservoirs, utts = dict.fromkeys(order), dict.fromkeys(order, 0)
    for group, feats in labelled_iter:
        if group not in utts:
            raise ValueError(f"balanced k-means: unexpected group {group!r} (groups: {order})")
        if reservoirs[group] is None:
            reservoirs[group] = Reservoir(per_group, feats.shape[1], seed + order.index(group))
        reservoirs[group].add(feats)
        utts[group] += 1
    empty = [g for g in order if reservoirs[g] is None]
    if empty:
        raise ValueError(f"balanced k-means: no features for group(s) {', '.join(map(str, empty))}")
    take = min(r.filled for r in reservoirs.values())
    if take * len(order) < k:
        raise ValueError(f"need at least k={k} frames over {len(order)} balanced groups, got "
                         f"{take} per group")
    rng = np.random.default_rng(seed)
    dim = reservoirs[order[0]].buf.shape[1]
    x, summary = np.empty((take * len(order), dim), np.float32), {}
    for i, group in enumerate(order):
        r = reservoirs[group]
        x[i * take:(i + 1) * take] = r.sample()[np.sort(rng.choice(r.filled, size=take,
                                                                    replace=False))]
        summary[group] = {"frames_seen": int(r.seen), "frames_used": int(take),
                          "utterances": int(utts[group])}
        reservoirs[group] = None                    # free it before the next copy
    log("[kmeans] balanced sample: " + ", ".join(
        f"{g} {v['frames_used']} of {v['frames_seen']} frames" for g, v in summary.items()))
    centroids, info = _run_kmeans(x, k, seed, log,
                                  seen=sum(v["frames_seen"] for v in summary.values()),
                                  utts=sum(utts.values()), max_frames=max_frames)
    info.update(groups=summary, balanced=True)
    return centroids, info


def import_sklearn_kmeans(path, md5=None):
    """Centroids of a joblib-pickled scikit-learn k-means (the format fairseq
    publishes, e.g. mhubert_base_vp_en_es_fr_it3_L11_km1000.bin) -> (k, dim)
    float32. joblib unpickles arbitrary objects, so `md5` (the published
    checksum) is checked before the file is opened."""
    import joblib
    path = pathlib.Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path}: no k-means file")
    if md5:
        got = hashlib.md5(path.read_bytes()).hexdigest()
        if got != md5:
            raise ValueError(f"{path.name}: MD5 {got}, expected {md5}; refusing to unpickle it")
    centers = getattr(joblib.load(path), "cluster_centers_", None)
    if centers is None or np.ndim(centers) != 2:
        raise ValueError(f"{path.name}: not a fitted scikit-learn k-means")
    return np.ascontiguousarray(centers, np.float32)


def assign(features, centroids):
    """Nearest centroid by squared L2. numpy or torch input -> int16 numpy ids."""
    if len(centroids) > MAX_INT16_UNITS:
        raise ValueError(f"{len(centroids)} centroids do not fit the int16 unit files")
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(features, torch.Tensor):
        c = torch.as_tensor(centroids, dtype=features.dtype, device=features.device)
        d = (features * features).sum(1, keepdim=True) - 2 * features @ c.T + (c * c).sum(1)
        return d.argmin(1).cpu().numpy().astype(np.int16)
    x = np.asarray(features, np.float32)
    c = np.asarray(centroids, np.float32)
    d = (x * x).sum(1, keepdims=True) - 2 * x @ c.T + (c * c).sum(1)
    return d.argmin(1).astype(np.int16)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def save_kmeans(out_dir, centroids, fit_info, teacher_info, manifests, spec=None):
    """Write the codebook and its fit metadata under the teacher's file names
    (default: the M2 km100 names)."""
    import sklearn
    spec = spec or TEACHERS[DEFAULT_TEACHER]
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / spec.centroids_file, np.asarray(centroids, np.float32))
    meta = {**teacher_info, **fit_info, "dim": int(centroids.shape[1]),
            "manifests": [str(m) for m in manifests], "sklearn": sklearn.__version__,
            "date": datetime.date.today().isoformat(),
            "centroids_sha256": file_sha256(out_dir / spec.centroids_file)}
    (out_dir / spec.meta_file).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def load_centroids(path, centroids_file=CENTROIDS_FILE):
    """Centroids file (or the directory holding `centroids_file`, by default
    km100_centroids.npy) -> (centroids, sha256)."""
    p = pathlib.Path(path)
    if p.is_dir():
        p = p / centroids_file
    if not p.is_file():
        raise FileNotFoundError(f"{p}: no centroids (fit them with extract_units.py --fit-kmeans)")
    c = np.load(p)
    if c.ndim != 2:
        raise ValueError(f"{p}: centroids must be (k, dim), got {c.shape}")
    return c.astype(np.float32), file_sha256(p)
