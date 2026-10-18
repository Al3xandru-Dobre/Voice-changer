"""HuBERT-Base layer-7 k-means(100) teacher (Milestone 2 / Stage 04).

Offline and non-causal: it labels training data and bounds quality from above.

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
import datetime
import hashlib
import json
import os
import pathlib

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


class HubertTeacher:
    def __init__(self, device="cuda", layer=LAYER, model_id=MODEL_ID, revision=REVISION,
                 truncate=True):
        paths.configure_model_caches()
        import torch
        from transformers import HubertModel
        self.torch = torch
        self.device = torch.device(device)
        self.model = HubertModel.from_pretrained(model_id, revision=revision,
                                                 cache_dir=hf_cache_dir()).eval().to(self.device)
        if self.model.config.do_stable_layer_norm:
            raise ValueError(f"{model_id}: stable-layer-norm encoders end in a LayerNorm, so "
                             "hidden_states[-1] is not the last layer's output")
        if truncate:
            self.model.encoder.layers = self.model.encoder.layers[:layer]
        self.layer, self.model_id, self.revision = layer, model_id, revision

    def features(self, wav):
        """16 kHz mono float array of 320*n samples -> (n, 768) float32 layer features."""
        torch = self.torch
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim != 1 or len(wav) % FRAME_SAMPLES or not len(wav):
            raise ValueError(f"need a non-empty 1-D signal of 320*n samples, got shape {wav.shape}")
        n = len(wav) // FRAME_SAMPLES
        x = torch.from_numpy(np.pad(wav, (PAD, PAD))).to(self.device)[None]
        with torch.inference_mode(), _fp32_matmuls(torch):
            h = self.model(x, output_hidden_states=True).hidden_states[self.layer][0]
        return fix_length(h.float().cpu().numpy(), n)

    def info(self):
        return {"model": self.model_id, "revision": self.revision, "layer": self.layer,
                "frame_rate_hz": FRAME_RATE_HZ, "frame_samples": FRAME_SAMPLES,
                "padding": f"{PAD} zero samples each side (frame i centred at 320*i+160)",
                "normalize": False, "precision": "fp32, TF32 off"}


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
    from sklearn.cluster import MiniBatchKMeans
    res, utts = None, 0
    for feats in feature_iter:
        if res is None:
            res = Reservoir(max_frames, feats.shape[1], seed)
        res.add(feats)
        utts += 1
    if res is None or res.filled < k:
        raise ValueError(f"need at least k={k} frames, got {0 if res is None else res.filled}")
    x = res.sample()
    log(f"[kmeans] fitting k={k} on {len(x)} of {res.seen} frames from {utts} utterances")
    # fairseq examples/hubert/simple_kmeans/learn_kmeans.py settings.
    km = MiniBatchKMeans(n_clusters=k, init="k-means++", max_iter=100, batch_size=10000, tol=0.0,
                         max_no_improvement=100, n_init=20, reassignment_ratio=0.0,
                         random_state=seed, compute_labels=False)
    km.fit(x)
    inertia = float(-km.score(x)) / len(x)
    return km.cluster_centers_.astype(np.float32), {
        "k": k, "seed": seed, "n_frames_used": int(len(x)), "n_frames_seen": int(res.seen),
        "n_utterances": utts, "max_frames": max_frames, "inertia_per_frame": inertia}


def assign(features, centroids):
    """Nearest centroid by squared L2. numpy or torch input -> int16 numpy ids."""
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


def save_kmeans(out_dir, centroids, fit_info, teacher_info, manifests):
    import sklearn
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / CENTROIDS_FILE, np.asarray(centroids, np.float32))
    meta = {**teacher_info, **fit_info, "dim": int(centroids.shape[1]),
            "manifests": [str(m) for m in manifests], "sklearn": sklearn.__version__,
            "date": datetime.date.today().isoformat(),
            "centroids_sha256": file_sha256(out_dir / CENTROIDS_FILE)}
    (out_dir / META_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def load_centroids(path):
    """Centroids file (or the directory holding km100_centroids.npy) ->
    (centroids, sha256)."""
    p = pathlib.Path(path)
    if p.is_dir():
        p = p / CENTROIDS_FILE
    if not p.is_file():
        raise FileNotFoundError(f"{p}: no centroids (fit them with extract_units.py --fit-kmeans)")
    c = np.load(p)
    if c.ndim != 2:
        raise ValueError(f"{p}: centroids must be (k, dim), got {c.shape}")
    return c.astype(np.float32), file_sha256(p)
