"""Speaker similarity: ECAPA-TDNN embeddings (SpeechBrain
spkrec-ecapa-voxceleb), cosine to the target references and to the source
utterance (leakage), equal error rate."""
import numpy as np

from . import paths

SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"   # HF commit at pinning time (2026-10-18)


class Ecapa:
    def __init__(self, device="cuda", batch_size=16):
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import FetchConfig, LocalStrategy
        # COPY: symlinks need Developer Mode / admin on Windows (WinError 1314).
        self.model = EncoderClassifier.from_hparams(
            source=SOURCE, savedir=str(paths.cache_dir("speechbrain", "spkrec-ecapa-voxceleb")),
            run_opts={"device": str(device)}, local_strategy=LocalStrategy.COPY,
            fetch_config=FetchConfig(revision=REVISION))
        self.device, self.batch_size = device, batch_size

    def embed(self, waves):
        """16 kHz mono arrays -> L2-normalised embeddings [N, 192]."""
        import torch
        order = np.argsort([len(w) for w in waves])     # similar lengths per batch: less padding
        out = np.zeros((len(waves), 192), np.float32)
        for s in range(0, len(order), self.batch_size):
            idx = order[s:s + self.batch_size]
            longest = max(len(waves[i]) for i in idx)
            batch = np.zeros((len(idx), longest), np.float32)
            lens = np.zeros(len(idx), np.float32)
            for j, i in enumerate(idx):
                batch[j, :len(waves[i])] = waves[i]
                lens[j] = len(waves[i]) / longest
            with torch.inference_mode():
                e = self.model.encode_batch(torch.from_numpy(batch).to(self.device),
                                            torch.from_numpy(lens).to(self.device))
            out[idx] = e.squeeze(1).float().cpu().numpy()
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    def close(self):
        from .asr import free_gpu
        self.model = None
        free_gpu()


def centroid(embeddings):
    """Mean of normalised embeddings, renormalised (a target voice's reference)."""
    c = np.mean(embeddings, axis=0)
    return c / np.linalg.norm(c)


def eer(scores, labels):
    """Equal error rate of `scores` for binary `labels` (1 = same speaker)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2)


def pair_stats(embeddings, speakers):
    """Same- vs different-speaker cosine over every pair (baseline check)."""
    sims = embeddings @ embeddings.T
    spk = np.asarray(speakers)
    iu = np.triu_indices(len(spk), k=1)
    same = spk[iu[0]] == spk[iu[1]]
    scores = sims[iu]
    both = same.any() and (~same).any()       # a one-speaker subset (--limit) has no EER
    return {"same_spk_cos": float(scores[same].mean()) if same.any() else float("nan"),
            "diff_spk_cos": float(scores[~same].mean()) if (~same).any() else float("nan"),
            "same_pairs": int(same.sum()), "diff_pairs": int((~same).sum()),
            "spk_eer": eer(scores, same.astype(int)) if both else float("nan")}
