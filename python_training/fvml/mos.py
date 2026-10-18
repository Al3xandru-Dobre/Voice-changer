"""Naturalness: UTMOS22 "strong learner" (SpeechMOS port, MIT), reference-free.

The weights come from the SpeechMOS v1.0.0 release and are checked against a
pinned SHA256 (torch.hub would fetch them unverified)."""
import hashlib
import urllib.request

import numpy as np

from . import audio, paths

MODEL = "UTMOS22-strong (SpeechMOS v1.2.0 @ ed25eac)"
WEIGHTS_URL = ("https://github.com/tarepan/SpeechMOS/releases/download/v1.0.0/"
               "utmos22_strong_step7459_v1.pt")
WEIGHTS_SHA256 = "38aa51ab79e2a4e09a1449758a4b37e9cbb2e8235a49662a732d33a9ba1e9bff"


def weights_path():
    path = paths.cache_dir("models") / WEIGHTS_URL.rsplit("/", 1)[1]
    if not path.is_file():
        tmp = path.with_name(path.name + ".part")
        urllib.request.urlretrieve(WEIGHTS_URL, tmp)
        tmp.replace(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != WEIGHTS_SHA256:
        path.unlink()
        raise RuntimeError(f"UTMOS weights SHA256 {sha}, expected {WEIGHTS_SHA256}")
    return path


class Utmos:
    def __init__(self, device="cuda"):
        import torch
        from speechmos.utmos22.strong.model import UTMOS22Strong
        self.model = UTMOS22Strong()
        self.model.load_state_dict(torch.load(weights_path(), map_location="cpu", weights_only=True))
        self.model.eval().to(device)
        self.device = device

    def score(self, waves):
        """One utterance at a time: the score is a time average, so padding
        would bias it."""
        import torch
        out = []
        with torch.inference_mode():
            for w in waves:
                x = torch.from_numpy(np.asarray(w, np.float32))[None].to(self.device)
                out.append(float(self.model(x, audio.SAMPLE_RATE).item()))
        return out

    def close(self):
        from .asr import free_gpu
        self.model = None
        free_gpu()
