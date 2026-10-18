"""StreamVC-style causal content encoder with three delay heads (Milestone 2 /
Stage 04).

16 kHz mono in, 50 Hz out (stride 2*4*5*8 = 320 samples, one frame per 20 ms).
A causal SoundStream trunk (every conv a StreamingConv1d) feeds a per-frame
channel LayerNorm; its D-dim output is the continuous `latents` (Milestone 3's
Mode B input). One trunk, three heads: head L sees trunk frames t-L..t and
predicts the HuBERT km100 teacher id of frame t-L, so one training run gives the
L = 0/1/2 variants. Logits stay pre-softmax (open decision 2: soft-unit fallback).

Frame i of the student closes on sample 320*i + 319, so its centre (320*i + 160)
is the centre of teacher frame i (fvml.teacher pads HuBERT by 40 samples each
side). Output index t of head L is the prediction for frame t - L: in a stream
the head's ids are L frames late, and that delay is its lookahead cost.
"""
import dataclasses
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .streaming import StreamingConv1d, StreamingModule, past_context_samples, state_numel

SAMPLE_RATE = 16000
FRAME_SAMPLES = 320
FRAME_RATE_HZ = SAMPLE_RATE / FRAME_SAMPLES     # 50.0
LOOKAHEAD = (0, 1, 2)                           # frames, per head
IGNORE = -100                                   # label of padded frames


def head_lookahead(head):
    """Lookahead in frames of a head given by index (0/1/2) or tokenizer name
    ('student-L1'); this is the contract's `lookahead_frames`."""
    if isinstance(head, str):
        if not head.startswith("student-L"):
            raise ValueError(f"not a student head: {head!r}")
        head = int(head.removeprefix("student-L"))
    if head not in range(len(LOOKAHEAD)):
        raise ValueError(f"no head {head}; heads are 0..{len(LOOKAHEAD) - 1}")
    return LOOKAHEAD[head]


def pad_to_frames(x, frame_samples=FRAME_SAMPLES):
    """Zero-pad the last axis up to a whole number of frames. Teacher labels and
    student inputs both use this, so frame i means the same samples in both."""
    n = -(-x.shape[-1] // frame_samples)
    extra = n * frame_samples - x.shape[-1]
    if not extra:
        return x
    if isinstance(x, torch.Tensor):
        return F.pad(x, (0, extra))
    import numpy as np
    return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, extra)])


@dataclasses.dataclass(frozen=True)
class EncoderConfig:
    channels: tuple = (32, 64, 128, 256, 256)   # stem width, then each block's output
    strides: tuple = (2, 4, 5, 8)
    kernel: int = 7
    dilations: tuple = (1, 3, 9)
    context_kernel: int = 3                     # 50 Hz context stack
    context_dilations: tuple = (1, 2)
    latent_dim: int = 64                        # D
    head_hidden: int = 256
    vocab: int = 100
    lookahead: tuple = LOOKAHEAD

    def to_dict(self):
        return {k: list(v) if isinstance(v, tuple) else v for k, v in dataclasses.asdict(self).items()}

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})


class ResidualUnit(nn.Module):
    """x + conv1x1(ELU(conv_k,d(ELU(x)))), both causal."""

    def __init__(self, channels, kernel, dilation):
        super().__init__()
        self.conv1 = StreamingConv1d(channels, channels, kernel, dilation=dilation)
        self.conv2 = StreamingConv1d(channels, channels, 1)

    def forward(self, x, stream=None):
        return x + self.conv2(F.elu(self.conv1(F.elu(x), stream)), stream)


class EncoderBlock(nn.Module):
    """Three residual units, then ELU + a strided causal conv (k = 2s)."""

    def __init__(self, cin, cout, stride, kernel, dilations):
        super().__init__()
        self.units = nn.ModuleList(ResidualUnit(cin, kernel, d) for d in dilations)
        self.down = StreamingConv1d(cin, cout, 2 * stride, stride=stride)

    def forward(self, x, stream=None):
        for u in self.units:
            x = u(x, stream)
        return self.down(F.elu(x), stream)


class Trunk(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c = cfg.channels
        self.stem = StreamingConv1d(1, c[0], cfg.kernel)
        self.blocks = nn.ModuleList(EncoderBlock(c[i], c[i + 1], s, cfg.kernel, cfg.dilations)
                                    for i, s in enumerate(cfg.strides))
        self.context = nn.ModuleList(ResidualUnit(c[-1], cfg.context_kernel, d)
                                     for d in cfg.context_dilations)
        self.proj = StreamingConv1d(c[-1], cfg.latent_dim, cfg.context_kernel)
        self.norm = nn.LayerNorm(cfg.latent_dim)    # over channels only: per frame, causal

    def forward(self, x, stream=None):
        """(B, 1, T) -> latents (B, T / 320, D)."""
        x = self.stem(x, stream)
        for b in self.blocks:
            x = b(x, stream)
        for u in self.context:
            x = u(x, stream)
        x = self.proj(F.elu(x), stream)
        return self.norm(x.transpose(1, 2))


class Head(nn.Module):
    """Head with lookahead L: conv over trunk frames t-L..t -> ELU -> 1x1 -> logits."""

    def __init__(self, dim, hidden, vocab, lookahead):
        super().__init__()
        self.lookahead = lookahead
        self.conv = StreamingConv1d(dim, hidden, lookahead + 1)
        self.out = StreamingConv1d(hidden, vocab, 1)

    def forward(self, latents_bdt, stream=None):
        return self.out(F.elu(self.conv(latents_bdt, stream)), stream).transpose(1, 2)


class ContentEncoder(StreamingModule):
    def __init__(self, cfg=EncoderConfig()):
        super().__init__()
        hop = 1
        for s in cfg.strides:
            hop *= s
        if hop != FRAME_SAMPLES:
            raise ValueError(f"strides {cfg.strides} give a {hop}-sample frame, need {FRAME_SAMPLES}")
        if len(cfg.channels) != len(cfg.strides) + 1:
            raise ValueError("channels needs one width per block plus the stem")
        self.cfg = cfg
        self.trunk = Trunk(cfg)
        self.heads = nn.ModuleList(Head(cfg.latent_dim, cfg.head_hidden, cfg.vocab, L)
                                   for L in cfg.lookahead)

    def _run(self, wav, stream):
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)
        if wav.dim() != 3 or wav.shape[1] != 1:
            raise ValueError(f"expected (B, T) or (B, 1, T) audio, got {tuple(wav.shape)}")
        if wav.shape[-1] % FRAME_SAMPLES:
            raise ValueError(f"audio length {wav.shape[-1]} is not a multiple of {FRAME_SAMPLES}")
        latents = self.trunk(wav, stream)
        x = latents.transpose(1, 2)
        return {"logits": [h(x, stream) for h in self.heads], "latents": latents}

    def forward(self, wav):
        """(B, T) or (B, 1, T), T % 320 == 0 -> {"logits": [3 x (B, T/320, vocab)],
        "latents": (B, T/320, D)}. Index t of head L predicts frame t - L."""
        return self._run(wav, None)

    def step(self, chunk, state):
        """One streaming step over frames_per_step * 320 samples. Returns
        (outputs for those frames, new state); the state dict is not mutated."""
        stream = self.stream(state)
        out = self._run(chunk, stream)
        return out, stream.result()

    # -- bookkeeping -------------------------------------------------------
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def past_context_samples(self):
        return past_context_samples(self.trunk)

    def state_numel(self, batch=1):
        return state_numel(self, batch)

    def macs_per_second(self):
        """Multiply-accumulates per second of audio (convs only, all heads)."""
        def conv_macs(conv, rate_in):
            k = conv.kernel_size[0]
            return k * conv.in_channels // conv.groups * conv.out_channels * rate_in / conv.step_stride

        # Execution order, tracking the sample rate each conv runs at.
        total, rate, t = 0.0, float(SAMPLE_RATE), self.trunk
        total += conv_macs(t.stem, rate)
        for b in t.blocks:
            for u in b.units:
                total += conv_macs(u.conv1, rate) + conv_macs(u.conv2, rate)
            total += conv_macs(b.down, rate)
            rate /= b.down.step_stride
        for u in t.context:
            total += conv_macs(u.conv1, rate) + conv_macs(u.conv2, rate)
        total += conv_macs(t.proj, rate)
        for h in self.heads:
            total += conv_macs(h.conv, rate) + conv_macs(h.out, rate)
        return total

    def summary(self):
        return {"params": self.num_params(), "macs_per_second": self.macs_per_second(),
                "past_context_samples": self.past_context_samples(),
                "state_floats": self.state_numel(), "state_tensors": len(self.state_layout()),
                "frame_rate_hz": FRAME_RATE_HZ, "lookahead_frames": list(self.cfg.lookahead)}

    @classmethod
    def from_checkpoint(cls, path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(EncoderConfig.from_dict(json.loads(ckpt["config"])))
        model.load_state_dict(ckpt["model"])
        return model, ckpt


# -- training targets ---------------------------------------------------------
def head_pairs(logits, labels, lookahead=LOOKAHEAD):
    """[(logits_L, targets_L)] with the delay applied: output t of head L is
    scored against labels[t - L], so its first L outputs have no target."""
    pairs = []
    n = labels.shape[1]
    for lg, L in zip(logits, lookahead):
        pairs.append((lg[:, L:], labels[:, :n - L]))
    return pairs


def distill_loss(logits, labels, lookahead=LOOKAHEAD):
    """Cross-entropy summed over heads (each a mean over its labelled frames).
    Returns (total, [per-head loss])."""
    per = [F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1), ignore_index=IGNORE)
           for lg, tg in head_pairs(logits, labels, lookahead)]
    return sum(per), per


@torch.no_grad()
def head_accuracy(logits, labels, lookahead=LOOKAHEAD):
    """[(correct, counted)] per head over frames whose label is not IGNORE."""
    out = []
    for lg, tg in head_pairs(logits, labels, lookahead):
        keep = tg != IGNORE
        out.append((int((lg.argmax(-1)[keep] == tg[keep]).sum()), int(keep.sum())))
    return out


def aligned_ids(model, wav, head):
    """Frame-aligned ids of one head for a whole utterance (offline): pads L
    frames of zeros at the end so index t is the head's id for frame t. Returns
    (ids (B, n) int64, logits (B, n, vocab))."""
    L = model.cfg.lookahead[head]
    wav = pad_to_frames(wav if wav.dim() == 2 else wav.squeeze(1))
    n = wav.shape[-1] // FRAME_SAMPLES
    lg = model(F.pad(wav, (0, L * FRAME_SAMPLES)))["logits"][head][:, L:L + n]
    return lg.argmax(-1), lg
