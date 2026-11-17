"""Causal S2UT translator: chunked Conformer encoder + Wait-k Transformer decoder
(Milestone 3 / Stage 05; Stage 06 exports exactly the step() functions below).

Spec: Documents/Milestone 3/05_causal_s2ut_translator_policy_gate.md, contract v2
in Documents/Milestone 3/02_translator_contract_v2_fixtures.md.

Geometry. Stage 1 `latents` arrive at 50 Hz ([B, T, 128]). The translator runs
at the chunk rate: chunk_frames = 4 frames (80 ms) are stacked into one source
step, and one decoder step emits 4 target unit frames (`stacked_4`). Positions
are counted in chunks everywhere (src_pos, tgt_pos, the ring windows, Wait-k).

Everything is relative, so a stream that never resets behaves like one that
does: there is no absolute position embedding. Each attention adds a learned
per-head bias indexed by a distance d in [0, W):

  encoder / decoder self-attention   d = i - j         query step i, key step j
  decoder cross-attention            d = g - 1 - j     g = source chunks read when
                                                        the target step is decoded

and a key is visible iff 0 <= d < W (W = max_src_steps / max_tgt_steps). That
one rule is causality, the sliding window and the Wait-k schedule at once:

  * forward() gets the schedule as `src_visible` [B, T_tgt] (g per target step;
    waitk_visible() builds min(w + k, S) for training) and masks full sequences.
  * step() keeps the last W keys/values in a ring. Slot s of a ring whose newest
    entry has index `last` holds the entry at distance d = (last - s) mod W,
    live iff d <= last. Cross-attention reads the encoder's memory ring with
    last = src_pos - 1, so it sees exactly the chunks the policy has let the
    encoder read: the Wait-k mask of step() is the policy itself.

Both paths run the same projections, biases and softmax, so they agree up to
float summation order (tests/test_translator.py measures it).

State is a dict of tensors passed in and returned (the house pattern of
fvml/streaming.py). No module buffer is mutated and no position is a Python
int: `src_pos` / `tgt_pos` are int64 [B] tensors and ring slots are selected
with tensor comparisons, so a traced step() cannot freeze a position (MEMORY.md
rule 6). State names are the contract's:

  translator_enc  src_pos, enc_kv_<l> [B, 2H, W_src, Dh] (K heads then V heads),
                  enc_conv_<l> [B, d_model, kernel - 1], and the shared memory
                  mem_k_<l> / mem_v_<l> [B, H, W_src, Dh], one pair per *decoder*
                  layer: the encoder owns the cross-attention key/value
                  projections and writes the newest chunk at slot src_pos mod W
  translator_dec  tgt_pos, dec_kv_<l> [B, 2H, W_tgt, Dh]; mem_* are read-only

Unit modes:
  stacked_4    U = 4 units per decoder step (contract v2 `units` [1, 4]).
  reduced_dur  U = 1 run-length-deduplicated unit + a duration class 1..16 per
               decoder step. A token that starts at target frame F is decoded
               during write w = F // 4, so its schedule is g = min(F // 4 + k, S).
               Its step() signature (extra prev_dur / duration) is not contract
               v2; it exists for the Stage 05 comparison.

Training-only pieces (S2utTranslator): per-language `asr_ctc_head` on encoder
block 4 and `s2tt_ctc_head` on encoder block 6, each Linear(d_model, 4 * V_spm)
(4 CTC frames per chunk). The 321-parameter `readiness_head` lives in the
encoder because CTC-adaptive Wait-k needs it at run time (`readiness_logit`).
"""
from dataclasses import asdict, dataclass
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .streaming import StreamingConv1d

UNIT_MODES = ("stacked_4", "reduced_dur")
DEFAULT_LANGS = (("en", 0), ("es", 1), ("fr", 2), ("de", 3))
# Masked attention logits. Finite on purpose: exp() underflows to exactly 0 in
# f32/f64/bf16, a fully masked row (no source read yet) gives a uniform average
# instead of NaN, and it survives a later f16 conversion (Milestone 4).
MASK_VALUE = -1.0e4
IGNORE = -100
LICENSE = "MIT"


@dataclass(frozen=True)
class TranslatorConfig:
    latent_dim: int = 128           # Stage 1 `latents` width
    chunk_frames: int = 4           # 20 ms frames per translation chunk
    d_model: int = 320
    heads: int = 4
    ffn_dim: int = 1280
    enc_layers: int = 6
    dec_layers: int = 6
    conv_kernel: int = 7            # causal depthwise conv, kernel - 1 chunks of left context
    max_src_steps: int = 128        # W_src, chunks (10.24 s)
    max_tgt_steps: int = 128        # W_tgt, decoder steps
    vocab: int = 1000               # target units; specials follow
    langs: tuple = DEFAULT_LANGS    # ((code, id), ...)
    unit_mode: str = "stacked_4"
    max_duration: int = 16          # reduced_dur: frames per deduplicated unit
    asr_ctc_layer: int = 4          # 1-based encoder block the ASR CTC head reads
    s2tt_ctc_layer: int = 6
    dropout: float = 0.1
    sample_rate: int = 16000
    frame_samples: int = 320

    def __post_init__(self):
        object.__setattr__(self, "langs", tuple((str(c), int(i)) for c, i in (
            self.langs.items() if isinstance(self.langs, dict) else self.langs)))
        if self.unit_mode not in UNIT_MODES:
            raise ValueError(f"unit_mode must be one of {UNIT_MODES} (got {self.unit_mode!r})")
        if self.d_model % self.heads:
            raise ValueError(f"d_model {self.d_model} is not a multiple of heads {self.heads}")
        if self.conv_kernel < 2:
            raise ValueError("conv_kernel must be >= 2")
        for name in ("asr_ctc_layer", "s2tt_ctc_layer"):
            if not 1 <= getattr(self, name) <= self.enc_layers:
                raise ValueError(f"{name} must be in 1..{self.enc_layers}")
        ids = sorted(i for _, i in self.langs)
        if ids != list(range(len(ids))):
            raise ValueError(f"language ids must be 0..{len(ids) - 1} (got {ids})")

    @property
    def bos(self):
        return self.vocab

    @property
    def eos(self):
        return self.vocab + 1

    @property
    def pad(self):
        return self.vocab + 2

    @property
    def n_classes(self):
        return self.vocab + 3

    @property
    def units_per_step(self):
        return self.chunk_frames if self.unit_mode == "stacked_4" else 1

    @property
    def head_dim(self):
        return self.d_model // self.heads

    @property
    def lang_ids(self):
        return dict(self.langs)

    @property
    def special_units(self):
        return {"bos": self.bos, "eos": self.eos, "pad": self.pad}

    @property
    def chunk_ms(self):
        return 1000.0 * self.chunk_frames * self.frame_samples / self.sample_rate

    def to_dict(self):
        d = asdict(self)
        d["langs"] = dict(self.langs)
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def tiny_config(**overrides):
    """A small config for tests and CPU smoke runs (same code paths, ~50k params)."""
    fields = dict(latent_dim=16, d_model=32, heads=2, ffn_dim=64, enc_layers=2, dec_layers=2,
                  conv_kernel=3, max_src_steps=8, max_tgt_steps=8, vocab=24,
                  asr_ctc_layer=1, s2tt_ctc_layer=2, dropout=0.0)
    return TranslatorConfig(**{**fields, **overrides})


# -- attention ------------------------------------------------------------------
def _split_heads(x, heads):
    b, t, d = x.shape
    return x.reshape(b, t, heads, d // heads).transpose(1, 2)          # [B, H, T, Dh]


def _merge_heads(x):
    b, h, t, dh = x.shape
    return x.transpose(1, 2).reshape(b, t, h * dh)


def _attend(q, k, v, bias, valid, drop):
    """softmax(q k^T / sqrt(Dh) + bias over the valid keys) v.
    q [B, H, T, Dh]; k, v [B, H, S, Dh]; bias and valid broadcast to [B, H, T, S]."""
    logits = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(q.shape[-1])) + bias
    logits = logits.masked_fill(~valid, MASK_VALUE)
    return torch.matmul(drop(torch.softmax(logits, dim=-1)), v)


def ring_write_mask(pos, window):
    """[B, W] bool: the slot that entry number `pos` (0-based) is written to."""
    slots = torch.arange(window, device=pos.device)
    return slots[None, :] == torch.remainder(pos, window)[:, None]


def ring_distance(last, window):
    """Distance of every ring slot from the newest entry, whose 0-based index is
    `last` (-1 for an empty ring), and whether the slot holds a live entry.
    Returns (dist [B, W] in 0..W-1, valid [B, W])."""
    slots = torch.arange(window, device=last.device)
    dist = torch.remainder(last[:, None] - slots[None, :], window)
    return dist, dist <= last[:, None]


class RingSelfAttention(nn.Module):
    """Causal sliding-window self-attention with a relative-distance bias."""

    def __init__(self, d_model, heads, window, dropout):
        super().__init__()
        self.heads, self.window = heads, window
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.rel_bias = nn.Parameter(torch.zeros(heads, window))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """[B, T, d] -> [B, T, d]; step i attends steps i - W + 1 .. i."""
        h = self.heads
        idx = torch.arange(x.shape[1], device=x.device)
        dist = idx[:, None] - idx[None, :]                               # i - j
        valid = (dist >= 0) & (dist < self.window)
        bias = self.rel_bias[:, dist.clamp(0, self.window - 1)]          # [H, T, T]
        y = _attend(_split_heads(self.q(x), h), _split_heads(self.k(x), h),
                    _split_heads(self.v(x), h), bias[None], valid[None, None], self.drop)
        return self.out(_merge_heads(y))

    def step(self, x, kv, pos):
        """x [B, 1, d]; kv [B, 2H, W, Dh]; pos [B] int64 = index of this step
        (entries written before it). Returns (y [B, 1, d], new kv)."""
        h = self.heads
        new = torch.cat([_split_heads(self.k(x), h), _split_heads(self.v(x), h)], dim=1)
        kv = torch.where(ring_write_mask(pos, self.window)[:, None, :, None], new, kv)
        dist, valid = ring_distance(pos, self.window)
        bias = self.rel_bias[:, dist].permute(1, 0, 2)[:, :, None, :]    # [B, H, 1, W]
        y = _attend(_split_heads(self.q(x), h), kv[:, :h], kv[:, h:], bias,
                    valid[:, None, None, :], self.drop)
        return self.out(_merge_heads(y)), kv


class RingCrossAttention(nn.Module):
    """Decoder side of the cross-attention: query + output projections and the
    recency bias. Keys/values come projected from the encoder (mem_k / mem_v)."""

    def __init__(self, d_model, heads, window, dropout):
        super().__init__()
        self.heads, self.window = heads, window
        self.q = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.rel_bias = nn.Parameter(torch.zeros(heads, window))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mem_k, mem_v, src_visible):
        """x [B, T, d]; mem_k / mem_v [B, H, S, Dh]; src_visible [B, T] int64 =
        source chunks read when each target step is decoded (the Wait-k schedule).
        Target step t attends source chunks g - W .. g - 1, g = src_visible[t]."""
        j = torch.arange(mem_k.shape[2], device=x.device)
        dist = src_visible[:, :, None] - 1 - j[None, None, :]            # [B, T, S]
        valid = (dist >= 0) & (dist < self.window)
        bias = self.rel_bias[:, dist.clamp(0, self.window - 1)].permute(1, 0, 2, 3)
        y = _attend(_split_heads(self.q(x), self.heads), mem_k, mem_v, bias,
                    valid[:, None], self.drop)
        return self.out(_merge_heads(y))

    def step(self, x, mem_k, mem_v, src_pos):
        """x [B, 1, d]; mem_k / mem_v [B, H, W, Dh] rings; src_pos [B] int64 =
        source chunks the encoder has written."""
        dist, valid = ring_distance(src_pos - 1, self.window)
        bias = self.rel_bias[:, dist].permute(1, 0, 2)[:, :, None, :]
        y = _attend(_split_heads(self.q(x), self.heads), mem_k, mem_v, bias,
                    valid[:, None, None, :], self.drop)
        return self.out(_merge_heads(y))


# -- blocks ---------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, ffn_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(F.silu(self.fc1(x)))))


class CausalConvModule(nn.Module):
    """Conformer conv module with a causal depthwise conv: pointwise + GLU ->
    depthwise (kernel k, left context k - 1 chunks) -> LayerNorm -> SiLU ->
    pointwise. LayerNorm, not BatchNorm: it is causal and identical in train/eval."""

    def __init__(self, d_model, kernel, dropout):
        super().__init__()
        self.pw1 = nn.Linear(d_model, 2 * d_model)
        self.depthwise = StreamingConv1d(d_model, d_model, kernel, groups=d_model)
        self.norm = nn.LayerNorm(d_model)
        self.pw2 = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cache=None):
        """[B, T, d] -> ([B, T, d], new cache). cache=None: offline (zero left
        padding); a [B, d, k - 1] cache: streaming."""
        y = F.glu(self.pw1(x), dim=-1).transpose(1, 2)
        if cache is None:
            y = self.depthwise(y)
        else:
            y, cache = self.depthwise.step(y, cache)
        y = self.pw2(F.silu(self.norm(y.transpose(1, 2))))
        return self.drop(y), cache


class ConformerBlock(nn.Module):
    """Macaron Conformer block: 1/2 FFN -> self-attention -> conv -> 1/2 FFN -> LN."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.norm_ff1, self.norm_att, self.norm_conv, self.norm_ff2, self.norm_out = (
            nn.LayerNorm(d) for _ in range(5))
        self.ff1 = FeedForward(d, cfg.ffn_dim, cfg.dropout)
        self.attn = RingSelfAttention(d, cfg.heads, cfg.max_src_steps, cfg.dropout)
        self.conv = CausalConvModule(d, cfg.conv_kernel, cfg.dropout)
        self.ff2 = FeedForward(d, cfg.ffn_dim, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def _run(self, x, attend, conv_cache):
        x = x + 0.5 * self.ff1(self.norm_ff1(x))
        a, kv = attend(self.norm_att(x))
        x = x + self.drop(a)
        c, conv_cache = self.conv(self.norm_conv(x), conv_cache)
        x = x + c
        x = x + 0.5 * self.ff2(self.norm_ff2(x))
        return self.norm_out(x), kv, conv_cache

    def forward(self, x):
        return self._run(x, lambda h: (self.attn(h), None), None)[0]

    def step(self, x, kv, conv_cache, pos):
        return self._run(x, lambda h: self.attn.step(h, kv, pos), conv_cache)


class DecoderBlock(nn.Module):
    """Pre-LN Transformer decoder block: self-attention -> cross-attention -> FFN."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.norm_self, self.norm_cross, self.norm_ff = (nn.LayerNorm(d) for _ in range(3))
        self.self_attn = RingSelfAttention(d, cfg.heads, cfg.max_tgt_steps, cfg.dropout)
        self.cross_attn = RingCrossAttention(d, cfg.heads, cfg.max_src_steps, cfg.dropout)
        self.ff = FeedForward(d, cfg.ffn_dim, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def _run(self, x, attend, cross):
        a, kv = attend(self.norm_self(x))
        x = x + self.drop(a)
        x = x + self.drop(cross(self.norm_cross(x)))
        return x + self.ff(self.norm_ff(x)), kv

    def forward(self, x, mem_k, mem_v, src_visible):
        return self._run(x, lambda h: (self.self_attn(h), None),
                         lambda h: self.cross_attn(h, mem_k, mem_v, src_visible))[0]

    def step(self, x, kv, tgt_pos, mem_k, mem_v, src_pos):
        return self._run(x, lambda h: self.self_attn.step(h, kv, tgt_pos),
                         lambda h: self.cross_attn.step(h, mem_k, mem_v, src_pos))


# -- encoder --------------------------------------------------------------------
class CausalTranslatorEncoder(nn.Module):
    """translator_enc: 4-frame chunk stacking -> d_model, + <src_lang>, causal
    chunked Conformer blocks, and the per-decoder-layer memory projections."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.in_proj = nn.Linear(cfg.chunk_frames * cfg.latent_dim, d)
        self.lang_emb = nn.Embedding(len(cfg.langs), d)
        self.in_norm = nn.LayerNorm(d)
        self.in_drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(ConformerBlock(cfg) for _ in range(cfg.enc_layers))
        self.mem_k = nn.ModuleList(nn.Linear(d, d) for _ in range(cfg.dec_layers))
        self.mem_v = nn.ModuleList(nn.Linear(d, d) for _ in range(cfg.dec_layers))
        self.readiness_head = nn.Linear(d, 1)

    def embed(self, latents, src_lang):
        """latents [B, T, D] (T a multiple of chunk_frames), src_lang [B] -> [B, T/4, d]."""
        b, t, dim = latents.shape
        c = self.cfg.chunk_frames
        if t % c or dim != self.cfg.latent_dim:
            raise ValueError(f"latents must be [B, {c}*n, {self.cfg.latent_dim}], "
                             f"got {tuple(latents.shape)}")
        x = self.in_proj(latents.reshape(b, t // c, c * dim)) + self.lang_emb(src_lang)[:, None, :]
        return self.in_drop(self.in_norm(x))

    def forward(self, latents, src_lang):
        """Offline: the whole utterance. Returns memory [B, S, d], hidden (one
        [B, S, d] per block), mem_k / mem_v (one [B, H, S, Dh] per decoder
        layer) and readiness_logit [B, S]."""
        x = self.embed(latents, src_lang)
        hidden = []
        for block in self.blocks:
            x = block(x)
            hidden.append(x)
        h = self.cfg.heads
        return {"memory": x, "hidden": hidden,
                "mem_k": [_split_heads(p(x), h) for p in self.mem_k],
                "mem_v": [_split_heads(p(x), h) for p in self.mem_v],
                "readiness_logit": self.readiness_head(x).squeeze(-1)}

    def state_names(self):
        cfg = self.cfg
        return (["src_pos"]
                + [n for i in range(cfg.enc_layers) for n in (f"enc_kv_{i}", f"enc_conv_{i}")]
                + self.shared_mem_names())

    def shared_mem_names(self):
        return [n for i in range(self.cfg.dec_layers) for n in (f"mem_k_{i}", f"mem_v_{i}")]

    def init_state(self, batch=1, device=None, dtype=None):
        cfg = self.cfg
        p = self.in_proj.weight
        kw = {"device": p.device if device is None else device,
              "dtype": p.dtype if dtype is None else dtype}
        state = {"src_pos": torch.zeros(batch, dtype=torch.int64, device=kw["device"])}
        for i in range(cfg.enc_layers):
            state[f"enc_kv_{i}"] = torch.zeros(batch, 2 * cfg.heads, cfg.max_src_steps,
                                                cfg.head_dim, **kw)
            state[f"enc_conv_{i}"] = torch.zeros(batch, cfg.d_model, cfg.conv_kernel - 1, **kw)
        for n in self.shared_mem_names():
            state[n] = torch.zeros(batch, cfg.heads, cfg.max_src_steps, cfg.head_dim, **kw)
        return state

    def step(self, latents, src_lang, state):
        """One chunk: latents [B, chunk_frames, D], src_lang [B], state ->
        ({readiness_logit [B, 1], memory [B, d]}, new state). `state` is not
        modified."""
        cfg = self.cfg
        pos = state["src_pos"]
        x = self.embed(latents, src_lang)                               # [B, 1, d]
        new = {"src_pos": pos + 1}
        for i, block in enumerate(self.blocks):
            x, new[f"enc_kv_{i}"], new[f"enc_conv_{i}"] = block.step(
                x, state[f"enc_kv_{i}"], state[f"enc_conv_{i}"], pos)
        write = ring_write_mask(pos, cfg.max_src_steps)[:, None, :, None]
        for i in range(cfg.dec_layers):
            new[f"mem_k_{i}"] = torch.where(write, _split_heads(self.mem_k[i](x), cfg.heads),
                                            state[f"mem_k_{i}"])
            new[f"mem_v_{i}"] = torch.where(write, _split_heads(self.mem_v[i](x), cfg.heads),
                                            state[f"mem_v_{i}"])
        out = {"readiness_logit": self.readiness_head(x)[:, 0, :], "memory": x[:, 0, :]}
        return out, {n: new[n] for n in self.state_names()}

    def contract(self, license=LICENSE):
        """Contract v2 of this encoder's step graph (role translator_enc)."""
        from . import contract as fc
        cfg = self.cfg
        return fc.ModelContract(
            contract_version=fc.CONTRACT_VERSION_V2, role="translator_enc",
            sample_rate=cfg.sample_rate, frame_samples=cfg.frame_samples,
            chunk_frames=cfg.chunk_frames, latent_dim=cfg.latent_dim,
            max_src_steps=cfg.max_src_steps, shared_mem_states=tuple(self.shared_mem_names()),
            langs=cfg.lang_ids, has_readiness_output=True,
            states=tuple(self.state_names()), license=license)


# -- decoder --------------------------------------------------------------------
class CausalTranslatorDecoder(nn.Module):
    """translator_dec: previous step's unit embeddings + <tgt_lang>, causal
    Transformer blocks cross-attending the encoder memory under the Wait-k
    schedule, the stacked unit head and the eos head."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d, u = cfg.d_model, cfg.units_per_step
        self.unit_emb = nn.Embedding(cfg.n_classes, d)
        self.in_proj = nn.Linear(u * d, d)
        self.lang_emb = nn.Embedding(len(cfg.langs), d)
        self.in_drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(DecoderBlock(cfg) for _ in range(cfg.dec_layers))
        self.out_norm = nn.LayerNorm(d)
        self.unit_head = nn.Linear(d, u * cfg.n_classes)
        self.eos_head = nn.Linear(d, 1)
        if cfg.unit_mode == "reduced_dur":
            self.dur_emb = nn.Embedding(cfg.max_duration + 1, d)        # 0 = <BOS>
            self.dur_head = nn.Linear(d, cfg.max_duration)              # class c = c + 1 frames

    def embed(self, prev_units, tgt_lang, prev_dur=None):
        """prev_units [B, T, U] int64, tgt_lang [B], prev_dur [B, T] int64
        (reduced_dur only) -> [B, T, d]."""
        b, t, u = prev_units.shape
        if u != self.cfg.units_per_step:
            raise ValueError(f"prev_units must have {self.cfg.units_per_step} units per step, "
                             f"got {u}")
        x = self.in_proj(self.unit_emb(prev_units).reshape(b, t, u * self.cfg.d_model))
        x = x + self.lang_emb(tgt_lang)[:, None, :]
        if self.cfg.unit_mode == "reduced_dur":
            if prev_dur is None:
                raise ValueError("reduced_dur needs prev_dur")
            x = x + self.dur_emb(prev_dur)
        return self.in_drop(x)

    def _heads(self, x):
        cfg = self.cfg
        b, t, _ = x.shape
        out = {"hidden": x,
               "logits": self.unit_head(x).reshape(b, t, cfg.units_per_step, cfg.n_classes),
               "eos_logit": self.eos_head(x).squeeze(-1)}
        if cfg.unit_mode == "reduced_dur":
            out["dur_logits"] = self.dur_head(x)
        return out

    def forward(self, prev_units, tgt_lang, mem_k, mem_v, src_visible, prev_dur=None):
        """Offline (teacher forced): prev_units [B, T, U], src_visible [B, T]
        (source chunks visible to each step; >= 1). Returns logits
        [B, T, U, vocab + 3], eos_logit [B, T], hidden [B, T, d] (and dur_logits
        [B, T, max_duration] for reduced_dur)."""
        x = self.embed(prev_units, tgt_lang, prev_dur)
        for i, block in enumerate(self.blocks):
            x = block(x, mem_k[i], mem_v[i], src_visible)
        return self._heads(self.out_norm(x))

    def state_names(self):
        return ["tgt_pos"] + [f"dec_kv_{i}" for i in range(self.cfg.dec_layers)]

    def shared_mem_names(self):
        return [n for i in range(self.cfg.dec_layers) for n in (f"mem_k_{i}", f"mem_v_{i}")]

    def init_state(self, batch=1, device=None, dtype=None):
        cfg = self.cfg
        p = self.in_proj.weight
        device = p.device if device is None else device
        state = {"tgt_pos": torch.zeros(batch, dtype=torch.int64, device=device)}
        for i in range(cfg.dec_layers):
            state[f"dec_kv_{i}"] = torch.zeros(batch, 2 * cfg.heads, cfg.max_tgt_steps,
                                                cfg.head_dim, device=device,
                                                dtype=p.dtype if dtype is None else dtype)
        return state

    def step(self, prev_units, tgt_lang, src_pos, mem, state, prosody_ctx=None, prev_dur=None):
        """One decoder step: prev_units [B, U] int64 (<BOS> x U first), tgt_lang
        [B], src_pos [B] int64 (translator_enc's state_out.src_pos), mem
        {mem_k_<l>, mem_v_<l>} (the encoder's rings, read-only), state.
        `prosody_ctx` [B, 2] is the contract input the Stage 06 prosody head
        consumes; it does not enter the unit path. Returns ({units [B, U], logits
        [B, U, vocab + 3], eos_prob [B, 1], eos_logit [B, 1], hidden [B, d],
        + duration [B, 1] / dur_logits for reduced_dur}, new state)."""
        pos = state["tgt_pos"]
        x = self.embed(prev_units[:, None, :], tgt_lang,
                       None if prev_dur is None else prev_dur.reshape(-1, 1))
        new = {"tgt_pos": pos + 1}
        for i, block in enumerate(self.blocks):
            x, new[f"dec_kv_{i}"] = block.step(x, state[f"dec_kv_{i}"], pos,
                                               mem[f"mem_k_{i}"], mem[f"mem_v_{i}"], src_pos)
        heads = self._heads(self.out_norm(x))
        out = {"logits": heads["logits"][:, 0], "units": heads["logits"][:, 0].argmax(-1),
               "eos_logit": heads["eos_logit"], "eos_prob": torch.sigmoid(heads["eos_logit"]),
               "hidden": heads["hidden"][:, 0]}
        if self.cfg.unit_mode == "reduced_dur":
            out["dur_logits"] = heads["dur_logits"][:, 0]
            out["duration"] = out["dur_logits"].argmax(-1, keepdim=True) + 1
        return out, {n: new[n] for n in self.state_names()}

    def contract(self, license=LICENSE, has_prosody_output=False):
        """Contract v2 of this decoder's step graph (role translator_dec).
        stacked_4 only: a reduced_dur step is not a contract v2 graph."""
        from . import contract as fc
        cfg = self.cfg
        if cfg.unit_mode != "stacked_4":
            raise ValueError("contract v2 describes stacked_4 decoders only")
        return fc.ModelContract(
            contract_version=fc.CONTRACT_VERSION_V2, role="translator_dec",
            sample_rate=cfg.sample_rate, frame_samples=cfg.frame_samples,
            chunk_frames=cfg.chunk_frames, vocab=cfg.vocab, special_units=cfg.special_units,
            max_src_steps=cfg.max_src_steps, max_tgt_steps=cfg.max_tgt_steps,
            shared_mem_states=tuple(self.shared_mem_names()), langs=cfg.lang_ids,
            has_prosody_output=has_prosody_output, states=tuple(self.state_names()),
            license=license)


# -- full model -----------------------------------------------------------------
class S2utTranslator(nn.Module):
    """Encoder + decoder + the training-only CTC heads.

    `ctc_vocab` maps a language code to its SentencePiece vocabulary size V. A
    CTC head is one Linear(d_model, chunk_frames * V) per language: the encoder
    never sees <tgt_lang>, so the S2TT head has to be chosen by target language.
    With ctc_vocab=None the model is the run-time translator only."""

    TRAINING_ONLY = ("asr_ctc_head.", "s2tt_ctc_head.")

    def __init__(self, cfg, ctc_vocab=None):
        super().__init__()
        self.cfg = cfg
        self.encoder = CausalTranslatorEncoder(cfg)
        self.decoder = CausalTranslatorDecoder(cfg)
        self.ctc_vocab = {str(k): int(v) for k, v in (ctc_vocab or {}).items()}
        unknown = sorted(set(self.ctc_vocab) - set(cfg.lang_ids))
        if unknown:
            raise ValueError(f"ctc_vocab has unknown language(s): {', '.join(unknown)}")
        for name in ("asr_ctc_head", "s2tt_ctc_head"):
            setattr(self, name, nn.ModuleDict({
                lang: nn.Linear(cfg.d_model, cfg.chunk_frames * v)
                for lang, v in sorted(self.ctc_vocab.items())}))

    def forward(self, latents, src_lang, prev_units, tgt_lang, src_visible, prev_dur=None):
        """Teacher-forced training pass. Returns the decoder outputs plus
        readiness_logit [B, S] and the encoder block outputs the CTC heads read
        (asr_hidden, s2tt_hidden: [B, S, d])."""
        enc = self.encoder(latents, src_lang)
        out = self.decoder(prev_units, tgt_lang, enc["mem_k"], enc["mem_v"], src_visible, prev_dur)
        out["readiness_logit"] = enc["readiness_logit"]
        out["asr_hidden"] = enc["hidden"][self.cfg.asr_ctc_layer - 1]
        out["s2tt_hidden"] = enc["hidden"][self.cfg.s2tt_ctc_layer - 1]
        return out

    def ctc_logits(self, head, hidden, lang):
        """hidden [n, S, d] of utterances in language `lang` -> [n, S * 4, V]."""
        n, s, _ = hidden.shape
        return getattr(self, head)[lang](hidden).reshape(n, s * self.cfg.chunk_frames, -1)

    def param_counts(self):
        """Run-time parameters (what ships: encoder + decoder) vs the
        training-only CTC projections."""
        training = sum(p.numel() for n, p in self.named_parameters()
                       if n.startswith(self.TRAINING_ONLY))
        total = sum(p.numel() for p in self.parameters())
        return {"runtime": total - training, "training_only": training, "total": total,
                "encoder": sum(p.numel() for p in self.encoder.parameters()),
                "decoder": sum(p.numel() for p in self.decoder.parameters())}

    def strip_training_heads(self):
        """Drop the CTC heads in place (Stage 06 does this before export)."""
        self.asr_ctc_head = nn.ModuleDict()
        self.s2tt_ctc_head = nn.ModuleDict()
        self.ctc_vocab = {}
        return self

    @classmethod
    def from_checkpoint(cls, path, map_location="cpu", runtime_only=False):
        """(model in eval mode, checkpoint dict) from a train_s2ut.py checkpoint."""
        ck = torch.load(path, map_location=map_location, weights_only=True)
        cfg = TranslatorConfig.from_dict(json.loads(ck["config"]))
        model = cls(cfg, None if runtime_only else json.loads(ck.get("ctc_vocab", "{}")))
        state = ck["model"]
        if runtime_only:
            state = {k: v for k, v in state.items() if not k.startswith(cls.TRAINING_ONLY)}
        model.load_state_dict(state)
        return model.eval(), ck


# -- targets --------------------------------------------------------------------
def rle_units(units, max_duration=16):
    """Run-length encode a frame-rate unit stream: (ids, durations), runs longer
    than max_duration split."""
    ids, durs = [], []
    for u in np.asarray(units, np.int64).reshape(-1).tolist():
        if ids and ids[-1] == u and durs[-1] < max_duration:
            durs[-1] += 1
        else:
            ids.append(u)
            durs.append(1)
    return np.asarray(ids, np.int64), np.asarray(durs, np.int64)


def expand_units(ids, durations):
    """Inverse of rle_units()."""
    return np.repeat(np.asarray(ids, np.int64), np.asarray(durations, np.int64))


def waitk_visible(write_index, k, src_chunks):
    """Wait-k schedule g = min(w + k, S): source chunks read when write number w
    (0-based; one write = one 80 ms target chunk) is decoded. k=None is the
    full-context (offline) schedule g = S. Always >= 1."""
    w = np.asarray(write_index, np.int64)
    s = max(int(src_chunks), 1)
    if k is None:
        return np.full(w.shape, s, np.int64)
    return np.clip(w + int(k), 1, s)


def build_targets(units, src_chunks, k, cfg):
    """Teacher-forcing layout of one utterance under the Wait-k schedule.

    `units`: its target unit ids at 50 Hz (N frames, each < vocab). `src_chunks`:
    S, source chunks including any appended silence tail. `k`: Wait-k lag in
    chunks, or None for full context.

    stacked_4: frames + <EOS>, padded with <PAD> to whole chunks, one chunk per
    decoder step. reduced_dur: run-length tokens + (<EOS>, 1), one token per step,
    scheduled by the write its first frame falls in. Under Wait-k the policy keeps
    writing while the source lasts (S - k + 1 writes), so when the target ends
    early the layout continues with all-<PAD> steps (eos target 1): the decoder
    is taught to stay silent, not to babble, until the source stops.

    Returns int64 arrays over T decoder steps: prev_units [T, U], target [T, U],
    eos [T] (1 from the step that carries <EOS>), src_visible [T], and for
    reduced_dur prev_dur [T] and dur [T] (frames, 1..max_duration); plus
    target_chunks = ceil(N / chunk_frames), the writes that carry speech."""
    units = np.asarray(units, np.int64).reshape(-1)
    if units.size and (units.min() < 0 or units.max() >= cfg.vocab):
        raise ValueError(f"target units must be in [0, {cfg.vocab})")
    c, s = cfg.chunk_frames, max(int(src_chunks), 1)
    n = int(units.size)
    writes = (n + 1 + c - 1) // c                  # writes that carry speech or <EOS>
    extra = 0 if k is None else max(0, s - int(k) + 1 - writes)
    if cfg.unit_mode == "stacked_4":
        body = np.full(writes * c, cfg.pad, np.int64)
        body[:n] = units
        body[n] = cfg.eos
        target = np.concatenate([body.reshape(writes, c),
                                 np.full((extra, c), cfg.pad, np.int64)])
        steps = writes + extra
        write_index = np.arange(steps)
        out = {"prev_units": np.concatenate([np.full((1, c), cfg.bos, np.int64), target[:-1]]),
               "target": target, "eos": (np.arange(steps) >= writes - 1).astype(np.int64)}
    else:
        ids, durs = rle_units(units, cfg.max_duration)
        n_tok = int(ids.size)
        start = np.concatenate([np.zeros(1, np.int64), np.cumsum(durs)])     # [n_tok + 1]
        ids = np.concatenate([ids, [cfg.eos], np.full(extra, cfg.pad, np.int64)])
        durs = np.concatenate([durs, [1], np.full(extra, c, np.int64)])
        write_index = np.concatenate([start[:n_tok] // c, [n // c],
                                      writes + np.arange(extra, dtype=np.int64)])
        out = {"prev_units": np.concatenate([[cfg.bos], ids[:-1]])[:, None],
               "target": ids[:, None],
               "eos": (np.arange(ids.size) >= n_tok).astype(np.int64),
               "prev_dur": np.concatenate([[0], durs[:-1]]), "dur": durs}
    out["src_visible"] = waitk_visible(write_index, k, s)
    out["target_chunks"] = (n + c - 1) // c
    return out


def collate_targets(layouts, cfg):
    """Pad build_targets() layouts to a batch of torch tensors. Padded steps
    have mask 0, target IGNORE, prev_units <PAD> and src_visible 1."""
    t_max = max(len(x["eos"]) for x in layouts)
    b, u = len(layouts), cfg.units_per_step
    batch = {"prev_units": torch.full((b, t_max, u), cfg.pad, dtype=torch.int64),
             "target": torch.full((b, t_max, u), IGNORE, dtype=torch.int64),
             "eos": torch.zeros(b, t_max), "mask": torch.zeros(b, t_max, dtype=torch.bool),
             "src_visible": torch.ones(b, t_max, dtype=torch.int64),
             "target_chunks": torch.tensor([x["target_chunks"] for x in layouts],
                                           dtype=torch.float32)}
    if cfg.unit_mode == "reduced_dur":
        batch["prev_dur"] = torch.zeros(b, t_max, dtype=torch.int64)
        batch["dur"] = torch.full((b, t_max), IGNORE, dtype=torch.int64)
    for i, x in enumerate(layouts):
        t = len(x["eos"])
        batch["prev_units"][i, :t] = torch.from_numpy(x["prev_units"])
        batch["target"][i, :t] = torch.from_numpy(x["target"])
        batch["eos"][i, :t] = torch.from_numpy(x["eos"]).float()
        batch["mask"][i, :t] = True
        batch["src_visible"][i, :t] = torch.from_numpy(x["src_visible"])
        if cfg.unit_mode == "reduced_dur":
            batch["prev_dur"][i, :t] = torch.from_numpy(x["prev_dur"])
            batch["dur"][i, :t] = torch.from_numpy(x["dur"]) - 1         # class = frames - 1
    return batch


# -- losses ---------------------------------------------------------------------
LOSS_WEIGHTS = {"unit": 1.0, "asr_ctc": 0.3, "s2tt_ctc": 0.3, "eos": 0.1, "readiness": 0.1}


def readiness_targets(nonblank_mass, src_mask, target_chunks):
    """Expected new target chunks per source chunk, in [0, 1]: the S2TT CTC
    non-blank mass of each chunk, rescaled so an utterance's chunks sum to the
    number of target chunks it must emit (before clamping).
    nonblank_mass [B, S] >= 0, src_mask [B, S] bool, target_chunks [B]."""
    mass = nonblank_mass * src_mask
    total = mass.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return (target_chunks[:, None] * mass / total).clamp(0.0, 1.0)


def _ctc_term(model, head, hidden, lang, tokens, src_chunks, blank):
    """Summed CTC loss of one head over the batch, the target token count, and
    the per-chunk non-blank mass [B, S] (detached)."""
    cfg = model.cfg
    b, s, _ = hidden.shape
    mass = torch.zeros(b, s, dtype=torch.float32, device=hidden.device)
    loss = torch.zeros((), dtype=torch.float32, device=hidden.device)
    count = 0
    codes = {i: code for code, i in cfg.lang_ids.items()}
    for lang_id in torch.unique(lang).tolist():
        code = codes[lang_id]
        if code not in getattr(model, head):
            raise KeyError(f"no {head} for language '{code}'")
        rows = torch.nonzero(lang == lang_id).flatten()
        logp = F.log_softmax(model.ctc_logits(head, hidden[rows], code).float(), dim=-1)
        targets = [torch.as_tensor(tokens[i], dtype=torch.int64) for i in rows.tolist()]
        lengths = torch.tensor([len(t) for t in targets], dtype=torch.int64)
        loss = loss + F.ctc_loss(
            logp.transpose(0, 1), torch.cat(targets).to(hidden.device),
            cfg.chunk_frames * src_chunks[rows], lengths.to(hidden.device),
            blank=blank[code], reduction="sum", zero_infinity=True)
        count += int(lengths.sum())
        nonblank = 1.0 - logp[..., blank[code]].detach().exp()          # [n, S * 4]
        mass[rows] = nonblank.reshape(len(rows), s, cfg.chunk_frames).sum(-1)
    return loss, count, mass


def s2ut_loss(model, out, batch, weights=None, label_smoothing=0.0):
    """Joint Stage 05 loss
        L = L_unit + 0.3 L_asr_ctc + 0.3 L_s2tt_ctc + 0.1 L_eos + 0.1 L_readiness.

    `out` is S2utTranslator.forward(); `batch` holds the collate_targets()
    tensors plus src_lang / tgt_lang [B], src_chunks [B] (valid source chunks),
    src_tokens / tgt_tokens (lists of SentencePiece id lists) and ctc_blank
    {lang: blank id}. Unit CE (and duration CE for reduced_dur) averages over
    target frames, CTC over target tokens, eos BCE over decoder steps, readiness
    BCE over source chunks. Returns {total, unit, asr_ctc, s2tt_ctc, eos,
    readiness, unit_acc} (and dur for reduced_dur)."""
    cfg = model.cfg
    w = {**LOSS_WEIGHTS, **(weights or {})}
    logits = out["logits"].float()
    target = batch["target"]
    terms = {"unit": F.cross_entropy(logits.reshape(-1, cfg.n_classes), target.reshape(-1),
                                     ignore_index=IGNORE, label_smoothing=label_smoothing)}
    with torch.no_grad():
        keep = target != IGNORE
        terms["unit_acc"] = ((logits.argmax(-1) == target) & keep).sum() / keep.sum().clamp_min(1)
    if cfg.unit_mode == "reduced_dur":
        terms["dur"] = F.cross_entropy(out["dur_logits"].float().reshape(-1, cfg.max_duration),
                                       batch["dur"].reshape(-1), ignore_index=IGNORE)
    mask = batch["mask"]
    terms["eos"] = F.binary_cross_entropy_with_logits(
        out["eos_logit"].float()[mask], batch["eos"][mask])

    src_chunks = batch["src_chunks"]
    s = out["readiness_logit"].shape[1]
    src_mask = torch.arange(s, device=src_chunks.device)[None, :] < src_chunks[:, None]
    asr, n_asr, _ = _ctc_term(model, "asr_ctc_head", out["asr_hidden"], batch["src_lang"],
                              batch["src_tokens"], src_chunks, batch["ctc_blank"])
    s2tt, n_s2tt, mass = _ctc_term(model, "s2tt_ctc_head", out["s2tt_hidden"], batch["tgt_lang"],
                                   batch["tgt_tokens"], src_chunks, batch["ctc_blank"])
    terms["asr_ctc"] = asr / max(n_asr, 1)
    terms["s2tt_ctc"] = s2tt / max(n_s2tt, 1)
    ready = readiness_targets(mass, src_mask, batch["target_chunks"])
    terms["readiness"] = F.binary_cross_entropy_with_logits(
        out["readiness_logit"].float()[src_mask], ready[src_mask])

    terms["total"] = (w["unit"] * (terms["unit"] + terms.get("dur", 0.0))
                      + w["asr_ctc"] * terms["asr_ctc"] + w["s2tt_ctc"] * terms["s2tt_ctc"]
                      + w["eos"] * terms["eos"] + w["readiness"] * terms["readiness"])
    return terms


# -- step graphs (ONNX; Stage 06 extends export_stateful_onnx.py with these) -----
def translator_enc_graph(encoder, license=LICENSE):
    """(fvml.export.StepGraph, contract) of one translator_enc chunk.
    Graph inputs: latents [1, 4, D], src_lang [1], state_in.*; outputs:
    readiness_logit [1, 1], state_out.* (mem_k_* / mem_v_* included)."""
    from . import export as fx
    c = encoder.contract(license)

    def step(ins, state):
        out, new = encoder.step(ins["latents"], ins["src_lang"], state)
        return {"readiness_logit": out["readiness_logit"]}, new

    return fx.StepGraph(encoder, step, c.expected_inputs(), c.expected_outputs(),
                        encoder.init_state(1)), c


def translator_dec_graph(decoder, encoder, license=LICENSE):
    """(fvml.export.StepGraph, contract) of one translator_dec step. Graph
    inputs: prev_units [1, 4], tgt_lang [1], prosody_ctx [1, 2], src_pos [1], the
    read-only mem_k_* / mem_v_* rings, state_in.*; outputs: units [1, 4],
    eos_prob [1, 1], state_out.*, plus the debug output `logits` that
    fvml.export.finalize() drops from the shipped file."""
    from . import export as fx
    from .contract import TensorSpec
    cfg = decoder.cfg
    c = decoder.contract(license)
    mem_init = encoder.init_state(1)
    mem_specs = [TensorSpec(n, "f32", tuple(mem_init[n].shape)) for n in c.shared_mem_states]

    def step(ins, state):
        out, new = decoder.step(ins["prev_units"], ins["tgt_lang"], ins["src_pos"],
                                {n: ins[n] for n in c.shared_mem_states}, state,
                                prosody_ctx=ins["prosody_ctx"])
        # Stage 05 has no prosody head yet: keep the contract input alive in the
        # traced graph without changing the value.
        eos_prob = out["eos_prob"] + 0.0 * ins["prosody_ctx"].sum(dim=1, keepdim=True)
        return {"units": out["units"], "eos_prob": eos_prob, "logits": out["logits"]}, new

    graph = fx.StepGraph(
        decoder, step, c.expected_inputs() + mem_specs, c.expected_outputs(), decoder.init_state(1),
        debug_outputs=[TensorSpec("logits", "f32", (1, cfg.chunk_frames, cfg.n_classes))])
    return graph, c
