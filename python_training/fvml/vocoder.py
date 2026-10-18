"""Causal, stateful, multi-speaker Unit-HiFi-GAN vocoder + MPD/MRD discriminators
(Milestone 2 / Stage 05).

Spec: Documents/Milestone 2/05_causal_vocoder_decision_gate.md.
Every convolution in the generator is a StreamingConv1d (left-padded by
(k - 1) * d) or a StreamingConvTranspose1d (right-trimmed, trim_right_ratio = 1.0,
with an explicit input-context cache). There is zero future lookahead and zero
OLA overlap, so chunked step() at 1/2/4 frames per step equals offline forward()
up to float reduction order.

Conditioning matches ModelContract v1 (Documents/Milestone 2/02_model_contract_fixtures.md):
  - `units` [B, T_u] int64 (discrete) or `content` [B, T_u, D] float32 (soft),
    upsampled by repetition to the 20 ms conditioning rate (x1 at 50 Hz, x4 at
    12.5 Hz Mimi)
  - `logf0`, `voicing`, `aperiodicity`, `energy` each [B, T_cond] float32 in the
    target speaker's domain (at training time: the speaker's own causal YIN
    features; at runtime: whitened source mapped via fvml.features.map_to_target)
  - `spk_emb` [B, 64] float32 driving FiLM (gamma, beta) after each upsampling
    stage. A learned speaker table nn.Embedding(n_speakers, 64) supplies these
    vectors during training and exports its rows to voices.json in Stage 06.
"""
import dataclasses
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from . import contract
from .streaming import (StreamingConv1d, StreamingConvTranspose1d, StreamingModule,
                        state_numel)

SAMPLE_RATE = 16000
COND_FRAME_SAMPLES = 320                        # 20 ms conditioning hop at 16 kHz
COND_INPUTS = contract.COND_INPUTS              # ("logf0", "voicing", "aperiodicity", "energy", "spk_emb")
FRAME_COND_INPUTS = ("logf0", "voicing", "aperiodicity", "energy")
LRELU_SLOPE = 0.1

DEFAULT_SPEAKERS = ("lj", "hifitts_92", "hifitts_8051", "hifitts_6097", "hifitts_9017")
SPEAKER_METADATA = {
    "lj": {
        "display": "LJSpeech (Linda Johnson)",
        "license": "Public domain",
        "source": "LJSpeech 1.1",
    },
    "hifitts_92": {
        "display": "Cori Samuel (Hi-Fi TTS #92)",
        "license": "CC BY 4.0",
        "source": "Hi-Fi TTS v0 (OpenSLR 109), reader 92",
        "attribution": "Hi-Fi Multi-Speaker English TTS Dataset (Bakhturina et al., 2021), reader Cori Samuel, CC BY 4.0.",
    },
    "hifitts_8051": {
        "display": "Maria Kasper (Hi-Fi TTS #8051)",
        "license": "CC BY 4.0",
        "source": "Hi-Fi TTS v0 (OpenSLR 109), reader 8051",
        "attribution": "Hi-Fi Multi-Speaker English TTS Dataset (Bakhturina et al., 2021), reader Maria Kasper, CC BY 4.0.",
    },
    "hifitts_6097": {
        "display": "Phil Benson (Hi-Fi TTS #6097)",
        "license": "CC BY 4.0",
        "source": "Hi-Fi TTS v0 (OpenSLR 109), reader 6097",
        "attribution": "Hi-Fi Multi-Speaker English TTS Dataset (Bakhturina et al., 2021), reader Phil Benson, CC BY 4.0.",
    },
    "hifitts_9017": {
        "display": "John Van Stan (Hi-Fi TTS #9017)",
        "license": "CC BY 4.0",
        "source": "Hi-Fi TTS v0 (OpenSLR 109), reader 9017",
        "attribution": "Hi-Fi Multi-Speaker English TTS Dataset (Bakhturina et al., 2021), reader John Van Stan, CC BY 4.0.",
    },
}


@dataclasses.dataclass(frozen=True)
class VocoderConfig:
    sample_rate: int = SAMPLE_RATE
    frame_samples: int = COND_FRAME_SAMPLES     # 320 for 50 Hz; 1280 for 12.5 Hz Mimi @ 16 kHz
    cond_frame_samples: int = COND_FRAME_SAMPLES
    content_kind: str = "discrete"              # "discrete" | "soft"
    vocab: int = 100                            # discrete unit vocabulary
    content_dim: int = 0                        # soft unit dimension (when content_kind == "soft")
    unit_dim: int = 128                         # unit embedding width
    cond_inputs: tuple = COND_INPUTS
    cond_hidden: int = 64                       # causal conditioning pre-net width
    cond_kernel: int = 3
    n_speakers: int = len(DEFAULT_SPEAKERS)
    spk_emb_dim: int = 64
    upsample_initial_channel: int = 128         # 128 = V2-class (~1.0M params); 256/512 = V1-class
    upsample_rates: tuple = (8, 5, 4, 2)        # product must equal cond_frame_samples (320)
    upsample_kernels: tuple = (16, 10, 8, 4)
    resblock_kernels: tuple = (3, 7, 11)
    resblock_dilations: tuple = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    pre_kernel: int = 7
    post_kernel: int = 7
    license: str = "MIT (code); CC BY 4.0 / Public Domain training corpora (see voices.json)"

    def __post_init__(self):
        if self.content_kind not in contract.CONTENT_KINDS:
            raise ValueError(f"unknown content_kind {self.content_kind!r}")
        if self.content_kind == "discrete" and self.vocab < 1:
            raise ValueError(f"vocab must be >= 1 for discrete content, got {self.vocab}")
        if self.content_kind == "soft" and self.content_dim < 1:
            raise ValueError(f"content_dim must be >= 1 for soft content, got {self.content_dim}")
        if self.frame_samples % self.cond_frame_samples != 0:
            raise ValueError(f"frame_samples ({self.frame_samples}) must be a multiple of "
                             f"cond_frame_samples ({self.cond_frame_samples})")
        up = 1
        for r in self.upsample_rates:
            up *= r
        if up != self.cond_frame_samples:
            raise ValueError(f"upsample_rates {self.upsample_rates} multiply to {up}, "
                             f"need cond_frame_samples={self.cond_frame_samples}")
        if len(self.upsample_kernels) != len(self.upsample_rates):
            raise ValueError("upsample_kernels and upsample_rates must have the same length")
        if len(self.resblock_dilations) != len(self.resblock_kernels):
            raise ValueError("resblock_dilations and resblock_kernels must have the same length")

    @property
    def unit_repeat(self):
        """Conditioning frames per content frame (1 at 50 Hz, 4 at 12.5 Hz)."""
        return self.frame_samples // self.cond_frame_samples

    def to_dict(self):
        d = dataclasses.asdict(self)
        return {k: ([list(x) if isinstance(x, tuple) else x for x in v]
                    if isinstance(v, tuple) else v)
                for k, v in d.items()}

    @classmethod
    def from_dict(cls, d):
        out = {}
        for k, v in d.items():
            if k == "resblock_dilations":
                out[k] = tuple(tuple(x) for x in v)
            elif isinstance(v, list):
                out[k] = tuple(v)
            else:
                out[k] = v
        return cls(**out)


class ConditioningPreNet(nn.Module):
    """Causal pre-net mapping [logf0, voicing, aperiodicity, energy] -> cond_hidden."""

    def __init__(self, in_dim, hidden, kernel):
        super().__init__()
        self.conv1 = StreamingConv1d(in_dim, hidden, kernel)
        self.conv2 = StreamingConv1d(hidden, hidden, kernel)

    def forward(self, cond_bct, stream=None):
        x = F.leaky_relu(self.conv1(cond_bct, stream), LRELU_SLOPE)
        return F.leaky_relu(self.conv2(x, stream), LRELU_SLOPE)


class FiLM(nn.Module):
    """Identity-centred Feature-wise Linear Modulation: x * (1 + gamma) + beta.

    Acts per channel and broadcasts over time, so it carries no temporal state
    and is linearly interpolable in `spk_emb` (Stage 09 timbre_blend).
    """

    def __init__(self, spk_dim, channels):
        super().__init__()
        self.proj = nn.Linear(spk_dim, 2 * channels)
        nn.init.normal_(self.proj.weight, 0.0, 0.01)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, spk_emb):
        gamma, beta = self.proj(spk_emb).chunk(2, dim=-1)
        return x * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)


class CausalResBlock1(nn.Module):
    """Causal HiFi-GAN ResBlock type 1: pairs of (dilated causal conv, d=1 causal conv)."""

    def __init__(self, channels, kernel, dilations=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            StreamingConv1d(channels, channels, kernel, dilation=d) for d in dilations
        )
        self.convs2 = nn.ModuleList(
            StreamingConv1d(channels, channels, kernel, dilation=1) for _ in dilations
        )

    def forward(self, x, stream=None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt, stream)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt, stream)
            x = x + xt
        return x


class CausalMRF(nn.Module):
    """Multi-Receptive Field Fusion over parallel CausalResBlock1 branches."""

    def __init__(self, channels, kernels, dilations):
        super().__init__()
        self.blocks = nn.ModuleList(
            CausalResBlock1(channels, k, d) for k, d in zip(kernels, dilations)
        )

    def forward(self, x, stream=None):
        out = self.blocks[0](x, stream)
        for b in self.blocks[1:]:
            out = out + b(x, stream)
        return out * (1.0 / len(self.blocks))


def _init_conv_weights(m, std=0.01):
    if isinstance(m, (StreamingConv1d, StreamingConvTranspose1d)):
        nn.init.normal_(m.weight, 0.0, std)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class UnitVocoder(StreamingModule):
    """Causal, stateful, multi-speaker Unit-HiFi-GAN generator."""

    def __init__(self, cfg=VocoderConfig()):
        super().__init__()
        self.cfg = cfg
        if cfg.content_kind == "discrete":
            self.unit_emb = nn.Embedding(cfg.vocab, cfg.unit_dim)
            nn.init.normal_(self.unit_emb.weight, 0.0, 0.02)
        else:
            self.unit_proj = nn.Linear(cfg.content_dim, cfg.unit_dim)
        self.speaker_table = nn.Embedding(cfg.n_speakers, cfg.spk_emb_dim)
        nn.init.normal_(self.speaker_table.weight, 0.0, 0.02)

        self.cond_net = ConditioningPreNet(len(FRAME_COND_INPUTS), cfg.cond_hidden, cfg.cond_kernel)
        self.pre = StreamingConv1d(cfg.unit_dim + cfg.cond_hidden, cfg.upsample_initial_channel,
                                   cfg.pre_kernel)

        self.ups = nn.ModuleList()
        self.films = nn.ModuleList()
        self.mrfs = nn.ModuleList()
        ch = cfg.upsample_initial_channel
        for r, k in zip(cfg.upsample_rates, cfg.upsample_kernels):
            ch_next = ch // 2
            self.ups.append(StreamingConvTranspose1d(ch, ch_next, k, stride=r))
            self.films.append(FiLM(cfg.spk_emb_dim, ch_next))
            self.mrfs.append(CausalMRF(ch_next, cfg.resblock_kernels, cfg.resblock_dilations))
            ch = ch_next
        self.post = StreamingConv1d(ch, 1, cfg.post_kernel)
        self.apply(_init_conv_weights)

    def speaker_embedding(self, spk):
        """Resolve integer speaker indices (B,) or pass through float vectors (B, spk_emb_dim)."""
        if spk.dim() == 1 and spk.dtype in (torch.int64, torch.int32):
            return self.speaker_table(spk)
        if spk.dim() == 2 and spk.shape[-1] == self.cfg.spk_emb_dim:
            return spk
        raise ValueError(f"expected speaker ids (B,) int or embeddings (B, {self.cfg.spk_emb_dim}) "
                         f"float, got shape {tuple(spk.shape)} dtype {spk.dtype}")

    def _embed_content(self, content):
        if self.cfg.content_kind == "discrete":
            if content.dim() != 2:
                raise ValueError(f"expected (B, T_u) unit ids, got {tuple(content.shape)}")
            u = self.unit_emb(content).transpose(1, 2)          # (B, unit_dim, T_u)
        else:
            if content.dim() != 3 or content.shape[-1] != self.cfg.content_dim:
                raise ValueError(f"expected (B, T_u, {self.cfg.content_dim}) soft units, "
                                 f"got {tuple(content.shape)}")
            u = self.unit_proj(content).transpose(1, 2)         # (B, unit_dim, T_u)
        rep = self.cfg.unit_repeat
        if rep > 1:
            b, c, t = u.shape
            u = u.unsqueeze(-1).expand(b, c, t, rep).reshape(b, c, t * rep)
        return u

    def _run(self, units, logf0, voicing, aperiodicity, energy, spk_emb, stream):
        u = self._embed_content(units)
        t_cond = u.shape[-1]
        for name, t in (("logf0", logf0), ("voicing", voicing),
                        ("aperiodicity", aperiodicity), ("energy", energy)):
            if t.dim() != 2 or t.shape[0] != u.shape[0] or t.shape[1] != t_cond:
                raise ValueError(f"{name} must have shape ({u.shape[0]}, {t_cond}), "
                                 f"got {tuple(t.shape)}")
        spk = self.speaker_embedding(spk_emb)
        if spk.shape[0] != u.shape[0]:
            raise ValueError(f"spk_emb batch {spk.shape[0]} != units batch {u.shape[0]}")

        cond = torch.stack([logf0, voicing, aperiodicity, energy], dim=1).to(u.dtype)
        c = self.cond_net(cond, stream)
        x = self.pre(torch.cat([u, c], dim=1), stream)
        for up, film, mrf in zip(self.ups, self.films, self.mrfs):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = up(x, stream)
            x = film(x, spk.to(x.dtype))
            x = mrf(x, stream)
        x = F.leaky_relu(x, LRELU_SLOPE)
        x = torch.tanh(self.post(x, stream))
        return x.squeeze(1)

    def forward(self, units, logf0, voicing, aperiodicity, energy, spk_emb):
        """Offline pass: (B, T_u) units + (B, T_cond) conditioning + (B, 64) spk_emb
        (or (B,) speaker ids) -> (B, T_cond * 320) audio."""
        return self._run(units, logf0, voicing, aperiodicity, energy, spk_emb, None)

    def step(self, units, logf0, voicing, aperiodicity, energy, spk_emb, state):
        """One streaming hop: returns (audio [B, hop], new_state). Input state is not mutated."""
        stream = self.stream(state)
        audio = self._run(units, logf0, voicing, aperiodicity, energy, spk_emb, stream)
        return audio, stream.result()

    def expand_speakers(self, new_count, init_from_row=None):
        """Grow `speaker_table` to `new_count` rows (Stage 10 fine-tune). Existing rows
        are preserved; new rows copy `init_from_row` when given."""
        old = self.cfg.n_speakers
        if new_count < old:
            raise ValueError(f"cannot shrink speaker_table from {old} to {new_count}")
        if new_count == old:
            return
        new_table = nn.Embedding(new_count, self.cfg.spk_emb_dim,
                                 device=self.speaker_table.weight.device,
                                 dtype=self.speaker_table.weight.dtype)
        with torch.no_grad():
            new_table.weight[:old].copy_(self.speaker_table.weight)
            if init_from_row is not None:
                new_table.weight[old:].copy_(self.speaker_table.weight[init_from_row])
            else:
                nn.init.normal_(new_table.weight[old:], 0.0, 0.02)
        self.speaker_table = new_table
        self.cfg = dataclasses.replace(self.cfg, n_speakers=new_count)

    # -- bookkeeping & contract --------------------------------------------
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def state_numel(self, batch=1):
        return state_numel(self, batch)

    def past_context_samples(self):
        """Receptive field into the past, in output audio samples (takes the max
        across parallel MRF branches at each upsampled rate)."""
        hop = self.cfg.cond_frame_samples
        total = (self.cond_net.conv1.context + self.cond_net.conv2.context + self.pre.context) * hop
        for up, mrf in zip(self.ups, self.mrfs):
            total += up.context * hop
            hop //= up.upsample_stride
            branch_ctx = max(sum(c1.context + c2.context for c1, c2 in zip(b.convs1, b.convs2))
                             for b in mrf.blocks)
            total += branch_ctx * hop
        total += self.post.context * hop
        return int(total)

    def macs_per_second(self):
        """Multiply-accumulates per second of output audio (generator convs + FiLM)."""
        def conv1d_macs(conv, rate):
            return conv.kernel_size[0] * conv.in_channels // conv.groups * conv.out_channels * rate

        def convt1d_macs(conv, rate_in):
            return conv.kernel_size[0] * conv.in_channels // conv.groups * conv.out_channels * rate_in

        rate = float(self.cfg.sample_rate) / self.cfg.cond_frame_samples
        total = conv1d_macs(self.cond_net.conv1, rate) + conv1d_macs(self.cond_net.conv2, rate)
        total += conv1d_macs(self.pre, rate)
        for up, film, mrf in zip(self.ups, self.films, self.mrfs):
            total += convt1d_macs(up, rate)
            rate *= up.upsample_stride
            total += film.proj.in_features * film.proj.out_features * (
                float(self.cfg.sample_rate) / self.cfg.cond_frame_samples
            )
            for b in mrf.blocks:
                for c1, c2 in zip(b.convs1, b.convs2):
                    total += conv1d_macs(c1, rate) + conv1d_macs(c2, rate)
        total += conv1d_macs(self.post, rate)
        return float(total)

    def contract(self, frames_per_step=1, license=None):
        """Build and validate the Stage 02 ModelContract for this generator."""
        c = contract.ModelContract(
            role="vocoder",
            sample_rate=self.cfg.sample_rate,
            frame_samples=self.cfg.frame_samples,
            frames_per_step=frames_per_step,
            lookahead_frames=0,
            content_kind=self.cfg.content_kind,
            vocab=self.cfg.vocab if self.cfg.content_kind == "discrete" else 0,
            content_dim=self.cfg.content_dim if self.cfg.content_kind == "soft" else 0,
            cond_frame_samples=self.cfg.cond_frame_samples,
            cond_inputs=self.cfg.cond_inputs,
            spk_emb_dim=self.cfg.spk_emb_dim,
            overlap_samples=0,
            states=tuple(n for n, _, _ in self.state_layout()),
            license=license or self.cfg.license,
        )
        c.validate()
        return c

    def voices_json(self, speakers, speaker_stats, extra_metadata=None):
        """Construct and validate the voices.json dictionary (schema 1) from the
        speaker table and per-speaker F0/energy stats."""
        extra_metadata = extra_metadata or {}
        if len(speakers) != self.cfg.n_speakers:
            raise ValueError(f"speakers list length {len(speakers)} != n_speakers {self.cfg.n_speakers}")
        weights = self.speaker_table.weight.detach().cpu().float().tolist()
        voices = []
        for idx, name in enumerate(speakers):
            if name not in speaker_stats:
                raise KeyError(f"missing F0/energy stats for speaker {name!r}")
            st = speaker_stats[name]
            meta = {**SPEAKER_METADATA.get(name, {}), **extra_metadata.get(name, {})}
            entry = {
                "name": name,
                "display": meta.get("display", name),
                "embedding": [float(v) for v in weights[idx]],
                "logf0_mean": float(st["logf0_mean"]),
                "logf0_std": float(st["logf0_std"]),
                "energy_mean": float(st["energy_mean"]),
                "energy_std": float(st["energy_std"]),
                "license": meta.get("license", "unknown"),
                "source": meta.get("source", name),
            }
            if meta.get("attribution"):
                entry["attribution"] = meta["attribution"]
            voices.append(entry)
        data = {"schema": 1, "voices": voices}
        contract.validate_voices(data, self.contract(1))
        return data

    def summary(self):
        return {
            "params": self.num_params(),
            "macs_per_second": self.macs_per_second(),
            "past_context_samples": self.past_context_samples(),
            "state_floats": self.state_numel(),
            "state_tensors": len(self.state_layout()),
            "sample_rate": self.cfg.sample_rate,
            "frame_samples": self.cfg.frame_samples,
            "cond_frame_samples": self.cfg.cond_frame_samples,
            "vocab": self.cfg.vocab,
            "n_speakers": self.cfg.n_speakers,
            "spk_emb_dim": self.cfg.spk_emb_dim,
        }

    @classmethod
    def from_checkpoint(cls, path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        cfg_data = ckpt["config"]
        if isinstance(cfg_data, str):
            cfg_data = json.loads(cfg_data)
        model = cls(VocoderConfig.from_dict(cfg_data))
        model.load_state_dict(ckpt["model"])
        return model, ckpt


# -- discriminators (training only; never exported) ---------------------------
class PeriodDiscriminator(nn.Module):
    """One sub-discriminator of HiFi-GAN's Multi-Period Discriminator."""

    def __init__(self, period, channels=(32, 128, 256, 512, 512), kernel=5, stride=3):
        super().__init__()
        self.period = period
        pad = (kernel - 1) // 2
        in_ch = (1, *channels[:-1])
        self.convs = nn.ModuleList([
            *[weight_norm(nn.Conv2d(ci, co, (kernel, 1), (stride, 1), padding=(pad, 0)))
              for ci, co in zip(in_ch[:-1], channels[:-1])],
            weight_norm(nn.Conv2d(in_ch[-1], channels[-1], (kernel, 1), 1, padding=(2, 0))),
        ])
        self.post = weight_norm(nn.Conv2d(channels[-1], 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        b, c, t = x.shape
        rem = t % self.period
        if rem:
            x = F.pad(x, (0, self.period - rem), mode="reflect")
            t = x.shape[-1]
        x = x.view(b, c, t // self.period, self.period)
        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            fmap.append(x)
        x = self.post(x)
        fmap.append(x)
        return x.flatten(1, -1), fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList(PeriodDiscriminator(p) for p in periods)

    def forward(self, y):
        scores, fmaps = [], []
        for d in self.discriminators:
            s, fm = d(y)
            scores.append(s)
            fmaps.append(fm)
        return scores, fmaps


class ResolutionDiscriminator(nn.Module):
    """One STFT sub-discriminator of the Multi-Resolution Discriminator (MRD)."""

    def __init__(self, n_fft, hop_length, win_length, channels=32):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, channels, (3, 9), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(channels, channels, (3, 3), padding=(1, 1))),
        ])
        self.post = weight_norm(nn.Conv2d(channels, 1, (3, 3), padding=(1, 1)))

    def _mag(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        with torch.amp.autocast(x.device.type, enabled=False):
            x32 = x.float()
            pad = (self.n_fft - self.hop_length) // 2
            x32 = F.pad(x32.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
            spec = torch.stft(x32, self.n_fft, hop_length=self.hop_length,
                              win_length=self.win_length, window=self.window.float(),
                              center=False, return_complex=True)
            mag = torch.sqrt(spec.real.square() + spec.imag.square() + 1e-9)
        return mag.unsqueeze(1).to(x.dtype)

    def forward(self, x):
        x = self._mag(x)
        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), LRELU_SLOPE)
            fmap.append(x)
        x = self.post(x)
        fmap.append(x)
        return x.flatten(1, -1), fmap


class MultiResolutionDiscriminator(nn.Module):
    """Multi-resolution STFT discriminator (UnivNet / HiFi-GAN MRD).
    Moderate FFT sizes fit within the 8 GB VRAM budget."""

    DEFAULT_RESOLUTIONS = (
        (512, 128, 512),
        (1024, 256, 1024),
        (2048, 512, 2048),
    )

    def __init__(self, resolutions=DEFAULT_RESOLUTIONS, channels=32):
        super().__init__()
        self.discriminators = nn.ModuleList(
            ResolutionDiscriminator(n_fft, hop, win, channels)
            for n_fft, hop, win in resolutions
        )

    def forward(self, y):
        scores, fmaps = [], []
        for d in self.discriminators:
            s, fm = d(y)
            scores.append(s)
            fmaps.append(fm)
        return scores, fmaps


class VocoderDiscriminator(nn.Module):
    """Combined MPD + MRD discriminator suite for Stage 05."""

    def __init__(self, periods=(2, 3, 5, 7, 11),
                 resolutions=MultiResolutionDiscriminator.DEFAULT_RESOLUTIONS):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator(periods)
        self.mrd = MultiResolutionDiscriminator(resolutions)

    def forward(self, y):
        s_p, f_p = self.mpd(y)
        s_r, f_r = self.mrd(y)
        return s_p + s_r, f_p + f_r


# -- losses (LSGAN + feature matching + multi-resolution mel L1) --------------
def hz_to_mel(hz):
    """Slaney-style mel scale (linear below 1 kHz, log above)."""
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    hz = torch.as_tensor(hz, dtype=torch.float64)
    mels = hz / f_sp
    above = hz >= min_log_hz
    mels = torch.where(above, min_log_mel + torch.log(torch.clamp(hz, min=1e-9) / min_log_hz) / logstep, mels)
    return mels


def mel_to_hz(mels):
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    mels = torch.as_tensor(mels, dtype=torch.float64)
    hz = f_sp * mels
    above = mels >= min_log_mel
    hz = torch.where(above, min_log_hz * torch.exp(logstep * (mels - min_log_mel)), hz)
    return hz


def mel_filterbank(sr, n_fft, n_mels, fmin=0.0, fmax=None):
    """Slaney-normalised triangular mel filterbank (n_mels, n_fft // 2 + 1) float32."""
    fmax = float(sr) / 2.0 if fmax is None else float(fmax)
    fftfreqs = torch.linspace(0.0, float(sr) / 2.0, n_fft // 2 + 1, dtype=torch.float64)
    mels = torch.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2, dtype=torch.float64)
    fpts = mel_to_hz(mels)
    fdiff = fpts[1:] - fpts[:-1]
    ramps = fpts.unsqueeze(1) - fftfreqs.unsqueeze(0)
    lower = -ramps[:-2] / fdiff[:-1].unsqueeze(1)
    upper = ramps[2:] / fdiff[1:].unsqueeze(1)
    weights = torch.clamp(torch.minimum(lower, upper), min=0.0)
    enorm = 2.0 / (fpts[2:n_mels + 2] - fpts[:n_mels])
    weights = weights * enorm.unsqueeze(1)
    return weights.float()


class MelSpectrogram(nn.Module):
    def __init__(self, sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=256, win_length=1024,
                 n_mels=80, fmin=0.0, fmax=8000.0):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        self.register_buffer("fb", mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax),
                             persistent=False)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
        with torch.amp.autocast(x.device.type, enabled=False):
            x32 = x.float()
            pad = (self.n_fft - self.hop_length) // 2
            x32 = F.pad(x32.unsqueeze(1), (pad, pad), mode="reflect").squeeze(1)
            spec = torch.stft(x32, self.n_fft, hop_length=self.hop_length,
                              win_length=self.win_length, window=self.window.float(),
                              center=False, return_complex=True)
            mag = torch.sqrt(spec.real.square() + spec.imag.square() + 1e-9)
            mel = torch.matmul(self.fb.float(), mag)
            return torch.log(torch.clamp(mel, min=1e-5))


class MultiResolutionMelLoss(nn.Module):
    """Multi-resolution log-mel L1 loss (Stage 05 spec: weight ~ 45)."""

    DEFAULT_SPECS = (
        # (n_fft, hop_length, win_length, n_mels)
        (512, 80, 320, 40),
        (1024, 160, 640, 80),
        (2048, 320, 1280, 80),
    )

    def __init__(self, sample_rate=SAMPLE_RATE, specs=DEFAULT_SPECS):
        super().__init__()
        self.mels = nn.ModuleList(
            MelSpectrogram(sample_rate, n_fft, hop, win, n_mels, 0.0, sample_rate / 2.0)
            for n_fft, hop, win, n_mels in specs
        )

    def forward(self, real, fake):
        losses = [F.l1_loss(m(real), m(fake)) for m in self.mels]
        return sum(losses) / len(losses), losses


def discriminator_loss(real_scores, fake_scores):
    """LSGAN discriminator loss: sum_k E[(D_k(y) - 1)^2 + D_k(y_hat)^2]."""
    total = 0.0
    for dr, dg in zip(real_scores, fake_scores):
        dr, dg = dr.float(), dg.float()
        total = total + torch.mean((1.0 - dr) ** 2) + torch.mean(dg ** 2)
    return total


def generator_adv_loss(fake_scores):
    """LSGAN generator adversarial loss: sum_k E[(D_k(y_hat) - 1)^2]."""
    total = 0.0
    for dg in fake_scores:
        total = total + torch.mean((1.0 - dg.float()) ** 2)
    return total


def feature_matching_loss(fmap_real, fmap_fake):
    """L1 feature-matching loss summed across discriminators and layers."""
    total = 0.0
    for dr, dg in zip(fmap_real, fmap_fake):
        for rl, gl in zip(dr, dg):
            total = total + torch.mean(torch.abs(rl.detach().float() - gl.float()))
    return total
