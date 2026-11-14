"""Model contract v1 & v2: everything the C++ engine needs to know about a model.

Specs:
  * Contract v1 (Mode A): Documents/Milestone 2/02_model_contract_fixtures.md
  * Contract v2 (Mode B): Documents/Milestone 3/02_translator_contract_v2_fixtures.md

The contract is one JSON string stored in the ONNX file's
metadata_props["fluxvoice.contract"].

src/pipeline/ModelContract.hpp implements the same rules, in the same order and
with the same error messages. The shared cases in tests/data/contracts/ (run by
python_training/tests/test_contract.py and tests/test_model_contract.cpp) keep
the two in step: change a rule or a message in both files at once.

Standard library only, so CTest can run the tests with any Python 3. The ONNX
helpers at the bottom import `onnx` lazily.
"""
from collections.abc import Mapping
import json
import math
from dataclasses import dataclass

CONTRACT_VERSION = 1
CONTRACT_VERSION_V2 = 2
SUPPORTED_CONTRACT_VERSIONS = (CONTRACT_VERSION, CONTRACT_VERSION_V2)
METADATA_KEY = "fluxvoice.contract"
MISSING_METADATA = f"model has no '{METADATA_KEY}' metadata"

ROLES_V1 = ("encoder", "vocoder")
ROLES_V2 = ("encoder", "translator_enc", "translator_dec", "vocoder")
ROLES = ROLES_V1
TRANSLATOR_ROLES = ("translator_enc", "translator_dec")

CONTENT_KINDS = ("discrete", "soft")
CONTENT_KEY = {"discrete": "vocab", "soft": "content_dim"}
COND_INPUTS = ("logf0", "voicing", "aperiodicity", "energy", "spk_emb")
FRAMES_PER_STEP = (1, 2, 4)
VOCODER_KEYS = ("cond_frame_samples", "cond_inputs", "spk_emb_dim", "overlap_samples")
V2_ONLY_KEYS = ("latent_dim", "chunk_frames", "max_src_steps", "max_tgt_steps",
                "shared_mem_states", "langs", "special_units",
                "has_readiness_output", "has_prosody_output")
ENC_VOC_ONLY_KEYS = ("frames_per_step", "lookahead_frames", "content_kind", "content_dim")
SPECIAL_UNIT_KEYS = ("bos", "eos", "pad")
DEC_FIXED_INPUTS = ("prev_units", "tgt_lang", "prosody_ctx", "src_pos")

STATE_IN = "state_in."
STATE_OUT = "state_out."
STATE_DTYPES = ("f32", "i64", "f16")        # what KVStateCache can back (Stage 07)
SHARED_MEM_DTYPES = ("f32", "f16")

# Every integer must fit int32 so both languages agree on the range.
INT_MIN, INT_MAX = -(2**31), 2**31 - 1

# Physical bounds. With these, every derived size (hop, chunk_samples,
# cond_frames, output_samples, algorithmic/onset latency) also fits int32 in
# both languages, and no contract can make the loader preallocate gigabytes.
MIN_SAMPLE_RATE, MAX_SAMPLE_RATE = 8000, 192000
MAX_LOOKAHEAD_FRAMES = 64
MAX_VOCAB = 1 << 20
MAX_CONTENT_DIM = 1 << 16
MAX_LATENT_DIM = 1 << 16
MAX_SPK_EMB_DIM = 1 << 12
MAX_CHUNK_FRAMES = 64
MAX_RING_STEPS = 4096
MAX_LANG_ID = 65535
MAX_SPECIAL_UNIT = MAX_VOCAB + 16
MIN_WAIT_K, MAX_WAIT_K = 1, 10


class ContractError(ValueError):
    """A contract, a bundle or a graph that breaks contract v1 or v2."""


class FrozenMap(Mapping):
    """Immutable, hashable mapping with deterministic sorted key iteration.

    Keeps frozen ModelContract instances hashable (BUG-005) while supporting
    standard dict indexing (`c.langs["en"]`, `dict(c.langs)`, equality with dict).
    """
    __slots__ = ("_items", "_dict")

    def __init__(self, data=()):
        items = tuple(sorted(dict(data).items(), key=lambda kv: str(kv[0])))
        self._items = items
        self._dict = dict(items)

    def __getitem__(self, key):
        return self._dict[key]

    def __iter__(self):
        return (k for k, _ in self._items)

    def __len__(self):
        return len(self._items)

    def __hash__(self):
        return hash(self._items)

    def __eq__(self, other):
        if isinstance(other, FrozenMap):
            return self._dict == other._dict
        if isinstance(other, Mapping):
            return self._dict == dict(other)
        return NotImplemented

    def __bool__(self):
        return bool(self._items)

    def __repr__(self):
        return f"FrozenMap({self._dict!r})"


EMPTY_MAP = FrozenMap()


def required_keys(role, content_kind="discrete", version=CONTRACT_VERSION):
    """Keys a contract must have, in the order they are checked."""
    if version == CONTRACT_VERSION_V2 and role == "translator_enc":
        return [
            "contract_version", "role", "sample_rate", "frame_samples",
            "chunk_frames", "latent_dim", "max_src_steps", "shared_mem_states",
            "langs", "has_readiness_output", "states", "license",
        ]
    if version == CONTRACT_VERSION_V2 and role == "translator_dec":
        return [
            "contract_version", "role", "sample_rate", "frame_samples",
            "chunk_frames", "vocab", "special_units", "max_src_steps",
            "max_tgt_steps", "shared_mem_states", "langs", "has_prosody_output",
            "states", "license",
        ]
    keys = [
        "contract_version", "role", "sample_rate", "frame_samples",
        "frames_per_step", "lookahead_frames", "content_kind",
        CONTENT_KEY[content_kind],
    ]
    if version == CONTRACT_VERSION_V2 and role == "encoder":
        keys.append("latent_dim")
    elif role == "vocoder":
        keys += VOCODER_KEYS
    return keys + ["states", "license"]


@dataclass(frozen=True)
class TensorSpec:
    """One graph input or output. shape uses -1 for a dynamic dimension and
    None when the graph gives no shape at all (unknown rank; () is a scalar)."""
    name: str
    dtype: str            # "f32" | "i64" | "f16" | "other"
    shape: tuple | None

    def to_dict(self):
        return {"name": self.name, "dtype": self.dtype,
                "shape": None if self.shape is None else list(self.shape)}

    @classmethod
    def from_dict(cls, d):
        shape = d.get("shape", ()) if isinstance(d, dict) else ()
        if not isinstance(d, dict) or not isinstance(d.get("name"), str) or \
                not isinstance(d.get("dtype"), str) or "shape" not in d or \
                not (shape is None or isinstance(shape, list) and all(_is_int(x) for x in shape)):
            raise ContractError(f"graph tensor must be {{name, dtype, shape}}, got {d!r}")
        return cls(d["name"], d["dtype"], None if shape is None else tuple(shape))


@dataclass(frozen=True)
class ModelContract:
    role: str
    sample_rate: int
    frame_samples: int
    frames_per_step: int = 0             # encoder / vocoder only (1, 2 or 4)
    lookahead_frames: int = 0            # encoder / vocoder only
    content_kind: str = "discrete"       # encoder / vocoder only
    license: str = ""
    vocab: int = 0                       # discrete enc/voc + translator_dec
    content_dim: int = 0                 # soft enc/voc only
    cond_frame_samples: int = 0          # vocoder only
    cond_inputs: tuple = ()              # vocoder only
    spk_emb_dim: int = 0                 # vocoder only
    overlap_samples: int = 0             # vocoder only
    states: tuple = ()
    contract_version: int = CONTRACT_VERSION
    # Contract v2 fields (Mode B simultaneous S2UT):
    latent_dim: int = 0                  # v2 encoder + translator_enc
    chunk_frames: int = 0                # translator_enc + translator_dec
    max_src_steps: int = 0               # translator_enc + translator_dec
    max_tgt_steps: int = 0               # translator_dec only
    shared_mem_states: tuple = ()        # translator_enc + translator_dec
    langs: Mapping = EMPTY_MAP           # translator_enc + translator_dec
    special_units: Mapping = EMPTY_MAP   # translator_dec only {"bos", "eos", "pad"}
    has_readiness_output: bool = False   # translator_enc only
    has_prosody_output: bool = False     # translator_dec only

    def __post_init__(self):
        # Lists/dicts from code-built contracts become immutable tuples / FrozenMaps,
        # so equality and hashing work (BUG-005).
        for key in ("cond_inputs", "states", "shared_mem_states"):
            val = getattr(self, key)
            if isinstance(val, list):
                object.__setattr__(self, key, tuple(val))
        for key in ("langs", "special_units"):
            val = getattr(self, key)
            if isinstance(val, Mapping) and not isinstance(val, FrozenMap):
                object.__setattr__(self, key, FrozenMap(val))

    # -- derived geometry ---------------------------------------------------
    @property
    def hop(self):
        """Samples consumed (encoder) or produced (vocoder) per Run."""
        return self.frames_per_step * self.frame_samples

    @property
    def chunk_samples(self):
        """Samples per translation chunk (translator_enc / translator_dec)."""
        return self.chunk_frames * self.frame_samples

    @property
    def cond_frames(self):
        return self.hop // self.cond_frame_samples if self.cond_frame_samples > 0 else 0

    @property
    def output_samples(self):
        """Vocoder `audio` output length: hop plus the OLA tail."""
        return self.hop + self.overlap_samples

    @property
    def content_name(self):
        """Encoder output = vocoder input: `units` (i64 ids) or `content` (f32)."""
        return "units" if self.content_kind == "discrete" else "content"

    def content_spec(self):
        if self.content_kind == "discrete":
            return TensorSpec("units", "i64", (1, self.frames_per_step))
        return TensorSpec("content", "f32", (1, self.frames_per_step, self.content_dim))

    def expected_inputs(self):
        """Non-state graph inputs, in contract order (states and translator_dec
        shared_mem_states have graph-defined shapes checked in validate_io)."""
        if self.role == "encoder":
            return [TensorSpec("audio", "f32", (1, self.hop))]
        if self.role == "translator_enc":
            return [
                TensorSpec("latents", "f32", (1, self.chunk_frames, self.latent_dim)),
                TensorSpec("src_lang", "i64", (1,)),
            ]
        if self.role == "translator_dec":
            return [
                TensorSpec("prev_units", "i64", (1, self.chunk_frames)),
                TensorSpec("tgt_lang", "i64", (1,)),
                TensorSpec("prosody_ctx", "f32", (1, 2)),
                TensorSpec("src_pos", "i64", (1,)),
            ]
        specs = [self.content_spec()]
        for name in self.cond_inputs:
            dim = self.spk_emb_dim if name == "spk_emb" else self.cond_frames
            specs.append(TensorSpec(name, "f32", (1, dim)))
        return specs

    def expected_outputs(self):
        if self.role == "encoder":
            outs = [self.content_spec()]
            if self.contract_version == CONTRACT_VERSION_V2:
                outs.append(TensorSpec("latents", "f32",
                                       (1, self.frames_per_step, self.latent_dim)))
            return outs
        if self.role == "translator_enc":
            if self.has_readiness_output:
                return [TensorSpec("readiness_logit", "f32", (1, 1))]
            return []
        if self.role == "translator_dec":
            outs = [TensorSpec("units", "i64", (1, self.chunk_frames))]
            if self.has_prosody_output:
                outs.append(TensorSpec("prosody", "f32", (1, self.chunk_frames, 4)))
            outs.append(TensorSpec("eos_prob", "f32", (1, 1)))
            return outs
        return [TensorSpec("audio", "f32", (1, self.output_samples))]

    # -- (de)serialisation --------------------------------------------------
    def to_dict(self):
        if self.role == "translator_enc":
            return {
                "contract_version": self.contract_version,
                "role": self.role,
                "sample_rate": self.sample_rate,
                "frame_samples": self.frame_samples,
                "chunk_frames": self.chunk_frames,
                "latent_dim": self.latent_dim,
                "max_src_steps": self.max_src_steps,
                "shared_mem_states": list(self.shared_mem_states),
                "langs": dict(self.langs),
                "has_readiness_output": self.has_readiness_output,
                "states": list(self.states),
                "license": self.license,
            }
        if self.role == "translator_dec":
            return {
                "contract_version": self.contract_version,
                "role": self.role,
                "sample_rate": self.sample_rate,
                "frame_samples": self.frame_samples,
                "chunk_frames": self.chunk_frames,
                "vocab": self.vocab,
                "special_units": dict(self.special_units),
                "max_src_steps": self.max_src_steps,
                "max_tgt_steps": self.max_tgt_steps,
                "shared_mem_states": list(self.shared_mem_states),
                "langs": dict(self.langs),
                "has_prosody_output": self.has_prosody_output,
                "states": list(self.states),
                "license": self.license,
            }
        d = {
            "contract_version": self.contract_version,
            "role": self.role,
            "sample_rate": self.sample_rate,
            "frame_samples": self.frame_samples,
            "frames_per_step": self.frames_per_step,
            "lookahead_frames": self.lookahead_frames,
            "content_kind": self.content_kind,
            CONTENT_KEY[self.content_kind]:
                self.vocab if self.content_kind == "discrete" else self.content_dim,
        }
        if self.contract_version == CONTRACT_VERSION_V2 and self.role == "encoder":
            d["latent_dim"] = self.latent_dim
        elif self.role == "vocoder":
            d.update(cond_frame_samples=self.cond_frame_samples,
                     cond_inputs=list(self.cond_inputs),
                     spk_emb_dim=self.spk_emb_dim,
                     overlap_samples=self.overlap_samples)
        d.update(states=list(self.states), license=self.license)
        return d

    def to_json(self):
        """Canonical form: sorted keys, no whitespace. Validates first."""
        self.validate()
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False)

    @classmethod
    def from_json(cls, text):
        """Parse and validate a contract string. Raises ContractError."""
        try:
            obj = json.loads(text, parse_constant=_reject_constant)
        except (ValueError, TypeError, RecursionError):
            raise ContractError("contract is not valid JSON") from None
        return cls.from_dict(obj)

    @classmethod
    def from_dict(cls, obj):
        if not isinstance(obj, dict):
            raise ContractError("contract must be a JSON object")
        version = _get_int(obj, "contract_version")
        if version not in SUPPORTED_CONTRACT_VERSIONS:
            raise ContractError(_version_message(version))
        role = _get_str(obj, "role")
        allowed_roles = ROLES_V1 if version == CONTRACT_VERSION else ROLES_V2
        if role not in allowed_roles:
            raise ContractError(_role_message(role, version))

        kind = "discrete"
        if role not in TRANSLATOR_ROLES:
            kind = _get_str(obj, "content_kind")
            if kind not in CONTENT_KINDS:
                raise ContractError(f"unknown content_kind '{kind}' (expected discrete or soft)")
            if version == CONTRACT_VERSION_V2 and role == "encoder" and kind != "discrete":
                raise ContractError(f"v2 encoder content_kind must be discrete (got {kind})")

        allowed = required_keys(role, kind, version)
        for key in sorted(obj):
            if key not in allowed:
                raise ContractError(_misplaced_key_message(key, version, role, kind))
        for key in allowed:
            if key not in obj:
                raise ContractError(f"missing key '{key}'")

        if role == "translator_enc":
            c = cls(
                contract_version=version,
                role=role,
                sample_rate=_get_int(obj, "sample_rate"),
                frame_samples=_get_int(obj, "frame_samples"),
                chunk_frames=_get_int(obj, "chunk_frames"),
                latent_dim=_get_int(obj, "latent_dim"),
                max_src_steps=_get_int(obj, "max_src_steps"),
                shared_mem_states=_get_str_list(obj, "shared_mem_states"),
                langs=_get_langs(obj),
                has_readiness_output=_get_bool(obj, "has_readiness_output"),
                states=_get_str_list(obj, "states"),
                license=_get_str(obj, "license"),
            )
        elif role == "translator_dec":
            c = cls(
                contract_version=version,
                role=role,
                sample_rate=_get_int(obj, "sample_rate"),
                frame_samples=_get_int(obj, "frame_samples"),
                chunk_frames=_get_int(obj, "chunk_frames"),
                vocab=_get_int(obj, "vocab"),
                special_units=_get_special_units(obj),
                max_src_steps=_get_int(obj, "max_src_steps"),
                max_tgt_steps=_get_int(obj, "max_tgt_steps"),
                shared_mem_states=_get_str_list(obj, "shared_mem_states"),
                langs=_get_langs(obj),
                has_prosody_output=_get_bool(obj, "has_prosody_output"),
                states=_get_str_list(obj, "states"),
                license=_get_str(obj, "license"),
            )
        else:
            c = cls(
                role=role,
                sample_rate=_get_int(obj, "sample_rate"),
                frame_samples=_get_int(obj, "frame_samples"),
                frames_per_step=_get_int(obj, "frames_per_step"),
                lookahead_frames=_get_int(obj, "lookahead_frames"),
                content_kind=kind,
                vocab=_get_int(obj, "vocab") if kind == "discrete" else 0,
                content_dim=_get_int(obj, "content_dim") if kind == "soft" else 0,
                latent_dim=(_get_int(obj, "latent_dim")
                            if version == CONTRACT_VERSION_V2 and role == "encoder" else 0),
                cond_frame_samples=_get_int(obj, "cond_frame_samples") if role == "vocoder" else 0,
                cond_inputs=_get_str_list(obj, "cond_inputs") if role == "vocoder" else (),
                spk_emb_dim=_get_int(obj, "spk_emb_dim") if role == "vocoder" else 0,
                overlap_samples=_get_int(obj, "overlap_samples") if role == "vocoder" else 0,
                states=_get_str_list(obj, "states"),
                license=_get_str(obj, "license"),
                contract_version=version,
            )
        c.validate()
        return c

    # -- rules --------------------------------------------------------------
    def validate(self):
        """Semantic checks on one contract. Raises ContractError."""
        self._check_types()
        if self.contract_version not in SUPPORTED_CONTRACT_VERSIONS:
            raise ContractError(_version_message(self.contract_version))
        allowed_roles = ROLES_V1 if self.contract_version == CONTRACT_VERSION else ROLES_V2
        if self.role not in allowed_roles:
            raise ContractError(_role_message(self.role, self.contract_version))

        if self.role not in TRANSLATOR_ROLES:
            if self.content_kind not in CONTENT_KINDS:
                raise ContractError(
                    f"unknown content_kind '{self.content_kind}' (expected discrete or soft)")
            if self.contract_version == CONTRACT_VERSION_V2 and self.role == "encoder" \
                    and self.content_kind != "discrete":
                raise ContractError(f"v2 encoder content_kind must be discrete (got {self.content_kind})")

        self._check_misplaced_fields()

        _in_range("sample_rate", self.sample_rate, MIN_SAMPLE_RATE, MAX_SAMPLE_RATE)
        _in_range("frame_samples", self.frame_samples, 1, MAX_SAMPLE_RATE)

        if self.role in TRANSLATOR_ROLES:
            _in_range("chunk_frames", self.chunk_frames, 1, MAX_CHUNK_FRAMES)
            if self.chunk_samples > self.sample_rate:
                raise ContractError(
                    f"chunk_samples ({self.chunk_samples} samples) is longer than one second at "
                    f"sample_rate {self.sample_rate}")
            _in_range("max_src_steps", self.max_src_steps, 1, MAX_RING_STEPS)
            if not self.shared_mem_states:
                raise ContractError("shared_mem_states must not be empty")
            seen_mem = set()
            for m in self.shared_mem_states:
                if not m:
                    raise ContractError("shared_mem_states names must not be empty")
                if m.startswith(STATE_IN) or m.startswith(STATE_OUT):
                    raise ContractError(
                        f"shared_mem_state '{m}' must be a base name, without the "
                        f"state_in./state_out. prefix")
                if m in seen_mem:
                    raise ContractError(f"duplicate shared_mem_state '{m}'")
                seen_mem.add(m)
            _validate_langs_map(self.langs)
            if self.role == "translator_enc":
                _in_range("latent_dim", self.latent_dim, 1, MAX_LATENT_DIM)
            else:
                _in_range("vocab", self.vocab, 1, MAX_VOCAB)
                _in_range("max_tgt_steps", self.max_tgt_steps, 1, MAX_RING_STEPS)
                _validate_special_units_map(self.special_units, self.vocab)
        else:
            if self.frames_per_step not in FRAMES_PER_STEP:
                raise ContractError(f"frames_per_step must be 1, 2 or 4 (got {self.frames_per_step})")
            if self.hop > self.sample_rate:
                raise ContractError(f"hop ({self.hop} samples) is longer than one second at "
                                    f"sample_rate {self.sample_rate}")
            _in_range("lookahead_frames", self.lookahead_frames, 0, MAX_LOOKAHEAD_FRAMES)
            if self.content_kind == "discrete":
                _in_range("vocab", self.vocab, 1, MAX_VOCAB)
            else:
                _in_range("content_dim", self.content_dim, 1, MAX_CONTENT_DIM)
            if self.contract_version == CONTRACT_VERSION_V2 and self.role == "encoder":
                _in_range("latent_dim", self.latent_dim, 1, MAX_LATENT_DIM)

        if self.role == "vocoder":
            _positive("cond_frame_samples", self.cond_frame_samples)
            if self.frame_samples % self.cond_frame_samples != 0:
                raise ContractError(
                    f"frame_samples ({self.frame_samples}) is not a multiple of "
                    f"cond_frame_samples ({self.cond_frame_samples})")
            seen = set()
            for name in self.cond_inputs:
                if name not in COND_INPUTS:
                    raise ContractError(f"unknown cond input '{name}'")
                if name in seen:
                    raise ContractError(f"duplicate cond input '{name}'")
                seen.add(name)
            _in_range("spk_emb_dim", self.spk_emb_dim, 0, MAX_SPK_EMB_DIM)
            if "spk_emb" in seen and self.spk_emb_dim == 0:
                raise ContractError("spk_emb_dim must be > 0 when cond_inputs lists spk_emb")
            if "spk_emb" not in seen and self.spk_emb_dim != 0:
                raise ContractError("spk_emb_dim must be 0 when cond_inputs does not list spk_emb")
            if self.overlap_samples < 0:
                raise ContractError(
                    f"overlap_samples must be >= 0 (got {self.overlap_samples})")
            if self.overlap_samples > self.hop:
                raise ContractError(
                    f"overlap_samples ({self.overlap_samples}) exceeds hop ({self.hop})")

        seen = set()
        for name in self.states:
            if not name:
                raise ContractError("state names must not be empty")
            if name.startswith(STATE_IN) or name.startswith(STATE_OUT):
                raise ContractError(
                    f"state '{name}' must be a base name, without the state_in./state_out. prefix")
            if name in seen:
                raise ContractError(f"duplicate state '{name}'")
            seen.add(name)

        if self.role == "translator_enc":
            if "src_pos" not in seen:
                raise ContractError("translator_enc states must include 'src_pos'")
            for m in self.shared_mem_states:
                if m == "src_pos":
                    raise ContractError("shared_mem_state 'src_pos' conflicts with position counter")
                if m not in seen:
                    raise ContractError(
                        f"translator_enc states must include shared_mem_state '{m}'")
        elif self.role == "translator_dec":
            if "tgt_pos" not in seen:
                raise ContractError("translator_dec states must include 'tgt_pos'")
            for m in self.shared_mem_states:
                if m in DEC_FIXED_INPUTS:
                    raise ContractError(f"shared_mem_state '{m}' conflicts with graph input")
                if m in seen:
                    raise ContractError(
                        f"translator_dec shared_mem_state '{m}' is read-only and must not be in states")

        if not self.license:
            raise ContractError("license must not be empty")

    def _check_misplaced_fields(self):
        v = self.contract_version
        if v == CONTRACT_VERSION:
            for key in V2_ONLY_KEYS:
                if getattr(self, key):
                    raise ContractError(_misplaced_key_message(key, v, self.role, self.content_kind))
        if self.role != "vocoder":
            for key in VOCODER_KEYS:
                if getattr(self, key):
                    raise ContractError(_misplaced_key_message(key, v, self.role, self.content_kind))
        if self.role in TRANSLATOR_ROLES:
            if self.frames_per_step != 0:
                raise ContractError(_misplaced_key_message("frames_per_step", v, self.role, "discrete"))
            if self.lookahead_frames != 0:
                raise ContractError(_misplaced_key_message("lookahead_frames", v, self.role, "discrete"))
            if self.content_kind != "discrete":
                raise ContractError(_misplaced_key_message("content_kind", v, self.role, "discrete"))
            if self.content_dim != 0:
                raise ContractError(_misplaced_key_message("content_dim", v, self.role, "discrete"))
            if self.role == "translator_enc":
                for key in ("vocab", "special_units", "max_tgt_steps", "has_prosody_output"):
                    if getattr(self, key):
                        raise ContractError(_misplaced_key_message(key, v, self.role, "discrete"))
            else:
                for key in ("latent_dim", "has_readiness_output"):
                    if getattr(self, key):
                        raise ContractError(_misplaced_key_message(key, v, self.role, "discrete"))
        else:
            if self.content_kind == "discrete" and self.content_dim != 0:
                raise ContractError(_misplaced_key_message("content_dim", v, self.role, "discrete"))
            if self.content_kind == "soft" and self.vocab != 0:
                raise ContractError(_misplaced_key_message("vocab", v, self.role, "soft"))
            for key in ("chunk_frames", "max_src_steps", "max_tgt_steps", "shared_mem_states",
                        "langs", "special_units", "has_readiness_output", "has_prosody_output"):
                if getattr(self, key):
                    raise ContractError(_misplaced_key_message(key, v, self.role, self.content_kind))
            if self.role == "vocoder" and self.latent_dim != 0:
                raise ContractError(_misplaced_key_message("latent_dim", v, self.role, self.content_kind))

    def _check_types(self):
        # from_dict guarantees these; contracts built in code (exporters) may not.
        for key in ("contract_version", "sample_rate", "frame_samples", "frames_per_step",
                    "lookahead_frames", "vocab", "content_dim", "cond_frame_samples",
                    "spk_emb_dim", "overlap_samples", "latent_dim", "chunk_frames",
                    "max_src_steps", "max_tgt_steps"):
            v = getattr(self, key)
            if not _is_int(v):
                raise ContractError(f"key '{key}' must be an integer")
            if not INT_MIN <= v <= INT_MAX:
                raise ContractError(f"key '{key}' is out of range")
        for key in ("has_readiness_output", "has_prosody_output"):
            if not isinstance(getattr(self, key), bool):
                raise ContractError(f"key '{key}' must be a boolean")
        for key in ("role", "content_kind", "license"):
            if not isinstance(getattr(self, key), str):
                raise ContractError(f"key '{key}' must be a string")
        for key in ("cond_inputs", "states", "shared_mem_states"):
            v = getattr(self, key)
            if not isinstance(v, (list, tuple)) or not all(isinstance(s, str) for s in v):
                raise ContractError(f"key '{key}' must be a list of strings")
        for key in ("langs", "special_units"):
            if not isinstance(getattr(self, key), Mapping):
                raise ContractError(f"key '{key}' must be an object")
        strings = [self.role, self.content_kind, self.license,
                   *self.cond_inputs, *self.states, *self.shared_mem_states,
                   *(k for k in self.langs if isinstance(k, str)),
                   *(k for k in self.special_units if isinstance(k, str))]
        for s in strings:
            try:
                s.encode("utf-8")
            except UnicodeEncodeError:   # e.g. a lone surrogate: not storable as metadata
                raise ContractError(f"string {s!r} is not valid UTF-8") from None


def validate_pair(enc, voc):
    """An encoder + vocoder that can run back to back (Mode A). Raises ContractError."""
    if enc.role != "encoder":
        raise ContractError(f"pair: first model must be an encoder (got {enc.role})")
    if voc.role != "vocoder":
        raise ContractError(f"pair: second model must be a vocoder (got {voc.role})")
    for key in ("sample_rate", "frame_samples", "frames_per_step", "content_kind",
                CONTENT_KEY[enc.content_kind]):
        a, b = getattr(enc, key), getattr(voc, key)
        if a != b:
            raise ContractError(f"{key} mismatch: encoder {a}, vocoder {b}")


def algorithmic_latency(enc, voc):
    """Samples from an input sample to the output describing it, for a valid pair.

    hop (framing) + (L_enc + L_voc) x frame_samples + overlap_samples. Device
    I/O and compute come on top (master latency budget).
    """
    return enc.hop + (enc.lookahead_frames + voc.lookahead_frames) * enc.frame_samples \
        + voc.overlap_samples


def validate_s2ut_bundle(enc, trans_enc, trans_dec, voc):
    """Validate a Mode B 4-model S2UT bundle (enc, trans_enc, trans_dec, voc)."""
    for pos, m, want in (("first", enc, "encoder"),
                         ("second", trans_enc, "translator_enc"),
                         ("third", trans_dec, "translator_dec"),
                         ("fourth", voc, "vocoder")):
        if m.role != want:
            article = "an" if want == "encoder" else "a"
            raise ContractError(f"s2ut bundle: {pos} model must be {article} {want} (got {m.role})")
    if enc.contract_version != CONTRACT_VERSION_V2:
        raise ContractError(
            f"s2ut bundle: encoder must use contract_version 2 (got {enc.contract_version})")
    if voc.content_kind != "discrete":
        raise ContractError(
            f"s2ut bundle: vocoder content_kind must be discrete (got {voc.content_kind})")
    validate_pair(enc, voc)
    for role_name, m in (("translator_enc", trans_enc), ("translator_dec", trans_dec)):
        if enc.sample_rate != m.sample_rate:
            raise ContractError(
                f"sample_rate mismatch: encoder {enc.sample_rate}, {role_name} {m.sample_rate}")
        if enc.frame_samples != m.frame_samples:
            raise ContractError(
                f"frame_samples mismatch: encoder {enc.frame_samples}, "
                f"{role_name} {m.frame_samples}")
    if enc.latent_dim != trans_enc.latent_dim:
        raise ContractError(
            f"latent_dim mismatch: encoder {enc.latent_dim}, translator_enc {trans_enc.latent_dim}")
    if trans_enc.chunk_frames != trans_dec.chunk_frames:
        raise ContractError(
            f"chunk_frames mismatch: translator_enc {trans_enc.chunk_frames}, "
            f"translator_dec {trans_dec.chunk_frames}")
    if trans_enc.chunk_frames % enc.frames_per_step != 0:
        raise ContractError(
            f"chunk_frames ({trans_enc.chunk_frames}) is not a multiple of "
            f"encoder frames_per_step ({enc.frames_per_step})")
    if trans_enc.max_src_steps != trans_dec.max_src_steps:
        raise ContractError(
            f"max_src_steps mismatch: translator_enc {trans_enc.max_src_steps}, "
            f"translator_dec {trans_dec.max_src_steps}")
    if trans_enc.shared_mem_states != trans_dec.shared_mem_states:
        raise ContractError("shared_mem_states mismatch between translator_enc and translator_dec")
    if trans_dec.vocab != voc.vocab:
        raise ContractError(
            f"vocab mismatch: translator_dec {trans_dec.vocab}, vocoder {voc.vocab}")
    if trans_enc.langs != trans_dec.langs:
        raise ContractError("langs mismatch between translator_enc and translator_dec")


def validate_s2ut_shared_io(trans_enc, enc_outputs, trans_dec, dec_inputs):
    """Cross-check that translator_enc's state_out.mem_* tensors match
    translator_dec's read-only mem_* graph inputs in dtype and shape."""
    outs = {t.name: t for t in enc_outputs}
    ins = {t.name: t for t in dec_inputs}
    for m in trans_enc.shared_mem_states:
        o = outs.get(STATE_OUT + m)
        i = ins.get(m)
        if o is None:
            raise ContractError(f"missing graph output '{STATE_OUT}{m}'")
        if i is None:
            raise ContractError(f"missing graph input '{m}'")
        if o.dtype != i.dtype:
            raise ContractError(
                f"shared_mem_state '{m}' dtype mismatch: translator_enc {o.dtype}, "
                f"translator_dec {i.dtype}")
        if o.shape is None or i.shape is None or tuple(o.shape) != tuple(i.shape):
            raise ContractError(
                f"shared_mem_state '{m}' shape mismatch: translator_enc {format_shape(o.shape)}, "
                f"translator_dec {format_shape(i.shape)}")


def s2ut_onset_latency_samples(enc, trans_enc, trans_dec, voc, wait_k):
    """Mode B algorithmic onset latency in samples (Milestone 3 / Stage 02):
    wait_k * chunk_frames * frame_samples + (L_enc + L_voc) * frame_samples + overlap_samples.
    """
    if not _is_int(wait_k) or not MIN_WAIT_K <= wait_k <= MAX_WAIT_K:
        raise ContractError(
            f"wait_k must be between {MIN_WAIT_K} and {MAX_WAIT_K} (got {wait_k!r})")
    return (wait_k * trans_enc.chunk_frames * enc.frame_samples
            + (enc.lookahead_frames + voc.lookahead_frames) * enc.frame_samples
            + voc.overlap_samples)


def validate_io(c, inputs, outputs):
    """Cross-check a graph's inputs/outputs (TensorSpec lists) against `c`."""
    readonly_mem = c.shared_mem_states if c.role == "translator_dec" else ()
    _check_side(c, "input", c.expected_inputs(), inputs, STATE_IN, readonly_mem)
    _check_side(c, "output", c.expected_outputs(), outputs, STATE_OUT, ())
    ins = {t.name: t for t in inputs}
    outs = {t.name: t for t in outputs}

    if c.role == "translator_dec":
        for m in c.shared_mem_states:
            t = ins.get(m)
            if t is None:
                raise ContractError(f"missing graph input '{m}'")
            _check_shared_mem_tensor(m, t, c.max_src_steps)

    for x in c.states:
        i, o = ins.get(STATE_IN + x), outs.get(STATE_OUT + x)
        if i is None and o is None:
            raise ContractError(
                f"missing state '{x}': the graph has neither state_in.{x} nor state_out.{x}")
        if o is None:
            raise ContractError(f"unpaired state '{x}': state_in.{x} has no state_out.{x}")
        if i is None:
            raise ContractError(f"unpaired state '{x}': state_out.{x} has no state_in.{x}")
        if i.dtype not in STATE_DTYPES:
            raise ContractError(f"state '{x}' has unsupported dtype {i.dtype}")
        if i.dtype != o.dtype:
            raise ContractError(f"state '{x}' changes dtype: state_in.{x} is {i.dtype}, "
                                f"state_out.{x} is {o.dtype}")
        # The loader sizes every state buffer once, from the graph.
        if i.shape is None:
            raise ContractError(f"state '{x}' has no shape in the graph")
        if any(d < 0 for d in i.shape):
            raise ContractError(f"state '{x}' has a dynamic shape {format_shape(i.shape)}")
        if any(d == 0 for d in i.shape):
            raise ContractError(f"state '{x}' has an empty dimension {format_shape(i.shape)}")
        if o.shape is None or tuple(i.shape) != tuple(o.shape):
            raise ContractError(f"state '{x}' changes shape: state_in.{x} is "
                                f"{format_shape(i.shape)}, state_out.{x} is "
                                f"{format_shape(o.shape)}")
        if (c.role == "translator_enc" and x == "src_pos") or \
                (c.role == "translator_dec" and x == "tgt_pos"):
            if i.dtype != "i64":
                raise ContractError(
                    f"position state '{x}' has dtype {i.dtype}, contract expects i64")
            if tuple(i.shape) != (1,):
                raise ContractError(
                    f"position state '{x}' has shape {format_shape(i.shape)}, "
                    f"contract expects [1]")
        if c.role == "translator_enc" and x in c.shared_mem_states:
            _check_shared_mem_tensor(x, i, c.max_src_steps)


def check_model(contract_text, inputs, outputs):
    """What a loader does: parse the metadata (None = absent), then check the
    graph against it. Returns the contract; raises ContractError."""
    if contract_text is None:
        raise ContractError(MISSING_METADATA)
    c = ModelContract.from_json(contract_text)
    validate_io(c, inputs, outputs)
    return c


def format_shape(shape):
    """"[1,320]"; dynamic dimensions print as "?", no shape at all as "unknown"."""
    if shape is None:
        return "unknown"
    return "[" + ",".join("?" if d < 0 else str(d) for d in shape) + "]"


# -- voices.json & langs.json -------------------------------------------------
VOICE_KEYS = ("name", "display", "embedding", "logf0_mean", "logf0_std", "energy_mean",
              "energy_std", "license", "source")
VOICE_OPTIONAL_KEYS = ("attribution", "lang")


def validate_voices(data, vocoder):
    """voices.json (schema 1) for a bundle whose vocoder contract is `vocoder`."""
    if not isinstance(data, dict):
        raise ContractError("voices.json must be a JSON object")
    schema = data.get("schema")
    if not _is_int(schema) or schema != 1:     # True == 1 and 1.0 == 1 in Python
        raise ContractError(f"unsupported voices.json schema {schema!r}")
    unknown = sorted(set(data) - {"schema", "voices"})
    if unknown:
        raise ContractError(f"voices.json: unknown key '{unknown[0]}'")
    voices = data.get("voices")
    if not isinstance(voices, list) or not voices:
        raise ContractError("voices.json: 'voices' must be a non-empty list")
    names = set()
    for i, v in enumerate(voices):
        where = f"voices[{i}]"
        if not isinstance(v, dict):
            raise ContractError(f"{where} must be an object")
        for key in sorted(v):
            if key not in VOICE_KEYS and key not in VOICE_OPTIONAL_KEYS:
                raise ContractError(f"{where}: unknown key '{key}'")
        for key in VOICE_KEYS:
            if key not in v:
                raise ContractError(f"{where}: missing key '{key}'")
        name = v["name"]
        if not isinstance(name, str) or not name or \
                not all(ch.isascii() and (ch.isalnum() or ch in "-_") for ch in name) or \
                name != name.lower():
            raise ContractError(f"{where}: name must be lowercase [a-z0-9_-]+ (got {name!r})")
        if name in names:
            raise ContractError(f"duplicate voice '{name}'")
        names.add(name)
        for key in ("display", "license", "source") + tuple(
                k for k in VOICE_OPTIONAL_KEYS if k in v):
            if not isinstance(v[key], str) or not v[key]:
                raise ContractError(f"voice '{name}': {key} must be a non-empty string")
        if "lang" in v and not _is_lang_code(v["lang"]):
            raise ContractError(
                f"voice '{name}': invalid language code '{v['lang']}' "
                f"(expected lowercase [a-z-]{{2,8}})")
        for key in ("logf0_mean", "logf0_std", "energy_mean", "energy_std"):
            if not _is_number(v[key]):
                raise ContractError(f"voice '{name}': {key} must be a finite number")
        for key in ("logf0_std", "energy_std"):
            if v[key] <= 0:
                raise ContractError(f"voice '{name}': {key} must be > 0")
        emb = v["embedding"]
        if not isinstance(emb, list) or not all(_is_number(x) for x in emb):
            raise ContractError(f"voice '{name}': embedding must be a list of numbers")
        if len(emb) != vocoder.spk_emb_dim:
            raise ContractError(f"voice '{name}': embedding has {len(emb)} values, "
                                f"the vocoder expects spk_emb_dim {vocoder.spk_emb_dim}")


LANG_ENTRY_KEYS = ("code", "id", "name")


def validate_langs(data, translator=None):
    """Validate langs.json (schema 1) and optionally cross-check against a
    translator_enc / translator_dec contract."""
    if not isinstance(data, dict):
        raise ContractError("langs.json must be a JSON object")
    schema = data.get("schema")
    if not _is_int(schema) or schema != 1:
        raise ContractError("unsupported langs.json schema (expected the integer 1)")
    unknown = sorted(set(data) - {"schema", "langs", "default_wait_k"})
    if unknown:
        raise ContractError(f"langs.json: unknown key '{unknown[0]}'")
    langs = data.get("langs")
    if not isinstance(langs, list) or not langs:
        raise ContractError("langs.json: 'langs' must be a non-empty list")
    codes = {}
    ids = set()
    for i, entry in enumerate(langs):
        if not isinstance(entry, dict):
            raise ContractError("langs.json: every language must be an object")
        for key in sorted(entry):
            if key not in LANG_ENTRY_KEYS:
                raise ContractError(f"langs[{i}]: unknown key '{key}'")
        code = entry.get("code")
        if not isinstance(code, str) or not _is_lang_code(code):
            raise ContractError("langs.json: 'code' must be a lowercase language tag")
        lid = entry.get("id")
        if not _is_int(lid) or not 0 <= lid <= MAX_LANG_ID:
            raise ContractError(
                f"langs.json: 'id' must be an integer in [0, {MAX_LANG_ID}] for '{code}'")
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ContractError(f"langs.json: 'name' must be a non-empty string for '{code}'")
        if code in codes:
            raise ContractError(f"langs.json: duplicate code '{code}'")
        if lid in ids:
            raise ContractError(f"langs.json: duplicate id for '{code}'")
        codes[code] = lid
        ids.add(lid)
    if "default_wait_k" in data:
        dk = data["default_wait_k"]
        if not isinstance(dk, dict):
            raise ContractError("langs.json: 'default_wait_k' must be an object")
        for pair_key in sorted(dk):
            parts = pair_key.split("->")
            if len(parts) != 2 or parts[0] not in codes or parts[1] not in codes \
                    or parts[0] == parts[1]:
                raise ContractError(
                    f"langs.json: invalid language pair '{pair_key}' in default_wait_k")
            kv = dk[pair_key]
            if not _is_int(kv) or not MIN_WAIT_K <= kv <= MAX_WAIT_K:
                raise ContractError(
                    f"langs.json: default_wait_k['{pair_key}'] must be an integer in "
                    f"[{MIN_WAIT_K}, {MAX_WAIT_K}]")
    if translator is not None and dict(translator.langs) != codes:
        raise ContractError("langs.json does not match translator contract langs")


# -- ONNX helpers (need the `onnx` package) ------------------------------------
def graph_io(model):
    """(inputs, outputs) TensorSpec lists of an onnx.ModelProto's graph."""
    return ([_value_spec(v) for v in model.graph.input],
            [_value_spec(v) for v in model.graph.output])


def read_contract_text(model):
    """The raw metadata string, or None when the model has none."""
    for p in model.metadata_props:
        if p.key == METADATA_KEY:
            return p.value
    return None


def load_onnx(path):
    """Read an .onnx file and check it like the C++ loader will. Returns
    (contract, model)."""
    import onnx
    try:
        model = onnx.load(str(path), load_external_data=False)
    except Exception as e:
        raise ContractError(f"{path}: invalid ONNX file: {e}") from None
    inputs, outputs = graph_io(model)
    return check_model(read_contract_text(model), inputs, outputs), model


def embed(model, contract_text):
    """Store `contract_text` as the contract, replacing any previous one."""
    keep = [(p.key, p.value) for p in model.metadata_props if p.key != METADATA_KEY]
    del model.metadata_props[:]
    for key, value in sorted(keep + [(METADATA_KEY, contract_text)]):
        p = model.metadata_props.add()
        p.key, p.value = key, value


def _value_spec(v):
    from onnx import TensorProto
    t = v.type.tensor_type
    dtype = {TensorProto.FLOAT: "f32", TensorProto.INT64: "i64",
             TensorProto.FLOAT16: "f16"}.get(t.elem_type, "other")
    if not t.HasField("shape"):
        return TensorSpec(v.name, dtype, None)        # unknown rank, not a scalar
    shape = tuple(d.dim_value if d.HasField("dim_value") else -1 for d in t.shape.dim)
    return TensorSpec(v.name, dtype, shape)


# -- internals -----------------------------------------------------------------
def _reject_constant(name):
    raise ValueError(f"{name} is not JSON")


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v):
    return (_is_int(v) or isinstance(v, float)) and math.isfinite(v)


def _is_lang_code(s):
    return isinstance(s, str) and 2 <= len(s) <= 8 and \
        all(("a" <= ch <= "z") or ch == "-" for ch in s)


def _version_message(version):
    return f"unsupported contract_version {version} (this build reads versions 1 and 2)"


def _role_message(role, version):
    if version == CONTRACT_VERSION:
        return f"unknown role '{role}' (expected encoder or vocoder)"
    return f"unknown role '{role}' (expected encoder, translator_enc, translator_dec or vocoder)"


def _misplaced_key_message(key, version, role, kind):
    if version == CONTRACT_VERSION and key in V2_ONLY_KEYS:
        return f"key '{key}' requires contract_version 2"
    if key in VOCODER_KEYS and role != "vocoder":
        return f"key '{key}' is not valid for role '{role}'"
    if key in V2_ONLY_KEYS:
        return f"key '{key}' is not valid for role '{role}'"
    if role in TRANSLATOR_ROLES and key in ENC_VOC_ONLY_KEYS + ("vocab",):
        return f"key '{key}' is not valid for role '{role}'"
    if key in CONTENT_KEY.values() and key != CONTENT_KEY[kind]:
        return f"key '{key}' is not valid for content_kind '{kind}'"
    return f"unknown key '{key}'"


def _get(obj, key):
    if key not in obj:
        raise ContractError(f"missing key '{key}'")
    return obj[key]


def _get_int(obj, key):
    v = _get(obj, key)
    if not _is_int(v):
        raise ContractError(f"key '{key}' must be an integer")
    if not INT_MIN <= v <= INT_MAX:
        raise ContractError(f"key '{key}' is out of range")
    return v


def _get_bool(obj, key):
    v = _get(obj, key)
    if not isinstance(v, bool):
        raise ContractError(f"key '{key}' must be a boolean")
    return v


def _get_str(obj, key):
    v = _get(obj, key)
    if not isinstance(v, str):
        raise ContractError(f"key '{key}' must be a string")
    return v


def _get_str_list(obj, key):
    v = _get(obj, key)
    if not isinstance(v, list) or not all(isinstance(s, str) for s in v):
        raise ContractError(f"key '{key}' must be a list of strings")
    return tuple(v)


def _get_langs(obj):
    v = _get(obj, "langs")
    if not isinstance(v, dict):
        raise ContractError("key 'langs' must be an object")
    _validate_langs_map(v)
    return FrozenMap(v)


def _validate_langs_map(langs):
    if not langs:
        raise ContractError("langs must not be empty")
    seen_ids = set()
    for code in sorted(langs, key=str):
        if not _is_lang_code(code):
            raise ContractError(
                f"langs: invalid language code '{code}' (expected lowercase [a-z-]{{2,8}})")
        lid = langs[code]
        if not _is_int(lid):
            raise ContractError(f"langs['{code}'] must be an integer")
        if not 0 <= lid <= MAX_LANG_ID:
            raise ContractError(
                f"langs['{code}'] must be between 0 and {MAX_LANG_ID} (got {lid})")
        if lid in seen_ids:
            raise ContractError(f"langs: duplicate language id {lid} for '{code}'")
        seen_ids.add(lid)


def _get_special_units(obj):
    v = _get(obj, "special_units")
    if not isinstance(v, dict):
        raise ContractError("key 'special_units' must be an object")
    for k in sorted(v):
        if k not in SPECIAL_UNIT_KEYS:
            raise ContractError(f"special_units: unknown key '{k}'")
    out = {}
    for k in SPECIAL_UNIT_KEYS:
        if k not in v:
            raise ContractError(f"special_units: missing key '{k}'")
        val = v[k]
        if not _is_int(val):
            raise ContractError(f"special_units['{k}'] must be an integer")
        if not INT_MIN <= val <= INT_MAX:
            raise ContractError(f"special_units['{k}'] is out of range")
        out[k] = val
    return FrozenMap(out)


def _validate_special_units_map(su, vocab):
    for k in sorted(su, key=str):
        if k not in SPECIAL_UNIT_KEYS:
            raise ContractError(f"special_units: unknown key '{k}'")
    vals = []
    for k in SPECIAL_UNIT_KEYS:
        if k not in su:
            raise ContractError(f"special_units: missing key '{k}'")
        val = su[k]
        if not _is_int(val):
            raise ContractError(f"special_units['{k}'] must be an integer")
        if not INT_MIN <= val <= INT_MAX:
            raise ContractError(f"special_units['{k}'] is out of range")
        if val < vocab:
            raise ContractError(f"special_units['{k}'] ({val}) must be >= vocab ({vocab})")
        if val > MAX_SPECIAL_UNIT:
            raise ContractError(
                f"special_units['{k}'] ({val}) exceeds max special unit {MAX_SPECIAL_UNIT}")
        vals.append(val)
    if len(set(vals)) != len(SPECIAL_UNIT_KEYS):
        raise ContractError(
            f"special_units values must be distinct "
            f"(got bos={su['bos']}, eos={su['eos']}, pad={su['pad']})")


def _check_shared_mem_tensor(name, spec, max_src_steps):
    if spec.dtype not in SHARED_MEM_DTYPES:
        raise ContractError(
            f"shared_mem_state '{name}' has unsupported dtype {spec.dtype} (expected f32 or f16)")
    if spec.shape is None:
        raise ContractError(f"shared_mem_state '{name}' has no shape in the graph")
    if any(d < 0 for d in spec.shape):
        raise ContractError(
            f"shared_mem_state '{name}' has a dynamic shape {format_shape(spec.shape)}")
    if any(d == 0 for d in spec.shape):
        raise ContractError(
            f"shared_mem_state '{name}' has an empty dimension {format_shape(spec.shape)}")
    if max_src_steps not in spec.shape:
        raise ContractError(
            f"shared_mem_state '{name}' shape {format_shape(spec.shape)} "
            f"does not contain max_src_steps ({max_src_steps})")


def _positive(key, v):
    if v <= 0:
        raise ContractError(f"{key} must be > 0 (got {v})")


def _in_range(key, v, lo, hi):
    if not lo <= v <= hi:
        raise ContractError(f"{key} must be between {lo} and {hi} (got {v})")


def _check_side(c, kind, expected, actual, prefix, readonly_mem=()):
    by_name = {t.name: t for t in actual}
    for e in expected:
        a = by_name.get(e.name)
        if a is None:
            raise ContractError(f"missing graph {kind} '{e.name}'")
        if a.dtype != e.dtype:
            raise ContractError(
                f"graph {kind} '{e.name}' has dtype {a.dtype}, contract expects {e.dtype}")
        if a.shape is None or tuple(a.shape) != tuple(e.shape):
            raise ContractError(f"graph {kind} '{e.name}' has shape {format_shape(a.shape)}, "
                                f"contract expects {format_shape(e.shape)}")
    names = {e.name for e in expected} | set(readonly_mem)
    for a in actual:
        if a.name in names:
            continue
        if a.name.startswith(prefix):
            if a.name[len(prefix):] in c.states:
                continue
            raise ContractError(f"graph {kind} '{a.name}' is not a state listed in the contract")
        raise ContractError(f"unexpected graph {kind} '{a.name}'")
