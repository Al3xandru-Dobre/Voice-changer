"""Model contract v1: everything the C++ engine needs to know about a model.

Spec: Documents/Milestone 2/02_model_contract_fixtures.md. The contract is one
JSON string stored in the ONNX file's metadata_props["fluxvoice.contract"].

src/pipeline/ModelContract.hpp implements the same rules, in the same order and
with the same error messages. The shared cases in tests/data/contracts/ (run by
python_training/tests/test_contract.py and tests/test_model_contract.cpp) keep
the two in step: change a rule or a message in both files at once.

Standard library only, so CTest can run the tests with any Python 3. The ONNX
helpers at the bottom import `onnx` lazily.
"""
import json
import math
from dataclasses import dataclass, field

CONTRACT_VERSION = 1
METADATA_KEY = "fluxvoice.contract"
MISSING_METADATA = f"model has no '{METADATA_KEY}' metadata"

ROLES = ("encoder", "vocoder")
CONTENT_KINDS = ("discrete", "soft")
CONTENT_KEY = {"discrete": "vocab", "soft": "content_dim"}
COND_INPUTS = ("logf0", "voicing", "aperiodicity", "energy", "spk_emb")
FRAMES_PER_STEP = (1, 2, 4)
VOCODER_KEYS = ("cond_frame_samples", "cond_inputs", "spk_emb_dim", "overlap_samples")
STATE_IN = "state_in."
STATE_OUT = "state_out."
STATE_DTYPES = ("f32", "i64", "f16")   # what KVStateCache can back (Stage 07)

# Every integer must fit int32 so both languages agree on the range.
INT_MIN, INT_MAX = -(2**31), 2**31 - 1

# Physical bounds. With these, every derived size (hop, cond_frames,
# output_samples, algorithmic latency) also fits int32 in both languages, and
# no contract can make the loader preallocate gigabytes.
MIN_SAMPLE_RATE, MAX_SAMPLE_RATE = 8000, 192000
MAX_LOOKAHEAD_FRAMES = 64
MAX_VOCAB = 1 << 20
MAX_CONTENT_DIM = 1 << 16
MAX_SPK_EMB_DIM = 1 << 12


class ContractError(ValueError):
    """A contract, a pair or a graph that breaks contract v1."""


def required_keys(role, content_kind):
    """Keys a v1 contract must have, in the order they are checked."""
    keys = ["contract_version", "role", "sample_rate", "frame_samples", "frames_per_step",
            "lookahead_frames", "content_kind", CONTENT_KEY[content_kind]]
    if role == "vocoder":
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
    frames_per_step: int
    lookahead_frames: int
    content_kind: str
    license: str
    vocab: int = 0                 # discrete only
    content_dim: int = 0           # soft only
    cond_frame_samples: int = 0    # vocoder only
    cond_inputs: tuple = ()        # vocoder only
    spk_emb_dim: int = 0           # vocoder only
    overlap_samples: int = 0       # vocoder only
    states: tuple = ()
    contract_version: int = CONTRACT_VERSION

    def __post_init__(self):
        # Lists from code-built contracts become tuples, so equality and
        # hashing work (from_dict already returns tuples).
        for key in ("cond_inputs", "states"):
            if isinstance(getattr(self, key), list):
                object.__setattr__(self, key, tuple(getattr(self, key)))

    # -- derived geometry ---------------------------------------------------
    @property
    def hop(self):
        """Samples consumed (encoder) or produced (vocoder) per Run."""
        return self.frames_per_step * self.frame_samples

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
        """Non-state graph inputs, in contract order (states come from the graph)."""
        if self.role == "encoder":
            return [TensorSpec("audio", "f32", (1, self.hop))]
        specs = [self.content_spec()]
        for name in self.cond_inputs:
            dim = self.spk_emb_dim if name == "spk_emb" else self.cond_frames
            specs.append(TensorSpec(name, "f32", (1, dim)))
        return specs

    def expected_outputs(self):
        if self.role == "encoder":
            return [self.content_spec()]
        return [TensorSpec("audio", "f32", (1, self.output_samples))]

    # -- (de)serialisation --------------------------------------------------
    def to_dict(self):
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
        if self.role == "vocoder":
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
        if version != CONTRACT_VERSION:
            raise ContractError(_version_message(version))
        role = _get_str(obj, "role")
        if role not in ROLES:
            raise ContractError(f"unknown role '{role}' (expected encoder or vocoder)")
        kind = _get_str(obj, "content_kind")
        if kind not in CONTENT_KINDS:
            raise ContractError(f"unknown content_kind '{kind}' (expected discrete or soft)")

        allowed = required_keys(role, kind)
        for key in sorted(obj):
            if key not in allowed:
                raise ContractError(_misplaced_key_message(key, role, kind))
        for key in allowed:
            if key not in obj:
                raise ContractError(f"missing key '{key}'")

        c = cls(
            role=role,
            sample_rate=_get_int(obj, "sample_rate"),
            frame_samples=_get_int(obj, "frame_samples"),
            frames_per_step=_get_int(obj, "frames_per_step"),
            lookahead_frames=_get_int(obj, "lookahead_frames"),
            content_kind=kind,
            vocab=_get_int(obj, "vocab") if kind == "discrete" else 0,
            content_dim=_get_int(obj, "content_dim") if kind == "soft" else 0,
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
        if self.contract_version != CONTRACT_VERSION:
            raise ContractError(_version_message(self.contract_version))
        if self.role not in ROLES:
            raise ContractError(f"unknown role '{self.role}' (expected encoder or vocoder)")
        if self.content_kind not in CONTENT_KINDS:
            raise ContractError(
                f"unknown content_kind '{self.content_kind}' (expected discrete or soft)")
        # Fields that belong to the other role / content kind must stay at
        # their defaults (from_dict already rejected them as keys).
        if self.content_kind == "discrete" and self.content_dim != 0:
            raise ContractError(_misplaced_key_message("content_dim", self.role, "discrete"))
        if self.content_kind == "soft" and self.vocab != 0:
            raise ContractError(_misplaced_key_message("vocab", self.role, "soft"))
        if self.role == "encoder":
            for key in VOCODER_KEYS:
                if getattr(self, key):
                    raise ContractError(_misplaced_key_message(key, "encoder", self.content_kind))

        _in_range("sample_rate", self.sample_rate, MIN_SAMPLE_RATE, MAX_SAMPLE_RATE)
        _in_range("frame_samples", self.frame_samples, 1, MAX_SAMPLE_RATE)
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
        if not self.license:
            raise ContractError("license must not be empty")

    def _check_types(self):
        # from_dict guarantees these; contracts built in code (exporters) may not.
        for key in ("contract_version", "sample_rate", "frame_samples", "frames_per_step",
                    "lookahead_frames", "vocab", "content_dim", "cond_frame_samples",
                    "spk_emb_dim", "overlap_samples"):
            v = getattr(self, key)
            if not _is_int(v):
                raise ContractError(f"key '{key}' must be an integer")
            if not INT_MIN <= v <= INT_MAX:
                raise ContractError(f"key '{key}' is out of range")
        for key in ("role", "content_kind", "license"):
            if not isinstance(getattr(self, key), str):
                raise ContractError(f"key '{key}' must be a string")
        for key in ("cond_inputs", "states"):
            v = getattr(self, key)
            if not isinstance(v, (list, tuple)) or not all(isinstance(s, str) for s in v):
                raise ContractError(f"key '{key}' must be a list of strings")
        for s in (self.role, self.content_kind, self.license, *self.cond_inputs, *self.states):
            try:
                s.encode("utf-8")
            except UnicodeEncodeError:   # e.g. a lone surrogate: not storable as metadata
                raise ContractError(f"string {s!r} is not valid UTF-8") from None


def validate_pair(enc, voc):
    """An encoder + vocoder that can run back to back. Raises ContractError."""
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


def validate_io(c, inputs, outputs):
    """Cross-check a graph's inputs/outputs (TensorSpec lists) against `c`."""
    _check_side(c, "input", c.expected_inputs(), inputs, STATE_IN)
    _check_side(c, "output", c.expected_outputs(), outputs, STATE_OUT)
    ins = {t.name: t for t in inputs}
    outs = {t.name: t for t in outputs}
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


# -- voices.json -------------------------------------------------------------
VOICE_KEYS = ("name", "display", "embedding", "logf0_mean", "logf0_std", "energy_mean",
              "energy_std", "license", "source")
VOICE_OPTIONAL_KEYS = ("attribution",)


def validate_voices(data, vocoder):
    """voices.json (schema 1) for a bundle whose vocoder contract is `vocoder`.

    Python only for now; the C++ reader arrives with `--list-voices` (Stage 08).
    """
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
    model = onnx.load(str(path), load_external_data=False)
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


def _version_message(version):
    return f"unsupported contract_version {version} (this build reads version {CONTRACT_VERSION})"


def _misplaced_key_message(key, role, kind):
    if key in VOCODER_KEYS and role != "vocoder":
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


def _positive(key, v):
    if v <= 0:
        raise ContractError(f"{key} must be > 0 (got {v})")


def _in_range(key, v, lo, hi):
    if not lo <= v <= hi:
        raise ContractError(f"{key} must be between {lo} and {hi} (got {v})")


def _check_side(c, kind, expected, actual, prefix):
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
    names = {e.name for e in expected}
    for a in actual:
        if a.name in names:
            continue
        if a.name.startswith(prefix):
            if a.name[len(prefix):] in c.states:
                continue
            raise ContractError(f"graph {kind} '{a.name}' is not a state listed in the contract")
        raise ContractError(f"unexpected graph {kind} '{a.name}'")
