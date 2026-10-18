# Stage 02 — Model contract & fixture models

> **Milestone 2** · [Master plan](00_master.md) · [← Stage 01](01_python_toolchain_eval_harness.md) · [Stage 03 →](03_causal_pitch_energy.md)

**Status:** ☑ done (2026-10-18)
**Depends on:** Milestone 1 / Stage 01 (CMake + CTest); Stage 01 (Python env for the generator script)
**Research:** §8 stateful export · §9 ORT deployment · §5 conditioning inputs

## Objective

Define once, in a versioned spec, everything the C++ engine needs to know about a
model:
- sample rate and frame geometry;
- lookahead;
- tensor names, shapes and dtypes;
- state pairs;
- conditioning inputs;
- the voices table.

The spec is embedded in each ONNX file and validated by both Python and C++.
Tiny generated **fixture models** that honour it let the C++ track (Stages 07–08)
start and stay testable in CI without any trained weights.

## Context & references

- No tensor-name contract exists today: `python_training/export_stateful_onnx.py`
  is a stub, so the C++ side has nothing to match against.
- `src/pipeline/PipelineTiming.hpp` hard-codes 16 kHz / 1280 / 320 / 320 samples.
  In VC mode the geometry must come from the model instead, so a 20/40/80 ms hop
  is a model property rather than a recompile.
- `src/pipeline/KVStateCache.hpp` holds fp32-only named buffers. Unit-id tensors
  are int64, so dtype support arrives in Stage 07.
- **Duplex constraint.** `AudioEngine::open` sets one `cfg.sampleRate` for capture
  and playback (`src/audio/AudioEngine.hpp:96`), so encoder and vocoder rates
  must match.
- **Reading metadata in C++.** sherpa-onnx reads model type and sample rate from
  ONNX `metadata_props` (`Ort::Session::GetModelMetadata()` →
  `LookupCustomMetadataMapAllocated`), research §8.

## Contract v1

Each model's contract is one JSON string stored in
`metadata_props["fluxvoice.contract"]`:

| Key | Type | Meaning |
|---|---|---|
| `contract_version` | int | `1`; a C++ build accepts exactly the versions it knows |
| `role` | `"encoder"` \| `"vocoder"` | Pipeline position (M3 adds `"translator"`) |
| `sample_rate` | int | Audio rate in/out; encoder and vocoder must match |
| `frame_samples` | int | Samples per model frame (320 = 20 ms @ 16 kHz; 1920 = 80 ms @ 24 kHz) |
| `frames_per_step` | int | Frames consumed/produced per `Run` (1, 2, 4); hop = `frames_per_step × frame_samples` |
| `lookahead_frames` | int | Output at step t describes input frame t − L |
| `content_kind` | `"discrete"` \| `"soft"` | Unit ids (int64) or continuous vectors |
| `vocab` / `content_dim` | int | Discrete vocabulary size, or soft dimension |
| `cond_frame_samples` | int | Frame size of the pitch/energy conditioning (vocoder) |
| `cond_inputs` | list | Vocoder conditioning inputs, from `logf0`, `voicing`, `aperiodicity`, `energy`, `spk_emb` |
| `spk_emb_dim` | int | Speaker-embedding size (vocoder) |
| `overlap_samples` | int | Extra trailing output samples for OLA; **0 by default** (stateful causal vocoder) |
| `states` | list | State base names `X`; graph has `state_in.X` → `state_out.X`, same shape/dtype |
| `license` | string | Licence summary (weights + training data), surfaced by `--list-voices` |

**Tensor names.**
- Encoder:
  - input `audio` `[1, hop]` f32
  - output `units` `[1, frames_per_step]` i64, or `content`
    `[1, frames_per_step, dim]` f32
- Vocoder:
  - input `units`
  - conditioning inputs named as in `cond_inputs`, `[1, cond_frames]` f32;
    `spk_emb` is `[1, D]`
  - output `audio` `[1, hop + overlap]` f32
- States are named `state_in.X` / `state_out.X` and enumerated in the contract.
  C++ resolves every name to an index once, at load time.

**Rules settled in implementation** (both validators, same order, same messages):

- **Strict keys.**
  - A v1 contract has exactly the keys of its role and content kind.
  - An unknown or misplaced key is an error, so a misspelt key is reported as
    misspelt rather than "missing".
  - Integers must be JSON integers, not booleans or `320.0`.
- **Physical bounds.** These keep every derived size in int32 and memory
  bounded:
  - `sample_rate` 8000–192000;
  - `frame_samples` 1–192000;
  - hop ≤ `sample_rate` (a hop is at most 1 s);
  - `lookahead_frames` 0–64;
  - `vocab` 1–2²⁰;
  - `content_dim` 1–2¹⁶;
  - `spk_emb_dim` 0–4096;
  - `frames_per_step` ∈ {1, 2, 4};
  - `overlap_samples` 0–hop.
- **Conditioning frames.** `frame_samples` must be a multiple of
  `cond_frame_samples`, so each content frame covers whole conditioning frames.
  This is the constructible form of "hop not a multiple of frame_samples": hop
  is `frames_per_step × frame_samples` by definition.
- **Speaker embedding.** `spk_emb_dim > 0` exactly when `cond_inputs` lists
  `spk_emb`.
- **Vocoder input name.** It mirrors the encoder output: `units` (i64,
  discrete) or `content` (f32, soft). The pair must agree on sample rate,
  frame size, frames per step, content kind and vocab/content_dim.
- **States.**
  - `states` holds base names, with no `state_in.`/`state_out.` prefix.
  - Each state is a same-dtype, same-shape pair.
  - Dtypes are f32, i64 or f16, which `KVStateCache` can back.
  - Shapes are static and non-empty. A tensor with no shape information is
    rejected; a scalar `[]` is allowed.
- **Graph I/O.** Graph inputs and outputs are exactly the contract's tensors
  plus the listed state pairs.
- **API.**
  - `check_model(contract_json | absent, inputs, outputs)` is what a loader
    calls (Stage 07).
  - A model without the metadata key fails with "model has no
    'fluxvoice.contract' metadata".
- **`voices.json`.**
  - Gains an optional `attribution` string (Stage 06's "licence and
    attribution text").
  - `name` is lowercase `[a-z0-9_-]+`, and the `embedding` length equals the
    vocoder's `spk_emb_dim`.
  - Python-only for now (`validate_voices`); the C++ reader arrives with
    `--list-voices` in Stage 08.

**Algorithmic latency**, in samples, of an encoder + vocoder pair:
`hop + (L_enc + L_voc) × frame_samples + overlap_samples`. Device I/O and compute
come on top (master budget).

**`voices.json`** ships in the bundle next to the models:
`{"schema": 1, "voices": [{"name", "display", "embedding": [D floats],
"logf0_mean", "logf0_std", "energy_mean", "energy_std", "license", "source"}]}`.
The F0 and energy statistics are what the run-time mapping needs (research §4.2).

**Bundle layout.**
- `models/<bundle>/encoder_s{1,2,4}.onnx`
- `models/<bundle>/vocoder_s{1,2,4}.onnx`
- `models/<bundle>/voices.json`
- `models/<bundle>/MODEL_CARD.md`

## Tasks

- [x] Write the spec above into `python_training/fvml/contract.py` (dataclass,
      `to_json`, `validate`, `validate_pair`, `algorithmic_latency`).
      *Added: `validate_io`, `check_model`, `validate_voices`, ONNX helpers
      (`graph_io`, `read_contract_text`, `embed`, `load_onnx`). Standard
      library only, except the lazily imported ONNX helpers.*
- [x] `src/pipeline/ModelContract.hpp`: an ORT-free parser + validator mirroring
      the Python one. Includes `parse(json) → ModelContract`, `validate()`,
      `validate_pair(enc, voc)` and `algorithmic_latency_samples()`.
      *Functions: `parse_contract`, `validate_contract`, `validate_pair`,
      `validate_io`, `check_model`, `algorithmic_latency_samples`. Each returns
      `bool` plus an error string and never throws (nlohmann is used with
      `allow_exceptions=false`). `ModelContract` exposes `hop()`,
      `cond_frames()`, `output_samples()`, `expected_inputs()` and
      `expected_outputs()` for Stage 07's bindings.*
  - [x] Add nlohmann/json **3.12.0** via `FetchContent` (release tarball URL +
        `URL_HASH SHA256`, SYSTEM include). Stage 09's presets reuse it.
        *(target `nlohmann_json_headers`; vendored fallback in
        `third_party/nlohmann_json/include`)*
- [x] **Shared contract test cases** in `tests/data/contracts/{valid,invalid}/*.json`,
      run by both the Python tests and the C++ `test_model_contract`.
      *(11 valid, 73 invalid; format in `tests/data/contracts/README.md`. The
      valid pairs mirror the master latency table.)*
  - [x] Invalid cases: missing key, unknown `contract_version`, encoder/vocoder
        sample-rate mismatch, `hop` not a multiple of `frame_samples`, an
        unpaired `state_in.X`, `overlap_samples > hop`, negative lookahead.
- [x] **`tools/make_fixture_models.py`.** Builds models with `onnx.helper` only (no
      torch), so the bytes are deterministic. Fixed producer name, no
      timestamps.
      *Opset 17 / IR 8: onnx 1.23 writes IR 14 by default, and ORT 1.31 reads
      at most IR 13. `--check` rebuilds the fixtures and compares them with
      disk. `--self-test` checks the committed files against the manifest
      using only the standard library (AGENTS.md rule 5; generating needs
      `onnx`, as this spec requires).*
  - [x] `identity_s{1,2,4}.onnx`: `audio` → `audio`, with a valid contract
        *Realised as the bundle `identity/encoder_s{1,2,4}.onnx` +
        `identity/vocoder_s{1,2,4}.onnx` + `voices.json`:*
        - *A single audio → audio graph matches neither role's tensor names.*
        - *The pair is a bit-exact passthrough: soft content of dim 320. The
          vocoder takes every conditioning input and ignores it.*
        - *It is also the "identity bundle" Stage 08 measures with.*
  - [x] `delay_s1.onnx`: one-hop delay via a state pair (exercises state
        round-trip) *(a vocoder with L = 1, paired with the identity encoder)*
  - [x] `accumulator_s1.onnx`: `state_out = state_in + sum(audio)`, output =
        state (closed form known) *(an encoder; `content = state_out.acc`)*
  - [x] Broken-metadata variants for Stage 07's negative tests *(10 in
        `broken/`, each with its exact error in `manifest.json`)*
  - [x] Output to `tests/data/fixtures/`, committed, plus a `README.md` naming
        the generator command
- [x] `test_model_contract` (ORT-free) added to the CTest loop in
      `CMakeLists.txt`. It parses every case and prints the algorithmic latency
      for each fixture pair. *Also added to CTest: `model_contract_py` and
      `fixture_models_selftest`.*
- [x] **Docs.**
  - [x] Update `models/README.md` (bundle layout replaces the
        `encoder_mimi.onnx` naming).
  - [x] Fix stale stub docstrings. They should point to this contract and to the
        Stage 04/05/06 plans:
    - `python_training/extract_units.py`: 12.5 Hz, `--tokenizer mimi`
      default, `--pitch rmvpe|swiftf0` *(since replaced outright by the Stage
      04 implementation)*
    - `export_stateful_onnx.py`: "80 ms chunk", opset 18
    - `train_unit_vocoder.py`: "single speaker"

## Implementation notes

- **One JSON key, not twenty metadata keys.** Versioning and adding fields stay
  trivial, and the same string is easy to dump with `onnx` in Python or with
  `--list-models` (Stage 08).
- **Fixtures stay tiny and deterministic.** Committed binaries are acceptable
  because they are < 100 KB in total and regenerate byte-identically. CI can
  check that (regenerate, then `git diff --exit-code`) once M1.5 / Stage 01
  lands.
- **No ORT in this stage.** `ModelContract.hpp` must compile in the ORT-less CI
  build. ONNX parsing of the metadata happens in Stage 07's `OrtModel`, which
  hands the JSON string to this parser.
- **Lookahead alignment is part of the contract.** With L > 0, the C++ pitch and
  energy features must be delayed L frames to line up with the content (Stage 03
  / 08).
- **Mimi, if it survives the kill test,** fits the same contract:
  `frame_samples = 1920`, `sample_rate = 24000`, KV states plus a position
  tensor state. `cond_frame_samples` stays 20 ms, giving 4 conditioning frames per
  content frame.

## Verification (2026-10-18)

Windows 11, MSVC 19.50 (VS 18), Ninja, `-DFLUXVOICE_WARNINGS_AS_ERRORS=ON`
(`/W4 /WX`): a fresh configure fetches nlohmann/json 3.12.0 (hash verified),
builds cleanly, and **CTest passes 42/42**. That includes:

- `model_contract` (C++, 11 valid + 73 invalid cases + the fixture manifest);
- `model_contract_py` (the same cases, system Python 3.14, standard library);
- `fixture_models_selftest`;
- `python_training_stage01_02`.

GCC/Clang were not run.

**Hand-computed algorithmic latency** of the fixture pairs, `hop + (L_enc +
L_voc) × frame_samples + overlap_samples` at 16 kHz. `test_model_contract`
prints and checks exactly these:

| Pair | hop | L_enc | L_voc | overlap | Latency |
|---|---|---|---|---|---|
| `identity_s1` | 320 | 0 | 0 | 0 | 320 samples = 20 ms |
| `identity_s2` | 640 | 0 | 0 | 0 | 640 samples = 40 ms |
| `identity_s4` | 1280 | 0 | 0 | 0 | 1280 samples = 80 ms |
| `delay_s1` (identity encoder s1 + `delay_s1`) | 320 | 0 | 1 | 0 | 640 samples = 40 ms |

**Fixtures.**

- Size: 20 files, 30 380 bytes (the limit is 100 KB).
- Byte-identity: `make_fixture_models.py --check` reports "up to date" after
  regeneration.
- In ONNX Runtime 1.31 (`python_training/tests/test_fixtures.py`):
  - the identity bundle is bit-exact at s1/s2/s4 with conditioning bound;
  - `delay_s1` returns the previous hop;
  - the accumulator matches its closed form over 1000 hops (constant 0.25:
    exact float32 sums);
  - every broken fixture loads as valid ONNX and is rejected with its
    manifest message.

**How the tests were checked.** Mutation runs:

- C++: removing the overlap ≤ hop rule, the unknown-key rule, the state
  shape-pair rule, the integer-type rule, or `L_voc` from the latency formula
  each fails `test_model_contract`.
- Python: the overlap rule, the unknown-key rule and the latency formula each
  fail `test_contract.py`.

**Review.** An independent review on 2026-10-18 filed BUG-003 to BUG-009
(unbounded geometry, leaked exception types, list fields, `voices.json`
schema `true`, shapeless states, `.gitignore`, CTest wiring) plus nine
mutation-test gaps; see
[REVIEW-M2-S01-S02](../Bug%20Report/REVIEW-M2-S01-S02_2026-10-18.md). All
are fixed in the code above and pinned by shared cases or Python tests.

**Notes.**

- `onnx.checker` (full check) rejects graph I/O with no shape field, so
  "state without a shape" has no `broken/` fixture. The validators still
  reject it (cases `io_*_no_shape`).
- Stage 07 must map ORT's unknown rank to `TensorSpec::has_shape = false`.
- An integer literal beyond 64 bits is "out of range" in Python but "must be
  an integer" in C++ (nlohmann parses it as a float). This is documented
  rather than unified.
- The contract v2 table (the M3 translator role) is left to Milestone 3 /
  Stage 02.

## Files touched

- `python_training/fvml/contract.py` (new), `src/pipeline/ModelContract.hpp` (new),
  `tests/test_model_contract.cpp` (new), `tests/data/contracts/**` (new),
  `tools/make_fixture_models.py` (new), `tests/data/fixtures/**` (new),
  `CMakeLists.txt` (nlohmann/json FetchContent, test registration),
  `models/README.md`, `python_training/{extract_units,export_stateful_onnx,train_unit_vocoder}.py`
  (docstrings only)
- Added: `python_training/tests/{test_contract,test_fixtures}.py`,
  `tests/data/{contracts,fixtures}/README.md`, `third_party/README.md`,
  `third_party/nlohmann_json/README.md`, `.gitattributes` (`*.onnx binary`, LF
  for fixture JSON)

## Definition of done

- Validator unit tests run in the ORT-less build and pass in both languages. Each
  invalid case is rejected with a specific message: missing key, version
  mismatch, encoder/vocoder sample-rate mismatch, unpaired state name.
- Fixtures total ≤ 100 KB and regenerate byte-identically from
  `tools/make_fixture_models.py`.
- `test_model_contract` prints the algorithmic latency for every fixture pair,
  matching the hand-computed value in this file's Verification section.

Met — see Verification. The CI workflow (`.github/workflows/build.yml`)
configures from scratch and runs `ctest`, so it picks up `model_contract`,
`model_contract_py` and `fixture_models_selftest` without changes. This has
not yet been observed on a CI runner.
