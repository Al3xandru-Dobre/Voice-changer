# Stage 02 — Model contract & fixture models

> **Milestone 2** · [Master plan](00_master.md) · [← Stage 01](01_python_toolchain_eval_harness.md) · [Stage 03 →](03_causal_pitch_energy.md)

**Status:** ☐ not started
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

- [ ] Write the spec above into `python_training/fvml/contract.py` (dataclass,
      `to_json`, `validate`, `validate_pair`, `algorithmic_latency`).
- [ ] `src/pipeline/ModelContract.hpp`: an ORT-free parser + validator mirroring
      the Python one. Includes `parse(json) → ModelContract`, `validate()`,
      `validate_pair(enc, voc)` and `algorithmic_latency_samples()`.
  - [ ] Add nlohmann/json **3.12.0** via `FetchContent` (release tarball URL +
        `URL_HASH SHA256`, SYSTEM include). Stage 09's presets reuse it.
- [ ] **Shared contract test cases** in `tests/data/contracts/{valid,invalid}/*.json`,
      run by both the Python tests and the C++ `test_model_contract`.
  - [ ] Invalid cases: missing key, unknown `contract_version`, encoder/vocoder
        sample-rate mismatch, `hop` not a multiple of `frame_samples`, an
        unpaired `state_in.X`, `overlap_samples > hop`, negative lookahead.
- [ ] **`tools/make_fixture_models.py`.** Builds models with `onnx.helper` only (no
      torch), so the bytes are deterministic. Fixed producer name, no
      timestamps.
  - [ ] `identity_s{1,2,4}.onnx`: `audio` → `audio`, with a valid contract
  - [ ] `delay_s1.onnx`: one-hop delay via a state pair (exercises state
        round-trip)
  - [ ] `accumulator_s1.onnx`: `state_out = state_in + sum(audio)`, output =
        state (closed form known)
  - [ ] Broken-metadata variants for Stage 07's negative tests
  - [ ] Output to `tests/data/fixtures/`, committed, plus a `README.md` naming
        the generator command
- [ ] `test_model_contract` (ORT-free) added to the CTest loop in
      `CMakeLists.txt`. It parses every case and prints the algorithmic latency
      for each fixture pair.
- [ ] **Docs.**
  - [ ] Update `models/README.md` (bundle layout replaces the
        `encoder_mimi.onnx` naming).
  - [ ] Fix stale stub docstrings. They should point to this contract and to the
        Stage 04/05/06 plans:
    - `python_training/extract_units.py`: 12.5 Hz, `--tokenizer mimi`
      default, `--pitch rmvpe|swiftf0`
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

## Files touched

- `python_training/fvml/contract.py` (new), `src/pipeline/ModelContract.hpp` (new),
  `tests/test_model_contract.cpp` (new), `tests/data/contracts/**` (new),
  `tools/make_fixture_models.py` (new), `tests/data/fixtures/**` (new),
  `CMakeLists.txt` (nlohmann/json FetchContent, test registration),
  `models/README.md`, `python_training/{extract_units,export_stateful_onnx,train_unit_vocoder}.py`
  (docstrings only)

## Definition of done

- Validator unit tests run in the ORT-less build and pass in both languages. Each
  invalid case is rejected with a specific message: missing key, version
  mismatch, encoder/vocoder sample-rate mismatch, unpaired state name.
- Fixtures total ≤ 100 KB and regenerate byte-identically from
  `tools/make_fixture_models.py`.
- `test_model_contract` prints the algorithmic latency for every fixture pair,
  matching the hand-computed value in this file's Verification section.
