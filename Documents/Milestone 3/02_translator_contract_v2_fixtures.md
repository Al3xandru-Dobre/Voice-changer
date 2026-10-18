# Stage 02 — Translator contract v2 & fixture models

> **Milestone 3** · [Master plan](00_master.md) · [← Stage 01](01_multilingual_data_simuleval_harness.md) · [Stage 03 →](03_multilingual_encoder_vocoder_upgrade.md)

**Status:** ☐ not started
**Depends on:** [Milestone 2 / Stage 02](../Milestone%202/02_model_contract_fixtures.md) (`ModelContract.hpp`, `fvml/contract.py`, `tools/make_fixture_models.py`)
**Research:** [M3 research](../Research_documents/M3_WAITK_S2UT_RESEARCH.md) §3.2 dual-output encoder · §4.2 chunk geometry · §8 two-graph ring-KV export

## Objective

Extend the Milestone 2 model contract (`contract_version: 2`, backward-compatible
with v1 Mode A bundles) so that Python and C++ agree on every detail of a Mode B
bundle:
- the dual-output Stage 1 encoder (`units` i64 for Mode A + `latents` f32 for
  Mode B);
- the two-graph Stage 2 translator (`role: "translator_enc"` and
  `role: "translator_dec"`);
- ring-buffer KV state shapes, cross-attention memory sharing (`mem_k_*`,
  `mem_v_*`), and tensor position counters (`src_pos`, `tgt_pos`);
- supported language tags (`langs.json`), chunk-to-frame ratio (`chunk_frames = 4`),
  special token IDs (`bos_unit`, `eos_unit`, `pad_unit`), and predicted prosody
  outputs (`prosody` `[1, chunk_frames, 4]` f32).

Generate deterministic ONNX fixture models for `translator_enc` and
`translator_dec` so the C++ runtime track (Stages 04, 07, 08) can be developed
and CI-tested without waiting for GPU training.

## Context & references

- [M2 / Stage 02](../Milestone%202/02_model_contract_fixtures.md) defined
  `metadata_props["fluxvoice.contract"]` (`contract_version: 1`) for `"encoder"`
  and `"vocoder"` roles.
- **Why two translator graphs (research §8.1):**
  - On `Decision::kRead`, C++ runs Stage 1 (`encoder`) + `translator_enc` only;
    `translator_dec` and `vocoder` are not called.
  - On `Decision::kWrite`, C++ runs Stage 1 + `translator_enc` +
    `translator_dec` + Stage 3 (`vocoder`).
  - On `Decision::kFlush`, C++ runs `translator_dec` + Stage 3 (`vocoder`) until
    flush completes.
- **Cross-attention memory sharing:** `translator_enc` writes the newest source
  chunk's keys and values into the ring-buffer tensors `state_out.mem_k_<l>` and
  `state_out.mem_v_<l>` at slot `src_pos mod max_src_steps`. `translator_dec`
  reads those exact buffers as graph inputs `mem_k_<l>` and `mem_v_<l>` (zero
  copy in C++ via shared `KVStateCache` buffers).

## Contract v2 specification

`ModelContract` accepts `contract_version` $\in \{1, 2\}$. Version 2 adds:

| Key | Role(s) | Type | Meaning |
|---|---|---|---|
| `contract_version` | all | int | `2` (v1 bundles remain valid for Mode A) |
| `role` | all | string | `"encoder"` \| `"translator_enc"` \| `"translator_dec"` \| `"vocoder"` |
| `latent_dim` | `encoder`, `translator_enc` | int | Dimension $D$ of Stage 1 continuous `latents` output (`[1, frames_per_step, D]` f32); must match between encoder and `translator_enc` |
| `chunk_frames` | `translator_enc`, `translator_dec` | int | Number of 20 ms frames per translation chunk (`4` = 80 ms @ 50 Hz) |
| `max_src_steps` | `translator_enc`, `translator_dec` | int | Ring-buffer capacity $W_{\text{src}}$ in chunks (`128` = 10.24 s) |
| `max_tgt_steps` | `translator_dec` | int | Decoder self-attention ring capacity $W_{\text{tgt}}$ in chunks (`128`) |
| `shared_mem_states` | `translator_enc`, `translator_dec` | list[str] | Cross-attention ring buffer names `mem_k_0..L`, `mem_v_0..L` updated by `translator_enc` and read by `translator_dec` |
| `langs` | `translator_enc`, `translator_dec` | map[str,int] | Supported ISO 639-1 language codes → integer token IDs, e.g. `{"en": 0, "es": 1, "fr": 2, "de": 3}` |
| `special_units` | `translator_dec` | object | `{"bos": 1000, "eos": 1001, "pad": 1002}`; regular vocoder units are `0 .. vocab - 1` |
| `has_readiness_output` | `translator_enc` | bool | True if graph outputs `readiness_logit` `[1, 1]` f32 for CTC-adaptive Wait-*k* |
| `has_prosody_output` | `translator_dec` | bool | True if graph outputs `prosody` `[1, chunk_frames, 4]` f32 (`z_f0`, `voicing`, `aperiodicity`, `z_energy`) |

**Tensor signatures (v2):**
- **`encoder` (dual-output v2):**
  - Input: `audio` `[1, hop]` f32 + `state_in.*`
  - Outputs: `units` `[1, frames_per_step]` i64 (for Mode A), `latents`
    `[1, frames_per_step, latent_dim]` f32 (for Mode B) + `state_out.*`
- **`translator_enc`:**
  - Inputs: `latents` `[1, chunk_frames, latent_dim]` f32, `src_lang` `[1]` i64,
    `state_in.src_pos` `[1]` i64, `state_in.enc_*`, `state_in.mem_*`
  - Outputs: `readiness_logit` `[1, 1]` f32, `state_out.src_pos` `[1]` i64,
    `state_out.enc_*`, `state_out.mem_*`
- **`translator_dec`:**
  - Inputs: `prev_units` `[1, chunk_frames]` i64, `tgt_lang` `[1]` i64,
    `prosody_ctx` `[1, 2]` f32, `src_pos` `[1]` i64 (from `translator_enc`),
    `state_in.tgt_pos` `[1]` i64, `mem_*` (read-only views of `translator_enc`'s
    output memory bank), `state_in.dec_*`
  - Outputs: `units` `[1, chunk_frames]` i64, `prosody` `[1, chunk_frames, 4]`
    f32, `eos_prob` `[1, 1]` f32, `state_out.tgt_pos` `[1]` i64,
    `state_out.dec_*`

**Mode B algorithmic onset latency** (samples):
`wait_k * chunk_frames * frame_samples + (L_enc + L_voc) * frame_samples + overlap_samples`.

**Bundle layout (v2):**
- `models/<bundle>/encoder_s{1,2,4}.onnx`
- `models/<bundle>/translator_enc.onnx`
- `models/<bundle>/translator_dec.onnx`
- `models/<bundle>/vocoder_s{1,2,4}.onnx`
- `models/<bundle>/voices.json`
- `models/<bundle>/langs.json` (`{"schema": 1, "langs": [{"code": "en", "id": 0, "name": "English"}, ...], "default_wait_k": {"es->en": 3, "fr->en": 3, "de->en": 4, ...}}`)
- `models/<bundle>/MODEL_CARD.md`

## Tasks

- [ ] **Update `python_training/fvml/contract.py` and `src/pipeline/ModelContract.hpp`:**
  - [ ] Support `contract_version: 2` while keeping `contract_version: 1` valid.
  - [ ] Add `validate_s2ut_bundle(enc, trans_enc, trans_dec, voc)`:
    - verifies `enc.latent_dim == trans_enc.latent_dim`;
    - verifies `trans_enc.chunk_frames == trans_dec.chunk_frames`;
    - verifies `trans_enc.shared_mem_states == trans_dec.shared_mem_states` and
      `max_src_steps` match;
    - verifies `trans_dec.vocab == voc.vocab` and `enc.sample_rate ==
      voc.sample_rate`;
    - verifies `langs` maps are non-empty and identical between `translator_enc`
      and `translator_dec`.
  - [ ] Add `s2ut_onset_latency_samples(enc, trans_enc, trans_dec, voc, wait_k)`.
- [ ] **Shared contract test vectors** in
      `tests/data/contracts/{valid,invalid}/*_v2.json`:
  - [ ] Valid v2 encoder + translator_enc + translator_dec + vocoder quartet.
  - [ ] Invalid cases: `latent_dim` mismatch, `shared_mem_states` mismatch,
        `vocab` mismatch between `translator_dec` and `vocoder`, missing
        `src_pos`/`tgt_pos` state, mismatched `langs` table, `chunk_frames` not a
        multiple of `frames_per_step`.
- [ ] **Extend `tools/make_fixture_models.py`** (pure `onnx.helper`, deterministic
      bytes):
  - [ ] Upgrade `identity_s{1,2,4}.onnx` (or add `dual_encoder_s4.onnx`) to emit
        both `units` and `latents` under contract v2.
  - [ ] `fixture_translator_enc.onnx`: writes incoming `latents` summary into a
        4-slot ring memory `mem_k_0` / `mem_v_0` at `src_pos mod 4` and increments
        `src_pos`.
  - [ ] `fixture_translator_dec.onnx`: reads `mem_k_0` / `mem_v_0` at
        `(src_pos - wait_k) mod 4`, emits deterministic `units` (`prev_units +
        tgt_lang`), synthetic `prosody`, increments `tgt_pos`, and raises
        `eos_prob = 1.0` after 3 flush steps.
  - [ ] Broken v2 fixture variants for Stage 07's negative loader tests.
- [ ] **Extend `tests/test_model_contract.cpp`** with v2 bundle validation and
      Mode B onset latency assertions.
- [ ] **Update `models/README.md`** to document the v2 bundle layout
      (`translator_enc.onnx`, `translator_dec.onnx`, `langs.json`).

## Implementation notes

- **Backward compatibility with M2.** A v1 bundle (no `translator_*.onnx`, no
  `latents` output on the encoder) must still load and run in `--vc` mode without
  warnings. Only `--s2ut` requires a v2 bundle.
- **Keep fixtures tiny.** Use `max_src_steps = 4`, `latent_dim = 8`,
  `chunk_frames = 4` in the committed test fixtures so all ONNX fixture files
  combined stay under 150 KB.

## Files touched

- `python_training/fvml/contract.py`, `src/pipeline/ModelContract.hpp`,
  `tests/test_model_contract.cpp`, `tests/data/contracts/**`,
  `tools/make_fixture_models.py`, `tests/data/fixtures/**`, `models/README.md`

## Definition of done

- Both Python (`pytest`) and C++ (`ctest -R model_contract`) pass all v1 and v2
  contract validation tests, rejecting every invalid v2 quartet with a specific
  diagnostic message.
- `tools/make_fixture_models.py` deterministically generates the v2 translator
  fixtures (byte-identical across runs, total fixtures $\le 150\text{ KB}$).
- `s2ut_onset_latency_samples` matches hand-computed values for $k \in \{1, 3,
  5, 10\}$ in `test_model_contract`.
