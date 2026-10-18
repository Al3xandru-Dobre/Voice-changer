# Stage 07 — C++ translator runtime & `fv_s2ut_file` offline runner

> **Milestone 3** · [Master plan](00_master.md) · [← Stage 06](06_translated_prosody_stateful_onnx_export.md) · [Stage 08 →](08_realtime_mode_b_pipeline.md)

**Status:** ☐ not started
**Depends on:** Stage 02 (contract v2 + translator fixtures), Stage 04 (`WaitKPolicy` + `UnitProsodyFifo`), [Milestone 2 / Stage 07](../Milestone%202/07_ort_cpp_runtime.md) (`OrtModel`, `KVStateCache`, `fv_vc_file`). Can start as soon as Stages 02 and 04 land, in parallel with Stages 03, 05 and 06; real-model golden checks use Stage 06's bundle.
**Research:** [M3 research](../Research_documents/M3_WAITK_S2UT_RESEARCH.md) §8.1 two-graph split · §9 C++ runtime & `WaitKPolicy` integration

## Objective

Extend `src/pipeline/OrtModel.{hpp,cpp}` to run the two-session S2UT translator
(`translator_enc.onnx` and `translator_dec.onnx`) with **zero-copy cross-attention
memory sharing** (`mem_k_*` / `mem_v_*`), prebuilt `Ort::IoBinding`s, and zero
steady-state heap allocations.

Add `tools/fv_s2ut_file.cpp`, an offline file → file simultaneous translation
runner that drives the full 4-session chain (`encoder → translator_enc →
WaitKPolicy → translator_dec → vocoder`) without an audio device, and verify
golden parity against Stage 06's Python ORT outputs.

## Context & references

- [M2 / Stage 07](../Milestone%202/07_ort_cpp_runtime.md) introduces `OrtModel`
  with two state banks (`A`, `B`) per session and two prebuilt `Ort::IoBinding`s
  (`A → B` and `B → A`) that flip on every `step()`.
- **Why `translator_enc` and `translator_dec` need a shared memory binding:**
  - `translator_enc` advances once per incoming 80 ms source chunk, flipping its
    active bank (`enc_parity_ ∈ {0, 1}`) and writing updated cross-attention keys
    and values into `mem_k_*` / `mem_v_*` in the output bank.
  - `translator_dec` may run 0 times (`kRead`), 1 time (`kWrite`), or multiple
    times (`kFlush` / adaptive catch-up) while reading the most recently written
    `mem_k_*` / `mem_v_*` bank from `translator_enc` and flipping its *own*
    independent decoder state parity (`dec_parity_ ∈ {0, 1}`).
  - Prebuilding **4 `Ort::IoBinding`s on `translator_dec`** indexed by
    `(enc_out_bank ∈ {A, B}, dec_parity ∈ {A→B, B→A})` at load time keeps every
    combinations' tensor views pre-bound — still **zero rebinding and zero copies**
    at run time!

## Tasks

- [ ] **Extend `src/pipeline/OrtModel.{hpp,cpp}` for contract v2 & shared memory:**
  - [ ] Support dual outputs on `"encoder"` (`units` i64 span + optional `latents`
        f32 span when `contract_version >= 2`).
  - [ ] Add `OrtModel::bind_shared_inputs(const OrtModel& producer,
        std::span<const std::string> state_names)` for `"translator_dec"`:
    - binds `translator_dec`'s `mem_k_*`, `mem_v_*` and `src_pos` inputs directly
      to `translator_enc`'s bank `A` and bank `B` `Ort::Value` tensors across a
      $2 \times 2$ (`enc_bank × dec_parity`) `IoBinding` table;
    - `translator_dec.step(enc_last_output_bank)` selects the matching prebuilt
      `IoBinding` in $O(1)$ without touching the heap.
  - [ ] Parse `langs.json` (and validate against `translator_enc.contract().langs`)
        to resolve `--src-lang <code|id>` and `--tgt-lang <code|id>` at load time.
  - [ ] `warm_up(n)` runs `n` dummy steps on both `translator_enc` and
        `translator_dec`, then zeroes all banks (`A` and `B`) and resets both
        parities to 0.
- [ ] **Extend `tests/test_ort_model.cpp` (ORT builds):**
  - [ ] Load the Stage 02 v2 quartet fixtures (`dual_encoder_s4`,
        `fixture_translator_enc`, `fixture_translator_dec`, `identity_s4`).
  - [ ] Verify that `translator_enc` writes to `mem_k_0`/`mem_v_0` across `kRead`
        steps and `translator_dec` reads the exact delayed slot `(src_pos - k) mod
        4` on `kWrite` and `kFlush` steps across all 4 `(enc_bank, dec_parity)`
        combinations.
  - [ ] Verify `reset()` restores both `src_pos` and `tgt_pos` and all ring banks
        to zero, producing bit-identical output on a second run.
  - [ ] Verify negative contract v2 fixtures (`latent_dim` mismatch,
        `shared_mem_states` mismatch, mismatched `langs`) are rejected at load.
- [ ] **`tools/fv_s2ut_file.cpp` (new offline S2UT runner, built when ORT is enabled):**
  - [ ] CLI:
        `fv_s2ut_file --model-dir <bundle> --src-lang <code> --tgt-lang <code> --voice <name> [--wait-k <1..10>] [--adaptive] [--frames-per-step <1|2|4>] [--dump-trace <jsonl>] [--repeat <N>] in.wav out.wav`
  - [ ] Runs the full streaming Mode B state machine on `in.wav` (plus trailing
        silence padding of 600 ms if needed to let `WaitKPolicy` complete the
        final `kFlush`).
  - [ ] When `--dump-trace <path.jsonl>` is passed, writes the per-chunk
        timestamp and unit-emission log consumed by `eval_s2ut.py` for SimulEval
        `AL` / `AL_CA` scoring.
  - [ ] Prints step counts (`reads`, `writes`, `flushes`), per-step
        $p_{50}/p_{95}/\max\;\mu\text{s}$ broken down by stage (`enc`,
        `trans_enc`, `trans_dec`, `voc`), algorithmic onset latency, and RSS
        before/after.
  - [ ] In Stage 08, `fv_s2ut_file` delegates its inner loop to
        `InferencePipeline` so offline and live Mode B share a single code path.
- [ ] **ORT-less CLI guard in `src/main.cpp`:**
  - [ ] `fluxvoice --s2ut ...` in an ORT-less build exits non-zero with the same
        clear diagnostic as `--vc`. Add CTest `cli_s2ut_without_ort`.
- [ ] **Golden parity check:** run `fv_s2ut_file` on Stage 06's
      `models/<bundle>/golden_s2ut/` inputs (`es→en` and `en→de`) and compare
      against Python ORT with `parity.py`.

## Implementation notes

- **Why a $2 \times 2$ `IoBinding` table on `translator_dec` is essential.**
  Because `translator_enc` and `translator_dec` do not step the same number of
  times (during `kRead`, only `translator_enc` flips its parity; during `kFlush`,
  only `translator_dec` flips its parity), their bank parities drift out of phase
  immediately. Indexing `bindings_[enc_out_bank][dec_parity_]` handles arbitrary
  `kRead` / `kWrite` / `kFlush` sequences with zero copies.
- **Trailing flush in file mode.** Real recordings often end abruptly right after
  the last spoken word. `fv_s2ut_file` feeds up to `kTailFlushMs + 100 ms` of
  zero samples at EOF so `WaitKPolicy` enters `kFlush` and drains the remaining
  $k$ chunks into `out.wav` rather than truncating the end of the translation.

## Files touched

- `src/pipeline/OrtModel.hpp`, `src/pipeline/OrtModel.cpp`,
  `tools/fv_s2ut_file.cpp` (new), `tests/test_ort_model.cpp`,
  `src/main.cpp` (ORT-less `--s2ut` guard), `CMakeLists.txt`

## Definition of done

- **ORT-less build:** all existing CTest cases + `test_waitk_policy` +
  `cli_s2ut_without_ort` pass.
- **ORT build:**
  - `test_ort_model` passes all v2 shared-memory `IoBinding` and `reset()` tests
    on the deterministic fixtures.
  - `fv_s2ut_file` reproduces Stage 06's Python ORT `golden_s2ut/` outputs for
    `es→en` and `en→de` with 100% identical `WaitKPolicy` decisions, 100%
    identical emitted unit IDs, and output waveform max $|\Delta| \le 10^{-5}$.
  - `fv_s2ut_file --repeat 50` (over 10,000 steps after warm-up) shows RSS growth
    $< 1\text{ MB}$.
