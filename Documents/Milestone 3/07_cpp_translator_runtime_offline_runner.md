# Stage 07 — C++ translator runtime & `fv_s2ut_file` offline runner

> **Milestone 3** · [Master plan](00_master.md) · [← Stage 06](06_translated_prosody_stateful_onnx_export.md) · [Stage 08 →](08_realtime_mode_b_pipeline.md)

**Status:** ◐ implemented and fixture-verified (2026-11-18); real-bundle checks pending Stage 06 / live rig
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

- [x] **Extend `src/pipeline/OrtModel.{hpp,cpp}` for contract v2 & shared memory:**
  - [x] Support dual outputs on `"encoder"` (`units` i64 span + optional `latents`
        f32 span when `contract_version >= 2`).
  - [x] Add `OrtModel::bind_shared_inputs(const OrtModel& producer,
        std::span<const std::string> state_names)` for `"translator_dec"`:
    - binds `translator_dec`'s `mem_k_*`, `mem_v_*` and `src_pos` inputs directly
      to `translator_enc`'s bank `A` and bank `B` `Ort::Value` tensors across a
      $2 \times 2$ (`enc_bank × dec_parity`) `IoBinding` table;
    - `translator_dec.step(enc_last_output_bank)` selects the matching prebuilt
      `IoBinding` in $O(1)$ without touching the heap.
  - [x] Parse `langs.json` (and validate against `translator_enc.contract().langs`)
        to resolve `--src-lang <code|id>` and `--tgt-lang <code|id>` at load time.
  - [x] `warm_up(n)` runs `n` dummy steps on both `translator_enc` and
        `translator_dec`, then zeroes all banks (`A` and `B`) and resets both
        parities to 0.
- [x] **Extend `tests/test_ort_model.cpp` (ORT builds):**
  - [x] Load the Stage 02 v2 quartet fixtures (`dual_encoder_s4`,
        `fixture_translator_enc`, `fixture_translator_dec`, `identity_s4`).
  - [x] Verify that `translator_enc` writes to `mem_k_0`/`mem_v_0` across `kRead`
        steps and `translator_dec` reads the exact delayed slot `(src_pos - k) mod
        4` on `kWrite` and `kFlush` steps across all 4 `(enc_bank, dec_parity)`
        combinations.
  - [x] Verify `reset()` restores both `src_pos` and `tgt_pos` and all ring banks
        to zero, producing bit-identical output on a second run.
  - [x] Verify negative contract v2 fixtures (`latent_dim` mismatch,
        `shared_mem_states` mismatch, mismatched `langs`) are rejected at load.
- [x] **`tools/fv_s2ut_file.cpp` (new offline S2UT runner, built when ORT is enabled):**
  - [x] CLI:
        `fv_s2ut_file --model-dir <bundle> --src-lang <code> --tgt-lang <code> --voice <name> [--wait-k <1..10>] [--adaptive] [--frames-per-step <1|2|4>] [--dump-trace <jsonl>] [--repeat <N>] in.wav out.wav`
  - [x] Runs the full streaming (zero-pad rule differs: see "As implemented") Mode B state machine on `in.wav` (plus trailing
        silence padding of 600 ms if needed to let `WaitKPolicy` complete the
        final `kFlush`).
  - [x] When `--dump-trace <path.jsonl>` is passed, writes the per-chunk
        timestamp and unit-emission log consumed by `eval_s2ut.py` for SimulEval
        `AL` / `AL_CA` scoring.
  - [x] Prints step counts (`reads`, `writes`, `flushes`), per-step
        $p_{50}/p_{95}/\max\;\mu\text{s}$ broken down by stage (`enc`,
        `trans_enc`, `trans_dec`, `voc`), algorithmic onset latency, and RSS
        before/after.
  - [x] In Stage 08, `fv_s2ut_file` delegates its inner loop to
        `InferencePipeline` so offline and live Mode B share a single code path.
- [x] **ORT-less CLI guard in `src/main.cpp`:**
  - [x] `fluxvoice --s2ut ...` in an ORT-less build exits non-zero with the same
        clear diagnostic as `--vc`. Add CTest `cli_s2ut_without_ort`.
- [ ] **Golden parity check (real bundle — pending Stage 06; the fixture twin below is done):** run `fv_s2ut_file` on Stage 06's
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

## As implemented (2026-11-18)

**Status: ◐ implemented and fixture-verified; real-bundle checks pending Stage 06 / live rig.**

### Files and APIs

- `src/pipeline/OrtModel.{hpp,cpp}`: dual `encoder` outputs (`units` + `latents`); `OrtModel::bind_shared_inputs(producer, names)` builds the 2 x 2 (`enc_bank x dec_parity`) `IoBinding` table on `translator_dec` (zero copy, zero rebinding); `warm_up_translator_pair(enc, dec, n)` warms both sessions, zeroes all banks and resets both parities. A producer reload or move is detected through a load-generation counter; the consumer then refuses to step (`kContractError`, "shared producer was reloaded; call bind_shared_inputs() again").
- `src/pipeline/ModelBundle.{hpp,cpp}`, `src/pipeline/LangCatalog.hpp` (new): `langs.json` parsing and validation against the translator contracts; `--src-lang` / `--tgt-lang` accept a code or an id.
- `tools/fv_s2ut_file.cpp` (new, ORT builds): drives `InferencePipeline` in `PipelineMode::kS2UT` (the Stage 08 delegation is done at once, one code path for offline and live). Extra options beyond the stage text: `--voice-b`, `--preset`, `--preset-dir`, `--param`, `--change <s>:<name>=<value>`, `--raw`, `--no-prosody-ctx`, `--max-tail-ms`, `--max-rss-growth-kb` (exit 3 on growth). Exit codes 0 ok / 1 error / 3 RSS growth.
- `src/main.cpp`: ORT-less `--s2ut` diagnostic (CTest `cli_s2ut_without_ort`).
- Golden twin: `python_training/s2ut_golden.py` (`make|check|--self-test`) + `python_training/fvml/s2ut_golden.py`, `fvml/s2utfile.py`; tests `python_training/tests/test_s2ut_golden.py`.
- CTest: `s2ut_file_help`, `s2ut_file_rejects_*` (driven by `tests/check_s2ut_file_cli.cmake`), `s2ut_golden_selftest`, `python_training_m3_stage07_golden` (venv), plus `test_ort_model` `test_s2ut_bundle_pipeline` and the v2 fixture cases.

### Deviations from the stage text

| Stage text | As implemented | Reason |
|---|---|---|
| Golden parity against Stage 06 `golden_s2ut/` with `parity.py` | Golden parity is a **Python ORT twin of the C++ pipeline** (`python_training/s2ut_golden.py`) run on the deterministic fixture quartet; bit-exact, max abs delta = 0 (9 C++ parity cases). The real-bundle `golden_s2ut/` + `parity.py` check is **still pending** | Stage 06 has no trained bundle yet; the twin pins every orchestration rule now |
| Fixed 600 ms trailing zero padding | `fv_s2ut_file` pads zero hops **until the pipeline is idle**, bounded by `tail_flush_ms + (max_flush_steps + k + 2) x chunk_ms` (override `--max-tail-ms`) | A fixed pad is too short for long utterances and wasteful for short ones; idle is the real completion condition |
| `<EOS>` rule (Stage 08 text: `eos_prob >= 0.5`) | `<EOS>` = the `<EOS>` unit **or** `eos_prob > 0.5` | Twin of `fvml/s2ut_stream.py`; the twin decides |
| Readiness | Enters the policy as `sigmoid(logit)` through one function, `readiness_from_logit()` | Mirrors the Python driver. [BUG-029](../Bug%20Report/BUG-029_ctc_adaptive_probability_never_catches_up.md) is still open: change this function and the Python driver together |
| Chunk speech detection | Any 20 ms input frame with RMS above -55 dBFS on the raw input (twin of `gate_s2ut.py --vad-dbfs`) | Matches the gate; the `voiced_frames` clause was dropped |
| `prosody_ctx` | Mean (`z_f0`, `z_energy`) over voiced (tracker voicing >= 0.5) frames of the current utterance; optional off (`--no-prosody-ctx`, zeros as the Python driver) for parity runs | Parity with the driver |
| Per-write loop | Extra writes in an adaptive double-write step re-check `decide()`, so nothing is decoded past an `<EOS>` heard in a pause | Prevents decoding past `<EOS>` |
| Model scope | Mode B requires a stateful vocoder (`overlap_samples == 0`) and one conditioning frame per content frame (constructor throws `std::invalid_argument`) | OLA and 20 ms conditioning with 80 ms content are Mode A shapes |
| Unit/prosody FIFO | Hop-aligned with `pad_to_multiple` after every push (BUG-028) and back-pressure (a write is deferred, never dropped) in C++ and the golden twin | See Stage 08 |
| `--dump-trace` | One JSON object per chunk step, first pass only | `--repeat` passes reuse the pipeline without re-warm-up |

## Verification (AGENTS rule 7; recorded 2026-11-18)

Commands: `cmake -S . -B build-<leg> -DFLUXVOICE_STRICT_WARNINGS=ON -DFLUXVOICE_BUILD_TESTS=ON -DFLUXVOICE_WARNINGS_AS_ERRORS=ON`, `cmake --build build-<leg> --config Release` (all targets, including `fluxvoice_header_checks`), `ctest --test-dir build-<leg> -C Release --output-on-failure`; venv leg adds `python_training_m3_stage07_golden`.

| Leg | Result |
|---|---|
| Linux GCC 13.3 + ORT 1.31.0 | PASS: warning-free `-Werror` build, hardware-free CTest (non-venv 104/104 at the lead's last gate; final total: see lead's final gate) |
| Linux Clang 18.1.3 + ORT 1.31.0 | PASS (same gate) |
| Linux GCC, no ORT | PASS (103/103 at the lead's last gate; final: see lead's final gate) |
| Linux Clang, no ORT | PASS (same gate) |
| Venv `python_training_m3_stage07_golden` | PASS: 27 tests incl. 9 C++ parity cases, bit-exact |
| Windows / MSVC, macOS / AppleClang | NOT RUN (CI pending) |
| Real-bundle golden parity (`golden_s2ut/`, `parity.py`) | NOT RUN (no trained bundle; Stage 06) |
| Live T - B onset, 10-min live run | NOT RUN (no rig) |
