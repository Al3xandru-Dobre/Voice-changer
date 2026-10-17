# Stage 07 — ONNX Runtime C++ runtime & offline file runner

> **Milestone 2** · [Master plan](00_master.md) · [← Stage 06](06_stateful_onnx_export.md) · [Stage 08 →](08_realtime_mode_a_pipeline.md)

**Status:** ☐ not started
**Depends on:** Stage 02 (contract + fixtures). It can start as soon as 02 lands,
in parallel with Stages 03–06. Real-model checks need Stage 06's bundle.
**Research:** §9 ORT deployment · §8 (metadata, fixed shapes)

## Objective

Bring ONNX Runtime into the C++ build properly:
- a pinned SDK;
- an imported target that copies the DLL next to each executable;
- a runtime version check.

Then build an `OrtModel` wrapper that:
- validates the embedded contract;
- runs a stateful model one step at a time, with zero allocations from
  FluxVoice code and no state copies (IoBinding ping-pong);
- never throws into the worker.

Add `fv_vc_file`, an offline file → file runner, so model execution is testable
and comparable to Python without any audio device.

## Context & references

- **Current ORT plumbing** (`CMakeLists.txt:68-76`, `:108-111`): an INTERFACE
  target `onnxruntime_sdk` from `-DONNXRUNTIME_ROOT`, plus the define
  `FLUXVOICE_WITH_ORT=1`. No source reads that define yet. There is no DLL copy
  or version check, and tests and tools don't link ORT.
- **INTERFACE targets can't carry a DLL location**, so `$<TARGET_RUNTIME_DLLS>`
  can't see them. An IMPORTED SHARED target can.
- **Packaging (research §9):**
  - The Windows SDK zip ships `include/` + `lib/` but no CMake config files.
  - Windows 11 has its own, older `onnxruntime.dll` in System32, which has been
    loaded instead of the app's copy.
  - v1.31.0 is the current release.
- **Placeholders this stage fills:**
  - `src/pipeline/KVStateCache.hpp:23-24` (`TODO(M2)`: bind each buffer to an
    `Ort::Value`). Its buffers are fp32-only today.
  - `src/pipeline/InferencePipeline.cpp:8-9` (`TODO(M2)`: `Ort::Session` for
    encoder + vocoder).
- **CI.** [M1.5 / Stage 01](../Milestone%201.5/01_ci_build_matrix.md) plans CI
  without ORT, so the ORT-less build must stay complete and green.

## Tasks

- [ ] **SDK pin.** Update `third_party/onnxruntime/README.md` with the exact
      version (v1.31.x at stage start), download URLs for win-x64 (zip),
      linux-x64 (tgz) and osx-arm64/universal2 (tgz), and their SHA256 sums.
- [ ] **CMake**, in a `cmake/OnnxRuntime.cmake` module:
  - [ ] `add_library(onnxruntime::onnxruntime SHARED IMPORTED)`:
    - Windows: `IMPORTED_IMPLIB` = `lib/onnxruntime.lib`, `IMPORTED_LOCATION` =
      `lib/onnxruntime.dll`
    - Linux: `lib/libonnxruntime.so.<ver>`
    - macOS: `lib/libonnxruntime.<ver>.dylib`
    - include dir as SYSTEM
  - [ ] Optional `FLUXVOICE_FETCH_ORT` (OFF by default): `FetchContent` with
        `URL` + `URL_HASH SHA256` for CI or fresh machines; otherwise
        `ONNXRUNTIME_ROOT` as today.
  - [ ] A post-build copy of `$<TARGET_RUNTIME_DLLS:tgt>` (+
        `onnxruntime_providers_shared.dll` if present) next to `fluxvoice`,
        `fv_vc_file` and the ORT tests. Linux/macOS: `BUILD_RPATH`/`INSTALL_RPATH`
        `$ORIGIN` / `@loader_path`.
  - [ ] `FLUXVOICE_WITH_ORT` defined on every target that links ORT.
- [ ] **`src/pipeline/OrtModel.{hpp,cpp}`** (the only translation units that
      include ORT headers):
  - [ ] Process-wide `OrtRuntime`: one `Ort::Env` (WARNING logging) created with
        **global thread pools**. `ThreadingOptions`: intra-op 1, inter-op 1,
        `SetGlobalSpinControl(0)`, `SetGlobalDenormalAsZero()`.
  - [ ] Session options: `DisablePerSessionThreads()`,
        `SetExecutionMode(ORT_SEQUENTIAL)`,
        `SetGraphOptimizationLevel(ORT_ENABLE_ALL)`.
  - [ ] **Version check:** compare `Ort::GetVersionString()` with the pinned
        version; refuse on mismatch with a message naming the DLL problem; log
        the version at load.
  - [ ] Read the contract from `metadata_props["fluxvoice.contract"]` and hand it
        to `ModelContract` (Stage 02). Cross-check every graph input/output name,
        shape and dtype against the contract.
  - [ ] **State banks.** Two preallocated banks (A, B) per state pair. Create
        `Ort::Value::CreateTensor` views once, then build **two `Ort::IoBinding`s**
        at load (in A → out B, in B → out A). `step()` runs the current binding
        with `Run(RunOptions, binding)` and flips. No rebinding, no copies.
  - [ ] Model inputs/outputs (audio, units, conditioning, `spk_emb`) are bound to
        fixed preallocated buffers that the caller fills or reads.
  - [ ] `warm_up(n)` runs n steps on zeros, then `reset()`. `reset()` zeroes
        both banks and resets parity.
  - [ ] Error policy: `step()` catches `Ort::Exception` and returns a status code
        plus the last error string. It never throws.
- [ ] **Extend `KVStateCache`** with a dtype (f32, i64, f16 storage as raw bytes)
      and keep it ORT-free; `OrtModel` uses it as the backing store for its
      banks. Add `tests/test_kv_state_cache.cpp` (ORT-free) covering
      registration, shape/size, dtype and `reset()`.
- [ ] **ORT-less behaviour.** `fluxvoice --vc ...` exits non-zero with "built
      without ONNX Runtime (configure with -DONNXRUNTIME_ROOT=... or
      -DFLUXVOICE_FETCH_ORT=ON)". Add a CTest `cli_vc_without_ort` that runs only
      in ORT-less builds.
- [ ] **`tests/test_ort_model.cpp`** (ORT builds only), using the Stage 02
      fixtures:
  - [ ] identity s1/s2/s4: output bit-exact to input
  - [ ] delay: one-hop delay across steps (state round-trip)
  - [ ] accumulator: matches its closed form over 1000 steps (exercises
        ping-pong parity)
  - [ ] each broken-metadata fixture is rejected with a specific error
  - [ ] `reset()` makes the model behave like a freshly loaded one
- [ ] **`tools/fv_vc_file.cpp`**, an offline runner:
  - [ ] Interface: `fv_vc_file --model-dir <bundle> --voice <name>
        [--frames-per-step N] in.wav out.wav`.
  - [ ] Reads mono WAV (16-bit PCM or float); resampling is the user's job, and
        a rate mismatch is rejected.
  - [ ] Runs encoder → vocoder step by step through `OrtModel`.
  - [ ] Prints per-step p50/p95/max µs and the algorithmic latency. Optional
        `--dump-units`; `--repeat N` loops the input N times and reports RSS
        before and after (leak check).
  - [ ] Stage 08 switches it to `InferencePipeline` so file and live runs share
        one code path.
- [ ] **Golden check.** `fv_vc_file` on Stage 06's `golden/` input reproduces the
      Python ORT chunked outputs.

## Implementation notes

- **Why `intra_op = 1`.** The run executes on the calling (worker) thread, so the
  worker's MMCSS/QoS priority (Stage 08) covers inference. No pool thread is
  left at normal priority, and none spins. Revisit only if the gate's compute
  budget is missed; measure a 2-thread pool before changing it.
- **Global thread pools** keep encoder + vocoder (+ the M3 translator) from each
  creating their own pools.
- **No ORT in headers seen by the rest of the program.** `OrtModel.hpp` exposes
  plain spans and status codes, and the ORT types stay in `OrtModel.cpp`. This
  keeps warnings-as-errors clean (ORT headers are SYSTEM includes) and lets the
  ORT-less build compile everything else.
- **Allocation scope.** The `operator new` counter technique can't see
  allocations inside `onnxruntime.dll` on Windows, and on Linux it would flag
  ORT's own arena. The zero-allocation DoD therefore covers FluxVoice code, and
  ORT itself is checked by "RSS flat after warm-up".
- **Optional optimised-model cache.** `SetOptimizedModelFilePath` can save load
  time later. Not needed for M2, since load happens before the stream starts.

## Files touched

- `CMakeLists.txt`, `cmake/OnnxRuntime.cmake` (new), `third_party/onnxruntime/README.md`,
  `src/pipeline/OrtModel.hpp` (new), `src/pipeline/OrtModel.cpp` (new),
  `src/pipeline/KVStateCache.hpp`, `src/main.cpp` (ORT-less `--vc` message),
  `tools/fv_vc_file.cpp` (new), `tests/test_kv_state_cache.cpp` (new),
  `tests/test_ort_model.cpp` (new)

## Definition of done

- **ORT-less build:** all existing CTest cases pass unchanged, plus
  `test_kv_state_cache`, `test_model_contract`, `test_pitch_tracker` and
  `cli_vc_without_ort`.
- **ORT build:**
  - `test_ort_model` passes: identity bit-exact, accumulator matches the closed
    form over 1000 steps, every negative contract case rejected.
  - `onnxruntime.dll` (or `.so`/`.dylib`) sits next to every ORT-linked
    executable.
  - The loaded version is logged, and a deliberately mismatched DLL is refused.
- `fv_vc_file` matches Stage 06's Python ORT chunked golden output within max
  |Δ| ≤ 1e-5.
- RSS grows < 1 MB over 10,000 steps after warm-up (`fv_vc_file --repeat`).
