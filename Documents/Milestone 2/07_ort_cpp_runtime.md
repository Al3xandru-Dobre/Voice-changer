# Stage 07 — ONNX Runtime C++ runtime & offline file runner

> **Milestone 2** · [Master plan](00_master.md) · [← Stage 06](06_stateful_onnx_export.md) · [Stage 08 →](08_realtime_mode_a_pipeline.md)

**Status:** ☑ done (2026-10-26)
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

- [x] **SDK pin.** Update `third_party/onnxruntime/README.md` with the exact
      version (v1.31.x at stage start), download URLs for win-x64 (zip),
      linux-x64 (tgz) and osx-arm64/universal2 (tgz), and their SHA256 sums.
- [x] **CMake**, in a `cmake/OnnxRuntime.cmake` module:
  - [x] `add_library(onnxruntime::onnxruntime SHARED IMPORTED)`:
    - Windows: `IMPORTED_IMPLIB` = `lib/onnxruntime.lib`, `IMPORTED_LOCATION` =
      `lib/onnxruntime.dll`
    - Linux: `lib/libonnxruntime.so.<ver>`
    - macOS: `lib/libonnxruntime.<ver>.dylib`
    - include dir as SYSTEM
  - [x] Optional `FLUXVOICE_FETCH_ORT` (OFF by default): `FetchContent` with
        `URL` + `URL_HASH SHA256` for CI or fresh machines; otherwise
        `ONNXRUNTIME_ROOT` as today.
  - [x] A post-build copy of `$<TARGET_RUNTIME_DLLS:tgt>` (+
        `onnxruntime_providers_shared.dll` if present) next to `fluxvoice`,
        `fv_vc_file` and the ORT tests. Linux/macOS: `BUILD_RPATH`/`INSTALL_RPATH`
        `$ORIGIN` / `@loader_path`.
  - [x] `FLUXVOICE_WITH_ORT` defined on every target that links ORT.
- [x] **`src/pipeline/OrtModel.{hpp,cpp}`** (the only translation units that
      include ORT headers):
  - [x] Process-wide `OrtRuntime`: one `Ort::Env` (WARNING logging) created with
        **global thread pools**. `ThreadingOptions`: intra-op 1, inter-op 1,
        `SetGlobalSpinControl(0)`, `SetGlobalDenormalAsZero()`.
  - [x] Session options: `DisablePerSessionThreads()`,
        `SetExecutionMode(ORT_SEQUENTIAL)`,
        `SetGraphOptimizationLevel(ORT_ENABLE_ALL)`.
  - [x] **Version check:** compare `Ort::GetVersionString()` with the pinned
        version; refuse on mismatch with a message naming the DLL problem; log
        the version at load.
  - [x] Read the contract from `metadata_props["fluxvoice.contract"]` and hand it
        to `ModelContract` (Stage 02). Cross-check every graph input/output name,
        shape and dtype against the contract.
  - [x] **State banks.** Two preallocated banks (A, B) per state pair. Create
        `Ort::Value::CreateTensor` views once, then build **two `Ort::IoBinding`s**
        at load (in A → out B, in B → out A). `step()` runs the current binding
        with `Run(RunOptions, binding)` and flips. No rebinding, no copies.
  - [x] Model inputs/outputs (audio, units, conditioning, `spk_emb`) are bound to
        fixed preallocated buffers that the caller fills or reads.
  - [x] `warm_up(n)` runs n steps on zeros, then `reset()`. `reset()` zeroes
        both banks and resets parity.
  - [x] Error policy: `step()` catches `Ort::Exception` and returns a status code
        plus the last error string. It never throws.
- [x] **Extend `KVStateCache`** with a dtype (f32, i64, f16 storage as raw bytes)
      and keep it ORT-free; `OrtModel` uses it as the backing store for its
      banks. Add `tests/test_kv_state_cache.cpp` (ORT-free) covering
      registration, shape/size, dtype and `reset()`.
- [x] **ORT-less behaviour.** `fluxvoice --vc ...` exits non-zero with "built
      without ONNX Runtime (configure with -DONNXRUNTIME_ROOT=... or
      -DFLUXVOICE_FETCH_ORT=ON)". Add a CTest `cli_vc_without_ort` that runs only
      in ORT-less builds.
- [x] **`tests/test_ort_model.cpp`** (ORT builds only), using the Stage 02
      fixtures:
  - [x] identity s1/s2/s4: output bit-exact to input
  - [x] delay: one-hop delay across steps (state round-trip)
  - [x] accumulator: matches its closed form over 1000 steps (exercises
        ping-pong parity)
  - [x] each broken-metadata fixture is rejected with a specific error
  - [x] `reset()` makes the model behave like a freshly loaded one
- [x] **`tools/fv_vc_file.cpp`**, an offline runner:
  - [x] Interface: `fv_vc_file --model-dir <bundle> --voice <name>
        [--frames-per-step N] in.wav out.wav`.
  - [x] Reads mono WAV (16-bit PCM or float); resampling is the user's job, and
        a rate mismatch is rejected.
  - [x] Runs encoder → vocoder step by step through `OrtModel`.
  - [x] Prints per-step p50/p95/max µs and the algorithmic latency. Optional
        `--dump-units`; `--repeat N` loops the input N times and reports RSS
        before and after (leak check).
  - [x] Stage 08 switches it to `InferencePipeline` so file and live runs share
        one code path.
- [x] **Golden check.** `fv_vc_file` on Stage 06's `golden/` input reproduces the
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
  `third_party/README.md`, `src/pipeline/OrtModel.hpp` (new),
  `src/pipeline/OrtModel.cpp` (new), `src/pipeline/KVStateCache.hpp`,
  `src/pipeline/ModelContract.hpp`, `src/main.cpp` (ORT-less `--vc` message),
  `tools/fv_vc_file.cpp` (new), `tests/test_kv_state_cache.cpp` (new),
  `tests/test_ort_model.cpp` (new), `tests/check_cli_vc_without_ort.cmake` (new),
  `python_training/tests/test_vc_file_golden.py` (new)

## Verification (2026-10-26)

Executed on Linux `x86_64` (Ubuntu 24.04, kernel `6.8.0-138-generic`, GCC 13.3.0,
Clang 18.1.3, CMake 3.28.3, ONNX Runtime C++ SDK `1.31.0`, Python 3.12.3):

| Matrix leg | Build dir | Compiler | Mode | Configure / Build / CTest | Result |
|---|---|---|---|---|---|
| `linux-gcc` (ORT-less) | `build-gcc` | GCC 13.3.0 (`-Wall -Wextra -Wpedantic -Werror`) | ORT-less | `cmake -S . -B build-gcc -DCMAKE_BUILD_TYPE=Release -DFLUXVOICE_STRICT_WARNINGS=ON -DFLUXVOICE_WARNINGS_AS_ERRORS=ON -DFLUXVOICE_BUILD_TESTS=ON && cmake --build build-gcc -j && ctest --test-dir build-gcc --output-on-failure` | **PASS** (58/58 C++/CLI tests + 4 Python training suites = 62/62) |
| `linux-clang` (ORT-less) | `build-clang` | Clang 18.1.3 (`-Wall -Wextra -Wpedantic -Werror`) | ORT-less | `CC=clang CXX=clang++ cmake -S . -B build-clang -DCMAKE_BUILD_TYPE=Release -DFLUXVOICE_STRICT_WARNINGS=ON -DFLUXVOICE_WARNINGS_AS_ERRORS=ON -DFLUXVOICE_BUILD_TESTS=ON && cmake --build build-clang -j && ctest --test-dir build-clang --output-on-failure` | **PASS** (58/58 C++/CLI tests + 4 Python training suites = 62/62) |
| `linux-gcc` (with ORT 1.31.0) | `build-gcc-ort` | GCC 13.3.0 (`-Wall -Wextra -Wpedantic -Werror`) | `FLUXVOICE_WITH_ORT=1` | `CC=gcc CXX=g++ cmake -S . -B build-gcc-ort ... -DONNXRUNTIME_ROOT=third_party/onnxruntime && cmake --build build-gcc-ort -j && ctest --test-dir build-gcc-ort --output-on-failure` | **PASS** (58/58 C++/CLI tests incl. `ort_model` + 5 Python suites incl. `python_training_stage07_golden` = 63/63) |
| `linux-clang` (with ORT 1.31.0) | `build-clang-ort` | Clang 18.1.3 (`-Wall -Wextra -Wpedantic -Werror`) | `FLUXVOICE_WITH_ORT=1` | `CC=clang CXX=clang++ cmake -S . -B build-clang-ort ... -DONNXRUNTIME_ROOT=third_party/onnxruntime && cmake --build build-clang-ort -j && ctest --test-dir build-clang-ort --output-on-failure` | **PASS** (58/58 C++/CLI tests incl. `ort_model` + 5 Python suites incl. `python_training_stage07_golden` = 63/63) |
| `windows-msvc` | — | MSVC | — | Not available on this Linux host; covered by `.github/workflows/build.yml` CI | **NOT RUN** locally |
| `macos-clang` | — | AppleClang | — | Not available on this Linux host; covered by `.github/workflows/build.yml` CI | **NOT RUN** locally |

**Golden parity & C++ cost (`fv_vc_file` vs. Stage 06 Python ORT `golden/` on 10.0 s clip, `L = 1`, 2 voices):**

| Bundle | Variant | Algorithmic latency | Unit IDs vs. `golden/units_s*.npy` | Waveform max $|\Delta|$ | Waveform SNR | Step `p50 / p95` (C++, 1 thread) |
|---|---|---|---|---|---|---|
| Tiny (`TINY_ENC` + `TINY_VOC`) | `s1` (20 ms) | 640 samples (40.0 ms) | **100% exact** (500/500) | $2.47 \times 10^{-6} \le 10^{-5}$ | 130.7 dB | 0.41 / 0.47 ms |
| Tiny (`TINY_ENC` + `TINY_VOC`) | `s2` (40 ms) | 960 samples (60.0 ms) | **100% exact** (500/500) | $2.70 \times 10^{-6} \le 10^{-5}$ | 130.7 dB | 0.60 / 0.72 ms |
| Tiny (`TINY_ENC` + `TINY_VOC`) | `s4` (80 ms) | 1600 samples (100.0 ms) | **100% exact** (500/500) | $2.70 \times 10^{-6} \le 10^{-5}$ | 130.7 dB | 1.00 / 1.11 ms |
| Full-size (`ContentEncoder` 16.9 MB, 21 states + `UnitVocoder` 4.8 MB, 80 states) | `s1` (20 ms) | 640 samples (40.0 ms) | **100% exact** (500/500) | $7.45 \times 10^{-8} \le 10^{-5}$ | 122.3 dB | 5.86 / **6.91 ms (35%)** |
| Full-size (`ContentEncoder` + `UnitVocoder`) | `s2` (40 ms) | 960 samples (60.0 ms) | **100% exact** (500/500) | $8.38 \times 10^{-8} \le 10^{-5}$ | 122.3 dB | 9.19 / **16.17 ms (40%)** |
| Full-size (`ContentEncoder` + `UnitVocoder`) | `s4` (80 ms) | 1600 samples (100.0 ms) | **100% exact** (500/500) | $6.71 \times 10^{-8} \le 10^{-5}$ | 123.2 dB | 15.48 / **18.81 ms (24%)** |

- **10,000-step leak check (`fv_vc_file --repeat 20` on 10 s input = 10,000 steps after warm-up):**
  RSS `before = 36068 KB`, `after = 36136 KB`, `delta = +68 KB` (`0.066 MB < 1 MB`). In `test_ort_model`, 10,000 steps of `identity_s1 + delay_s1` after warm-up gave `delta = 0 KB`.

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
