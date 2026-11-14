# Stage 08 — Execution providers (CUDA / DirectML or WinML / CoreML)

> **Milestone 4** · [Master plan](00_master.md) · [← Stage 07](07_mode_b_translation_wiring.md) · [Stage 09 →](09_quantization_fp16_int8.md)

**Status:** ☐ not started
**Depends on:** [M2 / Stage 07](../Milestone%202/07_ort_cpp_runtime.md) (`OrtModel`),
[Stage 05](05_gui_engine_host_mode_a.md) (GUI host), [Stage 06](06_live_latency_telemetry.md) (timing to compare against);
a real exported bundle ([M2 / Stage 06](../Milestone%202/06_stateful_onnx_export.md)) for the measurements
**Owner persona:** `@onnx-ort-runtime`

## Objective

Let the user choose where inference runs: CPU (default), or a GPU / NPU
execution provider. Selection is a Settings entry bound to ORT session options,
with an honest status when a provider is unavailable, and a measured
real-time-factor comparison against the CPU path.

## Context & verified starting point (2026-11-14)

- `src/pipeline/OrtModel.cpp:217-220` builds one fixed `Ort::SessionOptions`
  (sequential, all graph optimizations, one intra-op and one inter-op thread).
  There is no provider selection anywhere in `src/`, `frontend/` or `tools/`.
- `cmake/OnnxRuntime.cmake` imports the CPU SDK only (pinned 1.31.0, five
  archives with SHA-256). It already copies `onnxruntime_providers_shared`
  when the SDK ships one.
- `KVStateCache` and the ping-pong `Ort::IoBinding`s assume CPU memory for
  inputs, outputs and states.
- `third_party/onnxruntime/README.md` only says "prefer a GPU build in M4".
- The M4 master notes DirectML is in sustained engineering and asks for a
  Windows ML evaluation ([M2 research §9](../Research_documents/M2_TEXTLESS_VC_RESEARCH.md)).

## Tasks

- [ ] **Step 0 — verify against the pinned ORT release.** Before writing code,
      read the ORT 1.31 execution-provider documentation and record in this
      file: archive or package names for the CUDA, DirectML / WinML and CoreML
      builds; the C++ call that appends each provider; required runtime
      libraries (CUDA / cuDNN versions); whether each provider supports every
      operator in the exported graphs. Do not rely on remembered API names.
- [ ] **Decision record — Windows GPU path.** Compare DirectML EP with the
      Windows ML route (maintenance status, packaging, API surface, session
      options). Write the decision into the master's open-decisions table.
- [ ] **Runtime configuration** (`src/pipeline/OrtModel.{hpp,cpp}`):
  - [ ] `struct OrtRuntimeConfig { ExecutionProvider provider; int device_id; … }`
        passed to `OrtModel::load`. CPU stays the default and keeps today's
        exact options.
  - [ ] `available_providers()` wraps ORT's provider query for the UI.
  - [ ] A requested provider that fails to initialise is a **load error with
        the provider's message**. It is never replaced by CPU silently; the
        user (or `--ep auto`) chooses the fallback explicitly.
  - [ ] All provider code sits behind `FLUXVOICE_WITH_ORT` plus a per-provider
        macro (`FLUXVOICE_ORT_CUDA`, …) so CPU-only and ORT-less builds stay
        green.
- [ ] **Memory placement decision.** Two candidates, measured before choosing:
      (a) keep all bound buffers on the CPU and let the provider copy per hop;
      (b) keep recurrent state on the device and bind only audio, conditioning
      and units on the CPU. Record per-hop p95 for both on the s1/s2/s4
      variants. Hops are small, so copies may dominate.
- [ ] **Threading rules on GPU providers.** Confirm that warm-up covers
      provider initialisation and first-run compilation (run more than the 20
      CPU warm-up steps if the first runs are slow), and that the worker
      thread never blocks on a driver lock long enough to miss a deadline.
- [ ] **CMake:** `-DFLUXVOICE_ORT_FLAVOR=cpu|cuda|directml|coreml` selects the
      SDK archive (pinned URL + SHA-256 per flavour) and stages the provider
      libraries next to every executable. Default `cpu`.
- [ ] **CLI:** `--ep cpu|cuda|directml|coreml|auto` and `--ep-device N` on
      `fluxvoice` and `fv_vc_file`; `fluxvoice --list-eps`.
- [ ] **GUI (Settings → Inference):** provider dropdown built from
      `available_providers()`, device index, the active provider and its
      version in the status table. Changing it reloads the bundle through the
      Stage 05 lifecycle (stop → load → warm up → restart).
- [ ] **Benchmark tool** `tools/fv_bench.cpp`: per-stage and whole-hop p50 /
      p95 / max and RTF for a bundle, a provider and a frames-per-step
      variant, single worker thread, after warm-up. Prints JSON.
- [ ] **Parity:** `fv_vc_file --ep <gpu>` against the bundle's `golden/` with a
      provider-specific tolerance (GPU kernels are not bit-exact with CPU):
      unit agreement ≥ 99.9 % and audio SNR ≥ 40 dB against the CPU output.
- [ ] **Tests (hardware-free):**
  - [ ] Config parsing and the "no silent fallback" rule over a fake session
        factory (requested provider unavailable → error names the provider).
  - [ ] `--ep` CLI rejection cases in CTest (unknown provider, provider not
        built in).
  - [ ] ORT-less build: `--ep cuda` fails with the built-without-ORT message.

## Implementation notes

- **CPU may win at small hops.** A 20 ms hop of a V2-class vocoder is a small
  workload; transfer and scheduling overhead can exceed the kernel speed-up.
  The exit criterion says the GPU path must beat the RTF budget *with margin*,
  not that it must beat the CPU. Report both.
- **GPU contention.** A game or a browser sharing the GPU adds jitter the CPU
  path does not have. Measure p99 and the miss count under load, not only the
  mean.
- **Licences and size.** GPU SDKs are large and carry vendor runtime
  requirements. The default shipped build stays CPU-only.

## Files touched

- `src/pipeline/OrtModel.{hpp,cpp}`, `src/pipeline/ModelBundle.{hpp,cpp}`,
  `src/pipeline/KVStateCache.hpp` (only if option (b) wins),
  `cmake/OnnxRuntime.cmake`, `CMakeLists.txt`, `src/main.cpp`,
  `tools/fv_vc_file.cpp`, `tools/fv_bench.cpp` (new),
  `frontend/core/Settings.*`, `frontend/core/AppModel.*`,
  `frontend/imgui/ViewSettings.cpp`, `tests/test_ort_model.cpp`,
  `third_party/onnxruntime/README.md`, `.github/workflows/build.yml`

## Definition of done

- A provider can be selected from the CLI and the GUI; the active provider is
  visible in the status table and the `[stats]` header.
- An unavailable provider produces a clear error and never a silent CPU run.
- `fv_bench` table recorded in this file for CPU and at least one GPU provider
  on the s1/s2/s4 variants: p50, p95, max, RTF, misses over ≥ 5 minutes.
- GPU output meets the parity tolerance against the CPU output.
- CPU-only and ORT-less builds pass the four-leg gate unchanged.
