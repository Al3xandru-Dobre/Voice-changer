---
paths:
  - "src/pipeline/**"
  - "python_training/export_stateful_onnx.py"
  - "python_training/fvml/export.py"
  - "cmake/OnnxRuntime.cmake"
  - "tools/fv_vc_file.cpp"
  - "tools/fv_s2ut_file.cpp"
  - "third_party/onnxruntime/README.md"
---

# Stateful ONNX export & ORT C++ runtime rules (`@onnx-ort-runtime`)

Scope: `ModelContract`, stateful PyTorch -> ONNX export, Python↔C++ numerical parity, `OrtModel` / `KVStateCache` / `InferencePipeline` / `WaitKPolicy` / `LatencyGovernor`, Execution Providers and quantization (M4).
Rationale, by section: [`Documents/BEST_PRACTICES.md`](../../Documents/BEST_PRACTICES.md) §4.4 (export traps), §4.5 (ORT configuration), §4.7 (gates measure what ships), §4.8 (Mode B orchestration). As-implemented Mode B notes (banks, chunk step, FIFO, governor, Stage 09 switching): [`MEMORY.md`](../../MEMORY.md) §1, then the `Documents/Milestone 3/` stage files 07–09.

1. **Stateful ONNX Export Discipline:**
   - Expose every recurrent state (causal conv left-context, transformer KV caches, step/position counters) as explicit named graph inputs and outputs (`state_in_*` / `state_out_*`).
   - Never leave cache offsets or sequence positions as Python `int`s during `torch.onnx.export` — they freeze as trace constants and corrupt streaming output after frame 1.
   - Export fixed-shape graphs per hop variant (`s1`, `s2`, `s4`; and for M3, two-graph `translator_enc.onnx` + `translator_dec.onnx` with 128-chunk static ring-KV caches) and embed the `ModelContract` JSON in `ModelProto.metadata_props`.
   - Run chunked ORT vs. offline PyTorch parity tests for **> 2x the longest sliding attention window** (e.g., > 20 s for M2; >= 60 s / 6x window for M3 multi-utterance streams) to catch KV ring-buffer wrap drift, plus Python ORT vs. C++ `fv_vc_file` / `fv_s2ut_file` parity.
   - In Mode B, skip `translator_dec` and `vocoder` on `kRead` steps, and keep the output noise gate open during `kFlush` (driven by `WaitKPolicy` active output + predicted target energy, not live mic energy).
2. **Zero-Allocation Real-Time C++ Inference:**
   - Preallocate all input, output, and recurrent state buffers in `KVStateCache` at load time.
   - Build two `Ort::IoBinding` instances per session (state A->B and state B->A) and ping-pong them on alternating hops for zero per-hop allocation or rebinding.
   - Run warm-up passes on every `Ort::Session` **before** `engine.start()` so first-run kernel/arena initialization never causes a permanent startup underrun.
   - Catch `Ort::Exception` inside the model wrapper; the inference worker thread must never throw.
3. **Threading & Windows DLL Safety:**
   - Configure ORT with `intra_op_num_threads = 1`, `inter_op_num_threads = 1`, sequential execution, spin-wait disabled (`allow_spinning = "0"`), global thread pool with `SetGlobalDenormalAsZero()`, and elevate the worker thread (Windows MMCSS `"Pro Audio"`).
   - Copy the SDK `onnxruntime.dll` next to every executable on Windows, log `Ort::GetVersionString()`, and fail fast on a version mismatch to prevent `C:\Windows\System32\onnxruntime.dll` shadowing.
   - Wrap all ONNX Runtime headers and calls in `#ifdef FLUXVOICE_WITH_ORT` so ORT-less CI builds always succeed.
4. **Mode B orchestration (BEST_PRACTICES §4.8):** derive test expectations from the executable twin, not stage-text examples; protect every hop whose samples touch translated audio from governor drops (final-frame hop, its ramp-down successor, switch hops); a zero-copy cross-session binding must detect producer reload (generation counter) and refuse to step; accessors read by the UI thread are atomic; the control thread pre-validates whatever the worker's `set_params()` may refuse.
5. **Python↔C++ twins move together:** `WaitKPolicy.hpp` + `UnitProsodyFifo.hpp` ↔ `python_training/fvml/waitk_policy.py` (pinned by `tests/data/waitk/golden_trajectories.txt`; rerun `python_training/make_waitk_golden.py`), the Mode B runner ↔ `python_training/s2ut_golden.py`, `ModelContract.hpp` ↔ `fvml/contract.py`.
6. **Verify both build flavours:** a change here must pass the gate with and without ORT (`linux-gcc` and `linux-gcc-ort` presets; same for clang).
