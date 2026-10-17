# Milestone 4 — UI & Optimization

> Parent: [Development milestones](../../README.md#7-development-milestones) ·
> [Documents index](../README.md)

**Status:** ☐ draft — stage files to be written when M3 closes
**Goal from README:** *Add ImGui dashboard, FP16/INT8 quantization and GPU
execution-provider toggles.*

## Scope sketch

- **Device selectors** (cross-platform research §10, stage 5): two
  direction-specific dropdowns with an explicit "Follow system default"
  entry, a refresh action, and stale/error status. They consume
  [Milestone 1.5](../Milestone%201.5/00_master.md)'s `AudioDeviceRegistry` /
  `AudioController` — the UI submits endpoint key + snapshot generation,
  **never a row index**, and uses the same selection rules as the CLI.
  *Depends on M1.5 stages 03–05.*
- Dear ImGui app window (`src/ui/AppWindow.cpp` placeholder exists):
  translation ON/OFF, Wait-k slider (k ∈ [1..10]),
  `.onnx` model loading, live latency/underrun telemetry from
  `AudioEngine`'s atomic counters, and `StreamStatus` (backend, requested vs.
  resolved device, client vs. backend sample rate).
- Quantization: FP16 (`export_stateful_onnx.py --fp16`), then INT8 eval;
  measure quality vs. RTF trade-off.
- Execution providers: CUDA / DirectML / CoreML toggles via ORT session
  options (`third_party/onnxruntime/README.md`).

## Exit criteria

- Full pipeline controllable from the GUI; per-stage + end-to-end latency
  visible live; GPU path beats the real-time factor budget with margin.

## Stage files

_TBD._
