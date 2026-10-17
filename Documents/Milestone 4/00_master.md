# Milestone 4 — UI & Optimization

> Parent: [Development milestones](../../README.md#7-development-milestones) ·
> [Documents index](../README.md)

**Status:** ☐ draft — stage files to be written when M3 closes
**Goal from README:** *Add ImGui dashboard, FP16/INT8 quantization and GPU
execution-provider toggles.*

## Scope sketch

- Dear ImGui app window (`src/ui/AppWindow.cpp` placeholder exists):
  device dropdowns, translation ON/OFF, Wait-k slider (k ∈ [1..10]),
  `.onnx` model loading, live latency/underrun telemetry from
  `AudioEngine`'s atomic counters.
- Quantization: FP16 (`export_stateful_onnx.py --fp16`), then INT8 eval;
  measure quality vs. RTF trade-off.
- Execution providers: CUDA / DirectML / CoreML toggles via ORT session
  options (`third_party/onnxruntime/README.md`).

## Exit criteria

- Full pipeline controllable from the GUI; per-stage + end-to-end latency
  visible live; GPU path beats the real-time factor budget with margin.

## Stage files

_TBD._
