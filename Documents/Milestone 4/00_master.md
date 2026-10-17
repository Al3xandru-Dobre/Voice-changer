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
- **Voice sliders & preset picker.** They consume
  [M2 / Stage 09](../Milestone%202/09_voice_controls_presets.md)'s
  `VoiceParams` descriptor table (`fluxvoice --list-params --json`): one
  slider per descriptor (name, unit, range, default), voice A/B dropdowns
  from `voices.json`, preset load/save. No engine changes needed.
- Quantization: FP16 (`export_stateful_onnx.py --fp16`), then INT8 eval;
  measure quality vs. RTF trade-off.
- Execution providers: CUDA / DirectML / CoreML toggles via ORT session
  options (`third_party/onnxruntime/README.md`). DirectML is in sustained
  engineering; evaluate Windows ML (WinML) as the preferred Windows path
  ([M2 research §9](../Research_documents/M2_TEXTLESS_VC_RESEARCH.md)).
- Moved here from M2: 24/48 kHz output or bandwidth extension, input noise
  suppression, a delay-aligned monitor mix (replacing dry/wet, which M2
  excluded).

## Exit criteria

- Full pipeline controllable from the GUI; per-stage + end-to-end latency
  visible live; GPU path beats the real-time factor budget with margin.

## Stage files

_TBD._
