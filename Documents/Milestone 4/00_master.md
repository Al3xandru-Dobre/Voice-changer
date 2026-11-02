# Milestone 4 — UI & Optimization

> Parent: [Development milestones](../../README.md#7-development-milestones) ·
> [Documents index](../README.md)

**Status:** ◐ in progress — the desktop frontend (UI half) landed 2026-11-02;
quantization and execution-provider toggles (optimization half) remain.
**Goal from README:** *Add ImGui dashboard, FP16/INT8 quantization and GPU
execution-provider toggles.*

## Delivered: desktop frontend (`frontend/`, `fluxvoice_gui`)

Authoritative UX specification and implementation map:
[`frontend/README.md`](../../frontend/README.md). In short:

- **Stack:** SDL3 3.4.18 (window, OpenGL context, events, native file dialogs,
  system tray) + Dear ImGui v1.92.9, both fetched and pinned by CMake
  (`FLUXVOICE_BUILD_GUI`, on by default). No scripting runtime, no web view,
  no network access from the application.
- **Layering:** `frontend/core` is a toolkit-free model layer tested headless
  over a fake `AudioBackend` (`tests/test_frontend_model.cpp`);
  `frontend/platform` holds the tray seam; `frontend/imgui` the views; the UI
  thread is the control thread (it ticks `AudioController`).
- **Sections:** Live, Devices (endpoint key + snapshot generation, never a
  row index; feedback-guard preview; stream status table), Voices (sliders
  generated from `src/voice/VoiceParams.hpp`'s `kParams`, presets from
  `presets/*.json`), Voice Studio (M2 Stage 10 recording protocol as a
  six-step wizard with WAV analysis and job manifest), Translate (`langs.json`,
  same-language guard, Wait-k, Mode A/B), Catalogue (installed `voices.json`
  bundles + `index.json` catalogues with licence badges), Telemetry, Settings.
- **Minimised use:** system tray (icon state, tooltip, menu with engine, mute,
  mode, preset, voice and language-pair submenus), close-to-tray, start
  minimised, and a compact always-on-top Mini panel.
- **Shared descriptor table:** `src/voice/VoiceParams.hpp` + `Presets` and the
  five built-in presets implement the engine-side part of M2/M3 Stage 09 that
  the UI consumes; applying the parameters inside `InferencePipeline` is still
  that stage's work.

## Not yet delivered

- Inference wiring behind the controls (M2 Stages 07–09, M3 Stages 07–09):
  the UI delivers `VoiceParams` snapshots to the worker over the SPSC queue;
  the worker applies gain, mute and the limiter today.
- Quantization (FP16/INT8) and execution-provider toggles — see the scope
  sketch below; they become Settings entries bound to ORT session options.
- Global hotkeys, Windows taskbar thumbnail buttons, desktop notifications.

## Scope sketch

- **Device selectors** (cross-platform research §10, stage 5): two
  direction-specific dropdowns with an explicit "Follow system default"
  entry, a refresh action, and stale/error status. They consume
  [Milestone 1.5](../Milestone%201.5/00_master.md)'s `AudioDeviceRegistry` /
  `AudioController` — the UI submits endpoint key + snapshot generation,
  **never a row index**, and uses the same selection rules as the CLI.
  *Depends on M1.5 stages 03–05.*
- Dear ImGui app window (`src/ui/AppWindow.cpp` placeholder exists):
  translation ON/OFF (`translate`), Wait-k slider (`wait_k ∈ [1..10]`),
  source/target language dropdowns from `langs.json` (`src_lang`, `tgt_lang`
  — see [M3 / Stage 09](../Milestone%203/09_s2ut_controls_language_switching.md)),
  `.onnx` model loading, live latency/underrun/S2UT lag telemetry from
  `AudioEngine`'s atomic counters, and `StreamStatus` (backend, requested vs.
  resolved device, client vs. backend sample rate).
- **Voice sliders & preset picker.** They consume
  [M2 / Stage 09](../Milestone%202/09_voice_controls_presets.md) and
  [M3 / Stage 09](../Milestone%203/09_s2ut_controls_language_switching.md)'s
  `VoiceParams` descriptor table (`fluxvoice --list-params --json`): one
  slider per descriptor (name, unit, range, default), voice A/B dropdowns
  from `voices.json`, preset load/save. No engine changes needed.
- Quantization: FP16 (`export_stateful_onnx.py --fp16` across `encoder`,
  `translator_enc`, `translator_dec`, and `vocoder` — see
  [M3 / Stage 06](../Milestone%203/06_translated_prosody_stateful_onnx_export.md)),
  then INT8 eval; measure quality (WER + ASR-BLEU) vs. RTF trade-off.
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

| # | Stage | Output | Status |
|---|---|---|---|
| 01 | Frontend shell, tray & Mini panel | `frontend/`, `fluxvoice_gui`, `test_frontend_model` | ☑ (see [`frontend/README.md`](../../frontend/README.md)) |
| 02 | Device selectors over M1.5 | Devices panel, `DeviceSelectorModel` | ☑ |
| 03 | Voice controls, presets, Voice Studio, Translate, Catalogue | panels + `src/voice/VoiceParams.hpp`, `presets/` | ☑ UI; engine apply pending (M2/M3 Stage 09) |
| 04 | Quantization & execution providers | Settings entries, ORT session options | ☐ |
