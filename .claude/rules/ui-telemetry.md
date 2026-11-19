---
paths:
  - "frontend/**"
  - "src/voice/**"
  - "presets/**"
  - "tests/test_frontend_*.cpp"
---

# ImGui UI, voice controls & telemetry rules (`@ui-telemetry`)

Scope: the `VoiceParams` descriptor table (M2 Stage 09), preset JSON loader/saver, CLI/console voice controls, and the M4 desktop frontend in `frontend/` (core model layer, SDL3 tray seam, ImGui views, `fluxvoice_gui`). The dashboard lives only in `frontend/` (the `src/ui/AppWindow.cpp` stub was deleted in M4 Stage 04).
Rationale, by section: [`Documents/BEST_PRACTICES.md`](../../Documents/BEST_PRACTICES.md) §2.7 (unresolved device selection), §4.3 (no dry/wet mix), §4.8 (UI-thread accessors, pre-validation), §5.7 (user-typed names as paths). Layout and seams: [`frontend/README.md`](../../frontend/README.md). Specs: `Documents/Milestone 2/09_voice_controls_presets.md`, `Documents/Milestone 4/`.

1. **Single-Source `VoiceParams` Descriptor Table (M2 Stage 09 -> M4):** Drive CLI flags, interactive console commands, preset JSON serialization, and ImGui sliders from one shared `VoiceParams` descriptor table (`kParams` in `src/voice/VoiceParams.hpp`: name, unit, min, max, default, smoothing). Never duplicate parameter ranges or defaults in UI code.
2. **Conditioning-Domain Controls:** Controls act on vocoder conditioning (`logf0_tgt`, intonation scale, timbre blend `e_spk`, voicing) plus output gain/gate/limiter with per-frame/per-sample smoothing (no zipper noise). Never add a dry/wet audio mix slider (comb-filtering hazard).
3. **Safe Device Selectors (M1.5 -> M4):** Populate input/output dropdowns from `AudioDeviceRegistry` immutable snapshots. Submit selections as `(endpoint_key, snapshot_generation)` to `AudioController` — **never submit a dropdown row index**. Include an explicit "Follow system default" option and surface `StreamStatus` (backend, requested vs. resolved device, client vs. backend sample rate).
4. **Lock-Free Telemetry:** Read `frames_processed`, `input_overruns`, `output_underruns`, `output_drops`, `queued`, per-stage inference latency histograms, and RSS via lock-free atomics/snapshots on the UI thread without blocking the audio callback or inference worker.
5. **Frontend Layering (M4):** Keep `frontend/core` free of SDL/ImGui so it stays testable headless over `tests/FakeAudioBackend.hpp`; views only read `AppModel` and call its commands; the tray and Mini panel use the same commands. The UI thread is the control thread (`AppModel::tick` -> `AudioController::tick`); parameters reach the worker only through `EngineHost::push_params` (SPSC). Native dialogs are asynchronous and their callbacks may run off-thread: queue results and drain them on the UI thread. SDL3 and Dear ImGui stay pinned in `CMakeLists.txt` (`FLUXVOICE_BUILD_GUI`), compile as SYSTEM code in `fluxvoice_imgui`, and never receive the first-party warning flags.
6. **User-typed names that become paths (BUG-021):** the function that writes the path validates the name (`is_safe_project_name`); persist on commit, not per keystroke; never overwrite a project the user has not opened.
