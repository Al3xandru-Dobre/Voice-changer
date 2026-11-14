# Stage 04 — Build gate & frontend hardening

> **Milestone 4** · [Master plan](00_master.md) · [← Stage 03](03_voice_studio_translate_catalogue_panels.md) · [Stage 05 →](05_gui_engine_host_mode_a.md)

**Status:** ☐ not started — **do this first**
**Depends on:** nothing
**Owner persona:** `@qa-verifier`, with `@ui-telemetry` for the frontend fixes
**Reviews:** [REVIEW-M4-FRONTEND_2026-11-14](../Bug%20Report/REVIEW-M4-FRONTEND_2026-11-14.md),
[REVIEW-M2-S07-S09_2026-11-14](../Bug%20Report/REVIEW-M2-S07-S09_2026-11-14.md)

## Objective

Get the build gate green on all four native legs, and fix the defects the
2026-11-14 audit found in the delivered frontend, before any new Milestone 4
feature lands on top of them.

## Context (2026-11-14)

- `master` (`c299e9f`) fails two of four CI legs: `windows-msvc`
  ([BUG-018](../Bug%20Report/BUG-018_frontend_getenv_breaks_msvc_build.md), since
  the frontend pull request) and `macos-clang`
  ([BUG-019](../Bug%20Report/BUG-019_macos_qos_call_wrong_arity.md), since M2
  Stage 08).
- MSVC has never compiled `fluxvoice_gui` or `test_frontend_model`. More
  `/W4` diagnostics may appear once BUG-018 is fixed.
- CTest has not run on Windows for the last three commits.
- CI builds without ONNX Runtime on every leg, so `test_ort_model`, the DLL
  staging and the version check have no native Windows or macOS evidence.
- The development machine has no C++ toolchain (Visual Studio 18 was removed
  on 2026-11-12). Local verification is not possible until one is restored.
- The working tree holds uncommitted M2 Stage 09 and M3 Stage 02 work. This
  stage's fixes should land after, or be coordinated with, those commits.

## Tasks

### A. Build gate

- [ ] Restore a local Windows toolchain (MSVC + CMake + Ninja) or record that
      Windows evidence comes from CI only.
- [ ] Fix BUG-018 at the source (no `_CRT_SECURE_NO_WARNINGS`, no C4996
      suppression). Then fix every diagnostic MSVC reports in the GUI sources
      it compiles for the first time.
- [ ] Fix BUG-019 (three-argument `pthread_get_qos_class_np`, include
      `<pthread/qos.h>`).
- [ ] Add an ORT-enabled CI job per OS (`-DFLUXVOICE_FETCH_ORT=ON`) that
      builds and runs CTest, so `test_ort_model` and the DLL version check
      execute natively. Cache the SDK archive.
- [ ] Add a headless GUI smoke test to CTest where a software GL context is
      available (Linux: Xvfb + llvmpipe, as in `frontend/README.md` §8):
      `fluxvoice_gui --smoke-frames 40` for each section, 0 `[imgui-error]`
      lines.
- [ ] Make the `build` checks required for merging to `master`. (Repository
      setting; needs the owner.)
- [ ] Delete `src/ui/AppWindow.cpp` and its entry in `CMakeLists.txt:158`;
      update `AGENTS.md` §7, which still names it as the dashboard.

### B. Frontend defects (High)

- [ ] [BUG-020](../Bug%20Report/BUG-020_missing_remembered_device_becomes_default.md) —
      unresolved device policy must block Start.
- [ ] [BUG-021](../Bug%20Report/BUG-021_studio_draft_saved_per_keystroke_unvalidated_path.md) —
      validate the voice name before any path is built; no per-keystroke save.
- [ ] [BUG-022](../Bug%20Report/BUG-022_wrong_typed_json_field_terminates_gui.md) —
      typed JSON readers for all external files; top-level exception guard.

### C. Frontend defects (Medium)

- [ ] [BUG-023](../Bug%20Report/BUG-023_gui_param_queue_drops_newest_snapshot.md) —
      use `ParamChannel` (pending slot + `flush`).
- [ ] [BUG-024](../Bug%20Report/BUG-024_gui_queued_latency_not_rebaselined.md) —
      rebaseline the queued figure after queue discards; "n/a" in loopback.
- [ ] [BUG-025](../Bug%20Report/BUG-025_studio_commands_not_runnable.md) —
      label step 5 as unavailable until M2 Stage 10; fix the `.ps1` form.
- [ ] WAV parser bounds, non-finite samples, analysis off the UI thread
      (Stage 03 residual list).
- [ ] UTF-8 paths on Windows: convert SDL's UTF-8 strings with
      `std::filesystem::u8path` (or `std::u8string`), and write UTF-8 into JSON
      and ImGui. Add a test with a non-ASCII directory name.
- [ ] Preset save / restore rules and Wait-*k* persistence (Stage 03 residual
      list).

### D. Tests that would have caught the above

- [ ] The regression test listed in each bug report.
- [ ] Worker applies parameters: assert an output level change after
      `set_param(Gain)` and after mute (today only flags and generations are
      checked; removing the worker's apply lines passes the suite).
- [ ] Hostile-input table for every loader (`settings.json`, `voices.json`,
      `langs.json`, `index.json`, `draft.json`, `status.json`, WAV headers).

## Implementation notes

- Fix first-party warnings at their source. Do not weaken the shared warning
  gate (`AGENTS.md` rule 7).
- The frontend fixes touch `AppModel.cpp`, `EngineHost.cpp` and
  `VoiceStudio.cpp`, which Stage 05 also rewrites in part. Land Stage 04
  first, or give both stages one owner.
- Several agents edit this repository at the same time. `CMakeLists.txt` and
  `.github/workflows/build.yml` need a single owner for the duration.

## Files touched

- `frontend/core/Paths.cpp`, `src/util/WorkerPriority.hpp`,
  `frontend/core/AppModel.*`, `EngineHost.*`, `VoiceStudio.*`, `Settings.*`,
  `VoiceCatalog.cpp`, `TranslationModel.*`, `frontend/core/JsonRead.hpp` (new),
  `frontend/imgui/GuiApp.cpp`, `ViewStudio.cpp`, `ViewCatalogue.cpp`,
  `tests/test_frontend_model.cpp`, `tests/test_worker_priority.cpp`,
  `CMakeLists.txt`, `frontend/CMakeLists.txt`, `.github/workflows/build.yml`,
  `src/ui/AppWindow.cpp` (deleted), `AGENTS.md`, `frontend/README.md` §10

## Definition of done

- One CI run on `master` with all four legs green, ORT-less and with ORT,
  linked from this file with the commit hash and compiler versions.
- BUG-018 to BUG-025 are ☑ fixed in the [bug index](../Bug%20Report/00_index.md),
  each with its regression test (BUG-025 may close as "labelled unavailable").
- `frontend_model` passes on Windows MSVC for the first time, and the result
  is recorded in `frontend/README.md` §10.
