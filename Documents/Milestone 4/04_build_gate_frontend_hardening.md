# Stage 04 — Build gate & frontend hardening

> **Milestone 4** · [Master plan](00_master.md) · [← Stage 03](03_voice_studio_translate_catalogue_panels.md) · [Stage 05 →](05_gui_engine_host_mode_a.md)

**Status:** ◐ code complete and Linux-verified 2026-11-19 (GCC + Clang, with
and without ORT, headless GUI smoke); Windows MSVC / macOS AppleClang and the
new CI legs NOT RUN (no local toolchain, tree not pushed) — see
[evidence](#evidence-2026-11-19)
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

- [x] Restore a local Windows toolchain (MSVC + CMake + Ninja) or record that
      Windows evidence comes from CI only. *Recorded: Windows evidence comes
      from CI only (no local MSVC on the development machine).*
- [x] Fix BUG-018 at the source (no `_CRT_SECURE_NO_WARNINGS`, no C4996
      suppression). Then fix every diagnostic MSVC reports in the GUI sources
      it compiles for the first time. *Source fix in tree (`env_path`,
      `_wdupenv_s`); the "every MSVC diagnostic" half waits for the first
      `windows-msvc` run.*
- [x] Fix BUG-019 (three-argument `pthread_get_qos_class_np`, include
      `<pthread/qos.h>`).
- [x] Add an ORT-enabled CI job per OS (`-DFLUXVOICE_FETCH_ORT=ON`) that
      builds and runs CTest, so `test_ort_model` and the DLL version check
      execute natively. Cache the SDK archive. *`linux-gcc-ort`,
      `windows-msvc-ort`, `macos-clang-ort`; the archive lands in
      `ort-downloads/` (`FLUXVOICE_ORT_DOWNLOAD_DIR`, slashes normalised),
      cached under `ort-sdk-<os>-<hash of cmake/OnnxRuntime.cmake>`, still
      `URL_HASH`-verified. Not yet run on GitHub.*
- [x] Add a headless GUI smoke test to CTest where a software GL context is
      available (Linux: Xvfb + llvmpipe, as in `frontend/README.md` §8):
      `fluxvoice_gui --smoke-frames 40` for each section, 0 `[imgui-error]`
      lines. *`gui_smoke_<section>` ×8 + `gui_smoke_mini`, registered when
      `xvfb-run` exists, each with its own config/studio/models directory in
      the build tree; CI Linux legs install `xvfb libgl1-mesa-dri`.*
- [ ] Make the `build` checks required for merging to `master`. (Repository
      setting; needs the owner.)
- [x] Delete `src/ui/AppWindow.cpp` and its entry in `CMakeLists.txt:158`;
      update `AGENTS.md` §7, which still names it as the dashboard. *Also
      removed `src/ui/**` from the roster and the `@ui-telemetry` files.*

### B. Frontend defects (High)

- [x] [BUG-020](../Bug%20Report/BUG-020_missing_remembered_device_becomes_default.md) —
      unresolved device policy must block Start.
- [x] [BUG-021](../Bug%20Report/BUG-021_studio_draft_saved_per_keystroke_unvalidated_path.md) —
      validate the voice name before any path is built; no per-keystroke save.
- [x] [BUG-022](../Bug%20Report/BUG-022_wrong_typed_json_field_terminates_gui.md) —
      typed JSON readers for all external files; top-level exception guard.
      *2026-11-19: the guard is recoverable — one failing frame stops the
      stream in order (`AppModel::on_unexpected_exception`), shows the error
      and keeps the GUI running; 30 consecutive failing frames shut down in
      order (`GuiApp::run` returns 2). Every JSON number narrowed in
      `frontend/core` goes through a checked conversion
      (`json_to_integral` / `json_to_float` in `JsonRead.hpp`): non-finite,
      fractional-for-integral and out-of-range values become the default.*

### C. Frontend defects (Medium)

- [x] [BUG-023](../Bug%20Report/BUG-023_gui_param_queue_drops_newest_snapshot.md) —
      use `ParamChannel` (pending slot + `flush`). *"Applied = accepted" is
      Stage 05.*
- [x] [BUG-024](../Bug%20Report/BUG-024_gui_queued_latency_not_rebaselined.md) —
      rebaseline the queued figure after queue discards; "n/a" in loopback.
- [x] [BUG-025](../Bug%20Report/BUG-025_studio_commands_not_runnable.md) —
      label step 5 as unavailable until M2 Stage 10; fix the `.ps1` form.
- [x] WAV parser bounds, non-finite samples, analysis off the UI thread
      (Stage 03 residual list).
- [x] UTF-8 paths on Windows: convert SDL's UTF-8 strings with
      `std::filesystem::u8path` (or `std::u8string`), and write UTF-8 into JSON
      and ImGui. Add a test with a non-ASCII directory name. *`path_from_utf8`
      / `path_to_utf8` in `frontend/core/Paths.*` (`char8_t`, no deprecated
      `u8path`) at the SDL dialog, argv, JSON, TSV, command and ImGui
      boundaries; `test_utf8_paths` (`voces_ñ_日本`). On Linux the test pins
      behaviour only; the ANSI-code-page case is verified by MSVC alone.*
- [x] Preset save / restore rules and Wait-*k* persistence (Stage 03 residual
      list).

### D. Tests that would have caught the above

- [x] The regression test listed in each bug report (table below).
- [x] Worker applies parameters: assert an output level change after
      `set_param(Gain)` and after mute (today only flags and generations are
      checked; removing the worker's apply lines passes the suite).
      *`test_param_queue_latest_wins` (gain) and `test_mute_silences_output`
      (mute, < −60 dBFS, recovers on unmute).*
- [x] Hostile-input table for every loader (`settings.json`, `voices.json`,
      `langs.json`, `index.json`, `draft.json`, `status.json`, WAV headers).
      *`test_json_every_field_every_type` (every node → null / string / bool /
      float / object / array, plus 13 hostile numbers), `test_json_hostile_numbers`,
      `test_hostile_wavs`.*

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

## Evidence (2026-11-19)

Uncommitted tree on base `fe2fb68`; local machine Ubuntu 24.04 x86-64, CMake
3.28.3, Ninja 1.11.1. Every run below is `cmake --workflow --preset <p>`
(configure + build all targets incl. `fluxvoice_header_checks` and
`fluxvoice_gui`, strict warnings as errors, `ctest`, `python_training_*`
excluded).

| Leg | Result |
|---|---|
| Linux GCC 13.3.0, ORT-less (`linux-gcc`) | PASS, 121/121 (baseline 112 + 9 `gui_smoke_*`) |
| Linux Clang 18.1.3, ORT-less (`linux-clang`) | PASS, 121/121 |
| Linux GCC 13.3.0, ORT 1.31.0 (`linux-gcc-ort`) | PASS, 122/122 |
| Linux Clang 18.1.3, ORT 1.31.0 (`linux-clang-ort`) | PASS, 122/122 |
| Headless GUI smoke (Xvfb + llvmpipe, 8 sections + Mini) | PASS in all four trees |
| `FLUXVOICE_FETCH_ORT=ON` configure | PASS against a localhost mirror of the pinned archive (cache hit on reconfigure, corrupted archive re-downloaded by `URL_HASH`); the real GitHub URL NOT RUN (release assets blocked from this machine) |
| Windows MSVC, macOS AppleClang (ORT-less and ORT) | NOT RUN — no local toolchain; tree not pushed, so no CI run |
| Required `build` checks on `master` | NOT DONE — repository setting (owner) |

Tests that can fail (each rule broken once, the named test failed, the file
restored byte-identical):

| Rule broken | Failing test |
|---|---|
| Worker no longer applies mute | `frontend_model` (`test_mute_silences_output`) |
| `on_unexpected_exception` no longer stops the engine | `frontend_model` (`test_unexpected_exception_recovery`) |
| Studio `step` / `sample_rate` back to raw casts | `frontend_model` (`test_json_hostile_numbers`) |
| Settings ints back to `get<int>()` | `frontend_model` (`test_json_hostile_numbers`) |
| Telemetry view throws every frame | `gui_smoke_telemetry` fails on `[imgui-error]`; the GUI recovers per frame and exits with 2 after 30 frames |
| Smoke regex pointed at a printed warning | `gui_smoke_telemetry` |

Regression test per bug report:

| Bug | Test |
|---|---|
| BUG-020 | `test_app_model_restores_policy`, `test_device_policy_rules` (`tests/test_frontend_model.cpp`) |
| BUG-021 | `test_studio_drafts`, `test_project_dir` (`tests/test_frontend_studio.cpp`) |
| BUG-022 | `test_json_wrong_types`, `test_json_every_field_every_type`, `test_json_hostile_numbers`, `test_unexpected_exception_recovery` |
| BUG-023 | `test_param_queue_latest_wins` |
| BUG-024 | `test_queued_latency_rebaselined` |
| BUG-025 | `test_job_commands` (`tests/test_frontend_studio.cpp`) |

Open: the native legs above, the repository setting, and `GuiApp::open_path`
building `file://` URLs without percent-encoding (spaces and non-ASCII names
depend on the OS handler; unchanged behaviour, not a regression).
