# Milestone 4 — UI & Optimization

> Parent: [Development milestones](../../README.md#7-development-milestones) ·
> [Documents index](../README.md)

**Status:** ◐ in progress — the desktop frontend (UI half) landed 2026-11-02;
the GUI does not run inference yet, and quantization and execution-provider
toggles (optimization half) remain. The build gate is red on `master`
(Windows and macOS legs) as of 2026-11-14; see the
[verification snapshot](#verification-snapshot-2026-11-14).
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

## Verification snapshot (2026-11-14)

What the working tree and CI show, item by item. Method and evidence:
[REVIEW-M4-FRONTEND_2026-11-14](../Bug%20Report/REVIEW-M4-FRONTEND_2026-11-14.md).
No local build was possible on the review machine (no C++ toolchain
installed), so build evidence comes from CI run 42829208493 on `c299e9f`.

| Scope item | State | Evidence | Planned in |
|---|---|---|---|
| Frontend shell, tray seam, Mini panel | ◐ built and tested on Linux GCC + Clang; **does not build on Windows MSVC** ([BUG-018](../Bug%20Report/BUG-018_frontend_getenv_breaks_msvc_build.md)) | CI: `windows-msvc` red on the last 3 runs; `fluxvoice_gui` and `test_frontend_model` never compiled by MSVC | 01, 04 |
| Device selectors over M1.5 | ◐ model + view, headless tests; a missing remembered device becomes "Follow system default" ([BUG-020](../Bug%20Report/BUG-020_missing_remembered_device_becomes_default.md)) | `DeviceSelectorModel`, `ViewDevices.cpp`, `AppModel.cpp:320-341` | 02, 04 |
| Voice sliders from `kParams`, presets, voice A/B | ◐ UI complete; on GUI audio only gain, limiter and mute take effect | `EngineHost.cpp:113-117` | 03, 05 |
| Voice Studio, Catalogue, Translate panels | ◐ UI and model layer; no training launch, no downloads, no translator engine | `frontend/README.md` §1, §11 | 03, 07 |
| `.onnx` model loading from the GUI | ☐ bundles are listed (directory scan) but never loaded; the GUI links no ONNX Runtime | `frontend/CMakeLists.txt`; `VoiceCatalog.cpp:300-306` | 05 |
| GUI runs the Mode A pipeline | ☐ worker is a pass-through over the legacy `InferencePipeline()` shell at 80 ms chunks | `EngineHost.cpp:24`, `:136-142` | 05 |
| Live telemetry: counters, queued audio, RSS, meters | ☑ for the pass-through; the `queued` formula omits `prefilled` and `skipped` | `EngineHost.cpp:63-64` vs `ModeAWorker::snapshot()` | 06 |
| Per-stage + end-to-end latency visible live | ☐ no timing fields in `TelemetrySnapshot`; the engine has one whole-hop histogram | `Telemetry.hpp`; `InferencePipeline.cpp:249` | 06 |
| Translation toggle, Wait-*k*, language dropdowns | ◐ UI only; Milestone 3 is not started | `TranslationModel`, `ViewTranslate.cpp`; M3 master | 07 |
| S2UT lag telemetry | ☐ no Mode B worker to publish it | — | 07 |
| Execution providers (CUDA / DirectML or WinML / CoreML) | ☐ one fixed CPU `Ort::SessionOptions`; CPU SDK only in CMake | `OrtModel.cpp:217-220`; `cmake/OnnxRuntime.cmake` | 08 |
| FP16 export | ☐ no `--fp16` / `--precision` option | `export_stateful_onnx.py`; `fvml/export.py:69-70` (dtype names only) | 09 |
| INT8 evaluation | ☐ nothing | — | 09 |
| 24/48 kHz output or bandwidth extension, noise suppression, monitor mix | ☐ nothing | no matches in `src/`, `frontend/` | 10 |
| Global hotkeys, taskbar thumbnail buttons, notifications | ☐ nothing | no matches in `frontend/` | 11 |
| `src/ui/AppWindow.cpp` | stale placeholder, still compiled into `fluxvoice`; the real UI is `frontend/` | `CMakeLists.txt:158` | 04 |
| Exit: full pipeline controllable from the GUI | ☐ | needs 05 (and 07 for Mode B) | 12 |
| Exit: GPU path beats the RTF budget with margin | ☐ | needs 08 and a trained bundle (M2 Stage 06 real bundle pending) | 12 |

## Placement & dependencies

```text
Delivered:   01 shell/tray ── 02 devices ── 03 panels
                    │
Gate:        04 build gate & frontend hardening
                    │
Engine in GUI:  05 Mode A host ─► 06 latency telemetry ─► 07 Mode B wiring (needs M3 08–09)
                    │                    │
Optimization:       └──────► 08 execution providers ─► 09 FP16 / INT8 (needs a trained bundle)
                    │
Extras:        10 audio extensions      11 desktop integration
                    │                    │
Exit:          12 verification & exit report
```

- **Stage 04 comes first.** Nothing else should merge while two of the four
  native legs are red.
- **Stage 05 needs M2 Stages 08–09** in the tree (`ModeAWorker`,
  `ParamChannel`, `VoiceOutputChain`). Stage 09 is uncommitted working-tree
  work as of 2026-11-14.
- **Stages 08, 09 and 12 need a trained bundle.** M2 Stages 04–06 still list
  the long GPU runs and the real bundle as pending; `models/` holds only a
  README. The code for stages 08–09 can be built against the fixture models;
  the measurements cannot.
- **Stage 07 is blocked on Milestone 3** (not started).
- Stages 10 and 11 are independent of each other and of 08–09.

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

Measured in [Stage 12](12_verification_exit.md):

- [ ] Warnings-as-errors build + CTest green on Windows MSVC, Linux GCC,
      Linux Clang and macOS AppleClang, ORT-less and with ORT
- [ ] `fluxvoice_gui` converts live speech with a loaded bundle; every MVP
      control acts on GUI audio; tray and Mini panel drive the same commands
- [ ] Per-stage p50/p95/max, misses and an end-to-end estimate are visible
      live; the estimate is within ±10 ms of the measured `T − B`
- [ ] GUI `T − B` is at most 5 ms above the CLI's for the same bundle
- [ ] 30-minute GUI soak: 0 misses after warm-up, RSS growth ≤ 5 MB
- [ ] At least one GPU provider: per-hop p95 ≤ 50 % of the hop under load
- [ ] FP16 and INT8 each have a recorded accept/reject decision with numbers
- [ ] Mode B controls work from the GUI (only if Milestone 3 has landed;
      otherwise recorded as deferred with Stage 07 open)

## Key constraints

1. **Real-time rules are unchanged.** Nothing from the UI, the tray, a hotkey
   or a provider callback runs on the audio callback. Inference stays on the
   worker thread.
2. **One composition.** The CLI and the GUI run the same Mode A host. No
   second worker loop, parameter queue or output chain in `frontend/`.
3. **`frontend/core` stays toolkit-free** and testable headless. OS APIs live
   in `frontend/platform` behind per-OS guards.
4. **No silent substitution.** A missing device, bundle, provider or precision
   variant is an error the user sees, never a quiet fallback.
5. **`kParams` is the only source** of parameter names, ranges and defaults.
6. **The default shipped build is CPU, FP32, commercial-clean.** GPU providers
   and quantized variants are opt-in.
7. **Decisions are made on the shipped runtime** (ORT stateful timings, frozen
   manifests), with the acceptance rule written before the measurement.

## Open decisions

| # | Decision | Recommended default | Needed by |
|---|---|---|---|
| 1 | Windows GPU path | Evaluate WinML first; DirectML EP only if WinML cannot express the session options the pipeline needs | Stage 08 |
| 2 | GPU memory placement for states | Measure CPU-bound vs. device-bound state before choosing | Stage 08 |
| 3 | FP16 boundary dtype | Keep graph I/O and states `f32`; FP16 internals only | Stage 09 |
| 4 | INT8 scope | Encoder first; vocoder as a separate experiment that may be rejected | Stage 09 |
| 5 | Higher output rate | Decide between device-boundary resampling, bandwidth extension and a 24 kHz vocoder from a written comparison | Stage 10b |
| 6 | Noise suppressor | Chosen by latency, CPU cost and licence; none is pre-selected | Stage 10a |
| 7 | macOS as a release target | Not a release target; CI leg must still build | Stage 11, 12 |
| 8 | Fate of `src/ui/AppWindow.cpp` | Delete; the UI lives in `frontend/` | Stage 04 |

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Red CI legs hide further errors (MSVC has never compiled the GUI sources) | Stage 04 first; make the `build` checks required on `master` |
| No trained bundle when stages 08–09 are ready | Build and test against fixture models; keep the measurement tasks open rather than estimating |
| Milestone 3 slips | Stage 07 is isolated; the milestone can exit with Mode B recorded as deferred |
| GPU provider is slower than CPU at 20–40 ms hops | Exit criterion is the RTF budget, not "faster than CPU"; report both |
| GPU contention from games adds jitter | Measure p99 and misses under a second GPU load |
| Quantized vocoder degrades quality | Acceptance rule written in advance; "evaluated and rejected" is a valid outcome |
| Bundle loading blocks the UI/control thread | Loader thread while the stream is stopped; result drained on the UI thread |
| Two agents edit the same files (CMake, `EngineHost`, `main.cpp`) | One owner per stage; route shared-file edits through that owner |
| No local Windows toolchain on the development machine (2026-11-14) | Restore one before Stage 04; until then every Windows claim cites a CI run |

## Stage files

| # | Stage | Output | Status |
|---|---|---|---|
| 01 | [Frontend shell, tray & Mini panel](01_frontend_shell_tray_mini_panel.md) | `frontend/`, `fluxvoice_gui`, `test_frontend_model` | ◐ landed; Windows build broken ([BUG-018](../Bug%20Report/BUG-018_frontend_getenv_breaks_msvc_build.md)); see [`frontend/README.md`](../../frontend/README.md) |
| 02 | [Device selectors over M1.5](02_device_selectors.md) | Devices panel, `DeviceSelectorModel` | ◐ model + view landed; [BUG-020](../Bug%20Report/BUG-020_missing_remembered_device_becomes_default.md) open; hardware checks pending |
| 03 | [Voice, Studio, Translate & Catalogue panels](03_voice_studio_translate_catalogue_panels.md) | panels + `src/voice/VoiceParams.hpp`, `presets/` | ◐ UI landed; BUG-021/022/025 open in the Studio; engine apply landed for the CLI/fv_vc_file in M2 Stage 09; GUI EngineHost still pass-through |
| 04 | [Build gate & frontend hardening](04_build_gate_frontend_hardening.md) | four green legs; audit defects fixed | ☐ |
| 05 | [GUI engine host: Mode A inference](05_gui_engine_host_mode_a.md) | `ModeAHost`; `fluxvoice_gui` converts speech | ☐ |
| 06 | [Live latency & per-stage telemetry](06_live_latency_telemetry.md) | stage histograms; latency card; `[stats]` parity | ☐ |
| 07 | [Mode B in the GUI](07_mode_b_translation_wiring.md) | translation controls wired; lag telemetry | ☐ blocked on M3 |
| 08 | [Execution providers](08_execution_providers.md) | `--ep`, Settings → Inference, `fv_bench` | ☐ |
| 09 | [Quantization: FP16 & INT8](09_quantization_fp16_int8.md) | `--precision`, evaluation table, accept/reject | ☐ |
| 10 | [Audio extensions](10_audio_extensions.md) | noise suppression, 24/48 kHz, monitor mix | ☐ |
| 11 | [Desktop integration](11_desktop_integration.md) | global hotkeys, taskbar buttons, notifications | ☐ |
| 12 | [Verification & exit report](12_verification_exit.md) | measured exit criteria | ☐ |

Stage 04 in the earlier table ("Quantization & execution providers") is now
stages 08 and 09.
