# Milestone 4 — UI & Optimization

> Parent: [Development milestones](../../README.md#7-development-milestones) ·
> [Documents index](../README.md)

**Status:** ◐ in progress — the desktop frontend (UI half) landed 2026-11-02.
Stages 04–06 are code complete in the working tree (2026-11-19): the GUI runs
the Mode A pipeline through the same `ModeAHost` as the CLI and shows
per-stage and end-to-end latency live; Linux-verified only, see the
[2026-11-19 snapshot](#verification-snapshot-2026-11-19). Quantization and
execution-provider toggles (optimization half) remain. The build gate was red
on `master` (Windows and macOS legs) as of 2026-11-14 and has not been re-run
since; see the [2026-11-14 snapshot](#verification-snapshot-2026-11-14) and
the [2026-11-18 snapshot](#verification-snapshot-2026-11-18).
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

- Mode B in the GUI (Stage 07): the GUI loads encoder + vocoder only
  (`TranslatorPolicy::kNever`) and rejects `translate = 1`; Mode A inference
  and every voice control run through `ModeAHost` since Stage 05 (2026-11-19,
  in tree).
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
| Translation toggle, Wait-*k*, language dropdowns | ◐ UI only; the engine side landed in M3 Stage 09 (`set_params()` accepts translate / wait_k / language ids) but `EngineHost` does not route it ([handoff](#handoff-from-milestone-3--stage-09)) | `TranslationModel`, `ViewTranslate.cpp`; `InferencePipeline::set_params` | 07 |
| S2UT lag telemetry | ☐ engine publishes `s2ut_telemetry()` / `stage_timing()` (M3 Stage 09); the GUI does not read them yet | `InferencePipeline.hpp` | 07 |
| Execution providers (CUDA / DirectML or WinML / CoreML) | ☐ one fixed CPU `Ort::SessionOptions`; CPU SDK only in CMake | `OrtModel.cpp:217-220`; `cmake/OnnxRuntime.cmake` | 08 |
| FP16 export | ☐ no `--fp16` / `--precision` option | `export_stateful_onnx.py`; `fvml/export.py:69-70` (dtype names only) | 09 |
| INT8 evaluation | ☐ nothing | — | 09 |
| 24/48 kHz output or bandwidth extension, noise suppression, monitor mix | ☐ nothing | no matches in `src/`, `frontend/` | 10 |
| Global hotkeys, taskbar thumbnail buttons, notifications | ☐ nothing | no matches in `frontend/` | 11 |
| `src/ui/AppWindow.cpp` | stale placeholder, still compiled into `fluxvoice`; the real UI is `frontend/` | `CMakeLists.txt:158` | 04 |
| Exit: full pipeline controllable from the GUI | ☐ | needs 05 (and 07 for Mode B) | 12 |
| Exit: GPU path beats the RTF budget with margin | ☐ | needs 08 and a trained bundle (M2 Stage 06 real bundle pending) | 12 |

## Verification snapshot (2026-11-18)

Stages 01–03 residual work landed in the working tree (base `fe2fb68` plus
uncommitted changes) and was built and tested locally on Linux only. The
2026-11-14 snapshot above is kept as history.

What changed:

- **Stage 01.** BUG-018 fixed in tree (`env_path` in `frontend/core/Paths.cpp`,
  `_wdupenv_s` on Windows; that branch is compiled by no compiler yet). BUG-019
  was already fixed in tree. `Space` is scoped to a focused Live panel;
  `Ctrl+H` / `Ctrl+Q` documented; Mini panel position stored with an explicit
  flag (`"mini_position"`); a malformed `settings.json` becomes
  `settings.json.bad` with a visible warning and atomic saves; the GUI worker
  takes `WorkerPriority`.
- **Stage 02.** `selection_for_row` returns `std::optional`; a malformed stored
  policy is an unresolved choice, never Follow default; Apply never saves a
  placeholder; the status-bar dot turns red for an unresolved device. BUG-020
  unchanged (already fixed in tree).
- **Stage 03.** "Save as…" writes voice parameters only unless asked, built-in
  preset names are refused, the last preset is re-applied at start; Wait-*k*
  comes from `param_desc(ParamId::WaitK)` and survives restarts; `TranslatorCaps`
  (unknown until Stage 05/07 loads a bundle); Translate panel disabled for a
  Mode A bundle; bounded WAV parser, `NonFinite` verdict, off-thread analysis
  drained in `AppModel::tick`; BUG-021 residual (`project_dir` empty for an
  unsafe name); BUG-025 (unavailable steps labelled, scripts fixed); http(s)
  only catalogue links (`UrlPolicy.hpp`); RMS meters.

Evidence (local machine, Ubuntu 24.04 x86-64, CMake 3.28.3, Ninja 1.11.1):

| Leg | Result |
|---|---|
| Linux GCC 13.3.0 (`build-m4-gcc`) | strict + warnings-as-errors configure (GUI on, ORT off); all targets incl. `fluxvoice_header_checks` and `fluxvoice_gui` build; `ctest -E python_training` 105/105 PASS |
| Linux Clang 18.1.3 (`build-clang`) | same, PASS, 105/105 |
| Python CTest suites | 5/10 PASS, NOT attributable to this work: `python_training_m3_stage05` FAILS in `test_gate_s2ut.py::test_gate_runs_the_whole_chain_on_the_ort_clock` (AL bound; M3 gate code that was being edited concurrently in the same tree), and `stage03_04`, `stage05`, `stage09_10`, `m3_stage03` hit the CTest timeout under `-j4` load. No Python file is touched by the Stage 01–03 work; rerun serially on a quiet tree before citing |
| New suites | `frontend_shell`, `frontend_studio`, and `test_device_policy_rules`, `test_preset_and_translation_rules`, `test_studio_analysis_drained_by_tick` in `frontend_model` |
| Mutation spot-check | 7 rules broken one at a time (non-finite verdict, built-in preset name refusal, Wait-*k* pair-default override, malformed device policy, URL space/DEL rejection, PowerShell `$LASTEXITCODE` check, `AppModel::tick` draining Studio analysis); each caught by a failing suite |
| Windows MSVC, macOS AppleClang | NOT RUN (no toolchain on this machine; no CI run of this tree) |
| ORT-enabled builds | NOT RUN |
| Headless GUI smoke (`--smoke-frames`) | NOT RUN (no Xvfb) |
| Manual desktop checks (tray on Windows/KDE/GNOME/Wayland, native dialogs, close-to-tray, start minimised) | NOT RUN |
| Hardware device matrix | NOT RUN |

Still open for later stages: BUG-022's top-level guard, the "applied"
indicator, UTF-8 conversion of SDL dialog strings on Windows, deletion of
`src/ui/AppWindow.cpp`, CI jobs (Stage 04); chunk length and adaptive
capability from the bundle (Stages 05/07).

## Verification snapshot (2026-11-19)

Stages 04–06 landed in the working tree (base `fe2fb68` plus uncommitted
changes; nothing committed or pushed). Built and tested locally on Linux only;
the stage files hold the evidence and mutation tables
([04](04_build_gate_frontend_hardening.md#evidence-2026-11-19),
[05](05_gui_engine_host_mode_a.md#evidence-2026-11-19),
[06](06_live_latency_telemetry.md#evidence-2026-11-19)).

| Scope item | State after 2026-11-19 |
|---|---|
| Build gate | `src/ui/AppWindow.cpp` deleted; ORT CI leg per OS with a cached, hash-verified SDK; headless `gui_smoke_*` CTest cases (Xvfb + llvmpipe) incl. a fixture-bundle load. CI not yet run on this tree |
| Frontend defects (BUG-020…025) | fixed in tree with regression tests; BUG-022 guard recoverable; checked JSON number conversions; UTF-8 path boundary helpers |
| `.onnx` model loading from the GUI; GUI runs the Mode A pipeline | ☑ in tree: `ModeAHost` shared with `fluxvoice --vc/--s2ut`; loader thread; Bypass vs Mode A; "applied" = accepted. Live-mic conversion NOT RUN |
| Per-stage + end-to-end latency visible live | ☑ in tree: `stage_timings()` (enc, trans_enc, trans_dec, voc, feat, chain), `HostStats` estimate shared with the CLI, identity check vs measured occupancy, GUI `[stats]` line = CLI line |
| Live telemetry `queued` formula | ☑ full identity in Mode A (`prefilled`, `skipped`); Bypass rebaselined (BUG-024) |

| Leg | Result |
|---|---|
| Linux GCC 13.3.0 / Clang 18.1.3, ORT-less | PASS, 123/123 each |
| Linux GCC 13.3.0 / Clang 18.1.3, ORT 1.31.0 | PASS, 125/125 each |
| `fv_vc_file` / Mode B goldens (`linux-gcc-ort-training`, golden subset) | PASS, 4/4; the full `python_training_*` suites were not re-run (no Python file changed) |
| Windows MSVC, macOS AppleClang (ORT-less and ORT) | NOT RUN — no toolchain; tree not pushed |
| Live audio: GUI conversion, 10-minute identity run with a governor skip, GUI soak log | NOT RUN — needs audio hardware and a trained bundle |
| Required `build` checks on `master` | NOT DONE — repository setting |

## Handoff from Milestone 3 / Stage 09

What the engine now provides for the GUI
([M3 / Stage 09](../Milestone%203/09_s2ut_controls_language_switching.md)):

- **Bundle.** Load `fluxvoice::S2utBundle` (`src/pipeline/ModelBundle.hpp`)
  when the bundle directory has `translator_enc` / `translator_dec` and
  `langs.json`. It is non-movable: hold it by `std::unique_ptr`.
- **Pipeline.** Build the Mode B `InferencePipeline`
  (`src/pipeline/InferencePipeline.hpp`) with `S2utOptions.translate = false`
  so it starts as a voice changer, then toggle translation live.
- **Parameters.** Push full `VoiceParams` snapshots (`translate`, `wait_k`,
  `adaptive_policy`, `src_lang_id`, `tgt_lang_id` plus the voice fields)
  through `ParamChannel`. `set_params()` returns `false` (params unchanged)
  when it refuses a snapshot: `translate = 1` without translator models, or,
  only while `translate = 1`, a `src_lang_id` / `tgt_lang_id` that is not a
  translator language or `src == tgt`. With `translate = 0` the language ids are
  stored without checks (no effect until translation is switched on). So publish
  **acceptance, not delivery**
  ([BUG-023](../Bug%20Report/BUG-023_gui_param_queue_drops_newest_snapshot.md)).
  Query `has_translator()` and `mode()` to enable or grey out the controls.
- **Telemetry.** `s2ut_telemetry()` (reads / writes / flushes, utterances,
  EOS / cap ends, `lag_ms()`) and per-stage `stage_timing()` feed the lag
  and latency cards.
- **Descriptor dump.** `fluxvoice --list-params --json --model-dir <bundle>`
  includes the `langs` table (`{code, id, name}`) and `default_wait_k`, so the
  dropdowns and the Wait-*k* default come from one source.
- **Switching semantics.** A mode or language change runs a 20 ms crossfade and
  resets only the translator state; `wait_k` changes apply in place.

Work left for Milestone 4: `EngineHost` must route **every** `VoiceParams`
field to the worker (today only Gain / Limiter, `EngineHost.cpp` worker loop),
then wire Mode B (row 07): own the `S2utBundle`, surface `set_params()`
refusals in `TranslationModel::translation_message`, and read the Mode B
telemetry above.

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
- **Stage 07 needs Stage 05** (the GUI Mode A host). The Milestone 3 engine side
  (Stages 07–09) is in the tree; see the handoff below.
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
| 01 | [Frontend shell, tray & Mini panel](01_frontend_shell_tray_mini_panel.md) | `frontend/`, `fluxvoice_gui`, `test_frontend_model` | ◐ landed; residual fixes in tree 2026-11-18, Linux GCC + Clang only; BUG-018 fixed in tree, MSVC and macOS legs NOT RUN; manual desktop checks open; see [`frontend/README.md`](../../frontend/README.md) |
| 02 | [Device selectors over M1.5](02_device_selectors.md) | Devices panel, `DeviceSelectorModel` | ◐ model + view landed; [BUG-020](../Bug%20Report/BUG-020_missing_remembered_device_becomes_default.md) and residuals fixed in tree 2026-11-18 (Linux only); hardware matrix NOT RUN |
| 03 | [Voice, Studio, Translate & Catalogue panels](03_voice_studio_translate_catalogue_panels.md) | panels + `src/voice/VoiceParams.hpp`, `presets/` | ◐ UI landed; residuals, BUG-021 and BUG-025 fixed in tree 2026-11-18 (Linux only); BUG-022 fixed in tree, its top-level guard still shuts the GUI down (Stage 04); chunk length / adaptive capability unknown until a bundle is loaded (Stage 05/07); engine apply landed for the CLI/fv_vc_file in M2 Stage 09; GUI EngineHost still pass-through |
| 04 | [Build gate & frontend hardening](04_build_gate_frontend_hardening.md) | four green legs; audit defects fixed | ◐ code complete 2026-11-19; four Linux legs (GCC/Clang × ORT-less/ORT) + headless GUI smoke PASS; Windows/macOS legs and the new CI jobs NOT RUN (tree not pushed); required checks need the owner |
| 05 | [GUI engine host: Mode A inference](05_gui_engine_host_mode_a.md) | `ModeAHost`; `fluxvoice_gui` converts speech | ◐ code complete 2026-11-19; one host for CLI + GUI, CLI output identical on ~20 non-live paths; four Linux legs PASS incl. a GUI fixture-bundle load; Windows/macOS and live-mic conversion NOT RUN |
| 06 | [Live latency & per-stage telemetry](06_live_latency_telemetry.md) | stage histograms; latency card; `[stats]` parity | ◐ code complete 2026-11-19; six stage histograms, shared `HostStats` line and estimate, identity check, stats log; four Linux legs PASS; Windows/macOS and the live 10-minute run NOT RUN |
| 07 | [Mode B in the GUI](07_mode_b_translation_wiring.md) | translation controls wired; lag telemetry | ☐ engine side delivered by M3 Stage 09; needs 05 |
| 08 | [Execution providers](08_execution_providers.md) | `--ep`, Settings → Inference, `fv_bench` | ☐ |
| 09 | [Quantization: FP16 & INT8](09_quantization_fp16_int8.md) | `--precision`, evaluation table, accept/reject | ☐ |
| 10 | [Audio extensions](10_audio_extensions.md) | noise suppression, 24/48 kHz, monitor mix | ☐ |
| 11 | [Desktop integration](11_desktop_integration.md) | global hotkeys, taskbar buttons, notifications | ☐ |
| 12 | [Verification & exit report](12_verification_exit.md) | measured exit criteria | ☐ |

Stage 04 in the earlier table ("Quantization & execution providers") is now
stages 08 and 09.
