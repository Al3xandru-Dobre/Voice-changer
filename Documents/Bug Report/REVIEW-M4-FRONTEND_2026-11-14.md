# Review — Milestone 4 status and desktop frontend audit (2026-11-14)

> [Bug index](00_index.md) · Milestone: [M4 master](../Milestone%204/00_master.md) · Specification: [`frontend/README.md`](../../frontend/README.md) · Companion review: [M2 stages 07–09](REVIEW-M2-S07-S09_2026-11-14.md)

A brief audit of what Milestone 4 has delivered, against its master plan and
the frontend's own specification, followed by a code audit of `frontend/**`
(about 7 000 lines) and the engine headers it consumes. Defects are filed as
[BUG-018](BUG-018_frontend_getenv_breaks_msvc_build.md) and
[BUG-020](BUG-020_missing_remembered_device_becomes_default.md) to
[BUG-025](BUG-025_studio_commands_not_runnable.md).

**Result: the UI half of Milestone 4 exists and is well layered, but it is
not in a releasable state. It has never built on Windows MSVC, three High
defects are open in the model layer, and the GUI does not run inference. The
optimization half (execution providers, quantization) has not started.** The
stage plan written from this review is in
[`Documents/Milestone 4/`](../Milestone%204/00_master.md) (stages 01–12).

## 0. Method and limits

- **Nothing was built or run locally.** The review machine has no C++
  toolchain: the Visual Studio 18 directory is empty since 2026-11-12, and
  `cmake`, `cl`, `ninja`, `g++` and `clang++` are absent. A configure attempt
  failed at the first command. All four native legs are **NOT RUN** locally.
- **Build evidence is from CI**: GitHub Actions runs 42824977880 (`70b000b`,
  frontend pull request), 42826469000 (`86ad15b`, its merge) and 42829208493
  (`c299e9f`, M2 Stage 08). The failing-step logs of the last run were read.
- **Code was read, not executed.** A read-only audit pass covered
  `frontend/core`, `frontend/imgui`, `frontend/platform`, `frontend/app` and
  `tests/test_frontend_model.cpp`. Every finding that became a bug report was
  then re-read at the cited lines by the reviewer. The rows of §3 and §5 that
  did not become bug reports come from the audit pass and were not all
  re-read; they are marked.
- **The tree was moving.** Another agent edited `src/pipeline/**`,
  `python_training/**` and `tests/data/**` during the review (M2 Stage 09 and
  M3 Stage 02 work, uncommitted). `frontend/**` changed only in
  `OutputChain.hpp`, `CMakeLists.txt` and `README.md`. Line numbers are as
  read on 2026-11-14.
- Executed: `tools/vm_route.py`, `analyze_loopback.py` and `analyze_soak.py`
  `--self-test` (all exit 0, Python 3.14.7).

## 1. Master-plan verification

| Master claim | Verified state | Evidence |
|---|---|---|
| "Desktop frontend landed 2026-11-02" | True for the source. Builds and passes on Linux GCC and Clang. | CI legs green on all three runs |
| Stage 01 ☑ | **Not on Windows.** `windows-msvc` fails in the Build step on all three runs; `fluxvoice_gui` and `test_frontend_model` were never compiled by MSVC. The pull request was merged with the check red. | BUG-018 |
| `frontend/README.md` §10: Windows and macOS "NOT RUN here (CI)" | CI ran. Windows failed three times. macOS passed twice and fails at `c299e9f`. | [BUG-019](BUG-019_macos_qos_call_wrong_arity.md) (engine header) |
| Stage 02 ☑ device selectors | Implemented, with one rule violated: a missing remembered device becomes "Follow system default". | BUG-020 |
| Stage 03 ☑ UI; "GUI EngineHost still pass-through" | Accurate. The worker applies gain, limiter and mute; every other parameter is discarded. | `EngineHost.cpp:113-117` |
| "The UI delivers `VoiceParams` snapshots to the worker" | True, with a drop-newest defect in the GUI's own queue. | BUG-023 |
| `frontend/README.md` §1: "the worker records them" | False. Nothing is recorded; the snapshot is overwritten each drain. | `EngineHost.cpp:110-116` |
| Stage 04 ☐ quantization & execution providers | Accurate: nothing exists. One fixed CPU `Ort::SessionOptions`; no `--fp16`; no INT8 tooling. | `OrtModel.cpp:217-220`; `export_stateful_onnx.py` |
| "Not yet delivered: global hotkeys, taskbar buttons, notifications" | Accurate: no code. | no matches under `frontend/` |
| Scope: `.onnx` model loading | Not delivered and not listed as missing. Bundles are found by a directory scan and never loaded; the GUI does not link ONNX Runtime. | `frontend/CMakeLists.txt`; `VoiceCatalog.cpp:300-306` |
| Scope: per-stage and end-to-end latency, S2UT lag telemetry | Not delivered. `TelemetrySnapshot` has no timing fields. | `Telemetry.hpp` |
| Scope: 24/48 kHz, noise suppression, monitor mix | Not delivered: no code. | — |
| `src/ui/AppWindow.cpp` "placeholder exists" | Still a 13-line stub compiled into `fluxvoice`; `AGENTS.md` §7 still names it as the dashboard. | `CMakeLists.txt:158` |
| Exit criteria | None met. | master §Verification snapshot |

## 2. Rules the frontend obeys (checked, OK)

- No SDL or ImGui include in `frontend/core`.
- No dry/wet slider.
- No frontend code runs on the audio callback.
- A dropdown row is converted to `ExplicitEndpoint{key, generation}` in the
  same frame; the controller never receives an index.
- The feedback guard is previewed in `start_engine` and enforced by the
  controller before and after open.
- Native dialog callbacks only enqueue under a mutex; results are drained on
  the UI thread.
- The parameter queue has one producer (UI) and one consumer (worker); the
  worker is joined before the queues are discarded.
- Slider ranges and defaults come from `kParams` (Wait-*k* is the exception).
- Paths are shell-quoted correctly for both shells in the Studio commands.

## 3. Defects

| Tag | Defect | Severity | Re-read by reviewer |
|---|---|---|---|
| [BUG-018](BUG-018_frontend_getenv_breaks_msvc_build.md) | `std::getenv` under `/WX`: the frontend does not build on Windows MSVC | High | CI log + source |
| [BUG-020](BUG-020_missing_remembered_device_becomes_default.md) | A missing remembered device becomes "Follow system default"; the next Start opens the default device | High | yes |
| [BUG-021](BUG-021_studio_draft_saved_per_keystroke_unvalidated_path.md) | Studio saves a draft per keystroke into `studio/<unvalidated name>/`: prefix directories, overwrite of an existing project, path escape | High | yes |
| [BUG-022](BUG-022_wrong_typed_json_field_terminates_gui.md) | `json::value()` on a wrong-typed field throws; nothing catches; the GUI terminates | High | yes |
| [BUG-023](BUG-023_gui_param_queue_drops_newest_snapshot.md) | The GUI parameter queue drops the newest snapshot when full | Medium | yes |
| [BUG-024](BUG-024_gui_queued_latency_not_rebaselined.md) | Queued-latency readout keeps the residue of every previous stream | Medium | yes |
| [BUG-025](BUG-025_studio_commands_not_runnable.md) | Studio "exact commands" call a script and options that do not exist; the `.ps1` does not parse | Medium | yes |

Findings from the audit pass that are recorded here without a bug file. They
are scheduled in [Stage 03](../Milestone%204/03_voice_studio_translate_catalogue_panels.md)
and [Stage 04](../Milestone%204/04_build_gate_frontend_hardening.md):

| Finding | Severity | Where | Re-read |
|---|---|---|---|
| "Save as…" preset writes `translate`, `wait_k`, `adaptive_policy`; loading it later changes the translation state. A user preset named like a built-in is unreachable. | Medium | `AppModel.cpp:524` → `Presets.cpp:196`; `AppModel.cpp:487-488` | no |
| Startup restore: saved Wait-*k* replaced by the pair default for any pair except en→es; `last_preset` restores only the label; a preset's `wait_k` loses to the pair default. | Medium | `AppModel.cpp:107-119`, `:573-581` | partly (`:105-119` read) |
| WAV parser: header-derived sizes allocated before any bound; non-finite samples pass as clean and reach `std::sort`; analysis runs on the UI/control thread. | Medium | `VoiceStudio.cpp:110`, `:212-275`; `GuiApp.cpp:431-432` | no |
| UTF-8 paths on Windows: SDL's UTF-8 strings go through `fs::path(std::string)` (ANSI code page) and back into JSON and ImGui. A non-ASCII folder or user name fails; `json::dump()` may throw. | Medium, plausible | `GuiApp.cpp:398`; `VoiceStudio.cpp:454`, `:544-546` | no |
| GUI worker never takes `WorkerPriority`; it runs at normal priority, unlike the CLI. | Low | `EngineHost.cpp:94-98` vs `src/main.cpp:661` | yes |
| Wait-*k* range, default and the 80 ms chunk hard-coded in four places. | Low | `ViewTranslate.cpp:70`; `TranslationModel.cpp:148`; `Settings.cpp:145-148` | no |
| Catalogue `page` / `url` passed to `SDL_OpenURL` with no scheme check. | Low | `ViewCatalogue.cpp:95-98`; `GuiApp.cpp:437-439` | no |
| A `settings.json` with one syntax error yields defaults (warning on stderr, invisible in a `WIN32`-subsystem program) and is overwritten on exit. | Low | `Settings.cpp:108-111`; `app/main.cpp:130-131` | no |
| `OutputChain` (GUI) and `VoiceOutputChain` (engine) are two implementations: the GUI one has mute, no gate, no brightness and a hard limiter toggle. | Low (removed by Stage 05) | `OutputChain.hpp`; `VoiceOutputChain.hpp:99-100` | no |

## 4. Specification vs. code

The full table, panel by panel, is kept in the stage files so it is updated
where the work happens:
[Stage 01](../Milestone%204/01_frontend_shell_tray_mini_panel.md) (shell,
keyboard, Mini panel, settings),
[Stage 02](../Milestone%204/02_device_selectors.md) (devices),
[Stage 03](../Milestone%204/03_voice_studio_translate_catalogue_panels.md)
(voices, Studio, Translate, Catalogue, Telemetry).

Summary: of about 45 specification statements checked, roughly half are
implemented as written, a third are partial (for example peak meters without
RMS, a 60 s sparkline where 30 s is specified, no Apply button in Translate),
and eight are wrong or missing in a way a user would notice: preset save,
last-preset restore, Wait-*k* restore, draft persistence, runnable commands,
`status.json` robustness, the adaptive-policy toggle, and the
remembered-device rule.

## 5. Test gaps (`tests/test_frontend_model.cpp`)

1. **Policy restore** (`:788-804`) asserts the selection *became*
   `FollowSystemDefault`. It never starts the engine afterwards, so opening
   the default device passes. The test pins BUG-020 as expected behaviour.
2. **Worker applies parameters:** only `params_pushed` is checked. Removing
   the three apply lines in `EngineHost.cpp:113-115` passes the suite.
3. **Mute wiring:** flags and labels only. Removing `EngineHost.cpp:117`
   passes.
4. **Preset isolation and restart:** `save_preset_as` is checked for file
   existence; nothing inspects the saved keys or restores through a second
   `AppModel`.
5. **Studio outputs:** substring checks only. No name validation in
   `studio_save_draft`, no per-prefix directories, no overwrite case, no
   comparison with the real script options.
6. **Hostile inputs:** no WAV with an oversized `fmt` chunk, a huge channel
   count or NaN samples; no `status.json` / `draft.json` with wrong-typed or
   `null` fields.
7. **Queued identity across a restart:** never checked.
8. **Windows:** the whole suite has never run on MSVC.

## 6. Suggestions

**Do first** ([Stage 04](../Milestone%204/04_build_gate_frontend_hardening.md))

1. Fix BUG-018 and BUG-019; get one CI run with four green legs; make the
   checks required for merging.
2. Fix BUG-020, BUG-021 and BUG-022, each with its regression test.
3. Restore a local Windows toolchain. Today no Windows claim can be checked
   on the development machine.

**Then**

4. BUG-023, BUG-024, BUG-025 and the Medium rows of §3.
5. [Stage 05](../Milestone%204/05_gui_engine_host_mode_a.md): one shared Mode A
   host for the CLI and the GUI. It removes the second parameter queue, the
   second output chain and the second worker loop, which are the source of
   BUG-023, BUG-024 and the priority gap.

**Documentation**

6. `frontend/README.md` §1 ("the worker records them") and §10 (CI legs
   listed as NOT RUN) do not match reality. The master plan was updated in
   this review; the README was left for Stage 04, because another agent has
   uncommitted edits in it.
7. `Documents/README.md` still lists Milestone 2 as "Planned" (see the
   companion review).

## 7. Not verified

- No build, no CTest, no GUI run on any OS by the reviewer.
- Runtime behaviour taken from knowledge of the libraries, not executed:
  nlohmann `value()` / `dump()` exceptions, MSVC narrow `fs::path` encoding,
  PowerShell parsing of a leading quoted string, `SDL_OpenURL` on non-HTTP
  targets, the exact capacity of the moodycamel queue, ImGui `InputText`
  returning true on each edit.
- SDL tray callback threading: `SdlTray::on_entry` pushes to an
  unsynchronised deque (`SdlTray.cpp:82-86`). That is safe only if SDL calls
  it on the main thread; not checked against SDL 3.4.18.
- Not read in full: `Theme.cpp`, `Views.hpp`, `Widgets.hpp`, `Tray.hpp`,
  `SdlTray.hpp`, `TrayModel.hpp`, `VoiceCatalog.hpp`, `TranslationModel.hpp`,
  `Paths.hpp`.
- Whether `fluxvoice_header_checks` covers the new untracked headers
  (`ParamChannel.hpp`, `ParamSmoothing.hpp`, `VoiceOutputChain.hpp`).
- Colour contrast, real tray behaviour and Mini-panel dragging on real
  desktops.

## 8. Verdict

**Milestone 4 — ◐ in progress.** Stages 01–03 are code-complete for the UI
and should not be read as done: Stage 01 is red on Windows, Stage 02 breaks
its central rule in one path, and Stage 03 has three open defects in the
Studio. Stage 04 (build gate and hardening) is the next step, then Stage 05
(inference in the GUI). Stages 08–09 can be coded against fixture models but
cannot be measured until Milestone 2 produces a trained bundle.
