# Stage 01 — Frontend shell, tray & Mini panel

> **Milestone 4** · [Master plan](00_master.md) · [Stage 02 →](02_device_selectors.md)

**Status:** ◐ landed 2026-11-02 (`70b000b`); residual code tasks fixed in the working tree
2026-11-18 and verified on Linux GCC and Clang only. The Windows MSVC and macOS legs are
NOT RUN
([BUG-018](../Bug%20Report/BUG-018_frontend_getenv_breaks_msvc_build.md) fixed in tree, MSVC leg pending;
[BUG-019](../Bug%20Report/BUG-019_macos_qos_call_wrong_arity.md) fixed in tree, macOS leg pending);
manual desktop checks are open (Stage 12)
**Depends on:** [Milestone 1.5](../Milestone%201.5/00_master.md) stages 02–05 (`AudioTypes`, registry, controller)
**Specification:** [`frontend/README.md`](../../frontend/README.md) §4, §6–§8
**Review:** [REVIEW-M4-FRONTEND_2026-11-14](../Bug%20Report/REVIEW-M4-FRONTEND_2026-11-14.md)

## Objective

A desktop application (`fluxvoice_gui`) built from SDL3 and Dear ImGui that
composes the Milestone 1.5 engine, with a toolkit-free model layer, a system
tray, and a compact Mini panel. This file is the as-built record plus what is
still open.

## As built (read on 2026-11-14, updated 2026-11-18 for the working tree)

| Item | State | Where |
|---|---|---|
| SDL3 3.4.18 + Dear ImGui v1.92.9 pinned via CMake, compiled as SYSTEM code | ☑ | root `CMakeLists.txt`, `frontend/CMakeLists.txt` |
| `frontend/core` free of SDL / ImGui | ☑ no includes found | `frontend/core/**` |
| UI thread is the control thread; no frontend code on the audio callback | ☑ | `EngineHost.hpp` thread contract; `AppModel::tick` |
| Navigation rail, eight sections, status bar | ☑ | `frontend/imgui/GuiApp.cpp`, `View*.cpp` |
| Tray model (menu spec, icon state, tooltip, action queue) + SDL tray | ☑ | `frontend/core/TrayModel.*`, `frontend/platform/SdlTray.*` |
| Close-to-tray, start minimised, Mini panel | ☑ | `GuiApp.cpp`, `ViewMiniPanel.cpp` |
| Native dialogs: callbacks only enqueue, results drained on the UI thread | ☑ | `GuiApp.cpp` (native-dialog callbacks and their drain) |
| Keyboard: `Ctrl+1…8`, `Ctrl+M`, `M`, `Space`, `Ctrl+H`, `Ctrl+Q`, `Esc` | ◐ fixed in tree 2026-11-18. `Space` toggles the engine only when the Live section is shown in the main window and its content panel has keyboard focus; never in Mini mode, never while text input is active or a widget is held. `Ctrl+H` (hide to tray) and `Ctrl+Q` (quit) are global and documented, also in the Help menu. `Esc` closes the Save preset dialog (the only modal). Known limitation: after clicking the navigation rail, click the Live panel once before `Space` works | `GuiApp.cpp` (key handling); `tests/test_frontend_shell.cpp` |
| Mini panel remembers its position | ◐ fixed in tree 2026-11-18: `Settings::mini_has_position` + `mini_x/mini_y`; JSON `"mini_position": [x, y]` written only when set, negative coordinates allowed; legacy `mini_x`/`mini_y` read once (legacy `-1,-1` or a half pair = no position), migrated, never re-emitted. Restored only when the point lies inside a connected display, else the window manager places it | `Settings.cpp`; `GuiApp.cpp` (Mini panel) |
| Settings round-trip with unknown keys preserved | ◐ fixed in tree 2026-11-18: a malformed `settings.json` is renamed to `settings.json.bad` on load (an older `.bad` is replaced), defaults are used, a warning naming the `.bad` path is shown in the UI notices and on stderr. Saves are atomic (`settings.json.tmp` + rename); a save that finds a malformed target moves it to `.bad` first, and if that fails the save is refused and the file is left untouched | `Settings.cpp`; `app/main.cpp` |
| Argument handling before SDL initialises; `--smoke-frames`, `--screenshot` | ☑ | `frontend/app/main.cpp`; CTest `gui_help`, `gui_rejects_*` |
| Headless tests of the model layer and shell | ☑ on Linux GCC and Clang (2026-11-18); never built on Windows or macOS | `tests/test_frontend_model.cpp`, `tests/test_frontend_shell.cpp` (`frontend_shell`) |
| Worker thread priority | ◐ fixed in tree 2026-11-18: the GUI worker takes `WorkerPriority` (the same request as the CLI) for its lifetime; `EngineHost::priority_method()` / `priority_elevated()`; shown in a tooltip on the status-bar pill. Not measured on Windows or macOS | `EngineHost.cpp`; `GuiApp.cpp` status bar |
| Config directory from the environment | ◐ BUG-018 fixed in tree 2026-11-18: `env_path(const char*)` in `Paths.cpp` (declared in `Paths.hpp`); Windows uses `_wdupenv_s` and builds the path from the wide string (non-ASCII `%APPDATA%` kept), POSIX uses `std::getenv`; no warning suppression. The Windows branch has been compiled by no compiler yet | `frontend/core/Paths.cpp` |

## Build evidence

| Leg | Result | Source |
|---|---|---|
| Linux GCC 13.3 | PASS, 51/51 (2026-11-02, container) | `frontend/README.md` §10; CI green on `c299e9f` |
| Linux Clang 18.1 | PASS, 51/51 (2026-11-02, container) | same |
| macOS AppleClang | PASS at `70b000b` and `86ad15b`; **FAIL at `c299e9f`** ([BUG-019](../Bug%20Report/BUG-019_macos_qos_call_wrong_arity.md), engine header, not frontend code) | CI runs 42824977880, 42826469000, 42829208493 |
| Windows MSVC | **FAIL on all three runs**; `fluxvoice_gui` and `test_frontend_model` never compiled | same runs; BUG-018 |
| Local, 2026-11-14 | NOT RUN (no C++ toolchain on the machine) | — |
| Local Linux GCC 13.3.0, 2026-11-18 (`build-m4-gcc`, Ubuntu 24.04, CMake 3.28.3, Ninja 1.11.1, base `fe2fb68` + uncommitted tree) | PASS: strict + warnings-as-errors configure (GUI on, ORT off), all targets incl. `fluxvoice_header_checks` and `fluxvoice_gui`; `ctest -E python_training` 105/105 | `frontend/README.md` §10 |
| Local Linux Clang 18.1.3, 2026-11-18 (`build-clang`, same machine and tree) | PASS: same commands; 105/105 | same |
| Windows MSVC, 2026-11-18 | NOT RUN (no toolchain; no CI run of this tree) | — |
| macOS AppleClang, 2026-11-18 | NOT RUN (no toolchain; no CI run of this tree) | — |
| Headless GUI smoke (`--smoke-frames`), ORT-enabled build, 2026-11-18 | NOT RUN (no Xvfb; ORT off) | — |

## Residual tasks

Build and defect fixes are scheduled in
[Stage 04](04_build_gate_frontend_hardening.md). Items that belong to this
stage's scope (state on 2026-11-18):

- [ ] Windows MSVC and macOS legs green. BUG-018 and BUG-019 are fixed in the
      tree, but both legs are NOT RUN; a CI run of this tree is needed.
- [x] `Space` toggles the engine only when the Live panel has focus (fixed in
      tree, Linux-tested through `frontend_shell`; known limitation: click the
      Live panel once after using the navigation rail).
- [x] `Ctrl+H` / `Ctrl+Q` documented (README §4.10, Help menu).
- [x] Mini panel position restored for negative coordinates (explicit
      "has position" flag, `"mini_position"` key, legacy keys migrated).
- [x] Malformed `settings.json` kept as `settings.json.bad`, warning shown in
      the UI and on stderr, never overwritten.
- [x] Worker takes `WorkerPriority` (the GUI worker uses the same request as
      the CLI; Stage 05's shared host keeps it).
- [ ] Manual checks on real desktops: tray on Windows, KDE, GNOME with the
      AppIndicator extension and one Wayland session; native dialogs;
      close-to-tray; start minimised. NOT RUN; recorded in
      [Stage 12](12_verification_exit.md).

Not changed here and still open (Stage 04 and later): UTF-8 conversion of SDL
dialog strings on Windows, BUG-022's top-level guard (it still shuts down
instead of keeping the GUI running), the "applied" indicator, deletion of
`src/ui/AppWindow.cpp`, CI jobs.

## Definition of done

- All four native legs build `fluxvoice_gui` and pass `frontend_model`.
- The residual tasks above are closed or moved, with a link, to a later stage.
- `frontend/README.md` §10 lists a result (not "NOT RUN") for each CI leg.
