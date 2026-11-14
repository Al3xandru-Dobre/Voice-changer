# Stage 01 — Frontend shell, tray & Mini panel

> **Milestone 4** · [Master plan](00_master.md) · [Stage 02 →](02_device_selectors.md)

**Status:** ◐ landed 2026-11-02 (`70b000b`); the Windows MSVC build is broken
([BUG-018](../Bug%20Report/BUG-018_frontend_getenv_breaks_msvc_build.md)); residual tasks below
**Depends on:** [Milestone 1.5](../Milestone%201.5/00_master.md) stages 02–05 (`AudioTypes`, registry, controller)
**Specification:** [`frontend/README.md`](../../frontend/README.md) §4, §6–§8
**Review:** [REVIEW-M4-FRONTEND_2026-11-14](../Bug%20Report/REVIEW-M4-FRONTEND_2026-11-14.md)

## Objective

A desktop application (`fluxvoice_gui`) built from SDL3 and Dear ImGui that
composes the Milestone 1.5 engine, with a toolkit-free model layer, a system
tray, and a compact Mini panel. This file is the as-built record plus what is
still open.

## As built (verified by reading on 2026-11-14)

| Item | State | Where |
|---|---|---|
| SDL3 3.4.18 + Dear ImGui v1.92.9 pinned via CMake, compiled as SYSTEM code | ☑ | root `CMakeLists.txt`, `frontend/CMakeLists.txt` |
| `frontend/core` free of SDL / ImGui | ☑ no includes found | `frontend/core/**` |
| UI thread is the control thread; no frontend code on the audio callback | ☑ | `EngineHost.hpp` thread contract; `AppModel::tick` |
| Navigation rail, eight sections, status bar | ☑ | `frontend/imgui/GuiApp.cpp`, `View*.cpp` |
| Tray model (menu spec, icon state, tooltip, action queue) + SDL tray | ☑ | `frontend/core/TrayModel.*`, `frontend/platform/SdlTray.*` |
| Close-to-tray, start minimised, Mini panel | ☑ | `GuiApp.cpp`, `ViewMiniPanel.cpp` |
| Native dialogs: callbacks only enqueue, results drained on the UI thread | ☑ | `GuiApp.cpp:392-403`, `:425-432` |
| Keyboard: `Ctrl+1…8`, `Ctrl+M`, `M`, `Space` | ◐ implemented; `Space` acts whenever the section is Live (also in Mini mode), not only when the Live panel has focus; `Ctrl+H` / `Ctrl+Q` exist but are undocumented; no first-party `Esc` handling | `GuiApp.cpp:499-510` |
| Mini panel remembers its position | ◐ only when both coordinates are ≥ 0 (a panel on a monitor left of or above the primary is not restored) | `GuiApp.cpp:287-293`, `:367` |
| Settings round-trip with unknown keys preserved | ◐ works for a valid file; a file with one syntax error yields defaults and is overwritten on exit | `Settings.cpp:108-111`; `app/main.cpp:130-131` |
| Argument handling before SDL initialises; `--smoke-frames`, `--screenshot` | ☑ | `frontend/app/main.cpp`; CTest `gui_help`, `gui_rejects_*` |
| Headless tests of the model layer | ☑ on Linux; never built on Windows | `tests/test_frontend_model.cpp` |
| Worker thread priority | ☐ the GUI worker never takes `WorkerPriority`; it runs at normal priority, unlike the CLI | `EngineHost.cpp:94-98` vs `src/main.cpp:661` |

## Build evidence

| Leg | Result | Source |
|---|---|---|
| Linux GCC 13.3 | PASS, 51/51 (2026-11-02, container) | `frontend/README.md` §10; CI green on `c299e9f` |
| Linux Clang 18.1 | PASS, 51/51 (2026-11-02, container) | same |
| macOS AppleClang | PASS at `70b000b` and `86ad15b`; **FAIL at `c299e9f`** ([BUG-019](../Bug%20Report/BUG-019_macos_qos_call_wrong_arity.md), engine header, not frontend code) | CI runs 42824977880, 42826469000, 42829208493 |
| Windows MSVC | **FAIL on all three runs**; `fluxvoice_gui` and `test_frontend_model` never compiled | same runs; BUG-018 |
| Local, 2026-11-14 | NOT RUN (no C++ toolchain on the machine) | — |

## Residual tasks

Build and defect fixes are scheduled in
[Stage 04](04_build_gate_frontend_hardening.md). Items that belong to this
stage's scope:

- [ ] Windows MSVC and macOS legs green (BUG-018, BUG-019).
- [ ] `Space` toggles the engine only when the Live panel has focus, as the
      specification says, or the specification is changed to match.
- [ ] Document or remove `Ctrl+H` / `Ctrl+Q`.
- [ ] Restore the Mini panel position for negative coordinates (use a
      separate "has position" flag instead of `-1`).
- [ ] Malformed `settings.json`: keep the file (rename to
      `settings.json.bad`), show the warning in the UI, and do not overwrite.
- [ ] Worker takes `WorkerPriority` (folds into Stage 05's shared host).
- [ ] Manual checks on real desktops: tray on Windows, KDE, GNOME with the
      AppIndicator extension and one Wayland session; native dialogs;
      close-to-tray; start minimised. Recorded in [Stage 12](12_verification_exit.md).

## Definition of done

- All four native legs build `fluxvoice_gui` and pass `frontend_model`.
- The residual tasks above are closed or moved, with a link, to a later stage.
- `frontend/README.md` §10 lists a result (not "NOT RUN") for each CI leg.
