# Stage 11 — Desktop integration: global hotkeys, taskbar buttons, notifications

> **Milestone 4** · [Master plan](00_master.md) · [← Stage 10](10_audio_extensions.md) · [Stage 12 →](12_verification_exit.md)

**Status:** ☐ not started
**Depends on:** [Stage 01](01_frontend_shell_tray_mini_panel.md), [Stage 04](04_build_gate_frontend_hardening.md)
**Owner persona:** `@ui-telemetry`

## Objective

Add the three desktop conveniences the master lists as not delivered: global
hotkeys, Windows taskbar thumbnail buttons and desktop notifications. All
three sit behind the `frontend/platform` seam and post the same `TrayAction`s
the tray and the Mini panel use.

## Context & verified starting point (2026-11-14)

- `frontend/platform/` holds `Tray.hpp` (interface + null implementation) and
  `SdlTray.{hpp,cpp}`. There is no hotkey, taskbar or notification code in
  the repository.
- `frontend/README.md` §11 names the platform APIs: `RegisterHotKey` on
  Windows, the GlobalShortcuts portal on Wayland, `ITaskbarList3` for the
  thumbnail buttons.
- In-window shortcuts are specified in `frontend/README.md` §4.10. Their
  implementation status is recorded in
  [Stage 01](01_frontend_shell_tray_mini_panel.md).

## Tasks

- [ ] **Platform seam** (`frontend/platform/Hotkeys.hpp`,
      `Notifications.hpp`, `Taskbar.hpp`): small interfaces with null
      implementations, so `frontend/core` and the tests never see an OS
      header. Each real implementation is one translation unit guarded by
      `_WIN32` / `__APPLE__` / `__linux__`, linked per OS in CMake.
- [ ] **Global hotkeys:**
  - [ ] Actions: toggle engine, toggle mute, next / previous preset, toggle
        Mini panel, push-to-mute (hold).
  - [ ] Settings: one binding per action, a "record shortcut" control,
        conflict detection, and an explicit "disabled" state. Nothing is
        bound by default.
  - [ ] Windows: `RegisterHotKey` on the UI thread; a failed registration
        (already taken) is reported next to the binding.
  - [ ] Linux: X11 key grabs under X11; the GlobalShortcuts portal under
        Wayland where the desktop provides it; otherwise the Settings page
        states that global hotkeys are unavailable on this session.
  - [ ] macOS: record as NOT IMPLEMENTED unless a release target is declared.
  - [ ] Hotkey callbacks only enqueue a `TrayAction`. They never touch
        `AppModel` or the engine directly.
- [ ] **Windows taskbar thumbnail buttons** (`ITaskbarList3`): engine toggle
      and mute, with icons rendered by the existing procedural `TrayIcon`
      code. Re-add the buttons after the taskbar-created message.
- [ ] **Desktop notifications:** device lost / stream stopped, recovery
      succeeded, fatal pipeline error, Studio job finished. Rate-limited (one
      per event class per 10 s) and switchable per class in Settings. Use the
      tray balloon on Windows and the notification D-Bus interface on Linux;
      fall back to the in-app message list when neither exists.
- [ ] **Tests (hardware-free):**
  - [ ] Binding parser and conflict detection (round-trip through
        `Settings`, duplicate binding rejected).
  - [ ] A fake hotkey backend delivers events; `AppModel` receives the same
        actions as from the tray; push-to-mute press/release maps to mute
        on/off.
  - [ ] Notification policy: rate limit and per-class switches over a fake
        sink.

## Implementation notes

- A hotkey API callback may run on a different thread from the UI. Queue the
  action and drain it in `AppModel::tick`, as the native dialogs do.
- Global hotkeys are a keyboard hook from the user's point of view. Keep them
  opt-in and show what is bound.
- None of this may reach the audio callback or the worker.

## Files touched

- `frontend/platform/Hotkeys.hpp`, `HotkeysWin.cpp`, `HotkeysX11.cpp`,
  `HotkeysPortal.cpp`, `Notifications.hpp`, `Notifications*.cpp`,
  `Taskbar.hpp`, `TaskbarWin.cpp` (all new), `frontend/core/Settings.*`,
  `frontend/core/AppModel.*`, `frontend/core/TrayModel.*`,
  `frontend/imgui/ViewSettings.cpp`, `frontend/imgui/GuiApp.cpp`,
  `frontend/CMakeLists.txt`, `tests/test_frontend_model.cpp`,
  `frontend/README.md` (§4.10, §6, §11)

## Definition of done

- On Windows: bound hotkeys work while the window is hidden; the thumbnail
  buttons toggle engine and mute; a device-lost event raises one notification.
- On Linux: hotkeys work under X11, and under Wayland where the portal
  exists; the Settings page states the reason when they are unavailable.
- All new model-level tests pass on the four native legs; manual per-desktop
  checks are listed with PASS / FAIL / NOT RUN.
