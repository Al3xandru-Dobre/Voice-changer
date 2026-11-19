# Stage 02 — Device selectors over Milestone 1.5

> **Milestone 4** · [Master plan](00_master.md) · [← Stage 01](01_frontend_shell_tray_mini_panel.md) · [Stage 03 →](03_voice_studio_translate_catalogue_panels.md)

**Status:** ◐ model + view landed 2026-11-02; residual code tasks fixed in the working tree
2026-11-18 and tested on Linux GCC and Clang only
([BUG-020](../Bug%20Report/BUG-020_missing_remembered_device_becomes_default.md) fixed in tree, native legs pending);
hardware matrix NOT RUN (Stage 12)
**Depends on:** [M1.5 stages 03–05](../Milestone%201.5/00_master.md) (`AudioDeviceRegistry`, selection policy, `AudioController`)
**Specification:** [`frontend/README.md`](../../frontend/README.md) §2, §4.2, §4.8
**Review:** [REVIEW-M4-FRONTEND_2026-11-14](../Bug%20Report/REVIEW-M4-FRONTEND_2026-11-14.md)

## Objective

Two direction-specific device selectors with an explicit "Follow system
default" entry, a refresh action and stale / error status. The UI submits an
endpoint key plus the snapshot generation, never a row index, and never
replaces an explicit selection by itself.

## As built (read on 2026-11-14, updated 2026-11-18 for the working tree)

| Rule / feature | State | Where |
|---|---|---|
| Rows built from the immutable snapshot, "Follow system default" first | ☑ | `DeviceSelectorModel.cpp` |
| A row becomes `ExplicitEndpoint{key, generation}` in the same frame; the controller never sees an index | ☑ | `ViewDevices.cpp` → `AppModel::select_capture_row` / `select_playback_row` |
| Feedback guard previewed before start and enforced by the controller before and after open | ☑ | `AppModel` (feedback preview); `AudioController` (checks before and after open) |
| Capture mode (microphone / system-playback loopback, capability-gated) | ☑ | `ViewDevices.cpp`, `AppModel.cpp` |
| Refresh, snapshot generation, stale badge | ☑ | `ViewDevices.cpp` |
| Inspect (on-demand native-format probe) | ☑ | `ViewDevices.cpp`; `AppModel` (inspect command) |
| Stream status table | ☑ | `ViewDevices.cpp` |
| Device loss while running: no substitution (`AwaitingUser` / `DeviceNotFound`) | ☑ controller rule, covered by a headless test | `tests/test_frontend_model.cpp` |
| Remembered device missing at start: user is asked, nothing is substituted | ☑ BUG-020 fixed before 2026-11-18 and unchanged: an unresolved remembered device blocks Start; the tray shows the Error icon and tooltip | `AppModel.cpp` (policy restore) |
| Out-of-range row | ◐ fixed in tree 2026-11-18: `selection_for_row` returns `std::optional<DeviceSelection>`; only the Follow-default row yields `FollowSystemDefault`, an out-of-range row is `nullopt`. `AppModel::select_capture_row` / `select_playback_row` return false, raise an Error notice and keep the selection and any unresolved state | `DeviceSelectorModel.cpp`; `AppModel.cpp` |
| Stored policy that is not empty, `"default"` or `"name:<device>"` (for example `USB Mic`, `Name:USB Mic`, `name:`) | ◐ fixed in tree 2026-11-18: an unresolved explicit choice (Start refused, tray Error), never Follow default; a prefix-less name is later matched only as that exact device name | `AppModel.cpp` |
| Apply with an explicit endpoint that left the list before Apply | ◐ fixed in tree 2026-11-18: keeps the previously stored policy; no placeholder such as `name:(device no longer listed)` is ever saved | `AppModel.cpp` |
| Status-bar dot for an unresolved remembered device | ◐ fixed in tree 2026-11-18: turns red (it used to ignore the state) | `GuiApp.cpp` |

No code path in `frontend/` now produces `FollowSystemDefault` except the user
choosing it, a stored `"default"` or empty policy, or a capture-mode change
(which resets the capture list; existing behaviour, a user action). Tests:
`test_device_policy_rules` in `frontend_model` (GCC and Clang, 2026-11-18;
mutation check: breaking the malformed-policy rule fails the suite).

## Residual tasks

- [x] BUG-020 (unresolved-policy state; Start refused until the user
      chooses): fixed in tree, native legs pending.
- [x] An out-of-range row returns an error instead of Follow default.
- [x] The "choose again" state is visible from the tray (error icon and
      tooltip) while a policy is unresolved, and the status-bar dot is red.
- [ ] Hardware matrix through the GUI on each available OS. NOT RUN on every
      cell; recorded in [Stage 12](12_verification_exit.md).

### Hardware matrix (all NOT RUN as of 2026-11-18)

| Case | Windows | Linux | macOS |
|---|---|---|---|
| Unplug the selected input while running | NOT RUN | NOT RUN | NOT RUN |
| Unplug the selected output while running | NOT RUN | NOT RUN | NOT RUN |
| Change the system default while following it | NOT RUN | NOT RUN | NOT RUN |
| Two identical USB microphones (ambiguous name) | NOT RUN | NOT RUN | NOT RUN |

## Definition of done

- BUG-020 is fixed with its regression test (done in tree; native Windows and
  macOS legs pending).
- No code path in `frontend/` produces `FollowSystemDefault` except the user
  choosing it or a stored `"default"` policy.
- The hardware matrix rows are filled with PASS / FAIL / NOT RUN.
