# Stage 02 — Device selectors over Milestone 1.5

> **Milestone 4** · [Master plan](00_master.md) · [← Stage 01](01_frontend_shell_tray_mini_panel.md) · [Stage 03 →](03_voice_studio_translate_catalogue_panels.md)

**Status:** ◐ model + view landed 2026-11-02; one High defect open
([BUG-020](../Bug%20Report/BUG-020_missing_remembered_device_becomes_default.md)); hardware checks pending
**Depends on:** [M1.5 stages 03–05](../Milestone%201.5/00_master.md) (`AudioDeviceRegistry`, selection policy, `AudioController`)
**Specification:** [`frontend/README.md`](../../frontend/README.md) §2, §4.2, §4.8
**Review:** [REVIEW-M4-FRONTEND_2026-11-14](../Bug%20Report/REVIEW-M4-FRONTEND_2026-11-14.md)

## Objective

Two direction-specific device selectors with an explicit "Follow system
default" entry, a refresh action and stale / error status. The UI submits an
endpoint key plus the snapshot generation, never a row index, and never
replaces an explicit selection by itself.

## As built (verified by reading on 2026-11-14)

| Rule / feature | State | Where |
|---|---|---|
| Rows built from the immutable snapshot, "Follow system default" first | ☑ | `DeviceSelectorModel.cpp` |
| A row becomes `ExplicitEndpoint{key, generation}` in the same frame; the controller never sees an index | ☑ | `ViewDevices.cpp:136`, `:145` → `AppModel.cpp:254-255` |
| Feedback guard previewed before start and enforced by the controller before and after open | ☑ | `AppModel.cpp:212`; `AudioController.cpp:282`, `:309` |
| Capture mode (microphone / system-playback loopback, capability-gated) | ☑ | `ViewDevices.cpp`, `AppModel.cpp` |
| Refresh, snapshot generation, stale badge | ☑ | `ViewDevices.cpp` |
| Inspect (on-demand native-format probe) | ☑ | `ViewDevices.cpp:140`, `:149`; `AppModel.cpp:284-299` |
| Stream status table | ☑ | `ViewDevices.cpp` |
| Device loss while running: no substitution (`AwaitingUser` / `DeviceNotFound`) | ☑ controller rule, covered by a headless test | `tests/test_frontend_model.cpp` |
| **Remembered device missing at start: user is asked, nothing is substituted** | ☒ the selection silently becomes "Follow system default"; a later Start opens the default device | BUG-020; `AppModel.cpp:320-341` |
| Out-of-range row | maps to Follow default (`DeviceSelectorModel.cpp:66`); not reachable from the view, but it should be an error | — |

## Residual tasks

- [ ] Fix BUG-020 (unresolved-policy state; Start refused until the user
      chooses). Scheduled in [Stage 04](04_build_gate_frontend_hardening.md).
- [ ] An out-of-range row returns an error instead of Follow default.
- [ ] The "choose again" state is visible from the tray (error icon and
      tooltip) while a policy is unresolved, because the window may be hidden.
- [ ] Hardware matrix through the GUI on each available OS: unplug the
      selected input and output while running; change the system default
      while following it; two identical USB microphones (ambiguous name).
      Recorded in [Stage 12](12_verification_exit.md).

## Definition of done

- BUG-020 is fixed with its regression test.
- No code path in `frontend/` produces `FollowSystemDefault` except the user
  choosing it or a stored `"default"` policy.
- The hardware matrix rows are filled with PASS / FAIL / NOT RUN.
