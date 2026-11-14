# BUG-020 — A remembered device that is missing becomes "Follow system default"

> [Bug index](00_index.md) · Found by: [REVIEW-M4-FRONTEND_2026-11-14](REVIEW-M4-FRONTEND_2026-11-14.md)

| | |
|---|---|
| **Tag** | `BUG-020` |
| **Name** | *Quiet default* — the GUI substitutes the system default for a missing explicit device |
| **Reported** | 2026-11-14 |
| **Severity** | High (processed speech can be routed to speakers; violates the "never silently replace an explicit selection" rule) |
| **Status** | ☐ open |
| **Component** | `frontend/core/AppModel.cpp` (`selection_from_policy`, `restore_device_policy`, `start_engine`) |
| **Blocks** | [Milestone 4 / Stage 02](../Milestone%204/02_device_selectors.md), [Stage 04](../Milestone%204/04_build_gate_frontend_hardening.md) |

## Summary

Settings store an explicit device as `name:<display name>`. At start,
`selection_from_policy` looks the name up in the current snapshot. When the
name is missing or matches more than one endpoint, it sets `ok = false` and
**returns `FollowSystemDefault{}`** (`AppModel.cpp:333-336`).
`restore_device_policy` stores that value as the live selection
(`AppModel.cpp:340-341`).

`initialise()` reacts correctly once: it does not auto-start, shows a notice
and opens the Devices section (`AppModel.cpp:144-151`). But no "unresolved"
state is kept. Every later `start_engine()` — from Live, the tray's
"Engine on", the Mini panel or the `Space` key — calls
`engine_.start(capture_sel_, playback_sel_, …)` with the substituted default
(`AppModel.cpp:217`).

The notice text says "FluxVoice never substitutes another device". The code
then does exactly that on the next Start.

## Steps to reproduce (by reading; not executed — no toolchain on the review machine)

1. Settings: `playback_policy = "name:CABLE Input"`, `start_minimized = true`.
2. Start the GUI with the virtual cable absent (driver removed or renamed).
3. The window is hidden, so the notice and the Devices section are not seen.
4. Tray → "Engine on".

## Expected

Start is refused with `DeviceNotFound` for the remembered device, and the
user is asked to choose. This is the M1.5 decision 4 rule and
`frontend/README.md` §2 principle 2.

## Actual

The stream opens on the system default output. With a microphone and
speakers as defaults, converted speech plays on the speakers.

## Evidence

| Observation | Where |
|---|---|
| Missing / ambiguous name returns `FollowSystemDefault{}` | `frontend/core/AppModel.cpp:333-336` |
| The returned value becomes `capture_sel_` / `playback_sel_` | `frontend/core/AppModel.cpp:339-342` |
| `start_engine()` has no check for an unresolved policy | `frontend/core/AppModel.cpp:204-229` |
| The test pins the substitution as the expected result | `tests/test_frontend_model.cpp:801` (`holds_alternative<FollowSystemDefault>`) and never calls `start_engine()` afterwards |

## Root cause

`DeviceSelection` has two alternatives (follow default, explicit endpoint).
There is no value for "explicit, but not resolvable now", so the failure path
reused the nearest valid value.

## Suggested fix

- Keep the unresolved state: `std::optional<std::string> unresolved_capture_`
  / `unresolved_playback_` (the remembered name), set when `ok == false`.
- `start_engine()` refuses while either is set, with the typed
  `DeviceNotFound` message and the remembered name. Tray, Mini panel and
  hotkeys go through the same command, so they are covered.
- Selecting any row in Devices (including "Follow system default",
  explicitly) clears the unresolved state.
- A catalogue refresh re-tries the name; if it now resolves uniquely, the
  explicit endpoint is restored.
- The tray icon uses the error state while a policy is unresolved.

## Regression test

In `test_app_model_restores_policy`: after `initialise()` with a missing
`name:` policy, call `start_engine()` and `handle_tray_action` with a
`ToggleEngine` action;
assert both fail, `fb.open_calls == 0`, and the message names the remembered
device. Then add the device to the fake, refresh, and assert the explicit
endpoint is selected.

## Workaround

Open Devices and choose a device explicitly after any "remembered device is
no longer listed" notice, before starting the engine.
