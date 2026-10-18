# BUG-010 — `is_stream_live()` ignored the callback progress counter

> [Bug index](00_index.md)

| | |
|---|---|
| **Tag** | `BUG-010` |
| **Name** | *Deaf liveness* — a dead RT thread on a still-"started" device was reported as a live stream |
| **Reported** | 2026-10-18 |
| **Severity** | Medium (a stalled stream would be reported live and never recovered on some backends) |
| **Status** | ☑ fixed — stall detection added (`callback_stall_timeout`, default 2 s), covered by a fake-backend test |
| **Component** | `src/audio/AudioController.cpp` (`is_stream_live`, `tick`, `open_resolved_internal`) |
| **Blocks** | [Milestone 1.5 / Stage 05](../Milestone%201.5/05_refresh_recovery_controller.md) liveness task (checked `[x]` but only 3 of the 4 required signals were wired) |

## Summary

Stage 05 specifies stream liveness as the conjunction of **four** signals:
"controller state + notifications + backend state + **callback progress
counter** — not `dev_ok_`, and not *samples are non-zero*". The first
`AudioController` implementation used only the first three:

```cpp
bool AudioController::is_stream_live() const {
    if (status_.state != StreamState::Running) return false;   // controller state
    if (!backend_.is_stream_running()) return false;           // backend state
    if ((last_notifications_ & (kStopped | kInterruptionBegan)) != 0)
        return false;                                        // notifications
    return true;                                             // no progress check
}
```

`status_.callback_progress` was published for telemetry but never consumed by
the liveness decision, and `tick()` never reacted to a frozen counter. A
backend could therefore report `ma_device_state_started` while the real-time
thread had died (driver stall, USB suspend, backend-specific failure mode
miniaudio cannot observe), and the controller would sit in `Running` forever:
`is_stream_live()` true, no recovery, silence out.

## Steps to reproduce

Against `FakeControllableBackend` in `tests/test_audio_controller.cpp`:

1. `open_and_start(FollowSystemDefault, FollowSystemDefault)` → `Running`.
2. Never call `pump_callback` (the fake's way of simulating a dead RT thread;
   `stream_running` stays true — the backend still says "started").
3. Call `ctrl.tick(now)` repeatedly and read `is_stream_live()`.

## Expected

After a bounded stall interval the controller treats the stream as
interrupted: `is_stream_live()` → false, `Running` → `Recovering` with the
standard bounded-retry policy.

## Actual

`is_stream_live()` stayed true indefinitely; the controller never noticed the
frozen `frames_processed()` counter.

## Fix (2026-10-18)

- `ControllerConfig::callback_stall_timeout` (default 2000 ms, `0` disables).
- `tick()` tracks `router_.frames_processed()`: when the counter advances it
  resets `progress_since_`; when it does not advance for
  `callback_stall_timeout` it sets `callback_stalled_`.
- `callback_stalled_` joins the existing unexpected-stop/interruption policy
  row → quiesce → `Recovering` (bounded retries, unchanged).
- `is_stream_live()` returns false while `callback_stalled_` is set.
- Baselines are re-armed inside `open_resolved_internal`, so retries and
  reconfigures cannot carry a stale watermark into a fresh stream.
- Regression test `test_stalled_callback_is_not_live_and_recovers` verifies
  live → stalled → Recovering → retry → Running → live.

## Evidence

- Spec: `Documents/Milestone 1.5/05_refresh_recovery_controller.md`, tasks
  list — "Liveness = controller state + notifications + backend state +
  callback progress counter".
- `tests/test_audio_controller.cpp::test_stalled_callback_is_not_live_and_recovers`
  (passes after the fix; fails before — `is_stream_live()` never went false).

## Workaround (pre-fix)

None required to keep running, but there was no way to detect the failure
mode at all — the stream simply went silent while the CLI reported
`FluxVoice running`.
