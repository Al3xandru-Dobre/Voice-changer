# BUG-011 — Retryable pre-open catalog failures bypassed bounded recovery

> [Bug index](00_index.md)

| | |
|---|---|
| **Tag** | `BUG-011` |
| **Name** | *Retry bypass* — a transient enumeration failure at open died instantly instead of retrying |
| **Reported** | 2026-10-18 |
| **Severity** | Medium (one audio-server hiccup at startup killed the app; recovery machinery existed but was unreachable for this failure class) |
| **Status** | ☑ fixed — all pre-open failures now take the same classification as mid-stream failures, and `Recovering` can heal an empty catalog |
| **Component** | `src/audio/AudioController.cpp` (`open_and_start`, `reconfigure`, `on_context_recreated`, `tick`) |
| **Blocks** | [Milestone 1.5 / Stage 05](../Milestone%201.5/05_refresh_recovery_controller.md) failure table — "Device busy / temporary server loss → bounded retries with backoff" |

## Summary

Two related defects in the same code path:

1. `open_and_start()`, `reconfigure()`, and `on_context_recreated()` each run
   a catalog refresh before opening. On failure they classified the error as:

   ```cpp
   status_.state = is_non_retryable(err.category) ? StreamState::AwaitingUser
                                                  : StreamState::Error;
   ```

   So `DeviceBusy` / `EnumerationFailed` / `BackendUnavailable` — the exact
   categories the policy table says must take **bounded retries** — went to
   terminal `Error` immediately, while the *same* categories raised one step
   later inside `open_stream` correctly entered `Recovering`. The classification
   differed purely by which statement the failure happened to hit.

2. Had a retryable pre-open failure actually reached `Recovering`, it could
   never recover: the retry path calls `open_resolved_internal()`, whose first
   step is `registry_.resolve(...)` — and with `snapshot().generation == 0`
   that fails instantly with `DeviceChanged` (non-retryable) → `AwaitingUser`.
   The `Recovering` state had no way to rebuild the missing catalog itself.

## Steps to reproduce

Against `FakeControllableBackend` (`tests/test_audio_controller.cpp`):

```cpp
fb.enumerate_error = AudioError{AudioErrorCategory::DeviceBusy,
                                "enumerate devices", -19, "wasapi",
                                "audio server restarting"};
auto r = ctrl.open_and_start(FollowSystemDefault{}, FollowSystemDefault{},
                             CaptureMode::OrdinaryMicrophone, t0);
```

## Expected

`status_.state == StreamState::Recovering` with `retry_attempt == 0` and the
standard bounded back-off, matching the "device busy / temporary server loss"
row of the Stage 05 policy table. When the server returns, a later `tick()`
rebuilds the catalog and opens the stream.

## Actual

`status_.state == StreamState::Error` (terminal) on the first tick — no retry
ever attempted, even though `max_retries = 3`.

## Fix (2026-10-18)

- New `AudioController::enter_failure_state(err, now)` — the single
  classification for any failure that prevents (re)opening: non-retryable →
  `AwaitingUser`, retryable → `Recovering` + `schedule_next_retry` (or
  `Error` only when `max_retries == 0`). Used by all three pre-open paths.
- `tick()`'s `Recovering` section now refreshes the catalog first when
  `snapshot().generation == 0`, inside the retry attempt — a missing catalog
  can no longer bounce `Recovering` straight into `AwaitingUser`, and each
  failed heal consumes a bounded attempt.
- The two feedback-refusal paths now go through the shared `fail()` helper,
  so they also clear `resolved_capture`/`resolved_playback` like every other
  failure path (previously they left stale resolved keys in `status()`).
- Regression test `test_retryable_catalog_failure_enters_bounded_recovery`
  covers both directions: transient failure → `Recovering` → `Running`, and
  permanent failure → bounded retries → `Error`.

## Evidence

- `tests/test_audio_controller.cpp::test_retryable_catalog_failure_enters_bounded_recovery`
  (passes after the fix; before, the first `CHECK` saw `Error`, not
  `Recovering`).

## Workaround (pre-fix)

Restart `fluxvoice` when the audio server is mid-restart — the retry
machinery simply never ran for this failure class.
