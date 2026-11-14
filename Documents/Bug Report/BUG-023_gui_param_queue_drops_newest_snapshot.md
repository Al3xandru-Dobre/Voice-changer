# BUG-023 — The GUI parameter queue drops the newest snapshot when full

> [Bug index](00_index.md) · Found by: [REVIEW-M4-FRONTEND_2026-11-14](REVIEW-M4-FRONTEND_2026-11-14.md)

| | |
|---|---|
| **Tag** | `BUG-023` |
| **Name** | *Oldest wins* — a full queue keeps stale snapshots and discards new ones |
| **Reported** | 2026-11-14 |
| **Severity** | Medium (output gain and limiter can differ from the sliders while the UI reports "applied") |
| **Status** | ☐ open |
| **Component** | `frontend/core/EngineHost.cpp` (`push_params`), `frontend/core/AppModel.cpp` |
| **Blocks** | [Milestone 4 / Stage 05](../Milestone%204/05_gui_engine_host_mode_a.md) (replaces the queue with `ParamChannel`) |

## Summary

`EngineHost::push_params` enqueues with `try_enqueue` into a queue built with
capacity 64. On failure the snapshot is dropped and `params_pushed_` is not
advanced (`EngineHost.cpp:46-54`). The comment says "the newest snapshot wins
at the next push, so dropping is harmless". That holds only if another push
follows. The last change of a burst has no follower.

The worker is the only consumer and runs only while a stream is open. With
the engine stopped nothing drains the queue:

1. Engine stopped. The user drags sliders; each change pushes a snapshot.
2. After the queue fills, every later snapshot is dropped.
3. Start: the worker drains the queue and applies the newest *queued*
   snapshot, which is older than the user's last change.
4. `start_engine()` pushes once more after `engine_.start()`
   (`AppModel.cpp:227`). That push is also a `try_enqueue`, racing a worker
   that has just been created and may not have drained yet. If it loses, the
   stale values stay until the next slider move.

The "applied by worker" indicator compares two generations that both exclude
dropped snapshots, so it reads "applied".

## Status of the evidence

- Drop on full and the missing generation update: **confirmed** by reading.
- Step 4 losing the race: **plausible**, timing dependent, not executed.

## Expected

The newest snapshot always reaches the worker, as M2 Stage 09 specifies
("latest wins").

## Root cause

The GUI has its own queue. The engine's `src/voice/ParamChannel.hpp` already
solves this: `push()` parks the newest snapshot in a pending slot, `flush()`
retries it each control tick, and the CLI uses it (`src/main.cpp:583`).

## Suggested fix

Use `ParamChannel` in `EngineHost` and call `flush()` from `AppModel::tick`.
This is a task of [Stage 05](../Milestone%204/05_gui_engine_host_mode_a.md);
it can be done earlier in Stage 04 because it does not need the model host.

Related, found in the M2 audit: `ModeAWorker::pump` discards the result of
`pipeline_.set_params()` while `ParamChannel::drain` has already advanced
`applied_generation` (`ModeAWorker.hpp:74-75`, `ParamChannel.hpp:77`). A
rejected snapshot (voice index out of range after a bundle change) is then
reported as applied. Stage 05 must publish acceptance, not delivery.

## Regression test

With the engine stopped, push 200 snapshots with increasing gain, start the
engine over `FakeAudioBackend`, run the worker for a few hops, and assert the
output level corresponds to the **last** gain and that
`params_applied == params_pushed ==` the last generation.

## Workaround

Move any slider once after starting the engine.
