# Stage 05 — Refresh & recovery controller

> **Milestone 1.5** · [Master plan](00_master.md) · [← Stage 04](04_selection_policy_cli.md) · [Stage 06 →](06_routing_os_validation.md)

**Status:** ☐ not started
**Depends on:** Stage 04, Milestone 1 / Stage 03 (SPSC queues + worker)
**Research:** §5 "Notifications do not replace catalog refresh" · §7 · §8 · §10 stage 4

## Objective

An `AudioController` on a single control thread that keeps the catalog fresh,
reacts to device loss and default changes according to an explicit policy
table, and performs serialized stop → reconfigure → start transitions without
ever doing lifecycle work on a real-time thread.

## Stream lifecycle

```text
Idle -> Opening -> Running
                    |
                    v
                Recovering -> Opening
                    |
                    v
              AwaitingUser / Error

Running / Recovering / Error -> Stopping -> Idle
```

## Policy table

| Event | Behavior |
|---|---|
| Follow-default selection; OS default changes | Follow it; account for backend auto-routing before reopening (no duplicate restarts) |
| Explicit device disappears | Stop that route; show the missing selection; **do not** record another mic |
| Virtual output disappears | Stop output; request repair; **never** redirect processed speech to speakers |
| Permission denied | Explain the user action needed; no continuous retry; no substitute source |
| Device busy / temporary server loss | Bounded retries with backoff; show backend + reason |
| Context/backend recreated | Stop streams, invalidate keys, rebuild inventory, re-resolve saved policy |
| User picks another endpoint | Serialized stop / reconfigure / start |

## Tasks

- [ ] `AudioController` owning registry + device; all enumeration, open/close,
      retries on one control thread
- [ ] `notificationCallback` sets bits in a lock-free atomic pending-event
      bitset only; control thread drains it (`static_assert(is_always_lock_free)`)
- [ ] Periodic refresh (configurable, default **1 s**) + on-demand + on hint;
      coalesce bursts; bounded backoff on errors; stale snapshot retained
- [ ] Liveness = controller state + notifications + backend state + callback
      progress counter — not `dev_ok_`, and **not** "samples are non-zero"
      (the user may be silent)
- [ ] Switch transition: quiesce worker + callbacks → drain/discard both SPSC
      queues → reset OLA (and, from M2, model state) → reopen. No old-route PCM
      may leak into the new route.
- [ ] Log client vs. backend format: `ma_device.sampleRate` (client) vs.
      `capture/playback.internalSampleRate` (backend) — see M1 stage 05
- [ ] Bound queued-audio latency and define the overload/drop policy (queue
      capacity is a safety limit, not a target delay). Reuse
      [M2 / Stage 08](../Milestone%202/08_realtime_mode_a_pipeline.md)'s
      `LatencyGovernor` and its `InferencePipeline::reset()` contract (worker
      thread only, after quiesce). Whichever stage lands second adapts to the
      other.
- [ ] Fake-backend tests: notification bursts, retry limits, shutdown during
      recovery, no callback touching destroyed controller state, queue discard
      only after producers/consumers stop

## Implementation notes

- Notifications cover an **opened stream** (started/stopped/rerouted/
  interrupted), not system-wide hot-plug, and coverage varies per backend.
  Treat them as hints; polling is the portable baseline.
- **Never** restart or uninit a device inside `notificationCallback` or
  `dataCallback`.
- Notifications may arrive from several backend threads — do **not** push them
  into an SPSC queue.
- Keep `ma_context` / `ma_device` addresses stable for their lifetimes.
- Capture-only / loopback modes may need separate streams: today's callback
  assumes a non-null output buffer. Restructure explicitly; test null-buffer
  contracts.

## Files touched

- `src/audio/AudioController.{hpp,cpp}` (new), `src/audio/AudioEngine.hpp`,
  `src/main.cpp`, `src/pipeline/InferencePipeline.cpp` (`reset()` hook), tests

## Definition of done

- Every row of the policy table has a passing fake-backend test.
- Manual check on the dev machine: unplug/replug a USB mic and change the OS
  default while running — behavior matches the table; no RT-thread lifecycle calls.
