# Stage 03 — SPSC buffering & worker thread

> **Milestone 1** · [Master plan](00_master.md) · [← Stage 02](02_duplex_audio_io.md) · [Stage 04 →](04_overlap_add_dsp.md)

**Status:** ◐ skeleton landed
**Depends on:** Stage 02

## Objective

Route audio through `moodycamel::ReaderWriterQueue` SPSC queues with a worker
thread in the middle — the exact hand-off the ONNX stages will use. The
callback must stay allocation-free and lock-free end-to-end.

## Context & references

- RESEARCH_GUIDE §4: *"Using `std::mutex` inside an audio callback will cause
  glitches due to OS thread priority inversion."* — hence `try_enqueue` /
  `try_dequeue`, never the blocking variants.
- `src/audio/AudioEngine.hpp` — queues + counters implemented; the worker
  loop lives in `src/main.cpp` (placeholder for the M2 inference thread).

## Tasks

- [x] `in_q_` / `out_q_` `ReaderWriterQueue<float>`, capacity `1 << 16`
      (~4 s @ 16 kHz mono) allocated once at construction
- [x] Callback: capture → `in_q_.try_enqueue` (drop + count on overflow);
      `out_q_.try_dequeue` → device, zero-fill on underrun
- [x] Worker-side `read_input` / `write_output` drain/fill helpers
- [x] `main.cpp` worker thread: shovel input → output (loopback through queues)
- [ ] Chunk assembler: worker accumulates samples into exact 1,280-sample
      (80 ms) frames (+20 ms historical context) — the unit neural stages
      consume. Must be **independent of callback `frame_count`** (period sizes
      are hints); test with arbitrary callback sizes.
      *Design now, land with M2's encoder interface.*
- [ ] Verify queue-path loopback is glitch-free under CPU load
      (`stress-ng` / parallel build)

## Implementation notes

- **SPSC means SPSC.** `in_q_`: producer = audio callback, consumer = worker.
  `out_q_`: producer = worker, consumer = audio callback. Never let the UI
  thread touch them.
- Prefer non-blocking ops even if slightly slower per call — a bounded drop
  beats an unbounded stall on the RT thread.
- Underrun → **zero-fill** (silence), never repeat old audio (audible buzz).
  Overrun → drop newest, count it.
- Expose `input_overruns` / `output_underruns` atomics — printed on exit and
  later surfaced in the M4 telemetry UI.
- Worker sleep granularity: poll with ~2 ms backoff when empty (a 20 ms
  callback deposits 320 samples; busy-spinning burns a core for nothing).
- The `1 << 16` capacity is a **safety limit, not a target delay** — ~4 s of
  queued audio would be unusable latency. Bounding queued latency and the
  device-switch drain/reset sequence are specified in
  [Milestone 1.5 / Stage 05](../Milestone%201.5/05_refresh_recovery_controller.md).

## Files touched

- `src/audio/AudioEngine.hpp`, `src/main.cpp`

## Definition of done

- Queue-path loopback (`fluxvoice` default mode) is audible and stable.
- `output_underruns` ≈ startup transient only (worker fills faster than drain);
  `input_overruns` == 0 on an idle machine.
- No `malloc`/`new`, mutex, or logging call exists inside `data_callback`
  (audit by inspection; optionally verify with a `LD_PRELOAD` malloc counter).
