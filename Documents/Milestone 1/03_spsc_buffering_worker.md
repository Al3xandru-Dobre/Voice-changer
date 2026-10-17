# Stage 03 — SPSC buffering & worker thread

> **Milestone 1** · [Master plan](00_master.md) · [← Stage 02](02_duplex_audio_io.md) · [Stage 04 →](04_overlap_add_dsp.md)

**Status:** ☑ done (2026-10-17)
**Depends on:** Stage 02

## Objective

Route audio through `moodycamel::ReaderWriterQueue` SPSC queues with a worker
thread in the middle — the exact hand-off the ONNX stages will use. The
callback must stay allocation-free and lock-free end-to-end.

## Context & references

- RESEARCH_GUIDE §4: *"Using `std::mutex` inside an audio callback will cause
  glitches due to OS thread priority inversion."* — hence `try_enqueue` /
  `try_dequeue`, never the blocking variants.
- `src/audio/DuplexRouter.hpp` — queues, counters and the callback body;
  `AudioEngine` owns one and forwards to it. The worker loop lives in
  `src/main.cpp` (placeholder for the M2 inference thread).
- `src/audio/ChunkAssembler.hpp` + `src/pipeline/PipelineTiming.hpp` — the
  80 ms frame geometry shared with `InferencePipeline`.

## Tasks

- [x] `in_q_` / `out_q_` `ReaderWriterQueue<float>`, capacity `1 << 16`
      (~4 s @ 16 kHz mono) allocated once at construction
- [x] Callback: capture → `in_q_.try_enqueue` (drop + count on overflow);
      `out_q_.try_dequeue` → device, zero-fill on underrun
- [x] Worker-side `read_input` / `write_output` drain/fill helpers
- [x] `main.cpp` worker thread: shovel input → output (loopback through queues)
- [x] Chunk assembler (`ChunkAssembler`): exact 1,280-sample (80 ms) frames
      with 320 samples (20 ms) of carried-over context, **independent of
      callback `frame_count`**; allocation-free after construction. Tested
      against a reference slicer with irregular push patterns. Wired into the
      worker behind `--chunked`; *hooking frames into the encoder is M2.*
- [x] Verify queue-path loopback is glitch-free under CPU load
- [x] Unit tests (`ctest`): `duplex_router` (passthrough, in-order round trip
      with irregular callback sizes, underrun zero-fill, overrun counting,
      **zero heap allocations in `process()`**, concurrent callback + worker
      threads) and `chunk_assembler`
- [x] Worker counts `output drops` (samples `write_output` couldn't fit)

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

## Verification (2026-10-17)

Same dev machine as Stage 02; `tools/pw_loopback_check.sh` (tone → null sink →
fluxvoice → null sink → recorder → glitch analysis).

| Run | Duration | Overruns | Underruns | Drops | Recording |
|---|---|---|---|---|---|
| `--queue-loopback`, idle | 20 s | 0 | 320 (one period, startup) | 0 | clean, 440.00 Hz |
| `--chunked`, idle | 20 s | 0 | 1,280 (one hop, startup) | 0 | clean, 440.00 Hz |
| `--queue-loopback`, **24 busy loops on 24 cores** | 60 s | 0 | 320 (startup) | 0 | clean, 440.00 Hz |

Unit tests: 11/11 CTest pass on GCC 13 and Clang 18 with
`-DFLUXVOICE_WARNINGS_AS_ERRORS=ON`. `test_duplex_router` is also clean under
ThreadSanitizer and ASan+UBSan.

How the tests were checked: injecting a heap allocation into `process()` fails
the allocation check, and an off-by-one in `ChunkAssembler::advance()` fails
178 checks.

Notes:
- The allocation check replaces global `operator new`, so it catches C++
  allocations only, not a direct `malloc`. The callback body is visible in one
  header and makes no C allocation calls (checked by reading it).
- On this kernel TSan needs ASLR off: `setarch $(uname -m) -R ./test_duplex_router`.
  TSan builds skip the allocation test because TSan supplies its own `operator new`.

## Files touched

- `src/audio/DuplexRouter.hpp`, `src/audio/ChunkAssembler.hpp`,
  `src/pipeline/PipelineTiming.hpp` (new), `src/audio/AudioEngine.hpp`,
  `src/pipeline/InferencePipeline.cpp`, `src/main.cpp`,
  `tests/test_duplex_router.cpp`, `tests/test_chunk_assembler.cpp`,
  `tests/TestSupport.hpp` (new), `CMakeLists.txt`

## Definition of done

- Queue-path loopback (`fluxvoice` default mode) is audible and stable.
- `output_underruns` ≈ startup transient only (worker fills faster than drain);
  `input_overruns` == 0 on an idle machine.
- No `malloc`/`new`, mutex, or logging call exists inside `data_callback`
  (audit by inspection; optionally verify with a `LD_PRELOAD` malloc counter).

All three met — see Verification. The allocation audit is an automated test
rather than an `LD_PRELOAD` counter.
