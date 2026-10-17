# Milestone 1 — DSP Loopback & Audio Engine

> Parent: [Development milestones](../../README.md#7-development-milestones) ·
> [Documents index](../README.md)

**Status:** ◐ in progress (skeleton landed)
**Goal from README:** *Implement miniaudio device selector + SPSC ring buffer +
`OverlapAddProcessor` in C++. Verify click-free audio pass-through into a
virtual audio cable.*

## Why this milestone

Everything upstream (neural encoder, Wait-k translator, vocoder) assumes a
rock-solid real-time audio substrate: deterministic chunk delivery, lock-free
inter-thread hand-off, and glitch-free output. A defect here (priority
inversion, boundary clicks, buffer underflow) poisons every later stage and is
hardest to debug once neural models are in the loop — so we prove the plumbing
first with pure passthrough.

## Scope

**In scope**
- CMake build system + pinned dependency fetching
- miniaudio device enumeration & duplex stream @ 16 kHz mono float32, 20 ms callbacks
- Lock-free SPSC queues (`readerwriterqueue`) + inference-worker slot
- `OverlapAddProcessor` Hann crossfader + unit tests
- Virtual-audio-cable loopback routing + CLI device selection
- End-to-end latency measurement & soak test

**Out of scope** (later milestones)
- Stable device identity, refresh/hot-plug recovery, typed audio errors,
  Windows/macOS validation → [Milestone 1.5](../Milestone%201.5/00_master.md)
- Any ONNX Runtime inference (M2+)
- Wait-k policy / translation logic (M3)
- GUI (M4)

## Stages

| # | Stage | Output | Status |
|---|---|---|---|
| 01 | [Build system & dependencies](01_build_and_dependencies.md) | `cmake --build` produces `fluxvoice` binary | ☑ |
| 02 | [Duplex audio I/O](02_duplex_audio_io.md) | Enumerate devices; callback passthrough audible | ☑ |
| 03 | [SPSC buffering & worker thread](03_spsc_buffering_worker.md) | Queue-routed loopback, zero alloc/locks in callback | ☑ |
| 04 | [Overlap-Add DSP](04_overlap_add_dsp.md) | OLA crossfader + unit tests | ☑ |
| 05 | [Virtual cable loopback](05_virtual_cable_loopback.md) | Mic → fluxvoice → virtual cable → recorder | ◐ guard + `fv_probe` landed; cable check pending |
| 06 | [Latency & stability verification](06_latency_verification.md) | Measured latency budget + soak-test report | ◐ instruments landed; measurements pending |

## Acceptance criteria

- [ ] `fluxvoice --passthrough` runs glitch-free (no audible clicks/dropouts) for ≥ 30 min
- [ ] Queue-path loopback works with `input_overruns == 0` and bounded `output_underruns` during steady state
- [ ] Output routed to a virtual audio cable is recordable by a second application
- [ ] Measured input→output latency documented (target ≤ 60 ms on the passthrough path)
- [x] OLA unit tests pass; no heap allocation or mutex inside the audio callback

## Key constraints (real-time audio rules)

1. The miniaudio callback is a **real-time thread**: no `new`/`malloc`, no
   `std::mutex`, no `std::cout`/logging, no blocking calls. Violations cause
   priority inversion → glitches (see RESEARCH_GUIDE.md §4).
2. All sample hand-off between callback and worker goes through the
   `readerwriterqueue` SPSC queues — **single producer, single consumer** each
   direction.
3. Preallocate all buffers at `start()`/construction time.
4. `std::atomic` only for flags/counters shared with UI or worker threads.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| `ma_device_id` lifetime/ownership bugs | Copy ids into locals at `start()`; never retain `ma_device_info*` pointers |
| Sample-rate mismatch across endpoints | Fix everything to 16 kHz mono in M1; resampling is M4 polish |
| Callback overruns on loaded systems | 20 ms period default; expose `--sample-rate`/period config; count overruns instead of crashing |
| Underrun silence masking bugs | Count underruns atomically; report on exit; test with worker stalled |
| Callback sizes differ from the requested 20 ms period | Period size is a hint: handle any `frame_count`; build 80 ms chunks from accumulated samples, not callback counts |
| Capture/playback clocks drift on different devices | One duplex device keeps `out_q_` occupancy constant, so drift hides inside miniaudio/the OS ([Stage 06 finding](06_latency_verification.md#finding-out_q_-occupancy-cannot-show-clock-drift)); measure the route's delay over time with `fv_probe latency`; resample only if measured |

## References

- [Cross-platform audio I/O research](../Research_documents/CROSS_PLATFORM_AUDIO_IO_RESEARCH.md) — miniaudio 0.11.25 boundaries per OS, client vs. backend rates, period-size hints, clock drift (§5, §8).
- `RESEARCH_GUIDE.md` §4 — miniaudio device enumeration & duplex examples; why lock-free queues (priority inversion); sherpa-onnx for later stateful-ONNX patterns.
- `README.md` §4 — threading model and OLA reference implementation.
- `miniaudio.h` header docs — `ma_context_get_devices`, `ma_device_config_init(ma_device_type_duplex)`, `dataCallback` contract.
