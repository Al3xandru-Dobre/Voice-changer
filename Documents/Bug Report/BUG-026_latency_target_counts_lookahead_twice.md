# BUG-026 — The default latency target counts model lookahead twice

> [Bug index](00_index.md) · Found by: [REVIEW-M2-S07-S09_2026-11-14](REVIEW-M2-S07-S09_2026-11-14.md)

| | |
|---|---|
| **Tag** | `BUG-026` |
| **Name** | *Double lookahead* — the queue is prefilled for a delay the model already adds |
| **Reported** | 2026-11-14 |
| **Severity** | Medium (projected `T − B` is one lookahead above the master budget: 100 ms instead of 80 ms for A1, 140 ms instead of 100 ms for A2; no effect at lookahead 0) |
| **Status** | ◐ diagnosed by arithmetic — not measured (no trained bundle, no live run) |
| **Component** | `src/pipeline/InferencePipeline.cpp:89-92`, `src/pipeline/PipelineGeometry.hpp:21-23`, `src/pipeline/LatencyGovernor.hpp:44-48`; spec text in `Documents/Milestone 2/08_realtime_mode_a_pipeline.md:96-97` |
| **Blocks** | [Milestone 2 / Stage 08](../Milestone%202/08_realtime_mode_a_pipeline.md) live acceptance, [Stage 11](../Milestone%202/11_verification_exit.md) (`T − B` Must line) |

## Summary

The governor's default target, which is also the startup silence prefill, is

```cpp
LatencyGovernor::default_target(g.algorithmic_samples(), g.hop, g.sample_rate)
// = algorithmic + hop/2 + margin
// algorithmic_samples() = hop + lookahead_frames * frame_samples + overlap_samples
```

The target is a **queue occupancy**: `queued` counts samples resident between
capture and playback (`prefilled + underruns − overruns − drops − skipped`).
The framing hop is real occupancy, because the samples wait in the chunk
assembler. Compute time and the jitter margin need occupancy too.

Model lookahead is different. A model with lookahead `L` still consumes one
hop and emits one hop per step. Its output at step *t* describes frame
*t − L*. That delay lives in the model's content, not in the queues, and it
is present whatever the queue level is. Adding `L × frame` to the occupancy
target therefore pays for the lookahead a second time.

## Arithmetic (16 kHz, hop 20 ms, 10 ms device periods; master budget table)

| Config | Queue target (code) | Model lookahead | I/O floor | Projected `T − B` | Master budget |
|---|---|---|---|---|---|
| A0: L=0 | 20 + 10 + 5 = 35 ms | 0 | 25 | **60 ms** | 60 ms |
| A1: L=1 | 20 + **20** + 10 + 5 = 55 ms | 20 | 25 | **100 ms** | 80 ms |
| A2: L=2 | 20 + **40** + 10 + 5 = 75 ms | 40 | 25 | **140 ms** | 100 ms (the Must line) |

`tests/test_latency_governor.cpp:47` pins the A1 queue target:
`default_target(640, 320, 16000) == 880` samples = 55 ms.
`tests/test_mode_a_worker.cpp:56` shows `queued == target` after prefill, and
only for lookahead 0, where the two readings agree.

The M2 gate expects the winner to be L=1. With this target the gated
configuration would sit exactly on the 100 ms Must line before any real
device adds its own delay, and L=2 would miss it.

## Expected

Queue target = framing hop + compute budget + margin (+ OLA overlap if
enabled). Lookahead appears once, as the model's own delay.

## Root cause

The Stage 08 stage file says "target occupancy is … by default algorithmic
latency + compute budget + margin". "Algorithmic latency" in the master
budget is framing **plus lookahead**; that sum is right for the total
`T − B` and wrong for the part of it that must be held in the queues. The
implementation followed the sentence literally, and the tests pin the result.

## Suggested fix

- `PipelineGeometry`: add `queue_samples() = hop + overlap_samples` and keep
  `algorithmic_samples()` for reporting.
- `InferencePipeline` passes `queue_samples()` to `default_target`.
- Correct the sentence in `08_realtime_mode_a_pipeline.md` and the pinned
  values in `test_latency_governor.cpp` (`default_target(320, 320, 16000)`
  is the s1 target for every lookahead).
- Related, same family: `lat_ms` in `src/main.cpp:239-246` adds
  `algorithmic_samples()` to `queued`. `queued` already contains the framing
  hop, so the printed estimate is one hop too high. Use
  `lookahead + overlap + queued + device periods`.

## Verification needed

Arithmetic only so far. Confirm with `fv_vc_file` on a fixture pair with
lookahead 1: the first non-silent output sample of an impulse should arrive
`L × frame` after the queue delay, independent of the prefill level. Then
confirm with `fv_probe latency` once a route and a bundle exist.

## Regression test

`ModeAWorker` over fake models with `lookahead_frames = 1`: send an impulse,
assert its output position equals `target + L × frame_samples` and that
`target` does not change when only the lookahead changes.

## Workaround

Pass `--latency-target-ms` explicitly with the lookahead subtracted (A1: 35).
