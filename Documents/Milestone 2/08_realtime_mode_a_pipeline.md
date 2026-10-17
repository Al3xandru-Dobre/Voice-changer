# Stage 08 — Low-latency I/O & real-time Mode A pipeline

> **Milestone 2** · [Master plan](00_master.md) · [← Stage 07](07_ort_cpp_runtime.md) · [Stage 09 →](09_voice_controls_presets.md)

**Status:** ☐ not started
**Depends on:** Stage 03 (C++ pitch tracker), Stage 07 (`OrtModel`), Stage 06 (real
bundle, for the live runs only). Live latency runs also need a working cable route
([M1 / Stage 06](../Milestone%201/06_latency_verification.md),
[BUG-001](../Bug%20Report/BUG-001_voicemeeter_vaio_routing_wedge.md)).
**Research:** §3 latency · §10 real-time scheduling & fluent stream

## Objective

`fluxvoice --vc` converts live microphone audio into the target voice as a
**fluent stream**:
- the smallest stable device period;
- model geometry taken from the bundle, not compile-time constants;
- warm-up before the stream starts;
- a latency governor, so occasional spikes don't become permanent delay;
- soft underrun edges instead of clicks;
- an elevated worker priority;
- per-hop timing telemetry in the existing `[stats]` line.

## Context & references

- **Device period.** `AudioEngine::kDefaultPeriodMs = 20` is a `constexpr`
  (`src/audio/AudioEngine.hpp:46`), applied at `:97`. The period is a hint; the
  backend reports the real one in `StreamInfo`.
- **The worker** is an inline lambda in `src/main.cpp:253-284`. In `--chunked`
  mode it builds `[context | hop]` frames with `ChunkAssembler` and emits the hop
  unchanged (`:277-280`), which is the insertion point. `InferencePipeline` is
  header-less and unused (`src/pipeline/InferencePipeline.cpp:22-49`).
- **CLI parsing** handles only unsigned options (`src/main.cpp:168-179`).
  `tools/fv_probe.cpp` has string options (`--input-name`, `--csv`) to copy the
  pattern from.
- **Underruns are permanent latency** (M1 / Stage 06). The callback zero-fills
  on underrun (`src/audio/DuplexRouter.hpp:74-80`), a hard step.
- **The `queued` identity** `underruns − overruns − drops` lives in
  `src/main.cpp:128` and in `tools/analyze_soak.py` (docstring and checks). Any
  new way of discarding audio must join it.
- **Stdin.** The stop thread at `src/main.cpp:289-291` treats Enter as stop.
  Stage 09 extends it; this stage keeps it.
- **Shared ownership.** [M1.5 / Stage 05](../Milestone%201.5/05_refresh_recovery_controller.md)
  owns device switching and "bound queued-audio latency". It reuses this stage's
  governor and `reset()` contract.

## Tasks

- [ ] **Low-latency I/O first** (no model needed):
  - [ ] `--period-ms <5..40>` replaces the constexpr at `open()`. When ≤ 10 ms,
        also set `cfg.performanceProfile = ma_performance_profile_low_latency`.
        Print requested vs. actual internal period.
  - [ ] Measure passthrough and queue-loopback T−B at 10 and 20 ms with
        `fv_probe latency` and record the floors here. They decide whether a
        20 ms hop pays off. M1.5 / Stage 02's `StreamConfig` may later absorb this
        option.
- [ ] **CLI**, with a small string-option helper following fv_probe's pattern:
  - [ ] `--vc`
  - [ ] `--model-dir <bundle>`
  - [ ] `--voice <name>`
  - [ ] `--frames-per-step <1|2|4>`
  - [ ] `--latency-target-ms <ms>` (default computed, see governor)
  - [ ] `--list-voices`

  Rules:
  - `--sample-rate` must be omitted or equal the contract rate.
  - Models load and validate **before** any device opens, so CLI tests run
    without hardware.
  - Add `cli_rejects_*` cases: VC flags without `--vc`, missing bundle, unknown
    voice, bad frames-per-step.
- [ ] **`src/pipeline/InferencePipeline.{hpp,cpp}`:**
  - [ ] A `ModelStep` interface (`OrtModel` implements it; tests use a fake).
  - [ ] `PipelineGeometry {sample_rate, frame_samples, frames_per_step, hop,
        lookahead_frames, overlap_samples}` built from the validated contract
        pair. `PipelineTiming` stays only for `--chunked`.
  - [ ] Chain per hop:
    1. high-pass filter
    2. `PitchTracker` (Stage 03)
    3. feature delay line of L frames, aligning with the content lookahead
    4. encoder step
    5. map f0/energy into the voice's domain (`RunningNorm` + `voices.json`
       stats; `VoiceParams` hook for Stage 09)
    6. vocoder step
    7. `OverlapAddProcessor` only if `overlap_samples > 0`
    8. output gate hook (Stage 09)
    9. emit
  - [ ] `reset()` contract: called on the worker thread only, after the stream is
        quiesced. Resets model states, tracker, normaliser (to priors), OLA,
        governor.
  - [ ] `fv_vc_file` (Stage 07) switches to `InferencePipeline`.
- [ ] **Worker wiring.** Add `Mode::VC`; the worker drives `InferencePipeline` with
      a `ChunkAssembler(hop, context = 0)`. Load, validate and `warm_up()` every
      session **before** `engine.start()`.
- [ ] **`src/pipeline/LatencyGovernor.hpp`:**
  - [ ] Target occupancy is `--latency-target-ms`, by default algorithmic
        latency + compute budget + margin (master budget).
  - [ ] At startup, pre-fill the output queue with silence to the target level.
  - [ ] When `queued` stays above target + one hop for N hops, drop output
        samples **only while the output is quiet** (gate closed / low energy).
        Above a hard cap (target + 150 ms), drop regardless.
  - [ ] Count drops in a new `skipped` counter. The identity becomes `queued =
        underruns − overruns − drops − skipped`; update `src/main.cpp`,
        `stats_line` and `tools/analyze_soak.py` (FIELDS, docstring, self-test).
- [ ] **Underrun fade** in `DuplexRouter::process`. On underrun, ramp from the
      last output sample to 0 over about 2 ms; on resume, ramp in from 0. Only a
      few scalars of state. `test_duplex_router`'s no-allocation and counter
      tests still pass, plus a new test bounding the step at underrun edges.
- [ ] **Worker priority.**
  - [ ] Windows: `AvSetMmThreadCharacteristicsW(L"Pro Audio")` (link `avrt`),
        reverted on exit.
  - [ ] Linux: best-effort `SCHED_FIFO` (low priority) or nice −10, logging the
        result.
  - [ ] macOS: `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)`.
- [ ] **Telemetry.**
  - [ ] Per-hop `steady_clock` timing into a fixed-bin atomic histogram (50 µs
        bins up to 2× hop).
  - [ ] New `[stats]` fields: `inf_p50_us`, `inf_p95_us`, `inf_max_us`, `misses`
        (hops slower than the hop duration), `skipped`, `errors` (failed model
        steps), and `lat_ms` (in-process estimate = algorithmic + queued +
        reported device periods).
  - [ ] `analyze_soak.py --vc`: requires `misses` flat after warm-up and
        `queued` within target ± one hop. Underruns may still occur at startup.
- [ ] **Tests** (`tests/test_inference_pipeline.cpp`, `tests/test_latency_governor.cpp`;
      fake model, so CI runs them without ORT):
  - [ ] frames_per_step 1/2/4 produce identical output streams for an identity
        fake
  - [ ] an impulse comes out aligned with the declared lookahead
  - [ ] `reset()` makes the pipeline equal a freshly constructed one
  - [ ] no allocation in steady state (operator-new counter, fake model)
  - [ ] the governor drops only during quiet output, respects the hard cap, and
        keeps the identity exact

## Implementation notes

- **Context 0 in VC mode.** The stateful models and the tracker keep their own
  history. `ChunkAssembler`'s 320-sample context belonged to a stateless design
  and would only cost memory traffic.
- **Drop output during silence, never input.** Skipping model input would break
  state continuity. Dropping output samples while the gate is closed is
  inaudible and keeps the models' timeline intact.
- **Measure the plumbing through an identity bundle.** Stage 02's
  `identity_s{1,2,4}` fixtures pass the chirp through unchanged, so `fv_probe
  latency` measures the real VC plumbing (framing, queues, governor) before any
  model sounds different. Stage 11 adds an envelope marker for the real model.
- **README §4 lists the worker as "Normal" priority.** That is wrong after this
  stage; Stage 11 updates the root README pointer.
- **No inference failure takes the stream down.** If `step()` reports an error,
  the hop is output as faded silence and an `errors` counter increments. After K
  consecutive errors the app stops with the ORT message.

## Files touched

- `src/audio/AudioEngine.hpp` (period option, performance profile),
  `src/audio/DuplexRouter.hpp` (underrun fade), `src/main.cpp` (VC mode, CLI,
  warm-up, telemetry, identity), `src/pipeline/InferencePipeline.hpp` (new),
  `src/pipeline/InferencePipeline.cpp` (rewritten), `src/pipeline/LatencyGovernor.hpp`
  (new), `tools/fv_vc_file.cpp`, `tools/analyze_soak.py`, `CMakeLists.txt`
  (tests, `avrt`), `tests/test_inference_pipeline.cpp` (new),
  `tests/test_latency_governor.cpp` (new), `tests/test_duplex_router.cpp`,
  `Documents/Milestone 1.5/05_refresh_recovery_controller.md` (reuse note)

## Definition of done

- **Fake-model CI tests pass** for frames_per_step 1/2/4: lookahead alignment,
  `reset()` equivalence, steady-state zero allocation, governor behaviour.
- **Identity bundle via `fluxvoice --vc`:** `fv_probe latency` T−B equals
  queue-loopback T−B + hop ± 2 ms.
- **10-minute live run with the real bundle:**
  - 0 `misses` after warm-up;
  - `inf_p95_us` ≤ 50% of the hop;
  - `queued` within target ± one hop.
- Startup underruns ≤ one hop + one device period.
- Passthrough/queue floors at 10 ms and 20 ms periods are recorded in this file.
