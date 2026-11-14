# Stage 06 — Live latency & per-stage telemetry

> **Milestone 4** · [Master plan](00_master.md) · [← Stage 05](05_gui_engine_host_mode_a.md) · [Stage 07 →](07_mode_b_translation_wiring.md)

**Status:** ☐ not started
**Depends on:** [Stage 05](05_gui_engine_host_mode_a.md) (GUI runs `ModeAHost`)
**Owner persona:** `@ui-telemetry`

## Objective

Meet the exit criterion "per-stage + end-to-end latency visible live". The
Telemetry and Live panels show measured compute time per stage, deadline
misses, the governor's state and the queued-audio latency, all read through
lock-free snapshots.

## Context & verified starting point (2026-11-14)

- `frontend/core/Telemetry.hpp` carries frames, overruns, underruns, drops,
  queued samples, RSS, meters and two parameter generations. It has no timing
  fields.
- `EngineHost::telemetry()` computes `queued = underruns − overruns − drops`
  (`EngineHost.cpp:63-64`). The Mode A identity also has `prefilled` and
  `skipped` (`MEMORY.md` §2, `ModeAWorker::snapshot()`); the GUI formula is
  wrong as soon as the governor prefills or skips.
- The engine already measures what is needed: `InferencePipeline::timing()`
  returns a `TimingSnapshot` (samples, p50, p95, max, misses), and
  `WorkerSnapshot` carries `skipped` and `prefilled`.
- `InferencePipeline` owns one `TimingHistogram` for the whole hop
  (`InferencePipeline.cpp:249`, recorded at `:402`). There is no per-stage
  split (features, encoder, vocoder, output chain).
- The CLI already prints an estimate: `lat_ms` = algorithmic samples + queued
  + capture and playback device periods (`src/main.cpp:239-246`), and the
  `--vc` `[stats]` line adds `inf_p50_us`, `inf_p95_us`, `inf_max_us`,
  `misses`, `errors` and `lat_ms`. The GUI's `stats_line` has none of these.

## Tasks

- [ ] **Per-stage timing in the engine** (`src/pipeline/InferencePipeline.*`):
  - [ ] One `TimingHistogram` per stage: features (YIN + whitening), encoder
        `Run`, vocoder `Run`, output chain, plus the existing whole-hop one.
  - [ ] `InferencePipeline::stage_timing()` returns a fixed-size
        `std::array<TimingSnapshot, N>` with stable stage names. No allocation.
  - [ ] The clock reads stay on the worker thread; the UI reads the atomics.
- [ ] **`TelemetrySnapshot` extension:** stage snapshots, whole-hop snapshot,
      deadline misses, `skipped`, `prefilled`, governor target and hard cap,
      hop length, algorithmic latency (framing + lookahead from the contract).
- [ ] **Correct queued latency:** `EngineHost::telemetry()` takes `queued`
      from `WorkerSnapshot` in Mode A. Bypass mode keeps the M1 identity,
      rebaselined after every queue discard
      ([BUG-024](../Bug%20Report/BUG-024_gui_queued_latency_not_rebaselined.md)).
- [ ] **Settle the estimate's formula first.** The CLI's `lat_ms` adds the
      framing hop twice, and the default queue target includes model
      lookahead ([BUG-026](../Bug%20Report/BUG-026_latency_target_counts_lookahead_twice.md)).
      The GUI must show the corrected value, not copy the current one.
- [ ] **End-to-end estimate card (Live + Telemetry):** move the CLI's `lat_ms`
      computation into one shared function (in `ModeAHost`) and show the same
      number in the GUI, with its terms broken out (device periods, queued,
      algorithmic) and compute p95 beside it. Label it "estimated". The
      measured value comes only from `fv_probe latency` across two cables
      (Stage 12).
- [ ] **Telemetry panel:** a table of stages (p50 / p95 / max in ms, share of
      the hop), a 60 s compute sparkline, misses, skipped samples, governor
      target, and the identity check row using the full identity.
- [ ] **Copy `[stats]` line** produces exactly the line `src/main.cpp` prints
      in `--vc` mode, so `tools/analyze_soak.py` parses GUI soak logs.
- [ ] **Optional stats log:** Settings entry "Write stats log" appends the
      `[stats]` line once per second to a file from the UI thread.
- [ ] **Tests:**
  - [ ] Stage histograms: a fake clock drives known durations; p50/p95/max and
        the miss count match by construction (extends `test_timing_histogram`).
  - [ ] `telemetry().queued_samples` equals the worker snapshot with
        `prefilled > 0` and `skipped > 0` (fails with today's formula).
  - [ ] `stats_line` round-trips through `tools/analyze_soak.py --self-test`
        with a GUI-produced line added to the self-test corpus.

## Implementation notes

- A per-stage histogram costs four clock reads per hop. That is acceptable at
  20–40 ms hops; measure the overhead once and record it.
- `out_q_` occupancy still cannot show hardware clock drift. The panel must
  not present queue occupancy as a drift measurement.
- Meters and timing are published by the worker with relaxed atomics. The UI
  may see values from adjacent hops in one frame; that is fine for display.

## Files touched

- `src/pipeline/InferencePipeline.{hpp,cpp}`, `src/pipeline/TimingHistogram.hpp`,
  `src/pipeline/ModeAHost.{hpp,cpp}`, `frontend/core/Telemetry.{hpp,cpp}`,
  `frontend/core/EngineHost.cpp`, `frontend/imgui/ViewTelemetry.cpp`,
  `frontend/imgui/ViewLive.cpp`, `tests/test_timing_histogram.cpp`,
  `tests/test_inference_pipeline.cpp`, `tests/test_frontend_model.cpp`,
  `tools/analyze_soak.py`

## Definition of done

- While Mode A runs in the GUI, the Telemetry panel shows per-stage p50/p95/max,
  misses and the governor state, updating at least once per second.
- The identity row reads 0 difference during a 10-minute run that includes at
  least one governor skip.
- A GUI-produced `[stats]` log passes `tools/analyze_soak.py`.
- Steady-state hops still allocate 0 bytes (`AllocationCounter`).
