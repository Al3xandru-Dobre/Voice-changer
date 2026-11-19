# Stage 06 — Live latency & per-stage telemetry

> **Milestone 4** · [Master plan](00_master.md) · [← Stage 05](05_gui_engine_host_mode_a.md) · [Stage 07 →](07_mode_b_translation_wiring.md)

**Status:** ◐ code complete and Linux-verified 2026-11-19 (GCC + Clang, with
and without ORT); Windows MSVC / macOS AppleClang NOT RUN; the live
10-minute run and a soak log from a real stream NOT RUN (audio hardware) —
see [evidence](#evidence-2026-11-19)
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

- [x] **Per-stage timing in the engine** (`src/pipeline/InferencePipeline.*`):
  - [x] One `TimingHistogram` per stage: features (YIN + whitening), encoder
        `Run`, vocoder `Run`, output chain, plus the existing whole-hop one.
  - [x] `InferencePipeline::stage_timing()` returns a fixed-size
        `std::array<TimingSnapshot, N>` with stable stage names. No allocation.
  - [x] The clock reads stay on the worker thread; the UI reads the atomics.
- [x] **`TelemetrySnapshot` extension:** stage snapshots, whole-hop snapshot,
      deadline misses, `skipped`, `prefilled`, governor target and hard cap,
      hop length, algorithmic latency (framing + lookahead from the contract).
- [x] **Correct queued latency:** `EngineHost::telemetry()` takes `queued`
      from `WorkerSnapshot` in Mode A. Bypass mode keeps the M1 identity,
      rebaselined after every queue discard
      ([BUG-024](../Bug%20Report/BUG-024_gui_queued_latency_not_rebaselined.md)).
- [x] **Settle the estimate's formula first.** The CLI's `lat_ms` adds the
      framing hop twice, and the default queue target includes model
      lookahead ([BUG-026](../Bug%20Report/BUG-026_latency_target_counts_lookahead_twice.md)).
      The GUI must show the corrected value, not copy the current one.
- [x] **End-to-end estimate card (Live + Telemetry):** move the CLI's `lat_ms`
      computation into one shared function (in `ModeAHost`) and show the same
      number in the GUI, with its terms broken out (device periods, queued,
      algorithmic) and compute p95 beside it. Label it "estimated". The
      measured value comes only from `fv_probe latency` across two cables
      (Stage 12).
- [x] **Telemetry panel:** a table of stages (p50 / p95 / max in ms, share of
      the hop), a 60 s compute sparkline, misses, skipped samples, governor
      target, and the identity check row using the full identity.
- [x] **Copy `[stats]` line** produces exactly the line `src/main.cpp` prints
      in `--vc` mode, so `tools/analyze_soak.py` parses GUI soak logs.
- [x] **Optional stats log:** Settings entry "Write stats log" appends the
      `[stats]` line once per second to a file from the UI thread.
- [x] **Tests:**
  - [x] Stage histograms: a fake clock drives known durations; p50/p95/max and
        the miss count match by construction (extends `test_timing_histogram`).
  - [x] `telemetry().queued_samples` equals the worker snapshot with
        `prefilled > 0` and `skipped > 0` (fails with today's formula).
  - [x] `stats_line` round-trips through `tools/analyze_soak.py --self-test`
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

## As implemented (2026-11-19)

- **Stages.** `PipelineStage` gained `kFeatures` (`feat`: per-frame tracker /
  whitening / conditioning loop; in Mode B the per-frame VAD, tracker and
  control ramps) and `kOutputChain` (`chain`: `set_targets` + `process`),
  appended so the existing indices stay stable (`kPipelineStageCount` = 6).
  `stage_timings()` returns the six snapshots as a `std::array` (any thread,
  no allocation); the whole-hop histogram is unchanged. `fluxvoice`'s Mode B
  end-of-run summary now lists `feat` and `chain` too; `fv_s2ut_file` keeps
  its four model stages.
- **Clock.** `PipelineOptions::clock_us` (nullptr = `steady_clock` µs) lets
  the tests drive known durations through `FakeModelStep::step_us`; the
  per-stage cases match p50 / p95 / max and misses by construction.
- **One formatter, one estimate.** `src/pipeline/HostStats.hpp`:
  `format_stats_line` is the CLI line moved verbatim (pinned by literal golden
  strings in `test_mode_a_host`), `estimate_latency` keeps the CLI's exact
  `lat_ms` expression and exposes its terms. Settled formula (BUG-026): model
  delay + queued (framing hop included, once) + capture and playback device
  periods; compute time and the algorithmic latency are shown beside the
  estimate, never added to it. `ModeAHost::stats()` fills a `HostStats` for
  the CLI and the GUI; the diagnostic CLI modes and the GUI's Bypass fill the
  legacy fields only (`vc = false`).
- **Governor.** `InferencePipeline::governor_target_samples()` /
  `governor_hard_cap_samples()` mirror the effective values (Mode B burst
  cushion included) for any thread, refreshed per hop, on `reset()` and on
  `set_params()`.
- **Identity check.** `ModeAWorker` publishes `held` (read from the input
  queue, not yet written, dropped or skipped). The Telemetry row compares the
  identity (`queued`) with measured occupancy (input queue + output queue +
  held) and reads "consistent" when the difference is within one device
  period plus one hop — the two sides are sampled at different instants while
  running; at quiescence they are equal. A larger difference is an
  accounting error, not clock drift (the panel says queue occupancy cannot
  show drift).
- **GUI.** Telemetry: stage table (p50 / p95 / max in ms, share of the hop),
  whole-hop row, 60 s compute-p95 sparkline, misses, skipped / prefilled,
  governor target and hard cap, identity row, the `[stats]` line with Copy.
  Live: "Estimated end-to-end latency" with its terms broken out and compute
  p95 beside it. Settings: "Write stats log" (appends once per second of
  stream time from the UI thread; a write error turns it off with a notice).
- **Format check.** `tools/analyze_soak.py --check-format <log>` validates
  every `[stats]` line (legacy fields; complete VC / S2UT field sets; finite
  values; non-decreasing `t`) without judging soak criteria. CTest
  `gui_stats_parse` runs it on the lines `frontend_model` produces through the
  GUI's real `stats_line` path (Bypass and Mode A).

## Evidence (2026-11-19)

Same machine and base as [Stage 04](04_build_gate_frontend_hardening.md#evidence-2026-11-19);
`cmake --workflow --preset <p>` (all targets, strict warnings as errors).

| Leg | Result |
|---|---|
| Linux GCC 13.3.0 / Clang 18.1.3, ORT-less | PASS, 123/123 each (incl. `timing_histogram`, `inference_pipeline`, `mode_a_host`, `frontend_model`, `gui_stats_parse`) |
| Linux GCC 13.3.0 / Clang 18.1.3, ORT 1.31.0 | PASS, 125/125 each (also `gui_smoke_bundle`) |
| Goldens (`ctest --preset linux-gcc-ort-training -R "golden\|stage09_controls"`) | PASS, 4/4 (`stage07_golden`, `stage09_controls`, `m3_stage07_golden`, `s2ut_golden_selftest`) |
| `analyze_soak.py --self-test` / `--check-format` on the GUI sample (`9 lines: legacy 4, vc 5`) | PASS |
| Clock overhead: `steady_clock::now()` 15–29 ns; 4 reads added per hop → 0.0003–0.0006 % of a 20 ms hop | measured |
| `fv_vc_file`, identity fixture, 30 s × 40, 15 alternating runs: wall median 2.887 s → 2.906 s, hop p50 45.9 → 45.4 µs, p95 56.5 → 59.7 µs | within noise |
| Windows MSVC, macOS AppleClang | NOT RUN (no toolchain; tree not pushed) |
| 10-minute live run with a governor skip (identity row), GUI soak log from a real stream through `analyze_soak.py`, panels observed with a running engine | NOT RUN (audio hardware; no trained bundle) |

Tests that can fail (rule broken → failing check; each restored):

| Rule broken | Failing test |
|---|---|
| `feat` bracket closed after the encoder; Mode B `chain` bracket opened at the features start | `inference_pipeline` (fake-clock cases) |
| misses counted at `>=` | `timing_histogram`, `inference_pipeline` |
| governor mirror not republished in `set_params` | `inference_pipeline` |
| `occupancy` without `held`; `held` not refreshed after a hop | `mode_a_host`, `mode_a_worker` |
| `format_stats_line` precision changed | `mode_a_host` (three golden lines) |
| stage timings only with a translator (the old CLI behaviour) | `mode_a_host`, `frontend_model` |
| Mode A queued without `skipped` | `frontend_model` |
| GUI `stats_line` with its own format | `frontend_model` |
| latency card adds compute p95; Bypass card gets a model-delay term | `frontend_model` |
| identity verdict ignores the tolerance | `frontend_model` |
| stats log reopened per line; no rebase on reconfigure; write error not disabling it | `frontend_model` |
| `--check-format` VC / S2UT completeness, `skipped`/`prefilled`, monotonic `t`, empty file, parse errors | `analyze_soak_selftest` |
