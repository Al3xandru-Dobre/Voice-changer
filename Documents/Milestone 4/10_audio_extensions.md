# Stage 10 — Audio extensions: noise suppression, 24/48 kHz output, monitor mix

> **Milestone 4** · [Master plan](00_master.md) · [← Stage 09](09_quantization_fp16_int8.md) · [Stage 11 →](11_desktop_integration.md)

**Status:** ☐ not started
**Depends on:** [Stage 05](05_gui_engine_host_mode_a.md), [Stage 06](06_live_latency_telemetry.md);
[M2 / Stage 11](../Milestone%202/11_verification_exit.md) latency numbers as the baseline
**Owner persona:** `@rt-audio-dsp`, with `@ml-speech-trainer` for the bandwidth-extension decision

## Objective

Deliver the three items Milestone 2 moved here: input noise suppression,
24/48 kHz output (or bandwidth extension), and a delay-aligned monitor mix.
Each is optional at run time, off by default, and has a stated latency cost.

## Context & verified starting point (2026-11-14)

- None of the three exists in `src/` or `frontend/` (no suppressor, no
  resampler or bandwidth-extension model, no monitor path).
- The engine runs one duplex rate: encoder input rate = vocoder output rate =
  device client rate (M2 key constraint 4). Bundles are 16 kHz.
- M2 excluded a dry/wet mix slider because mixing dry and converted audio at
  different delays comb-filters. `AGENTS.md` keeps that rule for the UI.
- No research report covers these three items. The first task of each
  sub-stage is a short, dated decision note in this file.

Because the three items are independent, they may land separately; this file
tracks each with its own status line.

## 10a — Input noise suppression (☐)

- [ ] **Decision note:** candidate suppressors, their licence, frame size,
      added latency and CPU cost at 16 kHz on one thread. Reject any candidate
      with lookahead that breaks the Mode A budget or with a licence that
      taints the default bundle.
- [ ] Where it runs: on the worker, before the feature extractor and the
      encoder, on the same hop. Never in the audio callback.
- [ ] `noise_suppress` (toggle) and, if the algorithm has one, a strength
      parameter: new `kParams` rows, so the CLI, presets and the GUI get them
      from the descriptor table.
- [ ] State reset with the pipeline (`reset()`), preallocated buffers, 0
      allocations per hop.
- [ ] **Quality check:** WER and similarity on the frozen list with added
      noise at 20 / 10 / 5 dB SNR, suppression on and off. The gate threshold
      and the governor's "quiet" detection must still work with it on.
- [ ] Tests: 0-allocation hop, reset equivalence (a reset stream equals a
      fresh one), bypass is bit-exact when off.

## 10b — 24/48 kHz output or bandwidth extension (☐)

- [ ] **Decision note:** three routes, compared on latency, CPU and quality:
      (1) keep 16 kHz models and resample at the device boundary;
      (2) a causal bandwidth-extension stage after the vocoder;
      (3) a vocoder that emits 24 kHz from the same units.
      Route (1) adds no audio bandwidth; say so. Routes (2) and (3) need
      training work and belong to `python_training/`.
- [ ] The contract must express the output rate if it differs from the input
      rate. Today's validators enforce one rate; this is a contract change
      with fixtures in both validators.
- [ ] `AudioController` and `DuplexRouter` assume one client rate for both
      directions. Decide between two client rates and a worker-side
      resampler; either way the conservation identity must be restated for
      unequal rates.
- [ ] Tests: resampler or extension stage is causal (no lookahead beyond the
      declared amount), streaming equals offline, 0 allocations per hop.

## 10c — Delay-aligned monitor mix (☐)

- [ ] **Decision note:** what the monitor is for (hearing oneself converted,
      or checking sync against the dry voice), and which device it plays on.
      The converted stream to the virtual cable must stay untouched.
- [ ] The dry path is delayed by the pipeline's current latency (algorithmic
      + queued, from Stage 06) before mixing, so dry and converted are
      aligned. Without alignment the mix combs; that is the reason M2
      excluded dry/wet.
- [ ] Monitor output goes to a **second, explicitly selected** playback
      device. It never replaces the main output, and the feedback guard runs
      on it too.
- [ ] UI: a monitor device selector (endpoint key + generation), a monitor
      level, and a clear off state. No dry/wet slider on the main output.
- [ ] Tests: alignment within ±1 ms on a click train through the fake
      pipeline; monitor disabled leaves the main output bit-identical.

## Implementation notes

- Every item costs latency or compute. Each DoD below requires the measured
  cost, so the user-facing help text can state it.
- Keep each feature behind a descriptor row or a Settings entry. Presets that
  do not mention a new parameter leave it unchanged.

## Files touched (expected)

- `src/dsp/**` (new suppressor / resampler), `src/pipeline/InferencePipeline.*`,
  `src/pipeline/ModelContract.hpp`, `python_training/fvml/contract.py`,
  `src/voice/VoiceParams.hpp`, `src/audio/AudioController.*` (monitor device),
  `frontend/core/**`, `frontend/imgui/ViewDevices.cpp`, `ViewVoices.cpp`,
  `tests/**`, `presets/README.md`

## Definition of done

- Each landed item: off by default, a recorded decision note, measured added
  latency and per-hop cost, 0 allocations per hop, hardware-free tests, and
  the quality check listed in its section.
- An item that fails its decision note (latency, licence or quality) is
  closed as "evaluated, not shipped" with the numbers. That also satisfies
  this stage.
