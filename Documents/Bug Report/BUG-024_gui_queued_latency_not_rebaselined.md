# BUG-024 — The GUI's queued-latency figure is not rebaselined after a stop or reconfigure

> [Bug index](00_index.md) · Found by: [REVIEW-M4-FRONTEND_2026-11-14](REVIEW-M4-FRONTEND_2026-11-14.md)

| | |
|---|---|
| **Tag** | `BUG-024` |
| **Name** | *Sticky residue* — discarded samples stay in the latency readout |
| **Reported** | 2026-11-14 |
| **Severity** | Medium (the latency shown in the status bar, Live card, Mini panel, tray tooltip and `[stats]` line is wrong after the first restart and drifts further with each one) |
| **Status** | ☐ open |
| **Component** | `frontend/core/EngineHost.cpp` (`telemetry()`), `frontend/core/Telemetry.cpp` |
| **Blocks** | [Milestone 4 / Stage 06](../Milestone%204/06_live_latency_telemetry.md) |

## Summary

`EngineHost::telemetry()` computes

```cpp
queued = output_underruns − input_overruns − output_drops;   // EngineHost.cpp:63-64
```

from the router's lifetime counters and the host's lifetime `output_drops_`.
The identity is exact only **per stream**. `MEMORY.md` §2 states the rule:
"Rebaseline after queue discards".

Every stop, reconfigure and recovery discards both SPSC queues
(`AudioController.cpp:107`, `last_discarded_samples_ = router_.discard_queues()`)
and resets the chunk assembler. The samples that were resident are gone, but
the counters that described them are not reset, and the host never reads
`last_discarded_samples()`. The residue of each stream is carried into the
next.

Two further gaps in the same function:

- The figure is shown unqualified in system-playback loopback mode, where the
  capture and playback callbacks are independently clocked and the identity
  does not hold (`MEMORY.md` §2).
- It has no `prefilled` / `skipped` terms. That is correct for today's
  pass-through and wrong as soon as the GUI runs `ModeAWorker`
  ([Stage 05](../Milestone%204/05_gui_engine_host_mode_a.md)).

## Failure scenario (by reading; not executed)

Run with 40 ms queued. Press Stop, then Start. The new stream starts with
empty queues, but the readout starts at 40 ms plus whatever the new stream
adds. After several "Apply and restart" cycles in Devices the displayed
figure can exceed the 5120-sample cap the worker enforces, which is
impossible for a real queue.

## Expected

After any queue discard the readout starts from the new stream's baseline.

## Root cause

The engine side already does this: `ModeAWorker::reset()` stores
`underrun_base_` / `overrun_base_` and clears its drop, skip and prefill
counters (`ModeAWorker.hpp:53-61`). The GUI host predates it and kept the M1
diagnostic formula without the baseline.

## Suggested fix

- In `EngineHost::reset_dsp()` (called by the controller after the worker is
  quiesced and the queues are discarded): store the router's current underrun
  and overrun counts as baselines and zero `output_drops_`.
- `telemetry()` subtracts the baselines. Lifetime totals stay available as
  separate fields for the Telemetry panel.
- In loopback mode show the figure as "n/a (independent clocks)".
- When Stage 05 lands, take `queued` from `WorkerSnapshot` in Mode A.

## Regression test

Over `FakeAudioBackend`: run until `queued_samples` is non-zero, stop, start,
and assert `queued_samples == 0` before the first callback of the new stream.
Repeat through `reconfigure` and through a simulated device-loss recovery.

## Workaround

Restart the application to reset the readout. Audio is not affected; only
the displayed number is wrong.
