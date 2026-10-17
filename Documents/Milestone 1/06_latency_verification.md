# Stage 06 — Latency & stability verification

> **Milestone 1** · [Master plan](00_master.md) · [← Stage 05](05_virtual_cable_loopback.md)

**Status:** ◐ in progress — instruments landed (2026-10-17): `fv_probe latency`,
`fluxvoice --stats`, `tools/analyze_soak.py`. No measurements yet: they need a
virtual-cable route (Stage 05)
**Depends on:** Stages 02–05

## Objective

Quantify the M1 substrate: end-to-end latency, glitch counters under load, and
multi-hour stability. These numbers become the latency *floor* the neural
stages must budget against (Mode A target ~60–100 ms total).

## Tasks

- [ ] **Latency measurement.** Loop output back to a second input (physical
      jumper cable or a second cable pair), inject a click/impulse, measure
      samples between capture→playback edges. Report:
      - passthrough-mode latency
      - queue-mode latency
      - latency added by the virtual cable (Stage 05 setup vs. direct)
  - [x] Instrument: `fv_probe latency` (method below), detector unit-tested
        (CTest `latency_probe`); `tools/pw_latency_check.sh` automates it on Linux
- [ ] **Glitch audit under load.** Run queue-mode loopback while the machine
      is loaded (parallel build / `stress-ng`); record `input_overruns` and
      `output_underruns` deltas.
  - [x] Instrument: `fluxvoice --stats N` prints counters every N s;
        `tools/analyze_soak.py` reports the deltas after warm-up. (On Linux,
        `tools/pw_loopback_check.sh --load` from Stage 03 already does this.)
- [ ] **Soak test.** ≥ 30 min continuous loopback; counters stay flat after
      startup; RSS stable (no leaks in the RT path).
  - [x] Instrument: `--stats` lines carry `rss_kb`; `analyze_soak.py` checks
        counter deltas, RSS growth and the callback rate
- [ ] **Clock-drift check.** Repeat the soak with capture and playback on
      *different* devices (e.g. USB mic → virtual cable). ~~Sample `out_q_`
      occupancy periodically~~ — **this cannot show drift in the current
      design** (see below); measure the route's delay over time with
      `fv_probe latency` instead. Report the drift rate; do **not** mask it by
      enlarging buffers. *(Cross-platform research §8 — a resampling /
      clock-recovery stage is added only if this measures a problem.)*
- [ ] Write results into this file (table below) and update the master
      acceptance criteria.

## Finding: `out_q_` occupancy cannot show clock drift

fluxvoice opens **one miniaudio duplex device**. Every callback hands the
router `n` captured samples and asks for `n` playback samples. So the samples
held between capture and playback (both queues plus the worker) are fixed by
the counters alone:

```
queued = output_underruns − input_overruns − output_drops
```

That is exact, not an estimate — `--stats` prints it as `queued`. It changes
only when a counter does, so `out_q_` occupancy cannot creep with drift.
Two consequences:

- The software path's added delay is `queued / sample_rate`, fixed at startup
  (one period in queue mode, one 80 ms hop in chunked mode, per Stage 03's
  start-up underruns). **Every later underrun adds its length to the delay
  permanently**, because nothing catches up. Every overrun or drop removes
  its length. Bounding that is
  [M1.5 / Stage 05](../Milestone%201.5/05_refresh_recovery_controller.md) work.
- Drift between two physical devices is absorbed **inside miniaudio / the OS**,
  out of our counters' sight (miniaudio 0.11.25 source):
  - callback-driven backends (Core Audio, AAudio, …) and PulseAudio pass
    capture to playback through miniaudio's duplex ring buffer (5 capture
    periods, 2 prefilled). When it fills, captured frames are dropped silently.
    When it empties, the callback is fed silence.
  - WASAPI runs a blocking *read capture → callback → write playback* loop.
    The slower clock paces it, and the faster side's backlog grows in the
    WASAPI buffers until a data discontinuity (capture) or an engine underrun
    (playback).

  Either way, drift shows up only as **slowly growing delay plus an occasional
  dropout or click**. So measure it end-to-end: `fv_probe latency` over a long
  run (slope, sawtooth), and/or a long tone recording through
  `analyze_loopback.py`.

Making drift visible and correctable *inside* fluxvoice needs separate capture
and playback streams (research §8), which is a design change. It is not made
here; decide after the measurement.

## Measurement method

### Latency — `fv_probe latency`

`fv_probe` plays a 20 ms chirp (500 Hz – 3.5 kHz, survives the 16 kHz path)
every `--interval-ms` into one device and finds each one in another, on **one
duplex stream**: send and receive share a sample clock. The match is a
normalized cross-correlation, refined to a fraction of a sample. Delays must
stay below interval − 20 ms. Output: median / p95 / min / max, missed markers,
drift slope, optional per-marker CSV.

The figure includes fv_probe's own device buffers. So measure a baseline and
subtract it. Two cable pairs (cable 1 feeds fluxvoice, cable 2 receives it):

```bash
# B: baseline — cable 1 read straight back
fv_probe latency --output <cable 1 playback> --input <cable 1 capture> --count 60

# T: through fluxvoice (repeat with --passthrough, --queue-loopback, --chunked)
fluxvoice --input <cable 1 capture> --output <cable 2 playback> --duration 60 &
fv_probe latency --output <cable 1 playback> --input <cable 2 capture> --count 60
```

fluxvoice's input → output latency ≈ **T − B**: its own capture and playback
buffering, its 16 kHz resampling, its queues, and one cable. Consistency check:
T(queue) − T(passthrough) should equal the `queued` figure fluxvoice prints on
exit (Stage 03 saw 320 samples = 20 ms in queue mode, 1,280 = 80 ms chunked).

On Linux both cables are null sinks, and `tools/pw_latency_check.sh` runs the
whole table in one command: it creates two null sinks, measures the baseline,
then each mode, and prints medians, p95s and "added" (median − baseline).
*Written on the Windows machine: its control flow was run against stub
`pactl`/`fluxvoice`/`fv_probe` binaries, but not yet on a real PipeWire host.*
On Windows, use VB-Cable plus a second pair (e.g. VB-Cable A) and the commands
above. "Latency added by the virtual cable" needs a physical jumper (line out →
line in) as the reference. It is optional when no machine has one.

### Load audit and soak — `--stats` + `analyze_soak.py`

```bash
fluxvoice --input <mic> --output <cable> --duration 1800 --stats 10 > soak.log
python3 tools/analyze_soak.py soak.log      # stdlib only; exit 0 = criteria met
```

Feedback-safe: the output goes to the cable, not a speaker. For the load audit,
run the same while the machine is busy (a parallel `cmake --build`, `stress-ng`,
or `pw_loopback_check.sh --load` on Linux). `analyze_soak.py` ignores the first
10 s. It fails on any overrun/underrun/drop growth after that, or RSS growth
> 1 MB, and prints the callback rate against the system clock.

### Clock drift — long `fv_probe latency`

Run the through-fluxvoice measurement for ≥ 30 min, e.g.
`--count 1800 --interval-ms 1000 --csv drift.csv`, with capture and playback
on different devices. Read the `Drift` line (ms/min, ppm) and look for a
sawtooth in the CSV (delay creeping, then jumping back at a resync).
fv_probe's own two endpoints form a duplex stream too, so run the same long
measurement on the baseline route and compare. Two VB-Cables may share one
software clock and read ~0. A real mic in the route needs an acoustic or
line-level loop (fv_probe → speaker/line out → mic → fluxvoice → cable →
fv_probe).

## Instrument validation (offline, 2026-10-17)

CTest `latency_probe` passes the marker stream through simulated routes:

| Route | Recovered delay |
|---|---|
| 3217 samples, gain 0.2, white noise (≈ 15 dB SNR) | within 0.08 samples (< 2 µs at 48 kHz) |
| 2000 samples + 63-tap linear-phase low-pass (31 samples) | 2031.0 ± 0.05 |
| 2000 samples + 64-tap low-pass (31.5 samples) | 2031.5 ± 0.05 — sub-sample interpolation works |
| delay growing 3 samples/marker, one marker dropped | slope 720 ± 1 samples/min; the dropout is reported missed, not mis-timed |
| noise only | no false markers (correlation < 0.3; threshold 0.5) |

Breaking the window bookkeeping, the interpolation sign, the p95 rank, the
score threshold or the slope's time unit each fails the suite.
`analyze_soak.py --self-test` (CTest `analyze_soak_selftest`) checks its
verdicts on synthetic clean / late-underrun / leaking / too-short logs.

Live on the Windows dev machine: `fv_probe record` captured the microphone
correctly (Stage 05). Nothing was measured through a route: there is no
virtual cable, and playing chirps into the laptop speaker was not attempted
(audible).

## Results *(fill in)*

| Path | Median | p95 | Notes |
|---|---|---|---|
| fv_probe baseline (cable 1 read back) | — | — | subtract from the rows below |
| `--passthrough` direct | — | — | |
| queue loopback | — | — | expect passthrough + `queued` (20 ms) |
| via virtual cable | — | — | cable's own buffering included |

- Load test counters: overruns `—`, underruns `—`
- Clock drift (different devices): route delay trend `—` ms/min (`—` ppm);
  `out_q_` occupancy is constant by construction (see Finding)
- Soak test duration: `—`, final RSS: `—`

## Files touched

- `src/main.cpp` (`--stats`, exit summary with queued audio and RSS),
  `src/util/ProcessMemory.hpp` (new), `tools/fv_probe.cpp` (new),
  `tools/LatencyProbe.hpp` (new), `tools/analyze_soak.py` (new),
  `tools/pw_latency_check.sh` (new), `tests/test_latency_probe.cpp` (new),
  `CMakeLists.txt`, this doc

## Definition of done

- Numbers above filled; passthrough path ≤ ~60 ms end-to-end on the dev
  machine; master acceptance checklist in `00_master.md` can be ticked off.

Not met yet — the instruments are ready; the measurements need a cable route.
