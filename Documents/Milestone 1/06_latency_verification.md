# Stage 06 — Latency & stability verification

> **Milestone 1** · [Master plan](00_master.md) · [← Stage 05](05_virtual_cable_loopback.md)

**Status:** ☐ not started
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
- [ ] **Glitch audit under load.** Run queue-mode loopback while the machine
      is loaded (parallel build / `stress-ng`); record `input_overruns` and
      `output_underruns` deltas.
- [ ] **Soak test.** ≥ 30 min continuous loopback; counters stay flat after
      startup; RSS stable (no leaks in the RT path).
- [ ] Write results into this file (table below) and update the master
      acceptance criteria.

## Measurement method

1. Wire output → input (cable pair or virtual loopback of the cable).
2. Play a short impulse train from a tool (`pw-loopback`, `aplay` of a click
   WAV, or a `tools/` impulse generator).
3. In a debug build, timestamp (`std::chrono` / sample counters — *off* the RT
   thread) the impulse's capture frame and its playback frame.
4. Latency = output_t − input_t ± one callback period of jitter. Repeat ≥ 50
   times; report median & p95.

## Results *(fill in)*

| Path | Median | p95 | Notes |
|---|---|---|---|
| `--passthrough` direct | — | — | |
| queue loopback | — | — | |
| via virtual cable | — | — | cable's own buffering included |

- Load test counters: overruns `—`, underruns `—`
- Soak test duration: `—`, final RSS: `—`

## Definition of done

- Numbers above filled; passthrough path ≤ ~60 ms end-to-end on the dev
  machine; master acceptance checklist in `00_master.md` can be ticked off.
