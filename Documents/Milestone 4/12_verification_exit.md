# Stage 12 — Verification & exit report

> **Milestone 4** · [Master plan](00_master.md) · [← Stage 11](11_desktop_integration.md)

**Status:** ☐ not started
**Depends on:** Stages 04–09 (Stage 07 only if Milestone 3 has landed; Stages 10–11 as far as they shipped),
[M2 / Stage 11](../Milestone%202/11_verification_exit.md) (CLI latency, quality and soak numbers to compare against)
**Owner persona:** `@qa-verifier`

## Objective

Measure the milestone's exit criteria and record them: the full pipeline is
controllable from the GUI, per-stage and end-to-end latency are visible live,
and the GPU path beats the real-time-factor budget with margin.

## Tasks

- [ ] **Build gate, four native legs.** Warnings-as-errors all-target build
      plus CTest, ORT-less and with ORT, for Windows MSVC, Linux GCC, Linux
      Clang and macOS AppleClang. Record the commit, compiler versions and
      PASS / FAIL / NOT RUN per leg, with a link to the CI run. A leg that did
      not run is NOT RUN, not passed.
- [ ] **GUI controllability checklist** (manual, with a real bundle): start /
      stop, device selection and change while running, bundle load and swap,
      voice A/B, every MVP slider, presets, mute, Mode A ↔ B and language
      pair (if M3 landed), provider and precision selection, tray and Mini
      panel parity. One row per control: action, expected, observed.
- [ ] **Latency.** `fv_probe latency` baseline `B` on Cable 1 and total `T`
      through `fluxvoice_gui` (Cable 1 → GUI → Cable 2), for CPU and each
      shipped provider / precision. Compare `T − B` with the CLI's M2 Stage 11
      number for the same bundle: the GUI may add at most 5 ms.
- [ ] **The live estimate is honest.** The Stage 06 estimate card is within
      ±10 ms of the measured `T − B`.
- [ ] **Soak.** 30 minutes of live Mode A through the GUI with the window
      hidden to the tray: 0 deadline misses after warm-up, underruns flat,
      `queued` bounded, RSS growth ≤ 5 MB (`tools/analyze_soak.py` on the
      GUI's stats log). Repeat with the window visible and the Telemetry
      panel open, to show that rendering does not disturb the worker.
- [ ] **GPU real-time factor.** `fv_bench` for each provider and variant:
      per-hop p95 ≤ 50 % of the hop and p99 ≤ 75 % over ≥ 30 minutes, with a
      second GPU load running for part of the test. State the margin.
- [ ] **Quality did not regress** through the GUI path: WER on the frozen
      list recorded through Cable 2 is within +2 points of the offline number
      (same rule as M2).
- [ ] **Device-loss drill** through the GUI on each OS available: unplug the
      selected output; the stream stops with `DeviceNotFound`, nothing is
      substituted, and choosing a device restarts it.
- [ ] **Per-OS manual matrix:** real tray (Windows, KDE, GNOME with the
      AppIndicator extension, one Wayland session), native file dialogs,
      close-to-tray, start minimised, Mini panel always-on-top, hotkeys and
      notifications (Stage 11). PASS / FAIL / NOT RUN per cell.
- [ ] **Documentation:** update the master's status and stage table,
      `frontend/README.md` §10–§11, `Documents/README.md`, the root
      `README.md` milestone section, and `MEMORY.md` if a new invariant came
      out of this milestone. File every defect found as `BUG-NNN`.

## Definition of done

- Every exit criterion in the [master plan](00_master.md) has a measured row
  in this file, or an explicit, dated deviation with its reason.
- No row claims a platform or a measurement that was not run.
- Open defects are filed and linked; none is of High severity.
