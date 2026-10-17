# Stage 11 — Verification & exit report

> **Milestone 2** · [Master plan](00_master.md) · [← Stage 10](10_target_voice_finetune.md)

**Status:** ☐ not started
**Depends on:** Stage 08 (live pipeline), Stage 09 (controls), Stage 10 (own voice)
**Research:** §3 latency · §7 evaluation · §10 fluent stream

## Objective

Measure Mode A end to end and close the milestone. That means three numbers on
the user's machine:
- the latency budget, as measured T−B;
- intelligibility and voice quality, live vs. offline;
- stability over a 30-minute soak under load.

Then update the master and the stale project-wide docs.

## Context & references

- **Why a new marker.** `fv_probe latency` detects a 20 ms chirp by normalised
  cross-correlation (`tools/LatencyProbe.hpp`). A real VC model reshapes the
  chirp, so correlation fails. The identity bundle (Stage 08) covers the
  plumbing, and an onset/envelope marker covers the real model.
- **M1 / Stage 06 method** (baseline, per-mode medians/p95, "added" = median −
  baseline) and `analyze_soak.py` are reused. Stage 08 adds the `--vc` checks
  and the new fields.
- **Cable rig.** [BUG-001](../Bug%20Report/BUG-001_voicemeeter_vaio_routing_wedge.md)
  and [BUG-002](../Bug%20Report/BUG-002_voicemeeter_remote_write_lock.md) affect
  the Voicemeeter route. VB-Cable (+ a second pair) or the configured VAIO route
  must work before the live rows can be filled.
- **Stale claims to fix:**
  - root `README.md` §3: "Kyutai Mimi codec (recommended)", fixed 80 ms chunks,
    RMVPE/SwiftF0
  - root `README.md` §4: worker priority "Normal", 80 ms worker description
  - `RESEARCH_GUIDE.md` §2: Mimi as the default tokenizer

## Tasks

- [ ] **Envelope marker** in `tools/LatencyProbe.hpp` + `fv_probe latency --marker
      speech`:
  - [ ] Play a short recorded speech burst (≈ 150 ms, a plosive onset) every
        interval.
  - [ ] Detect the output onset by cross-correlating smoothed log-energy
        envelopes (robust to timbre and pitch change); sub-frame refinement.
  - [ ] `tests/test_latency_probe.cpp`: a simulated VC route (pitch-shift +
        formant-like filtering + noise + a known delay) recovers the delay within
        ± 2 ms.
- [ ] **Latency matrix** (fv_probe, 5 min per cell, median + p95):
  - [ ] rows: passthrough, queue loopback, VC identity bundle, VC real bundle
  - [ ] columns: device period 10 / 20 ms × hop of the gated variant (+ one
        neighbouring hop for comparison)
  - [ ] cross-check each VC cell against the in-process `lat_ms` estimate
- [ ] **Algorithmic latency offline.** `fv_vc_file` on an impulse/onset test file:
      measured onset shift vs. the contract's algorithmic latency.
- [ ] **Quality.**
  - [ ] Offline: `fv_vc_file` on `test_clean_200` → `eval_vc.py` (Neutral preset,
        2 public voices + "me").
  - [ ] Live: play the same utterances into the input side of the cable route →
        `fluxvoice --vc` → record the output with `fv_probe record` → align →
        `eval_vc.py`.
  - [ ] Live-mic source set (Stage 10): convert, then score WER (Whisper on the
        source as reference) and UTMOS.
- [ ] **Soak.** `fluxvoice --vc --duration 1800 --stats 10`, then `analyze_soak.py
      --vc`. Once idle, once under load (a parallel CPU-heavy build or stress tool
      on the other cores). Record worker CPU %.
- [ ] **Docs:**
  - [ ] Fill the master's latency budget with measured values. Tick acceptance
        criteria; set the status.
  - [ ] Update `Documents/README.md` status.
  - [ ] Root `README.md`: pointer notes in §3/§4 (encoder per the gate, hop
        configurable, YIN, MMCSS worker priority), the §6 quick start with
        `--vc`, and the Hi-Fi TTS attribution.
  - [ ] `RESEARCH_GUIDE.md`: a pointer to the M2 research report next to the
        Mimi entry.
  - [ ] [Milestone 3](../Milestone%203/00_master.md): fill "Inherited from M2"
        with the final unit rate, vocabulary and hop.

## Results *(fill in)*

| Path | Period (ms) | Hop (ms) | T−B median (ms) | T−B p95 (ms) | `lat_ms` estimate | Notes |
|---|---|---|---|---|---|---|
| passthrough | 10 | — | — | — | — | |
| queue loopback | 10 | — | — | — | — | |
| VC identity | 10 | — | — | — | — | |
| VC real | 10 | — | — | — | — | |
| VC real | 20 | — | — | — | — | |

| Quality | WER | CER | Tgt cos | Src cos | UTMOS | F0 PCC |
|---|---|---|---|---|---|---|
| Offline (`fv_vc_file`) | — | — | — | — | — | — |
| Live (cable) | — | — | — | — | — | — |
| Live-mic source set | — | — | — | — | — | — |

| Soak | Duration | misses | underruns Δ | skipped | `queued` range | RSS growth | Worker CPU |
|---|---|---|---|---|---|---|---|
| Idle | — | — | — | — | — | — | — |
| Loaded | — | — | — | — | — | — | — |

## Implementation notes

- **Identity before real.** If the identity bundle's T−B is already over budget,
  the problem is I/O or governor tuning, not the model. Fix it there first.
- **Live WER needs time alignment.** Trim the recorded output using the measured
  T−B before segmenting by utterance, or WER will reflect segmentation errors.
- **Load test.** Record what "under load" meant (tool, core count), so the number
  can be reproduced on another machine (M1.5 / Stage 06 matrix).

## Files touched

- `tools/LatencyProbe.hpp`, `tools/fv_probe.cpp` (speech marker),
  `tests/test_latency_probe.cpp`, `Documents/Milestone 2/00_master.md`,
  `Documents/README.md`, `README.md` (§3, §4, §6, credits), `RESEARCH_GUIDE.md`
  (pointer), `Documents/Milestone 3/00_master.md`

## Definition of done

- The latency matrix is filled with median and p95 T−B for queue, identity-VC
  and real VC at each measured period and hop. The gated configuration meets the
  master budget's **Must** line (≤ 100 ms; Mimi deviation ≤ 160 ms).
- The algorithmic latency measured with `fv_vc_file` is within ± 2 ms of the
  contract value.
- Live-cable WER is within +2 pts of the offline `fv_vc_file` WER on the same
  utterances.
- `analyze_soak.py --vc` passes for 30 min both idle and loaded:
  - 0 misses after warm-up;
  - underruns flat;
  - `queued` within target ± one hop;
  - RSS growth ≤ 5 MB.
- Every master acceptance criterion is ticked, or explicitly deferred with a
  reason.
