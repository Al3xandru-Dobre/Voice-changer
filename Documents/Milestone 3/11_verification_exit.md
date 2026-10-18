# Stage 11 — Verification & exit report

> **Milestone 3** · [Master plan](00_master.md) · [← Stage 10](10_cross_lingual_target_voice.md)

**Status:** ☐ not started
**Depends on:** Stage 08 (live Mode B pipeline), Stage 09 (S2UT controls), Stage 10 (cross-lingual "Me" voice)
**Research:** [M3 research](../Research_documents/M3_WAITK_S2UT_RESEARCH.md) §4 latency budget · §7 evaluation methodology

## Objective

Verify Mode B end to end across all 6 `En↔{Es,Fr,De}` translation directions and
close Milestone 3 with measured evidence on the dev machine:
- **First-audio onset latency ($T - B$)** and **tail-flush completion delay** via
  `fv_probe latency`;
- **Simultaneous Average Lagging (`AL` and `AL_CA`)**, **ASR-BLEU**, **ASR-chrF++**,
  **ECAPA speaker cosine**, and **UTMOS** — offline (`fv_s2ut_file`) vs. live
  virtual-cable routing;
- **30-minute Mode B soak tests** (idle and under CPU load) with continuous
  utterance/silence cycles and live language/mode switches;
- project-wide documentation updates handing off cleanly to Milestone 4.

## Context & references

- [M2 / Stage 11](../Milestone%202/11_verification_exit.md) added `fv_probe
  latency --marker speech` (short 150 ms plosive burst) for Mode A.
- **Why Mode B needs a multi-chunk phrase marker in `fv_probe latency`:**
  - A 150 ms burst is shorter than two 80 ms chunks ($k = 2$). At $k = 3$ (240 ms)
    or $k = 5$ (400 ms), a 150 ms burst only increments `t_src` by 2 chunks (`kRead`),
    and the translated output would only emerge after `kTailFlushMs = 400 ms` of
    silence (`kFlush`)!
  - To measure true **online `kWrite` onset latency ($T - B$)** at $k \in
    [1..10]$, `fv_probe latency --marker phrase` plays a **1.2-second spoken
    phrase** (15 chunks of 80 ms, followed by $\ge 1.5\text{ s}$ of silence so
    `kFlush` completes and `WaitKPolicy` resets before the next repetition!).
  - It then records **two delays per trial**:
    1. **Onset $T - B$:** first output speech energy edge minus input phrase onset
       edge (expected: passthrough $T - B + k \times 80\text{ ms} + L \times
       20\text{ ms} + \text{compute}$);
    2. **Flush tail lag:** output speech offset edge minus input phrase offset
       edge.

## Tasks

- [ ] **Phrase marker in `tools/LatencyProbe.hpp` & `tools/fv_probe.cpp`:**
  - [ ] Add `fv_probe latency --marker phrase [--phrase-ms 1200]
        [--interval-ms 3500]`:
    - generates or plays a deterministic 1.2 s speech-like modulated harmonic
      phrase with a sharp onset and offset;
    - detects both output **onset** (rising energy threshold above noise floor)
      and **offset** (falling edge at end of `kFlush`);
    - reports median / $p_{95}$ / min / max for `onset_delay_ms` and
      `flush_tail_ms`.
  - [ ] Extend `tests/test_latency_probe.cpp` with a simulated Wait-*k* route
        (buffers $k \times 80\text{ ms}$ before emitting, then flushes 400 ms
        after input offset) and verify both `onset_delay_ms` and `flush_tail_ms`
        are recovered within $\pm 5\text{ ms}$.
- [ ] **Mode B Latency Matrix (`fv_probe latency --marker phrase`, 60 trials per
      cell):**
  - [ ] Measure `onset_delay_ms` (median and $p_{95}$ $T - B$) and
        `flush_tail_ms` for $k \in \{1, 2, 3, 4, 5, 8\}$ and CTC-Adaptive (if
        enabled), cross-checking against the in-process `lat_ms + lag_ms`
        telemetry.
- [ ] **Translation Quality & SimulEval Latency Matrix (all 6 directions):**
  - [ ] **Offline (`fv_s2ut_file`):** run all 6 `test_{src}_{tgt}_200.tsv` lists +
        `reordering_torture_60.tsv` + `silence_hallucination_20.tsv` for both
        public voices and `--voice me`; score with `eval_s2ut.py`.
  - [ ] **Live virtual cable:** play the 200-utterance test sets (with 800 ms
        inter-utterance silence gaps so each utterance flushes and resets cleanly)
        through Cable 1 → `fluxvoice --s2ut` → Cable 2 → `fv_probe record` →
        segment & score with `eval_s2ut.py`.
  - [ ] **Live-mic source set (Stage 10):** translate the 20-prompt per-language
        real-mic recordings and score ASR-BLEU and UTMOS.
- [ ] **30-Minute Mode B Soak Tests (`analyze_soak.py --s2ut`):**
  - [ ] Feed a continuous looped speech + pause stream into `fluxvoice --s2ut
        --src-lang es --tgt-lang en --duration 1800 --stats 10`:
    - once on an idle machine;
    - once under heavy CPU load on the remaining cores;
    - include a scripted sequence of 20 live console transitions (`mode vc`,
      `mode s2ut`, `lang fr en`, `lang en de`, `waitk 2`, `waitk 5`) during the
      run.
- [ ] **Documentation & Milestone 4 handoff:**
  - [ ] Fill the Results tables below and the master latency table in
        `00_master.md`; tick acceptance criteria and update status in
        `Documents/README.md`.
  - [ ] Update root `README.md` (§2, §3 Stage 2, §6 Quick start with `--s2ut` and
        `fv_s2ut_file`, dataset attributions for CVSS-C, CoVoST 2, Europarl-ST,
        CSS10, CML-TTS).
  - [ ] Update `RESEARCH_GUIDE.md` §1 & §5 with a pointer to
        `Documents/Research_documents/M3_WAITK_S2UT_RESEARCH.md`.
  - [ ] Update [Milestone 4](../Milestone%204/00_master.md) with the final FP32
        per-stage CPU timings (`enc`, `trans_enc`, `trans_dec`, `voc`) that M4's
        FP16/INT8 quantization and GPU Execution Providers will optimize against.

## Results *(fill in)*

### 1. Onset & Flush Latency ($T - B$, 16 kHz, 80 ms chunks)

| Policy / $k$ | Algorithmic onset (ms) | Measured onset $T-B$ median (ms) | Onset $T-B$ $p_{95}$ (ms) | Flush tail lag median (ms) | In-process `lat_ms + lag_ms` | Notes |
|---|---:|---:|---:|---:|---:|---|
| $k = 1$ | 100 | — | — | — | — | |
| $k = 2$ | 180 | — | — | — | — | |
| $k = 3$ (default) | 260 | — | — | — | — | |
| $k = 4$ | 340 | — | — | — | — | |
| $k = 5$ | 420 | — | — | — | — | |
| $k = 8$ | 660 | — | — | — | — | |
| CTC-Adaptive | — | — | — | — | — | |

### 2. Translation Quality & SimulEval Average Lagging (200 utts / direction)

| Direction | Mode / Voice | $k$ | ASR-BLEU | ASR-chrF++ | SimulEval AL (ms) | $\text{AL}_{\text{CA}}$ (ms) | ECAPA `tgt_cos` | UTMOS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `Es → En` | Offline (public) | 3 | — | — | — | — | — | — |
| `Es → En` | Live cable (public) | 3 | — | — | — | — | — | — |
| `Es → En` | Offline (`--voice me`) | 3 | — | — | — | — | — | — |
| `Fr → En` | Offline (public) | 3 | — | — | — | — | — | — |
| `De → En` | Offline (public) | 4 | — | — | — | — | — | — |
| `En → Es` | Offline (`--voice me`) | 3 | — | — | — | — | — | — |
| `En → Fr` | Offline (`--voice me`) | 3 | — | — | — | — | — | — |
| `En → De` | Offline (`--voice me`) | 4 | — | — | — | — | — | — |

### 3. 30-Minute Mode B Soak

| Condition | Duration | `misses` | `underruns` $\Delta$ | `s2ut_f` (flushes) | `queued` range (ms) | RSS growth | Worker CPU % |
|---|---|---|---|---|---|---|---|
| Idle | 30 min | — | — | — | — | — | — |
| Under load + live switches | 30 min | — | — | — | — | — | — |

## Files touched

- `tools/LatencyProbe.hpp`, `tools/fv_probe.cpp`, `tests/test_latency_probe.cpp`,
  `Documents/Milestone 3/00_master.md`, `Documents/README.md`, `README.md`,
  `RESEARCH_GUIDE.md`, `Documents/Milestone 4/00_master.md`

## Definition of done

- `ctest -R latency_probe` passes the new `--marker phrase` Wait-*k* onset and
  flush-tail detector tests.
- Measured Mode B onset $T - B$ at default $k$ ($k \in \{3, 4\}$) is $\le
  500\text{ ms}$, matching the README target, and within $\pm 15\text{ ms}$ of
  the master budget formula.
- Live-cable ASR-BLEU across the tested directions is within $-2.0$ BLEU of
  offline `fv_s2ut_file`.
- `analyze_soak.py --s2ut` passes 30-minute idle and loaded runs: 0 `misses` after
  warm-up, 0 `errors`, underruns flat, `queued` bounded within target + burst
  cushion, and RSS growth $\le 5\text{ MB}$.
- All acceptance criteria in `Documents/Milestone 3/00_master.md` are ticked.
