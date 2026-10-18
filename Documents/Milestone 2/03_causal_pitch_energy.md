# Stage 03 — Causal pitch & energy features

> **Milestone 2** · [Master plan](00_master.md) · [← Stage 02](02_model_contract_fixtures.md) · [Stage 04 →](04_content_encoder_candidates.md)

**Status:** ◐ in progress (2026-10-18). Python + C++ tracker, golden vectors, tests and the accuracy check are done. Still open: the priors are provisional (dev-clean) until train-clean-100 is prepared (see Verification)
**Depends on:** Stage 01 (env, data, evaluation lists), Stage 02 (frame geometry, lookahead field)
**Research:** §4 pitch & energy · §3 latency (no hidden lookahead)

## Objective

Produce the prosody conditioning for the vocoder, with the same algorithm at
training time (Python) and at run time (C++):
- f0, aperiodicity, soft voicing and log-energy every 20 ms;
- computed **causally**: past samples plus at most the contract lookahead.

Add running normalisation (whitening) and the mapping into a target voice's
pitch/energy range. This mapping is what the Stage 09 pitch controls act on.

## Context & references

- **Why not the trackers the stubs name.** `python_training/extract_units.py`
  offers `--pitch rmvpe|swiftf0`, and both are non-causal. SwiftF0's own README
  says streaming needs 176 ms after a frame before it is final (research §4.1).
- **StreamVC's recipe** (research §2.4, §4.2): YIN at thresholds 0.05/0.10/0.15
  + CMND value + unvoiced flag, energy, and whitening with running averages. Its
  ablation shows whitening keeps the source's pitch range from leaking speaker
  identity.
- **Allocation check.** `tests/test_duplex_router.cpp` has a global
  `operator new` counter; the C++ tracker reuses the pattern.
- **No context from `ChunkAssembler`.** Its 320-sample context was meant for a
  stateless encoder. The tracker keeps its own past-sample ring, so VC mode can
  run with context 0 (Stage 08).

## Feature definition

| Output (per 20 ms frame) | Definition |
|---|---|
| `f0_hz` | YIN estimate at threshold 0.10 (first dip below 0.10, else the global CMND minimum) on a 40 ms window ending at the frame end (+ L frames if lookahead > 0; centred on the frame when L ≥ 1); search 55–1000 Hz; parabolic interpolation on the CMND. **0 unless the frame is voiced** (the voiced-decision row below) |
| `aperiodicity` | Minimum of the cumulative-mean-normalised difference (CMND) in the search range |
| `voicing` | Mean of the voiced flags at thresholds 0.05 / 0.10 / 0.15 → {0, ⅓, ⅔, 1} |
| voiced decision | `aperiodicity < 0.40` **and** `z_energy > −0.5`. Gates `f0_hz`, the held log-f0 and the log-f0 whitening. This rule is not in the original plan: at 0.10 alone, only 74% of frames agree with the reference (see Verification) |
| `log_energy` | log(mean square + ε) of the high-passed frame |
| `logf0_held` | log f0 on voiced frames; on unvoiced frames, holds the last voiced value, relaxing towards the running log-f0 mean with a 0.25 s time constant (causal; no interpolation across the future) |
| `z_logf0`, `z_energy` | `logf0_held` and `log_energy` whitened by the running statistics below (the statistics include the current frame) |

**Pre-filter.** A 2nd-order high-pass at about 70 Hz (biquad) before both YIN
and energy.

**Whitening.**
- An EMA mean and standard deviation of log-f0 over voiced frames, and of
  log-energy over all frames.
- The time constant is configurable (default 4 s), with a standard-deviation
  floor (0.05 for log-f0, 0.5 for log-energy).
- It starts from population priors computed on LibriSpeech train-clean-100 in
  this stage. **Provisional values** (dev-clean, 5.4 h, 40 speakers;
  `eval_pitch.py priors`): log-f0 mean **5.087** (162 Hz), std **0.427**;
  log-energy mean **−8.500**, std **3.829** (natural log of the mean square).
  The voiced decision uses the energy prior, so the log-f0 prior was iterated
  to a fixed point (3 runs, change < 0.01).

**Target mapping** (the C++ run-time path; training uses the target's own
features directly):
`logf0_tgt = μ_tgt + scale · σ_tgt · z_f0 + shift_st · ln2 / 12`,
`energy_tgt = μe_tgt + σe_tgt · z_e`. μ and σ come from `voices.json`; `scale`
and `shift_st` come from Stage 09 (defaults 1 and 0).

## Tasks

- [x] **`python_training/fvml/features.py`.** A streaming `PitchTracker` class
      (frame-in, features-out, explicit state) plus an offline wrapper that
      loops the streaming class. One code path, so offline equals streaming by
      construction. Also `RunningNorm` and `map_to_target`.
- [ ] **Population priors.** Log-f0 and log-energy mean/std over train-clean-100
      → constants recorded in this file and in `fvml/features.py`.
      *Provisional dev-clean values are in. Once train-clean-100 is prepared, rerun
      `eval_pitch.py priors --manifest librispeech-train-clean-100`, then
      `make_pitch_golden.py`, and update `PitchTracker.hpp`.*
- [x] **Accuracy check** against an offline reference tracker (non-causal is fine,
      e.g. a CREPE- or Harvest-based reference) on 30 min of LibriSpeech
      dev-clean (+ VCTK if prepared). The check is `eval_pitch.py accuracy`.
      The reference is `fvml/f0_ref.py` (Praat AC), the same one Stages
      01/05/09 use. VCTK is not prepared yet. Metrics:
  - [x] gross pitch error (> 20% off)
  - [x] voicing decision accuracy
  - [x] voiced log-f0 RMSE in cents
- [x] **Golden vectors** in `tests/data/pitch/` (`make_pitch_golden.py`, 161 KiB):
  - [x] a synthetic set: sawtooth glides 80→400 Hz, voiced/unvoiced
        alternation, white noise, silence, a clipped segment
  - [x] one short LibriSpeech test-clean utterance (CC BY 4.0, attributed in
        the folder README): `61-70968-0058`, 1.82 s
  - [x] expected per-frame features as little-endian float32 + a JSON header
        (frame size, thresholds, priors), for L = 0, 1, 2
- [x] **C++ port.**
  - [x] `src/dsp/Biquad.hpp`
  - [x] `src/dsp/PitchTracker.hpp`: header-only; preallocated history and CMND
        buffers; `push(const float* frame) → PitchFrame`; `reset()`; also the
        target mapping (`map_logf0_to_target`, `map_energy_to_target`)
  - [x] `src/dsp/RunningNorm.hpp`
- [x] **`tests/test_pitch_tracker.cpp`** (ORT-free, added to the CTest loop):
      CTest `pitch_tracker`, registered by the Stage 01/02 session (it owns
      `CMakeLists.txt`). Passing in Release and Debug, `/W4 /WX`.
  - [x] golden-vector comparison
  - [x] no allocation after construction (`operator new` counter)
  - [x] whitening converges to known stats on a synthetic AR(1) log-f0 track
  - [x] the target mapping reproduces closed-form values
- [x] **Timing.** A micro-benchmark (Release, one core) of `PitchTracker::push`
      per 20 ms frame, recorded in Verification.

## Implementation notes

- **Window vs. lookahead.** A 40 ms past window covers two periods at 50 Hz, the
  lowest f0 in range. With lookahead L ≥ 1 the window may be centred instead,
  which improves accuracy at no extra latency because the content encoder
  already waits L frames.
  - The contract's `lookahead_frames` decides which variant runs.
  - Both variants have golden vectors.
- **Use the same constants in both languages.** Thresholds, search range, HPF
  coefficients, EMA constants and priors live in one JSON header that ships with
  the golden vectors, and both implementations load it in tests.
- **Training features use the same tracker.** The vocoder (Stage 05) is
  conditioned on this causal tracker's output for the *target* speaker's audio,
  so it learns to render exactly what the run-time tracker will feed it, errors
  included.
- **Don't interpolate across unvoiced gaps.** Interpolation needs the future.
  Hold-and-relax is causal, and the `voicing` input tells the vocoder to ignore
  f0 there.
- **Gain invariance.** Whitening `log_energy` makes the conditioning independent
  of mic gain. The absolute output level is set by Stage 09's gain control, not
  by how loud the user speaks into the mic.

## Files touched

- `python_training/fvml/features.py` (new), `src/dsp/Biquad.hpp` (new),
  `src/dsp/PitchTracker.hpp` (new), `src/dsp/RunningNorm.hpp` (new),
  `tests/test_pitch_tracker.cpp` (new), `tests/data/pitch/**` (new),
  `CMakeLists.txt` (test registration)
- Added during implementation:
  - `python_training/eval_pitch.py` (priors + accuracy).
  - `python_training/make_pitch_golden.py` (golden vectors). It lives in
    `python_training/` rather than `tools/` because it needs numpy/scipy.
  - `python_training/tests/test_features.py`.

## Definition of done

- Against the offline reference on 30 min of speech: gross pitch error ≤ 5%,
  voicing accuracy ≥ 90%, voiced log-f0 RMSE ≤ 50 cents.
- The C++ tracker matches the golden vectors: |Δf0| ≤ 0.1 Hz on frames voiced in
  both, voicing identical on ≥ 99.5% of frames, |Δlog_energy| ≤ 1e-4.
- `PitchTracker::push` takes ≤ 0.3 ms per 20 ms frame on one core (Release),
  with zero allocations after construction.
- Python frame-by-frame output equals the whole-array output exactly.

## Verification

Measured 2026-10-18 on the dev laptop. Other jobs were running, so the
timings are pessimistic.

**Accuracy** (`eval_pitch.py accuracy --lookahead 0 1 2`): 30.0 min of
dev-clean, 229 utterances, all 40 speakers, held out from the tuning rows.
Each tracker frame is compared with the Praat frame nearest its window centre.

| Variant | Voicing accuracy (≥ 0.90) | Recall / false alarm | GPE (≤ 5%) | Fine RMSE (≤ 50 c) | RMSE incl. gross |
|---|---|---|---|---|---|
| L = 0 (past window) | **0.907** | 0.865 / 0.043 | **4.6%** (2.9% down, 1.7% up) | **30.2 c** | 374 c |
| L = 1 (centred) | **0.909** | 0.877 / 0.051 | **4.5%** | **31.5 c** | 370 c |
| L = 2 (centred) | **0.909** | 0.877 / 0.052 | **4.5%** | **31.5 c** | 370 c |

- "Fine RMSE" is over frames that are voiced in both and are not gross
  errors. An octave error is 1200 cents, so the all-frames RMSE measures GPE,
  not precision.
- **The margins are thin** (0.7 pt on voicing, 0.4 pt on GPE), and the
  reference is itself an estimate. Praat uses a 60 ms centred window, and about
  30% of frames lie within ±30 ms of a voicing change. Away from those
  transitions, the decision accuracy is about 0.91 even before tuning.
- **Why the voiced decision changed.** The plan's rule ("voiced at threshold
  0.10") agrees with the reference on only 0.74 of frames (recall 0.54). The
  StreamVC thresholds are kept for the soft `voicing` feature and for choosing
  the dip; the voiced decision is now a separate rule. Its constants were swept
  on rows 0, 60, … of dev-clean (45 utterances), which the accuracy set
  excludes:

  | Decision rule (tuning rows) | Accuracy | GPE | Fine RMSE |
  |---|---|---|---|
  | aperiodicity < 0.10 | 0.738 | 1.8% | 19 c |
  | < 0.35 | 0.860 | 8.7% | 34 c |
  | < 0.35, f0 = global min when no dip < 0.10 | 0.860 | 3.6% | 29 c |
  | < 0.40, z_energy > −0.5 (chosen) | 0.912 | 3.9% | 30 c |

  A 1–2 kHz low-pass before YIN was also tried. It did not help (best accuracy
  0.84–0.85).

**Parity.**
- C++ vs. the golden vectors, all 6 cases (synthetic and libri × L 0/1/2):
  identical after rounding to float32. That gives |Δf0| = 0 and
  |Δlog_energy| = 0, and voicing and the voiced decision agree on 100% of
  frames (DoD: ≤ 0.1 Hz, ≤ 1e-4, ≥ 99.5%).
- Python frame-by-frame output equals the whole-array output exactly
  (`test_streaming_equals_whole_array`, L = 0/1/2).
- Causality: changing the samples after the end of frame t + L leaves frames
  ≤ t untouched (`test_causal`).
- Mutation check of the C++ port: 8 mutants.
  - A no-op control passed.
  - Each of these failed the test: no parabolic interpolation, no energy
    gate, no hold-and-relax, an uncentred L ≥ 1 window, a frozen energy EMA,
    and `voiced_max_aperiodicity` 0.40 → 0.41.
  - One survived: starting the global-minimum scan one lag late
    (tau_min + 2). That only matters when the minimum sits at 941 Hz on the
    aperiodic path, which the vectors never exercise.

**Cost** (`test_pitch_tracker --bench 5000`, MSVC 19.50 `/O2`, one thread,
loaded machine):
- L = 0: `push()` mean **67 µs**, p50 63 µs, p99 165 µs per 20 ms frame.
  L = 1: mean 72 µs, p99 159 µs. Both are within the 0.3 ms DoD.
- 0 heap allocations across 182 pushes and a `reset()`.
- The Debug build is about 4× slower (mean 282 µs). The Python reference
  takes about 0.2 ms per frame.

**Tests.**
- `python_training/tests/test_features.py`: 26 passed. They cover:
  - golden ×6, streaming = offline ×3, causality ×3
  - tones 60–900 Hz within 0.5%; silence and noise; hold-and-relax
  - gain invariance of `z_energy` (−20 dB gives |Δz| < 0.02 after 8 time
    constants)
  - AR(1) whitening, the std floor, the closed-form mapping
  - config rejection and byte-identical regeneration
- `tests/test_pitch_tracker.cpp`: all checks pass in Release and Debug. On the
  AR(1) track the EMA reaches mean 4.790 (true 4.788) and std 0.195 (true
  0.200), with rms z 1.001.

**Open.**
- Recompute the priors on train-clean-100.
- Run the accuracy check on VCTK.
- `tests/test_features.py` runs under CTest as `python_training_stage03_04`,
  but only where `python_training/.venv` exists. CI needs a `uv sync --frozen`
  job to cover it (BUG-009).
