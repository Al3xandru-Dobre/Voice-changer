# Stage 03 — Causal pitch & energy features

> **Milestone 2** · [Master plan](00_master.md) · [← Stage 02](02_model_contract_fixtures.md) · [Stage 04 →](04_content_encoder_candidates.md)

**Status:** ☐ not started
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
| `f0_hz` | YIN estimate at threshold 0.10 on a 40 ms window ending at the frame end (+ L frames if lookahead > 0); search 55–1000 Hz; parabolic interpolation |
| `aperiodicity` | Minimum of the cumulative-mean-normalised difference (CMND) in the search range |
| `voicing` | Mean of the voiced flags at thresholds 0.05 / 0.10 / 0.15 → {0, ⅓, ⅔, 1} |
| `log_energy` | log(mean square + ε) of the high-passed frame |
| `logf0_held` | log f0 on voiced frames; on unvoiced frames, holds the last voiced value, relaxing towards the running mean (causal; no interpolation across the future) |

**Pre-filter.** A 2nd-order high-pass at about 70 Hz (biquad) before both YIN
and energy.

**Whitening.**
- An EMA mean and standard deviation of log-f0 over voiced frames, and of
  log-energy over all frames.
- The time constant is configurable (default about 4 s), with a standard-
  deviation floor.
- It starts from population priors computed on LibriSpeech train-clean-100 in
  this stage.

**Target mapping** (the C++ run-time path; training uses the target's own
features directly):
`logf0_tgt = μ_tgt + scale · σ_tgt · z_f0 + shift_st · ln2 / 12`,
`energy_tgt = μe_tgt + σe_tgt · z_e`. μ and σ come from `voices.json`; `scale`
and `shift_st` come from Stage 09 (defaults 1 and 0).

## Tasks

- [ ] **`python_training/fvml/features.py`.** A streaming `PitchTracker` class
      (frame-in, features-out, explicit state) plus an offline wrapper that
      loops the streaming class. One code path, so offline equals streaming by
      construction. Also `RunningNorm` and `map_to_target`.
- [ ] **Population priors.** Log-f0 and log-energy mean/std over train-clean-100
      → constants recorded in this file and in `fvml/features.py`.
- [ ] **Accuracy check** against an offline reference tracker (non-causal is fine,
      e.g. a CREPE- or Harvest-based reference) on 30 min of LibriSpeech
      dev-clean (+ VCTK if prepared). Metrics:
  - [ ] gross pitch error (> 20% off)
  - [ ] voicing decision accuracy
  - [ ] voiced log-f0 RMSE in cents
- [ ] **Golden vectors** in `tests/data/pitch/`:
  - [ ] a synthetic set: sawtooth glides 80→400 Hz, voiced/unvoiced
        alternation, white noise, silence, a clipped segment
  - [ ] one short LibriSpeech test-clean utterance (CC BY 4.0, attributed in
        the folder README)
  - [ ] expected per-frame features as little-endian float32 + a JSON header
        (frame size, thresholds, priors)
- [ ] **C++ port.**
  - [ ] `src/dsp/Biquad.hpp`
  - [ ] `src/dsp/PitchTracker.hpp`: header-only; preallocated ring, difference
        function and CMND buffers; `push(const float* frame) → PitchFrame`
  - [ ] `src/dsp/RunningNorm.hpp`
- [ ] **`tests/test_pitch_tracker.cpp`** (ORT-free, added to the CTest loop):
  - [ ] golden-vector comparison
  - [ ] no allocation after construction (`operator new` counter)
  - [ ] whitening converges to known stats on a synthetic AR(1) log-f0 track
  - [ ] the target mapping reproduces closed-form values
- [ ] **Timing.** A micro-benchmark (Release, one core) of `PitchTracker::push`
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

## Definition of done

- Against the offline reference on 30 min of speech: gross pitch error ≤ 5%,
  voicing accuracy ≥ 90%, voiced log-f0 RMSE ≤ 50 cents.
- The C++ tracker matches the golden vectors: |Δf0| ≤ 0.1 Hz on frames voiced in
  both, voicing identical on ≥ 99.5% of frames, |Δlog_energy| ≤ 1e-4.
- `PitchTracker::push` takes ≤ 0.3 ms per 20 ms frame on one core (Release),
  with zero allocations after construction.
- Python frame-by-frame output equals the whole-array output exactly.
