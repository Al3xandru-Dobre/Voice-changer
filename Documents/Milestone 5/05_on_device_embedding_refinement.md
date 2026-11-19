# Stage 05 — On-device embedding refinement (time-boxed)

> **Milestone 5** · [Master plan](00_master.md) · [← Stage 04](04_speaker_encoder_vocoder_gate.md) · [Stage 06 →](06_speaker_encoder_export_parity.md)

**Status:** ☐ not started
**Depends on:** [Stage 04](04_speaker_encoder_vocoder_gate.md) (chosen vocoder + speaker encoder),
[Stage 03](03_unseen_speaker_eval_harness.md) (lists and metrics)
**Owner persona:** `@ml-speech-trainer` (Phase A), `@onnx-ort-runtime` (Phase B)
**Time box:** Phase A one week. Phase B starts only if the gate passes.

## Objective

Find out whether a profile gets measurably closer to the user's voice when the
64 embedding values are optimised against the user's own recordings, with
every model weight frozen. If it does, choose how that optimisation runs on a
user's machine without a training toolchain. If it does not, record the
numbers and drop the Refined tier.

## Context & references

- The speaker encoder gives a one-pass estimate. Searching the embedding space
  for the point that best reconstructs the user's audio can only use
  information that is already in the recordings, and it changes nothing but
  the 64 numbers: the public voices cannot drift, the result is still a
  `profile.json`, and 64 parameters cannot memorise what was said.
- [M2 / Stage 10](../Milestone%202/10_target_voice_finetune.md) adapts the
  whole vocoder and needs hours of audio and a GPU. This stage is the
  smallest possible form of the same idea.
- ONNX Runtime's on-device training packages are marked deprecated (v1.20.1
  release notes). They are not an option.

## Method

```text
e* = argmin_e  Σ_clips  L_mel( Vocoder(units(x̃), cond(x), e),  x )
```

- `x` is a 2-second clip of the user's speech; `x̃` is its perturbed copy
  (formant, pitch and EQ perturbation from `fvml/augment.py`), so the units
  do not carry the timbre and the embedding has to.
- `cond(x)` is the clip's own F0, voicing, aperiodicity and energy: the
  target domain is the user's own.
- `L_mel` is the multi-resolution mel loss already used for training.
- Start from the Instant embedding `e0`. Keep `‖e − e0‖` inside a trust
  radius and the norm inside the range seen in training.
- Hold out 20 % of the clips; stop when their loss stops improving.

## Phase A — is it worth having? (Python, one week)

- [ ] `refine_profile.py`: Adam on `e` with PyTorch autograd, 300–1 000
      steps, batches of eight 2-second crops, early stopping, trust radius.
- [ ] Run on the Stage 03 speakers at 30 s, 60 s, 180 s and 600 s of
      enrolment audio (600 s where the corpus allows).
- [ ] Report against the Instant profile: Δ`tgt_cos`, ΔWER, ΔUTMOS, the
      share of speakers that improve, held-out loss curves, and the distance
      moved from `e0`.
- [ ] Check the curated voices are untouched (they must be: no weight
      changes) and that timbre blend still interpolates cleanly through a
      refined point.

### Gate (written first)

Productise refinement only if, at 180 s of enrolment audio:

- mean Δ`tgt_cos` ≥ +0.03 over Instant;
- at least 75 % of speakers improve;
- WER is no worse than +1 point and UTMOS no worse than −0.1.

Otherwise the result is "evaluated and rejected", the Refined tier is
removed from the master's tier table, and Stage 10 shows no "Refine" action.

## Phase B — how it runs on the device (only if the gate passes)

| Option | How | For | Against |
|---|---|---|---|
| 1. Exported gradient graph | One fixed-shape `refine_step.onnx`: (units, conditioning, target audio, `e`) → (loss, ∂loss/∂`e`); C++ runs Adam on 64 floats | Exact gradients; few steps | The exporter must be able to trace a gradient computation. Unknown on the pinned toolchain: one-day spike before relying on it |
| 2. Forward-only search | The shipped vocoder graph stepped over each crop from a zero state, plus a small `mel_loss.onnx`; SPSA or CMA-ES in a low-dimensional subspace (a PCA basis of training embeddings shipped as `speaker_basis.json`) | No new exporter risk; reuses graphs that are already tested | More forward passes; stochastic, so parity with Python is statistical |
| 3. ONNX Runtime training API | — | — | Deprecated upstream. **Rejected** |
| 4. Python sidecar | The Phase A script | Already exists | A 2 GB toolchain for end users. **Developer use only** |

- [ ] One-day spike on option 1; record the result either way.
- [ ] Implement the chosen option behind `EnrollJob` (Stage 07) as a second
      phase: cancellable, progress reported, below-normal thread priority.
- [ ] Budget: 180 s of audio in ≤ 10 minutes on four cores. Measure on the
      reference machine and record it.
- [ ] Live safety: run a refinement while Mode A converts; 0 deadline misses.
- [ ] Parity with Phase A on the frozen list with a fixed seed: mean
      `tgt_cos` within 0.01 of the Python result.
- [ ] The profile records `tier: "refined"` and, under `provenance`, the
      steps taken and the held-out loss before and after.

## Implementation notes

- **Privacy is unchanged.** Clips, units and conditioning stay in memory and
  are released when the job ends. Nothing new is stored.
- **Why a trust radius.** An unconstrained search can find a point that
  lowers the loss on these clips by leaving the region the vocoder was
  trained on. Such a point tends to sound worse on new sentences.
- **Why a gate at all.** Refinement adds minutes of waiting and a second code
  path. That is worth it only for a difference a listener could notice.
- **Rejection is a valid outcome** and costs one week.

## Results *(fill in)*

| Enrol (s) | Δ`tgt_cos` | % speakers improved | ΔWER | ΔUTMOS | Mean distance from `e0` |
|---|---|---|---|---|---|
| 30 | | | | | |
| 60 | | | | | |
| 180 | | | | | |
| 600 | | | | | |

Decision: … (accepted with option … / evaluated and rejected), date.

## Files touched

- `python_training/refine_profile.py` (new), `fvml/enroll.py`,
  `python_training/tests/test_refine_profile.py` (new); if accepted:
  `fvml/export.py`, `export_stateful_onnx.py` (`refine_step` or `mel_loss`
  graph, `speaker_basis.json`), `src/enroll/Refine.{hpp,cpp}`,
  `tests/test_enroll_refine.cpp`; this file (results, decision);
  [master plan](00_master.md) (tier table, open decision 7)

## Definition of done

- The results table is filled and the decision is recorded with its date.
- If accepted: the chosen option meets the time budget, the live-safety check
  and the parity threshold, each with a recorded number.
- If rejected: the master plan and Stage 10 are updated to remove the tier.
