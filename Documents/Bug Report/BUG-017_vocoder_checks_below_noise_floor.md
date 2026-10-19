# BUG-017 — Vocoder equivalence and export checks run on an output smaller than their own tolerance

> [Bug index](00_index.md) · Review: [REVIEW-M2-S05](REVIEW-M2-S05_2026-10-18.md)

| | |
|---|---|
| **Tag** | `BUG-017` |
| **Name** | *Quiet tolerance* — at random initialisation the generator's output peaks at 3.9e-5, so an absolute `max|Δ| ≤ 1e-4` check cannot fail |
| **Reported** | 2026-10-18 |
| **Severity** | Low–medium (the check that the Definition of Done relies on is meaningful on a trained checkpoint; but every random-init check, the unit tests and the quoted Verification pass criteria are vacuous, and broken caches pass them) |
| **Status** | ☑ fixed — verified 2026-10-19, see [Resolution](#resolution-verified-2026-10-19) |
| **Component** | `python_training/fvml/vocoder.py` (`_init_conv_weights`, std 0.01), `python_training/stream_check.py` (`VOCODER_MAX_ABS = 1e-4`, `check_vocoder`), `python_training/gate_report.py` (`export_smoke_test`: `voc_max_abs <= 1e-4`), `python_training/tests/test_vocoder.py` |
| **Blocks** | [Stage 05](../Milestone%202/05_causal_vocoder_decision_gate.md) streaming-equivalence and export-smoke evidence; regression protection for the `python_training_stage05` CTest entry |

## Environment

Windows 11, Python 3.12.13, torch 2.11.0+cu128, CPU, float32, tree as of
2026-10-18. Default `UnitVocoder()` (V2-class, 1.06 M parameters) with
`torch.manual_seed(0)`; the same seed and defaults the unit tests use.

## Summary

`_init_conv_weights` initialises every streaming convolution with `N(0, 0.01)`.
Through ~100 convolutions that gain is far below 1, so an untrained generator
emits almost silence:

```text
random-init output (5 s of real features):  max|y| = 3.88e-05   rms = 1.23e-05
```

The acceptance threshold used everywhere for this model is an *absolute*
`1e-4` (`VOCODER_MAX_ABS`; `voc_max_abs <= 1e-4` in the export smoke test;
`r["max_abs"] <= 1e-4` in `test_vocoder.py`). At this amplitude an all-zero
output passes. The stage file's Verification quotes `max|Δ| = 3.09e-11,
SNR 125.6 dB` for random init, and the *SNR* is informative, but nothing
asserts on the SNR.

## Steps to reproduce

Break a cache on purpose and run the vocoder-level check on the default model
(mutations applied by monkey-patching in a scratch script; the repository is
not touched):

```python
import numpy as np, torch
from fvml import vocoder, streaming
import stream_check                       # run from python_training/
torch.manual_seed(0)
m = vocoder.UnitVocoder().eval()
t = np.arange(30 * 16000) / 16000
wav = (0.2 * np.sin(2*np.pi*180*t)).astype(np.float32)
def run(tag):
    r = stream_check.check_vocoder(m, wav, fps_list=(1, 4))
    print(tag, r["pass"], {k: (f"{v['max_abs']:.1e}", f"{v['snr_db']:.0f} dB") for k, v in r["fps"].items()})
run("baseline")

orig_t = streaming.StreamingConvTranspose1d.step
def zero_cache_t(self, x, cache):
    y, c = orig_t(self, x, cache); return y, torch.zeros_like(c)
streaming.StreamingConvTranspose1d.step = zero_cache_t
run("transposed-conv cache dropped")
```

## Expected

A generator whose transposed convolutions (or dilated convolutions) forget
their left context is not streaming-equivalent to its offline pass, and the
check that exists to prove equivalence reports `pass = False`.

## Actual

Default model, 30 s, `fps` 1 and 4 (`check_vocoder` result):

| Mutation of the streaming caches | `pass` | max abs Δ | SNR |
|---|---|---|---|
| none (baseline) | True | 2.9e-11 | 125.6 dB |
| every `StreamingConv1d` cache dropped | **False**, but only through the boundary-slope criterion; `max_abs` = 5.1e-5 is inside the 1e-4 tolerance | 5.1e-5 | 8–14 dB |
| every `StreamingConvTranspose1d` cache dropped | **True** | 5.0e-7 | 48–54 dB |
| only the dilated `StreamingConv1d` caches dropped | **True** | 6.2e-8 | 60–66 dB |

In the repository's own suite (`pytest tests/test_vocoder.py tests/test_streaming.py -x`):

- the transposed-conv mutation and a "dilated-only" `StreamingConv1d` mutation
  both pass all **11** vocoder-level tests and are killed only by the
  **layer-level** tests in `tests/test_streaming.py`;
- CTest registers `test_vocoder.py` alone as `python_training_stage05`, and
  `test_streaming.py` under `python_training_stage03_04`, so a regression is
  still caught by the suite as a whole, but the Stage 05 entry by itself would
  stay green.

Same pattern in the ONNX smoke test (`gate_report.py:303`): the ORT-vs-PyTorch
comparison `voc_max_abs <= 1e-4` is made on an output of ≤ 4e-5, and it is
made on a single step from an all-zero state, so the caches are not exercised
at all. A graph that dropped FiLM or a residual branch would still pass.

What the check does reach: on a trained checkpoint (the 30-step CUDA smoke
checkpoint recorded in the stage file has a reference step of 0.39) the
absolute tolerance is a real constraint, and the 150k-step DoD run will be
meaningful. This bug is about the evidence available *before* that run and
about regression protection afterwards.

## Root cause

The weight init was chosen for training stability (small residual-branch
gain), and the tolerances were chosen for signals of order 0.1–1 without
checking the amplitude of the signal under test. The boundary criterion
(`b_step <= max_step_ref + 1e-6`) compares against the *global* maximum slope of
the reference, which does not scale down with a local discontinuity either.

## Fix

1. In `check_vocoder` and in the export smoke test, gate on a **relative**
   measure: `snr_db >= 90` (or `max_abs <= 1e-4 * max(1, ref_rms)`), keep the
   absolute bound as a second condition, and report `ref_peak` so a vacuous
   run is visible. With the SNR threshold at 90 dB the three mutants above
   (8–66 dB) all fail.
2. In the test fixture, scale the init so the offline output has `rms ≥ 0.05`
   (e.g. `nn.init.normal_(std=0.1)` on the convs for the test model only) and
   assert `ref.abs().max() > 1e-2` before comparing, so the test refuses to run
   in a vacuous regime.
3. Make the export smoke test run ≥ 3 hops with carried state (`state_out` fed
   back), not one hop from zeros.
4. Add a `past_context_samples()` test that measures the receptive field
   empirically (perturb frame 0, find the last changed sample) — it would also
   have caught the 806 ms vs 606 ms discrepancy in the stage file (see the
   review).

## Workaround

Until fixed, read the `snr_db` printed by `stream_check.py --role vocoder`
rather than the `ok`/`FAIL` flag, and require ≥ 90 dB; run
`python_training_stage03_04` together with `python_training_stage05` after any
change to `fvml/streaming.py`.

## Resolution (verified 2026-10-19)

1. `stream_check.check_vocoder` now gates on `snr_db >= 90.0` (`VOCODER_MIN_SNR_DB`),
   relative `max_abs <= 1e-4 * max(1.0, ref_rms)`, scale-aware boundary slope,
   and `ref_peak > 0.0`, and reports `ref_rms` and `ref_peak`.
2. `export_smoke_test` in `gate_report.py` runs `smoke_hops = 4` carried-state
   hops (feeding `state_out` back as `state_in` on every step) and requires
   `snr_db >= 50.0` (`MIN_VOCODER_SNR_DB`) and `ref_peak > 0.0` alongside the
   relative `max_abs` check.
3. `small_vocoder` in `tests/test_vocoder.py` initializes with `std=0.2` so
   test outputs have `ref_peak > 1e-2` and `ref_rms > 1e-3`;
   `test_vocoder_check_catches_broken_caches` verifies that dropping
   `StreamingConvTranspose1d` caches or dilated `StreamingConv1d` caches causes
   `check_vocoder` to fail (`pass = False`, `snr_db < 90 dB`).
4. `test_vocoder_whole_model_is_causal` empirically verifies that
   `past_context_samples() == 9696` (606 ms) matches the perturbation horizon of
   frame 0.
