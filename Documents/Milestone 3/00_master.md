# Milestone 3 — Wait-k S2UT Mode

> Parent: [Development milestones](../../README.md#7-development-milestones) ·
> [Documents index](../README.md)

**Status:** ☐ draft — stage files to be written when M2 closes
**Goal from README:** *Train/fine-tune causal S2UT transformer on discrete
units. Implement `WaitKPolicy` in C++ and connect between encoder and
vocoder.*

## Scope sketch

- Training path: fairseq direct-S2UT recipe or StreamSpeech codebase
  (RESEARCH_GUIDE §1); datasets CVSS / VoxPopuli (§5).
- Wait-k execution per README §3: READ / WRITE / tail-flush — the pure state
  machine already exists in `src/pipeline/WaitKPolicy.hpp` (unit-test it here).
- Alternative worth evaluating before committing to Wait-k: Hibiki's adaptive
  multistream approach (RESEARCH_GUIDE §1) — a short spike/comparison stage
  should open this milestone.
- C++: translator `Ort::Session` between encoder and vocoder; dual KV-caches;
  `<SILENCE/WAIT>` token handling during READ steps.
- **Inherited from M2** ([decision gate](../Milestone%202/05_causal_vocoder_decision_gate.md),
  [research §2.6](../Research_documents/M2_TEXTLESS_VC_RESEARCH.md)):
  - **Units the translator emits.** The M2 gate fixes the unit frame rate,
    vocabulary and hop: expected 50 Hz discrete HuBERT km100 ids (20 ms frames)
    rather than 12.5 Hz Mimi tokens. M2 / Stage 11 records the final values.
  - **Wait-k at 50 Hz.** `WaitKPolicy` counts k in 80 ms chunks today. At 50 Hz,
    k is counted in hops and sequences are 4× longer per second.
  - **Multilingual units.** The M2 teacher (HuBERT-Base) is English-only.
    mHuBERT-147 covers 147 languages but is CC-BY-NC-SA-4.0 (non-commercial);
    decide at this milestone's spike.
  - **Translator contract.** It plugs in through the M2 model contract
    (`role: "translator"`, state pairs, `ModelContract.hpp`) and the existing
    `OrtModel` runtime.

## Exit criteria

- Mode B live: mic → source units → target-language units → vocoder,
  ~300–500 ms; translation quality sanity-checked on held-out pairs.

## Stage files

_TBD — authored after the M2 stateful-ONNX patterns settle._
