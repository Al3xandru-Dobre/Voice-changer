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

## Exit criteria

- Mode B live: mic → source units → target-language units → vocoder,
  ~300–500 ms; translation quality sanity-checked on held-out pairs.

## Stage files

_TBD — authored after the M2 stateful-ONNX patterns settle._
