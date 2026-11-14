# Stage 09 — Quantization: FP16 export and INT8 evaluation

> **Milestone 4** · [Master plan](00_master.md) · [← Stage 08](08_execution_providers.md) · [Stage 10 →](10_audio_extensions.md)

**Status:** ☐ not started
**Depends on:** [M2 / Stage 06](../Milestone%202/06_stateful_onnx_export.md) with a **trained** bundle
(the real bundle is still pending there), [Stage 08](08_execution_providers.md) (FP16 only pays off on a GPU provider),
[M3 / Stage 06](../Milestone%203/06_translated_prosody_stateful_onnx_export.md) for the translator graphs
**Owner persona:** `@onnx-ort-runtime` with `@ml-speech-trainer` for the evaluation

## Objective

Produce FP16 and INT8 variants of each graph, measure quality against speed,
and expose a precision selector only for the variants that pass. The decision
is made from numbers on the shipped runtime, as with the M2 gate.

## Context & verified starting point (2026-11-14)

- `python_training/export_stateful_onnx.py` has no `--fp16` or INT8 option.
  `fvml/export.py:69-70` only maps the `f16` dtype name for tensors.
- `ModelContract.hpp` knows `TensorDtype::kF16` (used for the v2 shared-memory
  states). Contract v1 I/O is `f32` / `i64`.
- `OrtModel` and `KVStateCache` bind `f32` buffers.
- No bundle exists under `models/` (only `README.md`), so no measurement can
  be taken today.

## Tasks

- [ ] **FP16 export** (`export_stateful_onnx.py --precision fp16`):
  - [ ] Convert weights and internal tensors to float16 and **keep graph
        inputs, outputs and states as `f32`** at the boundary. The contract,
        `KVStateCache` and the C++ bindings then need no change.
  - [ ] Keep numerically sensitive nodes in FP32 through a block list (start
        with normalisation statistics and the final waveform layers; adjust
        from the parity results).
  - [ ] File naming `encoder_s{N}.fp16.onnx`, `vocoder_s{N}.fp16.onnx`
        (and `translator_enc` / `translator_dec` when M3 lands). Add a
        `precision` key to the embedded contract metadata and to
        `export_report.json`. An FP32-only bundle stays valid.
  - [ ] Parity gates against the FP32 PyTorch reference over ≥ 60 s: unit
        agreement ≥ 99.5 %, vocoder SNR ≥ 35 dB, end-to-end chain reported.
- [ ] **INT8 evaluation** (`python_training/quantize_onnx.py`, new):
  - [ ] Static quantization with a calibration set drawn from the frozen
        evaluation manifests (never from the test list). Recurrent states stay
        float at the graph boundary.
  - [ ] Quantize the encoder first; treat the vocoder as a separate
        experiment (transposed convolutions and waveform output are the known
        weak points). Record which operators stayed in float.
  - [ ] Streaming check: chunked INT8 ORT against the offline FP32 reference
        for more than 2× the longest context, to catch state drift that only
        appears after many hops.
- [ ] **Evaluation table** (`eval_vc.py`, and `eval_s2ut.py` once M3 exists)
      on the frozen manifests, for FP32 / FP16 / INT8 and each provider that
      Stage 08 enabled: WER, CER, ECAPA target similarity, UTMOS, F0 PCC,
      ASR-BLEU (Mode B), per-hop p95, RTF, model size.
- [ ] **Acceptance rule** (written before running, like the M2 gate):
      a variant ships only if WER rises by ≤ 1.0 point absolute, similarity
      drops by ≤ 0.02, UTMOS drops by ≤ 0.1, and p95 improves by ≥ 15 % on its
      target provider. Otherwise it is recorded as rejected, with the numbers.
- [ ] **Runtime:** `ModelBundle::load` takes a precision preference
      (`fp32 | fp16 | int8 | auto`) and reports which file it loaded. A missing
      variant is an error for an explicit request and a fall-through for
      `auto` (FP32 is always the last resort and is named in the log).
- [ ] **CLI and GUI:** `--precision` on `fluxvoice`, `fv_vc_file` and
      `fv_bench`; Settings → Inference shows only the precisions present in
      the active bundle, with the provider they were validated on.
- [ ] **Model card:** `MODEL_CARD.md` gains a precision section (parity,
      block list, evaluation row, validated providers).
- [ ] **Tests:**
  - [ ] Python: fixture graphs convert to FP16 with an `f32` boundary and pass
        the checker, the contract round-trip and a short parity run.
  - [ ] C++: `ModelBundle` precision selection over a fixture directory
        (explicit missing → error; `auto` → documented order).
  - [ ] Contract: a `precision` value outside the allowed set is rejected by
        both validators, with a fixture under `tests/data/contracts/invalid/`.

## Implementation notes

- **FP16 on the CPU provider is usually slower**, because most CPU kernels
  compute in FP32 and insert casts. Treat FP16 as a GPU-provider variant and
  say so in the UI.
- **INT8 may be rejected.** A neural vocoder is sensitive to quantization
  noise. "INT8 evaluated and rejected, numbers attached" satisfies this
  stage's scope ("then INT8 eval").
- **Gate hygiene** (`MEMORY.md` rule 11): measure on ORT stateful timings, use
  a relative tolerance on a non-silent signal, and report any converter
  fallback.

## Files touched

- `python_training/export_stateful_onnx.py`, `python_training/fvml/export.py`,
  `python_training/quantize_onnx.py` (new), `python_training/fvml/contract.py`,
  `python_training/tests/test_export.py`, `src/pipeline/ModelContract.hpp`,
  `src/pipeline/ModelBundle.{hpp,cpp}`, `src/main.cpp`, `tools/fv_vc_file.cpp`,
  `tools/fv_bench.cpp`, `frontend/core/Settings.*`,
  `frontend/imgui/ViewSettings.cpp`, `models/README.md`,
  `tests/data/contracts/**`

## Definition of done

- FP16 variants of every graph export, pass the parity gates and load in C++
  with no contract or binding change.
- The evaluation table for FP32 / FP16 / INT8 is recorded in this file with
  the acceptance rule applied to each row.
- The precision selector works from the CLI and the GUI and never loads a
  variant that the bundle does not contain without saying so.
