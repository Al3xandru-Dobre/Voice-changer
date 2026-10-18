# Stage 06 — Stateful ONNX export & parity

> **Milestone 2** · [Master plan](00_master.md) · [← Stage 05](05_causal_vocoder_decision_gate.md) · [Stage 07 →](07_ort_cpp_runtime.md)

**Status:** ◐ in progress (2026-10-18). The exporter, the bundle checks,
`voices.json`, the model card, parity, cost and golden I/O are written and
tested, and the full flow has run on 60 s of dev-clean with random-weight
models. Still waiting for a trained student (Stage 04), a trained vocoder and
the gate decision (Stage 05), which the real bundle needs (see Verification).
**Depends on:** Stage 02 (contract), Stage 05 (gated encoder + fine-tuned vocoder)
**Research:** §8 stateful export · §2.2 (KV-ring pitfalls, if Mimi won)

## Objective

Export the gate's winning encoder head and the fine-tuned vocoder as **fixed-shape
stateful ONNX graphs**:
- one graph per frames-per-step variant (1/2/4);
- every recurrent tensor exposed as a `state_in.X` / `state_out.X` pair;
- the contract embedded in each file.

Package them as a model bundle with `voices.json` and a model card. Prove that
ONNX Runtime run chunk by chunk reproduces PyTorch run offline.

## Context & references

- `python_training/export_stateful_onnx.py` is a stub. Its checklist (wrap
  states, export, `onnx.checker`, ORT parity) stays valid; this stage implements
  it against the Stage 02 contract.
- **Exporter.** PyTorch ≥ 2.9 defaults to `torch.onnx.export(..., dynamo=True)`;
  `dynamo=False` keeps the legacy TorchScript exporter (research §8).
- **Pitfalls to design out:**
  - Python-int cache offsets frozen by tracing (garbage audio, NaNs).
  - Graphs fixed to the trace length.
  - KV ring-buffer wrap drift that only appears after the attention window (a
    community Mimi export drifted after 10 s).
- Stage 04/05 modules already run through `StreamingConv1d.step()` with explicit
  caches. Export wraps `step()`, not `forward()`.

## Tasks

- [x] **`export_stateful_onnx.py`, real implementation.**
  - [x] CLI: `--checkpoint`, `--role {encoder,vocoder}`, `--frames-per-step
        {1,2,4}`, `--bundle models/<name>/`. Additions:
        - `--head` (encoder delay head);
        - `--license` (encoder checkpoints store none);
        - `--exporter`;
        - `--parity-*`;
        - `--gate-report` / `--gate-variant`;
        - `--voice`;
        - `--cost-steps`;
        - `--skip-parity`.
  - [x] Wrapper modules flatten every cache into ordered `state_in.X` inputs and
        return `state_out.X` with identical shape and dtype. Names are stable and
        derived from module paths.
  - [x] Static shapes everywhere (batch 1, one hop). Any position or step counter
        is a **tensor state**, never a Python int.
  - [x] Export with `dynamo=True`. On a capture failure, fall back to
        `dynamo=False` and record which exporter produced each file (model card).
  - [x] Run `onnx.shape_inference` + `onnx.checker.check_model(full_check=True)`.
  - [x] Embed `metadata_props["fluxvoice.contract"]` from the checkpoint's stored
        fields; validate it with `fvml.contract`.
  - [x] Write the encoder and vocoder for s1/s2/s4 into the bundle.
- [x] **`voices.json`.** One entry per speaker-table row: embedding vector, log-f0
      and energy mean/std (from Stage 05's per-speaker stats), display name,
      source dataset, licence and attribution text.
- [x] **`MODEL_CARD.md`** per bundle: training data + licences (Hi-Fi TTS
      attribution), gate numbers, exporter path, torch/onnx/onnxruntime
      versions, contract dump, parity results.
- [x] **Parity** (in `fvml/export.py`, run by the exporter; see Design
      decisions):
  - [x] Input: ≥ 60 s of concatenated dev-clean speech, longer than 2× any
        attention or KV window.
  - [x] Encoder: PyTorch `forward()` offline vs. ORT chunked (s1/s2/s4, states
        fed back): argmax agreement, max |Δlogit|.
  - [x] Vocoder: PyTorch offline vs. ORT chunked on the same units and
        conditioning: SNR, max |Δ|.
  - [x] End to end: audio → encoder (ORT) → vocoder (ORT) vs. the PyTorch chain.
- [x] **Golden I/O for C++.** A 10 s input WAV + the expected ORT-Python chunked
      outputs (units + audio) in `models/<bundle>/golden/`. Stage 07's
      `fv_vc_file` is checked against these.
- [x] **Cost.** ORT-Python per-step p95 on one core with `intra_op_num_threads=1`,
      sequential execution, s1/s2/s4.
- [ ] **Real bundle.** Run both roles on the trained student and the post-gate
      vocoder, with `--gate-report`. This needs Stages 04/05.

## Implementation notes

- **Fixed shapes per hop.** One file per frames-per-step keeps every shape
  static, so ORT can plan memory once and allocate nothing per run. The bundle
  is small, so three variants cost little disk. C++ picks the variant matching
  `--frames-per-step`.
- **State order and names are the contract.** C++ resolves names to indices at
  load (Stage 07). Reordering is harmless, but renaming means a contract change.
- **Discrete argmax can flip on near-ties.** An agreement threshold (≥ 99.9%) is
  the right check for unit ids, and logit |Δ| catches real drift.
- **If Mimi won:**
  - export its KV cache as a ring buffer with an absolute-position tensor state
    and a mask computed from it;
  - parity must run for at least 2 × 10 s (the 25 Hz × 250-step window) — the
    60 s requirement already covers that.
- **FP32 only.** FP16/INT8 are Milestone 4 work, and their quality/speed
  trade-off is measured there.

## Files touched

- `python_training/export_stateful_onnx.py` (real implementation),
  `python_training/fvml/export.py` (new: step graphs, export + checks, ORT
  streaming, parity, conditioning, cost, voices),
  `python_training/tests/test_export.py` (new), `models/<bundle>/**`
  (generated, git-ignored), `models/README.md` (bundle example). Not
  extended: `python_training/parity.py`. Its two-file compare already covers
  Stage 07's C++-vs-golden check.

## Definition of done

- Every exported file passes `onnx.checker` full check, and `fvml.contract`
  accepts its embedded contract. The encoder/vocoder pair passes
  `validate_pair`.
- ORT chunked vs. PyTorch offline over ≥ 60 s:
  - encoder argmax agreement ≥ 99.9%;
  - vocoder SNR ≥ 50 dB given identical inputs;
  - the end-to-end chain's unit agreement and waveform SNR are reported (a
    near-tie argmax flip changes the audio legitimately, so this is not a
    threshold).
- ORT-Python per-step p95 with `intra_op_num_threads=1` is within the gate's
  compute budget (≤ 50% of hop) for the chosen variant.

## Design decisions

- **State names are module paths.** One `state_in.<path>` /
  `state_out.<path>` pair per `StreamingConv1d` / `StreamingConvTranspose1d`
  with context > 0, for example `state_in.trunk.blocks.0.down`. Order is
  registration order (`fvml.streaming.state_layout`). The convs' caches are
  the whole recurrent state, so neither model has a step or position counter.
  Parity runs thousands of hops from a fresh state, which is what would catch
  one frozen into the trace.
- **The encoder graph holds the trunk and the chosen head only.** The other
  heads and their caches are left out. For L1 that is 21 states / 42 566
  floats. `units` is computed in the graph by ArgMax (i64).
- **Logits for parity without breaking the contract.** Contract v1 rejects
  extra outputs. The exporter therefore exports `logits` as a graph output,
  then drops it from the shipped file. The tensor is still computed, since it
  feeds ArgMax. For parity, `fvml.export.with_outputs` adds it back in memory,
  so parity runs on the shipped file itself.
- **Exporter.**
  - `--exporter auto` tries `dynamo=True` with `external_data=False`, so each
    role is a single file.
  - On any exception it falls back to TorchScript at opset 18 and records the
    one-line reason in `export_report.json` and the model card.
  - Trace warnings are recorded the same way. The dynamo exporter's console
    output is swallowed (its emoji crash cp1252 pipes).
  - IR is capped at 13, and opset > 26 is refused: ORT 1.31.
  - Today's dynamo path gives opset 20 / IR 10. TorchScript gives 18 / 8.
- **Vocoder API.** The exporter uses `fvml.vocoder.UnitVocoder` (Stage 05) as
  written:
  - `contract(fps, license)`, whose states are the model's state layout;
  - `step(units, **cond, state=)` and `forward(units, **cond)`, with `cond`
    keyed by the contract's `cond_inputs`;
  - `voices_json(speakers, speaker_stats, extra)`, fed from the checkpoint's
    `speakers` / `speaker_stats` / optional `speaker_metadata`.
  
  An optional checkpoint `license` (or `--license`) overrides the config's.
  `overlap_samples` > 0 is refused: stateful causal vocoders need no OLA.
- **Conditioning = the run-time path.** For the vocoder, chain and golden
  inputs, `fvml.features.PitchTracker` is pushed one 20 ms frame at a time
  with lookahead L = the encoder head's. Push *t* returns frame *t − L*, the
  frame the encoder's id at *t* describes. The features are then mapped with
  `map_to_target` (scale 1, shift 0). Stage 08 does the same in C++, so
  Stage 07's golden check compares like with like.
- **Parity lives in `fvml/export.py`, not `parity.py`.** `parity.py` belongs to
  Stage 01 and stays a two-file compare. That is exactly what Stage 07 needs
  for `fv_vc_file` output vs. `golden/`. The chunked-ORT runners reuse
  `fvml.parity.compare`.
- **Budget gate = the chosen variant.** Encoder + vocoder p95 per hop is
  measured for every variant. Only the gate's variant is a gate: from
  `--gate-variant`, or from the winner in `gate_report.py`'s JSON. If a gate
  report is given, the exported head's lookahead must equal the winner's.
- **Re-exports.** A run with the same checkpoint merges variants into the
  bundle. A run with a different checkpoint deletes that role's variants it
  did not re-export, and the chain results, so the bundle never mixes models.

## Verification

Measured 2026-10-18 on a shared CPU (other sessions were training and testing,
so cost varies run to run).

- **Tests:** `tests/test_export.py`, 20 passed in about 35 s. The full
  `python_training` suite gives 157 passed. Coverage:
  - state names = trunk + one head;
  - every file passes the checker and contract;
  - IR capping, and refusal of opset 27;
  - a stale contract is caught;
  - dynamo export + parity for both roles;
  - fallback reason recorded;
  - parity at s1/s2;
  - conditioning equals `extract()` shifted by L and mapped to the voice;
  - voices.json from the checkpoint;
  - CLI end to end (bundle, golden reproducible, stale-variant cleanup);
  - gate-variant / head verdicts.
  
  Mutation checks:
  - a stream that drops its states fails parity (max |Δlogit| > 1e-2);
  - units one frame late fail the 50 dB gate.
- **Full-size dry run.** The default student (L1) and the default
  `UnitVocoder` (1.06 M params), random weights. 60 s of dev-clean, one
  utterance per speaker in turn:
  - Every file is dynamo-exported (opset 20 / IR 10). Encoder 16.9 MB, 21
    states; vocoder 4.8 MB, 80 states.
  - All pass the checks. Pairs are valid: algorithmic latency 40 / 60 / 100 ms
    at s1 / s2 / s4.
  - Encoder parity: unit agreement **1.00000**, max |Δlogit| **1.7–2.1e-6**
    (s1/s2/s4).
  - Vocoder parity: SNR 135 dB. The random-init vocoder is nearly silent, so
    the tests use an audible tiny vocoder (RMS > 0.01): ≥ 126 dB at s1/s2.
  - Chain: unit agreement 1.00000.
  - golden/ written for s1/s2/s4.
- **Cost** (ORT 1.31, one thread, per hop; p50 / p95 ms, random weights):

  | | s1 (20 ms) | s2 (40 ms) | s4 (80 ms) |
  |---|---|---|---|
  | encoder | 4.5 / 7.3 | 9.0 / 12.2 | 16.2 / 24.2 |
  | vocoder | 3.0 / 5.1 | 3.7 / 6.2 | 5.6 / 9.4 |
  | encoder + vocoder | 6.8 / **10.9 (54%)** | 12.0 / **17.2 (43%)** | 21.6 / **29.2 (36%)** |

  - The default pair misses the 50% budget at s1 on this loaded machine and
    fits at s2/s4. The gate (Stage 05) measures the same thing for its
    decision.

**Open:**
- Run both roles on the trained student and post-gate vocoder with
  `--gate-report`. Record the real numbers here and in the bundle's model
  card.
- The gate's cost and Stage 07's C++ numbers will tell whether s1 needs a
  cheaper `--channels` student.
- Mimi is not exportable by this exporter: no ring-KV wrapper, and its
  encoder role is ContentEncoder-only. This stays moot unless Mimi survives
  Stage 04's kill test.
