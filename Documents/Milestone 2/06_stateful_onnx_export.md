# Stage 06 — Stateful ONNX export & parity

> **Milestone 2** · [Master plan](00_master.md) · [← Stage 05](05_causal_vocoder_decision_gate.md) · [Stage 07 →](07_ort_cpp_runtime.md)

**Status:** ☐ not started
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

- [ ] **`export_stateful_onnx.py`, real implementation.**
  - [ ] CLI: `--checkpoint`, `--role {encoder,vocoder}`, `--frames-per-step
        {1,2,4}`, `--bundle models/<name>/`.
  - [ ] Wrapper modules flatten every cache into ordered `state_in.X` inputs and
        return `state_out.X` with identical shape and dtype. Names are stable and
        derived from module paths.
  - [ ] Static shapes everywhere (batch 1, one hop). Any position or step counter
        is a **tensor state**, never a Python int.
  - [ ] Export with `dynamo=True`. On a capture failure, fall back to
        `dynamo=False` and record which exporter produced each file (model card).
  - [ ] Run `onnx.shape_inference` + `onnx.checker.check_model(full_check=True)`.
  - [ ] Embed `metadata_props["fluxvoice.contract"]` from the checkpoint's stored
        fields; validate it with `fvml.contract`.
  - [ ] Write the encoder and vocoder for s1/s2/s4 into the bundle.
- [ ] **`voices.json`.** One entry per speaker-table row: embedding vector, log-f0
      and energy mean/std (from Stage 05's per-speaker stats), display name,
      source dataset, licence and attribution text.
- [ ] **`MODEL_CARD.md`** per bundle: training data + licences (Hi-Fi TTS
      attribution), gate numbers, exporter path, torch/onnx/onnxruntime
      versions, contract dump, parity results.
- [ ] **Parity** (`parity.py`, extended):
  - [ ] Input: ≥ 60 s of concatenated dev-clean speech, longer than 2× any
        attention or KV window.
  - [ ] Encoder: PyTorch `forward()` offline vs. ORT chunked (s1/s2/s4, states
        fed back): argmax agreement, max |Δlogit|.
  - [ ] Vocoder: PyTorch offline vs. ORT chunked on the same units and
        conditioning: SNR, max |Δ|.
  - [ ] End to end: audio → encoder (ORT) → vocoder (ORT) vs. the PyTorch chain.
- [ ] **Golden I/O for C++.** A 10 s input WAV + the expected ORT-Python chunked
      outputs (units + audio) in `models/<bundle>/golden/`. Stage 07's
      `fv_vc_file` is checked against these.
- [ ] **Cost.** ORT-Python per-step p95 on one core with `intra_op_num_threads=1`,
      sequential execution, s1/s2/s4.

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
  `python_training/parity.py` (extended), `models/<bundle>/**` (generated,
  git-ignored), `models/README.md` (bundle example)

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
