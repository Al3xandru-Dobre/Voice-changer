# Milestone 2 — Textless VC Mode (Encoder + Unit-Vocoder)

> Parent: [Development milestones](../../README.md#7-development-milestones) ·
> [Documents index](../README.md)

**Status:** ☐ draft — stage files to be written when M1 closes
**Goal from README:** *Export causal encoder + custom Unit-HiFi-GAN to ONNX.
Run Mode A (voice changer without translation) in C++ with stateful KV/CNN
padding caches.*

## Scope sketch

- Choose encoder path: **Mimi semantic codebook** (recommended) vs. causal
  HuBERT + k-means — see RESEARCH_GUIDE §2.
- Record 1–3 h target-voice dataset; run `python_training/extract_units.py`.
- Train causal Unit-HiFi-GAN (`train_unit_vocoder.py`) — all convolutions
  causal (RESEARCH_GUIDE §3, StreamVoice/X-VC).
- Export both stages stateful (`export_stateful_onnx.py`); acquire ONNX
  Runtime SDK (`third_party/onnxruntime/README.md`).
- C++: `Ort::Session` wrapper, `KVStateCache`-backed state round-trip,
  wire into the M1 worker thread → OLA → output queue.

## Exit criteria

- Mode A runs live: mic → units → vocoder → target voice out, ~60–100 ms.
- Stateful ONNX round-trip verified (parity vs. PyTorch on chunked input).

## Stage files

_TBD — will be authored as `01_<stage>.md …` following the Milestone 1 pattern
once M1 verification numbers are in._
