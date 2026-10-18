# models/

Model bundles produced by `python_training/export_stateful_onnx.py` (Milestone 2
/ Stage 06). Large binaries — git-ignored; only this README is tracked.

One directory per bundle:

| File | Contents |
|---|---|
| `models/<bundle>/encoder_s{1,2,4}.onnx` | Causal content encoder, one fixed-shape graph per frames-per-step variant (20/40/80 ms hops at 50 Hz frames) |
| `models/<bundle>/vocoder_s{1,2,4}.onnx` | Causal, stateful, multi-speaker unit vocoder, same variants |
| `models/<bundle>/voices.json` | Target voices: speaker embedding, log-f0 / energy statistics, licence, source, attribution |
| `models/<bundle>/MODEL_CARD.md` | Training data, licences and attribution, evaluation numbers |
| `models/<bundle>/golden/` | 10 s reference input + expected outputs for the C++ parity check (Stage 07) |

Every `.onnx` carries its **model contract** in
`metadata_props["fluxvoice.contract"]`: sample rate, frame geometry,
lookahead, tensor names, state pairs and licence
([Stage 02 spec](../Documents/Milestone%202/02_model_contract_fixtures.md)).
The engine validates it, and checks that encoder and vocoder pair up, before
any audio device opens. A tiny generated bundle that honours the contract lives
in [`tests/data/fixtures/identity/`](../tests/data/fixtures/README.md).

Milestone 3 adds the Wait-k translator (`role: "translator"`) to the same
layout.

## Building a bundle (Stage 06)

One run per role. Both runs write into the same directory
([Stage 06 spec](../Documents/Milestone%202/06_stateful_onnx_export.md)):

```powershell
cd python_training
.venv\Scripts\python.exe export_stateful_onnx.py --role encoder `
    --checkpoint ..\checkpoints\<student>\best.pt --head student-L1 `
    --license "<licence text>" --bundle ..\models\<bundle>
.venv\Scripts\python.exe export_stateful_onnx.py --role vocoder `
    --checkpoint ..\checkpoints\<vocoder>\best.pt --bundle ..\models\<bundle> `
    --gate-report ..\data\reports\<gate>.json
```

What each `.onnx` is:
- one hop at batch 1 with static shapes;
- every causal-conv cache is a `state_in.<module path>` / `state_out.<module path>`
  pair, e.g. `state_in.trunk.blocks.0.down`;
- there is no step or position counter.

Encoders run the trunk plus the chosen delay head and output `units` (argmax,
int64). Their logits are not a graph output.

Each run checks every file:
1. strict shape inference and the full `onnx.checker`;
2. opset ≤ 26 and IR ≤ 13, for ONNX Runtime 1.31;
3. `fvml.contract.load_onnx`, plus `validate_pair` for each s1/s2/s4 pair.

Each run also measures:
- **Parity** over ≥ 60 s of dev-clean, ONNX Runtime chunked vs. PyTorch
  offline:
  - encoder: unit agreement ≥ 99.9%;
  - vocoder: SNR ≥ 50 dB;
  - the end-to-end chain is reported, not gated.
- **Cost**: per-hop p95 on one thread. The gate-chosen variant must fit in 50%
  of the hop.

Besides the table above, the bundle holds:

| File | Contents |
|---|---|
| `export_report.json` | Everything the model card shows, machine-readable: checkpoints and hashes, exporter per file (dynamo, or TorchScript with the reason), contracts, parity, cost, versions |
| `golden/manifest.json` | Voice, alignment rule, tolerances (units exact, audio max abs Δ ≤ 1e-5) and the variants |
| `golden/input.wav` | 10 s of 16 kHz float input |
| `golden/units_s{1,2,4}.npy` | ONNX Runtime chunked encoder ids (int64) |
| `golden/cond.npy` | Vocoder conditioning per 20 ms frame (`[logf0, voicing, aperiodicity, energy]`, f32): Stage 03 tracker with lookahead L, mapped to the voice (scale 1, shift 0) |
| `golden/audio_s{1,2,4}.wav` | ONNX Runtime chunked encoder → vocoder output, float WAV |

Alignment: hop *t* feeds the vocoder the encoder's id at *t* and the tracker's
push *t*. Both describe frame *t − L*.
