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
