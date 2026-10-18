# Fixture models (Milestone 2 / Stage 02)

Tiny ONNX graphs that honour [model contract v1](../../../Documents/Milestone%202/02_model_contract_fixtures.md),
so the C++ runtime (Stages 07-08) can be built and tested without trained
weights. Generated, never hand-edited:

    uv run --project python_training python tools/make_fixture_models.py          # write
    uv run --project python_training python tools/make_fixture_models.py --check  # must say "up to date"

The bytes regenerate identically (`onnx.helper` only, opset 17 / IR 8, fixed
producer, no timestamps). CC0-1.0: they contain no trained weights.

| File | Role | What it does |
|---|---|---|
| `identity/encoder_s{1,2,4}.onnx` | encoder, soft content (dim 320) | `audio [1,hop]` → `content [1,N,320]` (a reshape) |
| `identity/vocoder_s{1,2,4}.onnx` | vocoder | `content` → `audio [1,hop]`; takes `logf0`, `voicing`, `aperiodicity`, `energy` `[1,N]` and `spk_emb [1,4]` and ignores them |
| `identity/voices.json` | — | voices `identity` and `identity_b` (for `--voice` and blending tests) |
| `delay_s1.onnx` | vocoder, L = 1 | `audio(t) = content(t−1)` via `state_in.buf` → `state_out.buf` |
| `accumulator_s1.onnx` | encoder | `state_out.acc = state_in.acc + Σ audio`; `content = state_out.acc` |
| `broken/*.onnx` | — | valid ONNX that breaks the contract; `manifest.json` records the exact error each must be rejected with |
| `manifest.json` | — | SHA256, contract string, graph I/O and expected error per file; the pairs and their algorithmic latency |

`identity/` is a complete bundle in the `models/<bundle>/` layout, so an
encoder → vocoder run is a bit-exact passthrough at s1/s2/s4 (Stage 08 measures
the VC plumbing with it). Pairs and their hand-computed algorithmic latency
(`hop + (L_enc + L_voc) × frame_samples + overlap_samples`):

| Pair | Latency |
|---|---|
| `identity_s1` | 320 samples (20 ms) |
| `identity_s2` | 640 samples (40 ms) |
| `identity_s4` | 1280 samples (80 ms) |
| `delay_s1` (identity encoder s1 + `delay_s1`) | 640 samples (40 ms) |

For a closed-form accumulator check, feed values whose partial sums are exact
in float32 (e.g. a constant 0.25: 80 per hop), so the result does not depend on
the order in which ONNX Runtime's ReduceSum adds.

Checked by `tests/test_model_contract.cpp` (manifest, no ORT) and
`python_training/tests/test_fixtures.py` (runs them in ONNX Runtime).
