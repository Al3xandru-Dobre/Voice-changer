# models/

Model bundles produced by `python_training/export_stateful_onnx.py` (Milestone 2
/ Stage 06 for Mode A v1; Milestone 3 / Stage 06 for Mode A + Mode B v2). Large
binaries — git-ignored; only this README is tracked.

One directory per bundle:

| File | Contents |
|---|---|
| `models/<bundle>/encoder_s{1,2,4}.onnx` | Causal content encoder, one fixed-shape graph per frames-per-step variant (20/40/80 ms hops at 50 Hz frames). Contract v1 emits `units` (or `content`); contract v2 emits both discrete `units` i64 (Mode A) and continuous `latents` f32 (Mode B). |
| `models/<bundle>/translator_enc.onnx` | *(Contract v2 / Mode B)* Causal chunked Conformer encoder (`chunk_frames = 4` = 80 ms @ 50 Hz) writing cross-attention ring memory `mem_k_*` / `mem_v_*` (`max_src_steps = 128`) indexed by `state_in/out.src_pos`. |
| `models/<bundle>/translator_dec.onnx` | *(Contract v2 / Mode B)* Causal Transformer decoder reading `translator_enc`'s shared `mem_k_*` / `mem_v_*` bank, emitting target `units` `[1, chunk_frames]` i64, predicted target `prosody` `[1, chunk_frames, 4]` f32, and `eos_prob` `[1, 1]` f32. |
| `models/<bundle>/vocoder_s{1,2,4}.onnx` | Causal, stateful, multi-speaker unit vocoder, same `s{1,2,4}` variants. |
| `models/<bundle>/voices.json` | Target voices: speaker embedding, log-f0 / energy statistics, licence, source, optional `attribution` and `lang`. |
| `models/<bundle>/langs.json` | *(Contract v2 / Mode B)* Supported ISO 639-1 language codes, integer token IDs, display names, and per-pair `default_wait_k` (`schema: 1`). |
| `models/<bundle>/MODEL_CARD.md` | Training data, licences and attribution, evaluation numbers. |
| `models/<bundle>/golden/` | 10 s reference input + expected Mode A outputs for the C++ parity check (M2 Stage 07). |
| `models/<bundle>/golden_s2ut/` | *(Mode B)* Multi-utterance reference inputs + expected step decisions, units, prosody and output WAVs for `fv_s2ut_file` parity (M3 Stages 06–07). |

Every `.onnx` carries its **model contract** in
`metadata_props["fluxvoice.contract"]`:
- **Contract v1** ([M2 Stage 02 spec](../Documents/Milestone%202/02_model_contract_fixtures.md)): Mode A-only bundles (`encoder` + `vocoder` + `voices.json`). Remains 100% valid in `--vc` mode.
- **Contract v2** ([M3 Stage 02 spec](../Documents/Milestone%203/02_translator_contract_v2_fixtures.md)): Dual-mode bundles (`encoder` + `translator_enc` + `translator_dec` + `vocoder` + `voices.json` + `langs.json`), validated before opening any audio device by `validate_pair` (Mode A) and `validate_s2ut_bundle` (Mode B).

Tiny deterministic generated bundles that honour contracts v1 and v2 live in
[`tests/data/fixtures/identity/`](../tests/data/fixtures/README.md) and
[`tests/data/fixtures/s2ut/`](../tests/data/fixtures/README.md).

## Building a bundle (M2 Stage 06 / M3 Stage 06)

One run per role, writing into the same bundle directory
([M2 Stage 06 spec](../Documents/Milestone%202/06_stateful_onnx_export.md),
[M3 Stage 06 spec](../Documents/Milestone%203/06_translated_prosody_stateful_onnx_export.md)):

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
- one hop (or 80 ms translation chunk) at batch 1 with static shapes;
- every causal-conv cache, ring-buffer KV state, and translator position counter
  (`src_pos`, `tgt_pos`) is an explicit `state_in.<name>` / `state_out.<name>`
  pair;
- `translator_dec.onnx` reads `translator_enc.onnx`'s cross-attention memory
  buffers (`mem_k_*`, `mem_v_*`) directly as zero-copy shared inputs.

Each run checks every file:
1. strict shape inference and the full `onnx.checker`;
2. opset ≤ 26 and IR ≤ 13, for ONNX Runtime 1.31;
3. `fvml.contract.load_onnx`, plus `validate_pair` (Mode A) and
   `validate_s2ut_bundle` + `validate_s2ut_shared_io` (Mode B).

Each run also measures:
- **Parity** over ≥ 60 s of speech (wrapping any sliding ring-KV window > 5× and
  exercising multi-utterance `kFlush` resets), ONNX Runtime chunked vs. PyTorch:
  - encoder: unit agreement ≥ 99.9%, `latents` max `|Δ| ≤ 1e-5`;
  - translator: `translator_enc` max `|Δ| ≤ 1e-5`, `translator_dec` unit argmax
    agreement ≥ 99.8% and prosody max `|Δ| ≤ 1e-4`;
  - vocoder: SNR ≥ 50 dB.
- **Cost**: per-hop / per-chunk p95 on one thread (`intra_op_num_threads = 1`),
  which must fit in 50% of the step duration.

Besides the table above, the bundle holds:

| File | Contents |
|---|---|
| `export_report.json` | Everything the model card shows, machine-readable: checkpoints and hashes, exporter per file (dynamo, or TorchScript with the reason), contracts, parity, cost, versions |
| `golden/manifest.json` | Voice, alignment rule, tolerances (units exact, audio max abs Δ ≤ 1e-5) and the variants |
| `golden/input.wav` | 10 s of 16 kHz float input |
| `golden/units_s{1,2,4}.npy` | ONNX Runtime chunked encoder ids (int64) |
| `golden/cond.npy` | Vocoder conditioning per 20 ms frame (`[logf0, voicing, aperiodicity, energy]`, f32): Stage 03 tracker with lookahead L, mapped to the voice (scale 1, shift 0) |
| `golden/audio_s{1,2,4}.wav` | ONNX Runtime chunked encoder → vocoder output, float WAV |

Alignment (Mode A): hop *t* feeds the vocoder the encoder's id at *t* and the
tracker's push *t*. Both describe frame *t − L*.
