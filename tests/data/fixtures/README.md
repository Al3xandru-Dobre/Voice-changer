# Fixture models (Milestone 2 / Stage 02 & Milestone 3 / Stage 02)

Tiny ONNX graphs that honour [model contract v1](../../../Documents/Milestone%202/02_model_contract_fixtures.md)
(Mode A) and [model contract v2](../../../Documents/Milestone%203/02_translator_contract_v2_fixtures.md)
(Mode B simultaneous S2UT), so the C++ runtime can be built and tested without
trained weights. Generated, never hand-edited:

    uv run --project python_training python tools/make_fixture_models.py          # write
    uv run --project python_training python tools/make_fixture_models.py --check  # must say "up to date"
    python -I tools/make_fixture_models.py --self-test                            # stdlib-only check

The bytes regenerate identically (`onnx.helper` only, opset 17 / IR 8, fixed
producer, no timestamps). CC0-1.0: they contain no trained weights.

| File | Role & Contract | What it does |
|---|---|---|
| `identity/encoder_s{1,2,4}.onnx` | v1 encoder, soft content (dim 320) | `audio [1,hop]` → `content [1,N,320]` (a reshape) |
| `identity/vocoder_s{1,2,4}.onnx` | v1 vocoder | `content` → `audio [1,hop]`; takes `logf0`, `voicing`, `aperiodicity`, `energy` `[1,N]` and `spk_emb [1,4]` and ignores them |
| `identity/voices.json` | — | voices `identity` and `identity_b` (for `--voice` and blending tests) |
| `delay_s1.onnx` | v1 vocoder, L = 1 | `audio(t) = content(t−1)` via `state_in.buf` → `state_out.buf` |
| `accumulator_s1.onnx` | v1 encoder | `state_out.acc = state_in.acc + Σ audio`; `content = state_out.acc` |
| `s2ut/encoder_s{1,2,4}.onnx` | v2 dual-output encoder (`vocab=16`, `latent_dim=8`) | `audio [1,hop]` → `units [1,N]` i64 + `latents [1,N,8]` f32 |
| `s2ut/translator_enc.onnx` | v2 `translator_enc` (`chunk_frames=4`, `max_src_steps=4`) | Writes chunk `latents` summary into 4-slot ring `mem_k_0` / `mem_v_0` at `src_pos mod 4` and increments `src_pos` |
| `s2ut/translator_dec.onnx` | v2 `translator_dec` (`vocab=16`, `bos=16, eos=17, pad=18`) | Reads delayed `mem_k_0` / `mem_v_0` slot, emits `(prev_units + tgt_lang) mod 16` + `prosody [1,4,4]`, increments `tgt_pos`, raises `eos_prob = 1.0` after 3 flush steps |
| `s2ut/vocoder_s{1,2,4}.onnx` | v2 discrete vocoder (`vocab=16`) | `units [1,N]` + prosody conditioning → `audio [1,hop]` |
| `s2ut/voices.json`, `s2ut/langs.json` | — | v2 bundle voice and bidirectional `{en,es,fr,de}` language tables |
| `broken/*.onnx` | — | valid ONNX that breaks contract v1 or v2; `manifest.json` records the exact error each must be rejected with |
| `s2ut_mismatch/translator_enc_latent16.onnx` | v2 `translator_enc`, `latent_dim=16` (`latents`, `enc_conv`, `mem_*` sized 16) | loads and runs alone; replacing `translator_enc` in `s2ut_s4` breaks the bundle (`latent_dim mismatch: encoder 8, translator_enc 16`) and `translator_dec` cannot bind its `[1,4,8]` memory to the `[1,4,16]` ring |
| `s2ut_mismatch/translator_dec_mem1.onnx` | v2 `translator_dec`, `shared_mem_states = ["mem_k_1", "mem_v_1"]` | loads alone; breaks the bundle (`shared_mem_states mismatch between translator_enc and translator_dec`) and cannot bind to `translator_enc` (no `mem_k_1` state) |
| `s2ut_mismatch/translator_dec_langs.onnx` | v2 `translator_dec`, `langs = {en: 0, es: 1, fr: 2, it: 3}` | loads alone; breaks the bundle (`langs mismatch between translator_enc and translator_dec`) |
| `manifest.json` | — | SHA256, contract string, graph I/O and expected error per file; Mode A pairs, Mode B quartets with latencies and Mode B quartet mismatches |

`s2ut_mismatch/` fixtures are ordinary valid models in `manifest.json`'s `models`
list (`"error": null`, so the broken-fixture loops skip them). The top-level
`s2ut_mismatch` list records, per file, the quartet it is swapped into
(`"quartet": "s2ut_s4"`), the role it `"replaces"`, and the exact
`validate_s2ut_bundle()` error. `--self-test` re-derives every error from the
manifest; `tests/test_ort_model.cpp` loads each file in ONNX Runtime and checks
the same error, plus the `bind_shared_inputs()` refusals.

## Hand-computed latencies

Mode A pairs (`hop + (L_enc + L_voc) × frame_samples + overlap_samples`):

| Pair | Latency |
|---|---|
| `identity_s1` | 320 samples (20 ms) |
| `identity_s2` | 640 samples (40 ms) |
| `identity_s4` | 1280 samples (80 ms) |
| `delay_s1` (identity encoder s1 + `delay_s1`) | 640 samples (40 ms) |

Mode B S2UT quartets (`wait_k × chunk_frames × frame_samples + (L_enc + L_voc) × frame_samples + overlap_samples`, with `chunk_frames = 4`, `frame_samples = 320`, `L_enc = L_voc = overlap = 0`):

| `wait_k` | Onset latency |
|---:|---|
| `1` | 1280 samples (80 ms) |
| `3` | 3840 samples (240 ms) |
| `5` | 6400 samples (400 ms) |
| `10` | 12800 samples (800 ms) |

Checked by `tests/test_model_contract.cpp` (manifest, no ORT),
`python_training/tests/test_contract.py` (stdlib),
`python_training/tests/test_fixtures.py` (runs both v1 and v2 chains in ONNX Runtime), and
`tests/test_ort_model.cpp` (the v2 quartet in the C++ runtime: zero-copy `mem_*`/`src_pos`
sharing across all four `(translator_enc bank, translator_dec parity)` bindings).
