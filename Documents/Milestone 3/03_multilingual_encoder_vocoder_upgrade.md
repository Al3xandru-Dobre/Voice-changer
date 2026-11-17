# Stage 03 — Multilingual encoder & vocoder upgrade

> **Milestone 3** · [Master plan](00_master.md) · [← Stage 02](02_translator_contract_v2_fixtures.md) · [Stage 04 →](04_waitk_policy_streaming_state_machine.md)

**Status:** ◐ implemented & smoke-tested (2026-11-15). The teacher registry, the balanced K=1000 fit, K=1000 labelling with causal prosody, the dual-output D=128 student, student-latent and silence-bank extraction, the multilingual vocoder data path, the probe and scoring hooks, and the contract-v2 dual-output export are in and unit-tested on synthetic data and random weights. The streaming parity DoD holds on random weights. Pending on corpora, the 1.1 GB `mhubert-vp` checkpoint and GPU time: the 100 h codebook fit, the probe table, both training runs and every quality number in the Definition of done. See [Verification](#verification-2026-11-15-windows--code--smoke-runs-only).
**Depends on:** Stage 01 (multilingual manifests, CSS10/CML-TTS), Stage 02 (contract v2 dual-output encoder spec), [Milestone 2 / Stages 04–05](../Milestone%202/05_causal_vocoder_decision_gate.md) (student encoder + Unit-HiFi-GAN codebase)
**Research:** [M3 research](../Research_documents/M3_WAITK_S2UT_RESEARCH.md) §3 multilingual unit & encoder representation · §10 licences

## Objective

Upgrade Stage 1 (the 50 Hz causal conv encoder) and Stage 3 (the causal
multi-speaker Unit-HiFi-GAN vocoder) from English-only (`HuBERT-Base` km100) to
unified `{en, es, fr, de}` coverage without breaking Mode A latency or quality:
1. Run a **multilingual teacher probe** comparing Meta's MIT-licensed
   `mhubert_base_vp_en_es_fr_it3` (layer-11 $K=1000$, refitted on balanced
   `{en,es,fr,de}`) against `utter-project/mHuBERT-147` (CC-BY-NC-SA-4.0).
2. Train the **dual-output 50 Hz causal conv student** so a single forward pass
   emits both `latents` `[1, frames_per_step, D]` f32 (for Mode B's S2UT
   translator) and discrete `units` `[1, frames_per_step]` i64 (for Mode A VC).
3. Upgrade the **causal multi-speaker Unit-HiFi-GAN vocoder** to the multilingual
   unit codebook ($K=1000$) on `{en, es, fr, de}` studio voices (LJSpeech +
   Hi-Fi TTS + CSS10 + CML-TTS) so any voice embedding can synthesize speech in
   all 4 languages.

## Context & references

- [M2 / Stage 04](../Milestone%202/04_content_encoder_candidates.md) trained the
  English 50 Hz StreamVC-style causal student against `facebook/hubert-base-ls960`
  (km100). An English-only teacher under-represents Spanish/French nasal and
  palatal phones and German front-rounded vowels (`/y:/`, `/ø:/`) and dorsal
  fricatives (`/ç/`, `/x/`).
- **Licence constraint (research §3.1, §10):** `mhubert_base_vp_en_es_fr_it3` is
  released under the fairseq **MIT** licence and is proven in Fairseq S2UT,
  StreamSpeech (`Fr/Es/De→En`) and UTUT (`En↔Es/Fr/De/It/Nl`). `mHuBERT-147` is
  **CC-BY-NC-SA-4.0**. We probe both and use `mhubert-vp` with a balanced
  4-language $K=1000$ k-means codebook as the commercial-clean default.
- **Why dual outputs (`latents` + `units`):** Mode A needs discrete `units` for
  zero-leakage voice conversion; Mode B needs continuous speaker-independent
  `latents` (`D = 128`, layer-normalised pre-classification bottleneck) so the
  translator receives soft phonetic distributions rather than hard quantization
  errors.

## Tasks

Ticked = code done and covered by tests (synthetic data / random weights). The
runs that need corpora, the teacher checkpoint or a GPU are not ticked, even
where the tooling for them is finished.

- [ ] **Multilingual teacher probe (`python_training/probe_units.py` extended):**
  - [ ] Extract 50 Hz layer-11 features from `mhubert_base_vp_en_es_fr_it3` on a
        balanced 100 h `{en, es, fr, de}` subset (25 h per language from
        LibriSpeech, CoVoST 2 / Europarl-ST, CSS10). *(Tooling done: `fvml/teacher.py`
        loads the fairseq checkpoint without fairseq; `extract_units.py --fit-kmeans
        --teacher mhubert-vp --balance-langs en,es,fr,de --hours 100`. Not run: no
        corpora and no checkpoint on this machine.)*
  - [ ] Fit `MiniBatchKMeans(n_clusters=1000)` →
        `python_training/units/mhubert_vp_en_es_fr_de_km1000.npy`. *(Balanced fit
        implemented and tested (`teacher.fit_kmeans_balanced`). The file does not exist yet.)*
  - [ ] Compare against `mhubert-vp` stock km1000 and (if `--include-noncommercial`
        is enabled) `utter-project/mHuBERT-147` on held-out dev sets across all 4
        languages. *(Tooling done: `extract_units.py --import-kmeans` for the stock
        `.bin`, the `mhubert-147` teacher behind the flag, the per-language probe.)*
    - [x] Probe code: linear-probe phone purity & PNMI per language (`en`, `es`, `fr`,
          `de`), plus the ratio to a language-specific codebook (`--lang-reference`).
          ASR resynthesis is scored by `eval_multilingual.py resynth` (below).
    - [x] Probe code: 100-speaker speaker-ID leakage probe accuracy with a pass flag
          at $\le 5\%$ (`--max-speaker-accuracy`).
  - [ ] Record the probe table in this file and freeze the winning MIT codebook.
        *(`probe_units.py --markdown` writes the table. It stays empty below
        until the probe runs.)*
- [ ] **Label all S2UT & vocoder manifests (`extract_units.py` extended):**
  - [x] Code: target discrete `units` ($K=1000$, `int16` `.npy`) + Stage 03
        causal YIN prosody (`logf0`, `voicing`, `aperiodicity`, `log_energy`) for
        monolingual and paired (`--side src|tgt`) manifests.
  - [ ] Run it for all CVSS-C, synthesized CoVoST 2 / Europarl-ST, LJSpeech,
        Hi-Fi TTS, CSS10 and CML-TTS manifests (needs the Stage 01 corpora and the codebook).
- [ ] **Dual-output causal student (`fvml/content_encoder.py`,
      `train_content_encoder.py`):**
  - [x] Widen the bottleneck from $D=64$ to $D=128$ (still $\le 6\text{M}$ params)
        with a `LayerNorm(128)` output port `latents` immediately before the
        $K=1000$ linear head(s). *(`ce.MULTILINGUAL`: **5,135,160** params.)*
  - [ ] Train on balanced `{en, es, fr, de}` source speech with MUSAN noise + RIR
        reverb + gain/EQ augmentation against the clean teacher's $K=1000$ IDs
        (using the winning lookahead $L$ from M2 Stage 05). *(The trainer supports
        it (`--latent-dim 128 --lang-ratios balanced`, per-language top-1/top-5
        validation) and ran a 2-step CPU smoke. The real run is pending.)*
  - [x] Verify streaming `step()` vs. offline `forward()` parity for both
        `latents` (max $|\Delta| \le 10^{-5}$) and `units` (argmax agreement
        $\ge 99.9\%$). *(Verified on random weights over 60 s. Rerun on the trained
        checkpoint: `stream_check.py --checkpoint`.)*
  - [x] Pre-extract (or on-the-fly compute) student `latents` on the S2UT
        training manifests so Stage 05 trains on the exact distribution the C++
        pipeline produces at run time. *(`extract_units.py --save-latents --side src`,
        plus the `--silence-latents` bank. Tested against `train_s2ut.py`'s reader.)*
- [ ] **Multilingual causal Unit-HiFi-GAN vocoder (`fvml/vocoder.py`,
      `train_unit_vocoder.py`):**
  - [x] Expand unit embedding table to `vocab = 1000`, `dim = 128`.
  - [ ] Train on multi-speaker `{en, es, fr, de}` studio audio (LJSpeech +
        Hi-Fi TTS English speakers + CSS10 `es`/`fr`/`de` + curated CML-TTS
        `es`/`fr`/`de` speakers; 12–16 speakers total in `voices.json`, each with
        a `lang` metadata tag and F0/energy stats). *(The `lang` tag per voice and the
        per-speaker stats are implemented and tested. Training is pending.)*
  - [x] Sampling ratio: 40% `en`, 20% `es`, 20% `fr`, 20% `de` so English Mode A
        quality does not regress. *(`--lang-ratios m3-vocoder`.)*
- [ ] **Mode A non-regression & multilingual resynthesis check:**
  - [ ] Score English Mode A VC on `test_clean_200` with `eval_vc.py` and compare
        against the M2 Stage 05/11 numbers. *(Hook done: `eval_multilingual.py mode-a`. No checkpoints yet.)*
  - [ ] Score resynthesis (student `units` + target speaker embedding → vocoder)
        on 100 held-out clips each of `en`, `es`, `fr`, `de` (Whisper WER/CER in
        each language + UTMOS + ECAPA target cosine). *(Hook done: `eval_multilingual.py
        lists` / `resynth`, `fvml.evaluate` `language=`. No checkpoints yet.)*
- [x] **Contract-v2 dual-output encoder export** (from Stage 02):
      `export_stateful_onnx.py --contract-version 2` writes `encoder_s{1,2,4}.onnx`
      with `units` + `latents`. ORT parity gates both ports. v1 stays the default.

### Probe table

To be filled by `probe_units.py --markdown` (pending run 3 below). No cell has
been measured.

| Codebook | Teacher | K | Licence | en purity / PNMI | es purity / PNMI | fr purity / PNMI | de purity / PNMI | Purity vs. language-specific (min) | Speaker-ID acc. (chance) |
|---|---|---|---|---|---|---|---|---|---|
| `mhubert_vp_en_es_fr_de_km1000` (balanced refit) | mhubert-vp L11 | 1000 | MIT | | | | | | |
| `mhubert_base_vp_en_es_fr_it3_L11_km1000` (stock) | mhubert-vp L11 | 1000 | MIT | | | | | | |
| `mhubert147_en_es_fr_de_km1000` (research only) | mHuBERT-147 L11 | 1000 | CC-BY-NC-SA-4.0 (NC) | | | | | | |

## Implementation notes

- **Why train Stage 05 on student `latents`, not teacher features.** At run time,
  Stage 2 (`translator_enc`) receives the causal student's `latents`. Training
  Stage 2 directly on student `latents` (with slight Gaussian/spec-augment noise)
  eliminates any teacher-student mismatch at the encoder-translator boundary.
- **Cross-lingual FiLM conditioning.** Do not pass a language token into the
  vocoder! Phonetics come strictly from the $K=1000$ unit sequence, and vocal
  tract timbre comes strictly from `spk_emb` via FiLM. That way an English
  speaker's `spk_emb` naturally renders Spanish, French, or German units without
  being pulled toward a native training speaker's timbre.

## As built

These decisions interpret the task list where it leaves room, or deviate from
it. The module docstrings (`fvml/teacher.py`, `fvml/units.py`,
`fvml/content_encoder.py`, `extract_units.py`, `train_content_encoder.py`,
`train_unit_vocoder.py`, `eval_multilingual.py`) are the reference.

- **The multilingual setup is opt-in. M2 is unchanged.** Every default is still the
  M2 English setup: teacher `hubert-base` (layer 7, km100, `km100_centroids.npy`),
  `EncoderConfig()` D=64 / K=100 (4,293,612 params), `VocoderConfig()` vocab 100,
  contract v1 export, and M2 units metadata and checkpoints with no new keys. The M3
  setup is chosen by flags: `--teacher mhubert-vp`, tokenizer
  `teacher-mhubert-km1000`, `--latent-dim 128` (the vocabulary always follows the
  labels), `--lang-ratios`, `--contract-version 2`. Programmatic callers that build
  their own `extract()` namespace (M2 scripts, tests) get the new options filled
  with M2 values (`extract_units.with_defaults`). One M2 behaviour did change:
  `stream_check.py` now also gates the `latents` port at $10^{-5}$ for the M2
  encoder. It passes (5.96e-6 on random weights, see Verification).
- **Teachers** (`fvml/teacher.py::TEACHERS`):
  - `mhubert-vp` ships only as a fairseq checkpoint, and fairseq does not install on
    Python 3.12. `load_fairseq_hubert()` reads it without fairseq: a restricted
    unpickler builds tensors from an allowlist and turns every other pickled object
    into an inert stub, so no foreign code runs. The keys are then mapped onto a
    transformers `HubertModel` (Base layout) and loaded strictly; a missing, extra or
    mis-shaped tensor is an error.
  - `--download-teacher` fetches the 1.1 GB file through the dataset machinery, which
    pins its SHA-256 in `datasets.lock.json`.
  - The stock `..._L11_km1000.bin` is joblib, so `--import-kmeans` checks the
    published MD5 before it unpickles anything.
  - `mhubert-147` uses layer 11 to mirror `mhubert-vp`; its model card names no unit
    layer.
- **Licence isolation (NC).** `mhubert-147` and anything derived from it raises
  `LicenseError` without `--include-noncommercial`. This covers its units, a student
  distilled from them, a vocoder trained on them, the probe, latents, the silence
  bank and the export. Its outputs must also sit under a directory named `nc`. The
  taint travels in `units_meta.json` and in the checkpoints' `labels_meta`
  (`noncommercial: true`). A research export gets a licence suffix saying it must
  not be distributed. Commercial and NC unit directories cannot be mixed: both
  `fvml.units.read_units_dirs` and the vocoder trainer's `read_all_units_dirs`
  refuse the mix.
- **Balanced codebook.** `--hours` is split evenly over `--balance-langs`. Each
  language fills its own reservoir of `max_frames / n_langs` frames, and all
  reservoirs are cut to the smallest one, so every language gives the k-means the
  same number of frames. The fit uses fairseq `simple_kmeans` settings, and
  per-language frame counts are recorded in the codebook's JSON. LibriSpeech rows
  carry no `lang`, so they need `--default-lang en`.
- **Units directories.** Stage 03 adds these files and fields:
  - `<id>.prosody.npy` float32 `(frames, 4)` holds the causal YIN tracker
    (`fvml/features.py`, lookahead 0 by default) on the **clean** audio. **The
    `logf0` column is `logf0_held`**: log-F0 held through unvoiced frames, the form
    the vocoder is conditioned on.
  - `prosody_stats.json` holds per-speaker log-F0 / energy mean and std, with the
    same estimator and floors as the vocoder trainer.
  - `<id>.latents.npy` is the student's `latents`, **float16 by default**
    (`--save-latents f32` for full precision). Measured f16 error on the smoke
    corpus is ≤ 5e-3; `train_s2ut.py` casts to f32 and adds its own noise.
  - Rows now carry `lang`, `pair` and `side`.
  - A paired S2UT row (`--side src|tgt`) keeps its `id` as `utt_id`. The target side
    is named `<dataset>_tgt_<lang>` (the corpus's single synthetic voice).
  - Prosody can be added to a labelled directory later without rerunning the
    tokenizer.
- **Student.** The M2 trunk already ended in a per-frame `LayerNorm(D)` whose output
  all three heads read. With D=128 that output is the `latents` port. `latents[t]`
  describes frame `t` with no delay, while head L's `units` are L frames late
  (`content_encoder.encode_utterance`).
  - The budget is 5,135,160 params ≤ 6M: trunk 4,166,784; three K=1000 heads 968,376.
  - Validation reports top-1 and top-5 per head, overall and per language, with
    `dod` flags at 65% / 90%. `--val-limit` is split evenly between languages.
  - The last validation is stored in the checkpoint (`val`).
- **Language-ratio sampling** (`fvml/sampling.py`). The presets are `balanced`
  (student, 25% each) and `m3-vocoder` (40/20/20/20); explicit
  `en=0.4,es=0.2,...` also works. Each language receives exactly its share of draws,
  and utterances inside a language keep their length weighting. A language with no
  items, or an item with no language, is an error rather than a silent fallback.
- **Vocoder.**
  - The unit embedding stays `dim = 128` (the M2 default). Only `vocab` grows to
    1000: 1,173,201 params with 5 voices, against 1,058,001 at vocab 100.
  - There is no language input anywhere: not in `forward`/`step`, `VocoderConfig`,
    the parameters or the contract (a test checks all four).
  - `lang` is voices.json metadata only, validated as a language code. It comes
    from the manifest rows (one language per voice, enforced) and only fills gaps
    in the built-in / `--speaker-metadata` tables. The same applies to `license`
    and `source`.
  - Validation also logs the mel loss per language, so an English regression is
    visible during training.
- **Scoring hooks** (`eval_multilingual.py`).
  - `lists` builds deterministic, speaker-round-robin held-out lists of 3–12 s clips
    with a transcript, from the `test` subsets, plus reference clips for the new
    voices.
  - `resynth` runs the Stage 05 gate's streaming chain (`gate_report.convert_eval_list`).
    It scores with `fvml.evaluate`: Whisper is forced to the list's language and the
    diacritic-preserving normaliser is used for es/fr/de.
  - `mode-a` compares English VC on `test_clean_200` with an M2 baseline. Without a
    baseline the verdict is *incomplete*, never a pass.
  - `--converted` scores existing WAVs, e.g. from the C++ `fv_vc_file`.
- **Contract-v2 encoder export.** The graph emits `units` and `latents` from one
  trunk pass. The only state is conv caches, so there are no positions in the graph
  (MEMORY.md rule 6). ORT parity is gated on unit agreement ≥ 99.9% and on
  `latents` max $|\Delta| \le 10^{-4} \times$ RMS of a non-silent port (rule 11).
  A v2 vocoder carries the v2 stamp only (its tensors are unchanged). Re-exporting
  a checkpoint at another contract version replaces the stale variants.
- **Silence-latent bank** (`extract_units.py --silence-latents OUT.npy --checkpoint
  <student>`, new). This is the `[T, D]` float32 bank that `train_s2ut.py
  --silence-latents` cuts its silence tails from.
  - The default input is 60 s of synthetic room tone: 2 s of digital silence, then
    2 s segments of low-passed Gaussian noise at RMS levels drawn in −90…−45 dBFS
    (`--silence-dbfs=`), plus any `--manifest` recordings scaled into the same range.
  - The encoder first runs one receptive field (22 frames) of silence, and those
    frames are dropped. The bank therefore holds the steady state after silence,
    not the zero-state start-up transient.
  - A JSON sidecar records the checkpoint SHA-256, the levels and the licence.
  - Real room-tone recordings would match the run-time distribution better than
    synthetic noise. Pass them with `--manifest` once there are some.

## Verification (2026-11-15, Windows — code + smoke runs only)

Windows 11, CPython 3.12.13, torch 2.11.0+cu128, onnxruntime 1.31.0,
transformers 5.20.1, scikit-learn 1.9.1, run from `python_training/` with
`PYTHONIOENCODING=utf-8`, CPU only. There is no multilingual corpus, no
`mhubert-vp` / `mHuBERT-147` weights and no trained M2/M3 checkpoint on this
machine: **nothing below is a quality result, and the probe table above is empty
on purpose.**

- **Unit tests, Stage 03** (`.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
  tests/test_multilingual.py`, about 2 min): **25 passed** (23 test functions, one
  parametrized over fps 1/2/4). They run on synthetic audio, tiny models and a
  fake teacher. They cover:
  - the teacher registry and NC guard, a fairseq-format checkpoint round trip,
    checkpoint fetch, the balanced k-means and the stock k-means import (MD5);
  - language sampling and paired-row handling;
  - the M3 student's size, port and LayerNorm, and `step()` = `forward()` at fps
    1/2/4;
  - an end-to-end run: balanced fit → K=1000 labelling with prosody (checked
    against `fvml.features`) → paired `--side tgt` → student training with
    per-language top-1/top-5 → `--save-latents` f16/f32 (equal to the streaming
    `step()` latents, ≤ 1e-5 f32) → vocoder training with language ratios and
    voices.json `lang` → **`train_s2ut.join_items` + `S2utDataset` reading those
    directories and the silence bank**;
  - NC isolation, the vocoder's lack of a language input, the v2 export (graph,
    contract, ORT parity, CLI bundle), the probe with per-language and reference
    ratios, and the scoring verdicts and CLI;
  - the silence bank, and M2-era `extract()` callers.
- **Whole suite** (`.venv\Scripts\python.exe -m pytest -q -rfEs -p no:cacheprovider tests`,
  5 min 45 s): **316 passed, 7 skipped, 0 failed**. The skips are the opt-in 60 s
  translator run and the tests that need the built `fv_vc_file` /
  `test_waitk_policy` binaries. Before this stage was finished, the suite had 1
  failure: `test_target_voice.py::test_extract_units_perturbation` (an M2-era
  `extract()` caller without the new options), fixed by `extract_units.with_defaults`.
- **Streaming parity, Definition of done** (`stream_check.py --multilingual`).
  - Setup: full-size M3 student (D=128, K=1000, 5,135,160 params), random weights,
    60.0 s = 3,000 frames of concatenated LibriSpeech dev-clean. `step()` at 1/2/4
    frames per step against one `forward()`:

    | fps | max $|\Delta\text{latents}|$ | max $|\Delta\text{logit}|$ L0 / L1 / L2 | unit argmax agreement (all heads) |
    |---|---:|---:|---:|
    | 1 | 6.91e-6 | 1.91e-6 / 1.79e-6 / 1.79e-6 | 100.000 % |
    | 2 | 6.91e-6 | 1.82e-6 / 1.85e-6 / 1.87e-6 | 100.000 % |
    | 4 | 6.68e-6 | 1.85e-6 / 1.76e-6 / 1.73e-6 | 100.000 % |

  - Both bounds hold ($\le 10^{-5}$, $\ge 99.9\%$). The latents margin is only about
    1.4×. The LayerNorm divides by each frame's std, so float32 accumulation-order
    noise in the trunk is scaled up when pre-norm activations are small (as on a
    random init). Rerun on the trained checkpoint.
  - M2 regression (`stream_check.py`, D=64/K=100): latents 5.96e-6 / 5.96e-6 /
    5.72e-6, logits ≤ 1.79e-6, agreement 100 %: passes.
  - M3 vocoder (`stream_check.py --multilingual --role vocoder --seconds 30`,
    vocab 1000): max $|\Delta|$ 5.1e-11, SNR 137.3 dB at fps 1/2/4, boundary
    steps within the reference's. The random-init vocoder is near-silent
    (RMS 7.7e-5), so this is a code-path check only (BUG-017).
- **Full-size ORT export dry run (random weights).** Run `export_stateful_onnx.py
  --role encoder --contract-version 2 --head student-L1` then `--role vocoder
  --contract-version 2` into a scratch bundle, then the same pair at the M2 config
  and contract v1 for a same-session comparison. 60 s dev-clean parity, ORT
  1.31, one intra-op thread:
  - The M3 v2 bundle's `encoder_s{1,2,4}.onnx` are dynamo-exported (opset 20 /
    IR 10, 21 states, 42,630 state floats) and `units` + `latents` are graph
    outputs.
  - Encoder parity at s1/s2/s4: unit agreement 1.00000, max $|\Delta\text{logit}|$
    1.83e-6 / 2.04e-6 / 2.00e-6, **max $|\Delta\text{latents}|$ 6.44e-6 /
    6.14e-6 / 6.68e-6** at latents RMS 1.00. That is within the 1e-4 × RMS gate,
    and within 1e-5 absolute too.
  - Vocoder (vocab 1000) parity: SNR 135.3 dB. Pairs are valid at algorithmic
    latency 40 / 60 / 100 ms. Chain unit agreement 1.00000. Verdict: every gate
    passed.
  - The M2 v1 rerun also passes every parity check (encoder agreement 1.00000,
    max $|\Delta\text{logit}|$ ≤ 2.09e-6; vocoder SNR 121.6 dB). All four exports
    report "every gate passed". The 50 % cost rule is reported, not gated, by the
    exporter.

  Cost, p95 ms per hop, same session (p50 in brackets):

  | | s1 (20 ms) | s2 (40 ms) | s4 (80 ms) |
  |---|---|---|---|
  | M3 encoder (D=128, K=1000, v2) | 10.65 (4.49) | 17.16 (7.32) | 27.28 (13.74) |
  | M3 vocoder (vocab 1000) | 6.99 (4.65) | 10.11 (5.36) | 14.82 (7.29) |
  | **M3 chain** | **18.24 (91%)** | **23.11 (58%)** | **35.27 (44%)** |
  | M2 encoder (D=64, K=100, v1) | 11.23 (6.13) | 21.74 (12.58) | 39.48 (26.18) |
  | M2 vocoder (vocab 100) | 7.38 (3.85) | 10.43 (7.94) | 15.75 (10.57) |
  | M2 chain | 17.72 (89%) | 28.35 (71%) | 46.47 (58%) |

  This laptop was loaded and noisy during the runs: the M2 pair came out *slower*
  than the M3 pair, although M3 does strictly more work. So these numbers cannot
  decide the compute-budget DoD. What they do show:
  - The M3 encoder costs 3.06 GMAC/s against M2's 3.02 (+1.4 %, all in the
    K=1000 heads).
  - The M3 vocoder has +115k params, all in the embedding table, which is a lookup.
  - Both pairs miss the 50 % rule at s1 (M2 Stage 06 measured the same) and
    pass it at s4.
  - Stage 05 leaves about 17 ms p95 for encoder s4 + vocoder s4 beside the
    translator. **Neither the M2 pair nor the M3 pair fits that**: M2 Stage 06's
    quieter measurement was already 29.2 ms. This is flagged for Stage 05/08, not
    something Stage 03 created.
  - Rerun with the trained weights on an idle machine (pending run 9) and in C++.

### Pending runs (in order)

Each needs the inputs named in its comment; none has been run.

```powershell
cd python_training
# 0. Stage 01: es/fr/de corpora + manifests (CSS10, CML-TTS, CoVoST 2 / Europarl-ST, CVSS-C) and
#    MFA TextGrids for held-out dev clips in each language (the phone probe needs them).
# 1. Teacher + balanced K=1000 codebook (downloads the 1.1 GB MIT checkpoint once; GPU):
.venv\Scripts\python.exe extract_units.py --fit-kmeans --teacher mhubert-vp --download-teacher --balance-langs en,es,fr,de --hours 100 --default-lang en --side src --subsets train --manifest librispeech-train-clean-100.jsonl,<es>.jsonl,<fr>.jsonl,<de>.jsonl --out-dir units/
#    -> units/mhubert_vp_en_es_fr_de_km1000.npy (+ .json). Language-specific references (one per language, same budget):
.venv\Scripts\python.exe extract_units.py --fit-kmeans --teacher mhubert-vp --balance-langs es --hours 25 --side src --subsets train --manifest <es>.jsonl --out-dir units/ref_es/
#    Stock codebook for the comparison (download https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin first):
.venv\Scripts\python.exe extract_units.py --import-kmeans <path>\mhubert_base_vp_en_es_fr_it3_L11_km1000.bin --teacher mhubert-vp --out-dir units/stock/
# 2. Label the held-out dev clips with each codebook (and, research only, mHuBERT-147 under an `nc` directory):
.venv\Scripts\python.exe extract_units.py --tokenizer teacher-mhubert-km1000 --centroids units/ --default-lang en --manifest <dev manifests> --subsets test --out-dir data/units/probe/balanced
.venv\Scripts\python.exe extract_units.py --tokenizer teacher-mhubert-km1000 --centroids units/stock/mhubert_base_vp_en_es_fr_it3_L11_km1000.npy --default-lang en --manifest <dev manifests> --subsets test --out-dir data/units/probe/stock
.venv\Scripts\python.exe extract_units.py --tokenizer teacher-mhubert-km1000 --centroids units/ref_es/ --manifest <es dev manifest> --subsets test --out-dir data/units/probe/ref_es   # one per language
.venv\Scripts\python.exe extract_units.py --fit-kmeans --teacher mhubert-147 --include-noncommercial --balance-langs en,es,fr,de --hours 100 --default-lang en --side src --subsets train --manifest <same as 1> --out-dir data/nc/units/km147/
.venv\Scripts\python.exe extract_units.py --tokenizer teacher-mhubert147-km1000 --include-noncommercial --centroids data/nc/units/km147/ --default-lang en --manifest <dev manifests> --subsets test --out-dir data/nc/units/probe/m147
# 3. The probe table (paste the Markdown above; repeat --lang-reference for en/fr/de):
.venv\Scripts\python.exe probe_units.py --units data/units/probe/balanced --units data/units/probe/stock --units data/nc/units/probe/m147 --include-noncommercial --alignments <MFA TextGrids> --lang-reference es=data/units/probe/ref_es --speaker-probe --speakers 100 --markdown ..\runs\probe\m3_probe.md --out ..\runs\probe\m3_probe.json
# 4. Label the training corpora (K=1000 + prosody), per manifest <key>:
.venv\Scripts\python.exe extract_units.py --tokenizer teacher-mhubert-km1000 --centroids units/ --prosody --default-lang en --manifest <key>.jsonl --out-dir data/units/mhubert/<key>
.venv\Scripts\python.exe extract_units.py --tokenizer teacher-mhubert-km1000 --centroids units/ --prosody --side tgt --manifest s2ut/<key>.jsonl --out-dir data/units/mhubert/<key>.tgt
.venv\Scripts\python.exe extract_units.py --tokenizer teacher-mhubert-km1000 --centroids units/ --side src --manifest s2ut/<key>.jsonl --out-dir data/units/mhubert/<key>.src
# 5. Dual-output student (GPU), then its 60 s parity on the trained weights:
.venv\Scripts\python.exe train_content_encoder.py --latent-dim 128 --lang-ratios balanced --default-lang en --side src --train-subsets train --val-subsets test,dev --train-manifest <en/es/fr/de manifests> --train-units <matching data/units/mhubert dirs> --val-manifest <...> --val-units <...> --musan-manifest musan.jsonl --rir-manifest rirs.jsonl --run-name ce_m3
.venv\Scripts\python.exe stream_check.py --checkpoint ..\checkpoints\ce_m3\best.pt
# 6. Student latents + silence bank for Stage 05:
.venv\Scripts\python.exe extract_units.py --tokenizer student-L1 --checkpoint ..\checkpoints\ce_m3\best.pt --save-latents --side src --manifest s2ut/<key>.jsonl --out-dir data/units/student/<key>.src
.venv\Scripts\python.exe extract_units.py --silence-latents data/units/student/silence.latents.npy --checkpoint ..\checkpoints\ce_m3\best.pt
# 7. Multilingual vocoder (GPU), 40/20/20/20, 12-16 voices:
.venv\Scripts\python.exe train_unit_vocoder.py --lang-ratios m3-vocoder --default-lang en --train-subsets train --train-manifest ljspeech.jsonl,hifitts.jsonl,<css10-*>,<cml-tts-*> --train-units <matching data/units/mhubert dirs> --run-name voc_m3
# 8. Scoring (the DoD numbers): held-out lists, resynthesis in 4 languages, English Mode A vs M2:
.venv\Scripts\python.exe eval_multilingual.py lists --manifest <es manifests> --lang es --voice-refs <es voice> --out-dir eval_lists/resynth
.venv\Scripts\python.exe eval_multilingual.py resynth --encoder-checkpoint ..\checkpoints\ce_m3\best.pt --vocoder-checkpoint ..\checkpoints\voc_m3\best.pt --candidate student-L1:s1 --list en=eval_lists/test_clean_200.tsv --list es=eval_lists/resynth/resynth_es_100.tsv --list fr=eval_lists/resynth/resynth_fr_100.tsv --list de=eval_lists/resynth/resynth_de_100.tsv --voice en=lj --voice es=<es voice> --voice fr=<fr voice> --voice de=<de voice> --targets eval_lists/resynth/targets_m3.tsv --out ..\runs\eval\m3_resynth
.venv\Scripts\python.exe eval_multilingual.py mode-a --encoder-checkpoint ..\checkpoints\ce_m3\best.pt --vocoder-checkpoint ..\checkpoints\voc_m3\best.pt --voices lj,hifitts_92 --baseline-summary <M2 eval_vc summary.json or gate report> --out ..\runs\eval\m3_mode_a
# 9. Bundle (contract v2) and the shipped-runtime cost on the trained weights:
.venv\Scripts\python.exe export_stateful_onnx.py --role encoder --contract-version 2 --checkpoint ..\checkpoints\ce_m3\best.pt --head student-L1 --license "MIT" --bundle ..\models\m3\
.venv\Scripts\python.exe export_stateful_onnx.py --role vocoder --contract-version 2 --checkpoint ..\checkpoints\voc_m3\best.pt --bundle ..\models\m3\
```

## Files touched

- `python_training/probe_units.py`, `python_training/extract_units.py`,
  `python_training/fvml/content_encoder.py`,
  `python_training/train_content_encoder.py`,
  `python_training/fvml/vocoder.py`, `python_training/train_unit_vocoder.py`,
  `python_training/fvml/teacher.py`, `python_training/fvml/units.py`,
  `python_training/fvml/evaluate.py`, `python_training/fvml/export.py`,
  `python_training/export_stateful_onnx.py`, `python_training/stream_check.py`,
  `python_training/fvml/sampling.py` (new), `python_training/eval_multilingual.py` (new),
  `python_training/tests/test_multilingual.py` (new), this file
- Pending: `python_training/units/mhubert_vp_en_es_fr_de_km1000.npy` (new, written by pending run 1)

## Definition of done

- Balanced $K=1000$ `mhubert-vp` codebook reaches phone purity $\ge 90\%$ of
  language-specific codebooks across all 4 languages (`en`, `es`, `fr`, `de`),
  and student top-1/top-5 unit accuracy reaches $\ge 65\% / \ge 90\%$ on held-out
  dev sets in all 4 languages. — *not measured (runs 1–5)*
- Dual-output student streaming `step()` matches offline `forward()`: `latents`
  max $|\Delta| \le 10^{-5}$ and `units` argmax agreement $\ge 99.9\%$ over 60 s.
  — *met on random weights (6.9e-6, 100 %); recheck on the trained checkpoint (run 5)*
- Multilingual vocoder resynthesis achieves Whisper WER $\le 8\%$ (`en`, `es`),
  $\le 10\%$ (`fr`, `de`), and UTMOS $\ge 3.3$ across all 4 languages. — *not measured (run 8)*
- Mode A English VC on `test_clean_200` remains $\le 10\%$ WER and within $+1.0$
  pt of the M2 English-only vocoder. — *not measured (run 8)*
- Single-core CPU per-step $p_{95}$ for dual-output encoder + vocoder stays
  within M2's compute budget. — *random-weight ORT measurement only, see Verification*
