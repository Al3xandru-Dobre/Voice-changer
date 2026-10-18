# Stage 03 — Multilingual encoder & vocoder upgrade

> **Milestone 3** · [Master plan](00_master.md) · [← Stage 02](02_translator_contract_v2_fixtures.md) · [Stage 04 →](04_waitk_policy_streaming_state_machine.md)

**Status:** ☐ not started
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

- [ ] **Multilingual teacher probe (`python_training/probe_units.py` extended):**
  - [ ] Extract 50 Hz layer-11 features from `mhubert_base_vp_en_es_fr_it3` on a
        balanced 100 h `{en, es, fr, de}` subset (25 h per language from
        LibriSpeech, CoVoST 2 / Europarl-ST, CSS10).
  - [ ] Fit `MiniBatchKMeans(n_clusters=1000)` →
        `python_training/units/mhubert_vp_en_es_fr_de_km1000.npy`.
  - [ ] Compare against `mhubert-vp` stock km1000 and (if `--include-noncommercial`
        is enabled) `utter-project/mHuBERT-147` on held-out dev sets across all 4
        languages:
    - [ ] ASR resynthesis / linear-probe phone purity & PNMI per language (`en`,
          `es`, `fr`, `de`)
    - [ ] 100-speaker speaker-ID leakage probe accuracy (must stay near chance,
          $\le 5\%$)
  - [ ] Record the probe table in this file and freeze the winning MIT codebook.
- [ ] **Label all S2UT & vocoder manifests (`extract_units.py` extended):**
  - [ ] Extract target discrete `units` ($K=1000$, `int16` `.npy`) + Stage 03
        causal YIN prosody (`logf0`, `voicing`, `aperiodicity`, `log_energy`) for
        all CVSS-C, synthesized CoVoST 2 / Europarl-ST, LJSpeech, Hi-Fi TTS,
        CSS10, and CML-TTS manifests.
- [ ] **Dual-output causal student (`fvml/content_encoder.py`,
      `train_content_encoder.py`):**
  - [ ] Widen the bottleneck from $D=64$ to $D=128$ (still $\le 6\text{M}$ params)
        with a `LayerNorm(128)` output port `latents` immediately before the
        $K=1000$ linear head(s).
  - [ ] Train on balanced `{en, es, fr, de}` source speech with MUSAN noise + RIR
        reverb + gain/EQ augmentation against the clean teacher's $K=1000$ IDs
        (using the winning lookahead $L$ from M2 Stage 05).
  - [ ] Verify streaming `step()` vs. offline `forward()` parity for both
        `latents` (max $|\Delta| \le 10^{-5}$) and `units` (argmax agreement
        $\ge 99.9\%$).
  - [ ] Pre-extract (or on-the-fly compute) student `latents` on the S2UT
        training manifests so Stage 05 trains on the exact distribution the C++
        pipeline produces at run time.
- [ ] **Multilingual causal Unit-HiFi-GAN vocoder (`fvml/vocoder.py`,
      `train_unit_vocoder.py`):**
  - [ ] Expand unit embedding table to `vocab = 1000`, `dim = 128`.
  - [ ] Train on multi-speaker `{en, es, fr, de}` studio audio (LJSpeech +
        Hi-Fi TTS English speakers + CSS10 `es`/`fr`/`de` + curated CML-TTS
        `es`/`fr`/`de` speakers; 12–16 speakers total in `voices.json`, each with
        a `lang` metadata tag and F0/energy stats).
  - [ ] Sampling ratio: 40% `en`, 20% `es`, 20% `fr`, 20% `de` so English Mode A
        quality does not regress.
- [ ] **Mode A non-regression & multilingual resynthesis check:**
  - [ ] Score English Mode A VC on `test_clean_200` with `eval_vc.py` and compare
        against the M2 Stage 05/11 numbers.
  - [ ] Score resynthesis (student `units` + target speaker embedding → vocoder)
        on 100 held-out clips each of `en`, `es`, `fr`, `de` (Whisper WER/CER in
        each language + UTMOS + ECAPA target cosine).

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

## Files touched

- `python_training/probe_units.py`, `python_training/extract_units.py`,
  `python_training/fvml/content_encoder.py`,
  `python_training/train_content_encoder.py`,
  `python_training/fvml/vocoder.py`, `python_training/train_unit_vocoder.py`,
  `python_training/units/mhubert_vp_en_es_fr_de_km1000.npy` (new)

## Definition of done

- Balanced $K=1000$ `mhubert-vp` codebook reaches phone purity $\ge 90\%$ of
  language-specific codebooks across all 4 languages (`en`, `es`, `fr`, `de`),
  and student top-1/top-5 unit accuracy reaches $\ge 65\% / \ge 90\%$ on held-out
  dev sets in all 4 languages.
- Dual-output student streaming `step()` matches offline `forward()`: `latents`
  max $|\Delta| \le 10^{-5}$ and `units` argmax agreement $\ge 99.9\%$ over 60 s.
- Multilingual vocoder resynthesis achieves Whisper WER $\le 8\%$ (`en`, `es`),
  $\le 10\%$ (`fr`, `de`), and UTMOS $\ge 3.3$ across all 4 languages.
- Mode A English VC on `test_clean_200` remains $\le 10\%$ WER and within $+1.0$
  pt of the M2 English-only vocoder.
- Single-core CPU per-step $p_{95}$ for dual-output encoder + vocoder stays
  within M2's compute budget.
