# Stage 10 — Cross-lingual target-voice transfer ("Me" voice in `En↔{Es,Fr,De}`)

> **Milestone 3** · [Master plan](00_master.md) · [← Stage 09](09_s2ut_controls_language_switching.md) · [Stage 11 →](11_verification_exit.md)

**Status:** ☐ not started
**Depends on:** Stage 03 (multilingual vocoder), Stage 06 (export pipeline), [Milestone 2 / Stage 10](../Milestone%202/10_target_voice_finetune.md) (the user's recorded 1–3 h English "Me" dataset)
**Research:** [M3 research](../Research_documents/M3_WAITK_S2UT_RESEARCH.md) §3.3 multilingual vocoder & cross-lingual voice preservation · §5 prosody transfer

## Objective

Enable the user's own recorded voice (`--voice me`, recorded in English during
M2 Stage 10) to serve as the target voice across **all 6 `En↔{Es,Fr,De}`
translation directions** — so the user can speak English and be heard speaking
Spanish, French, or German in their *own* vocal timbre and pitch range, or speak
Spanish/French/German and be heard in English in their own voice.

Also record a short **multilingual live-mic source evaluation set** (15–20
sentences per language read or played into the real microphone) to test real-room
deployment in Stage 11.

## Context & references

- [M2 / Stage 10](../Milestone%202/10_target_voice_finetune.md) recorded 1–3 h of
  clean English speech from the user and fine-tuned a `"me"` speaker row in the
  M2 English vocoder.
- **How cross-lingual voice transfer works without recording 3 hours each of
  Spanish, French, and German:**
  - In Stage 03's multilingual Unit-HiFi-GAN, phonetic identity comes exclusively
    from the language-shared $K=1000$ `mhubert-vp` unit sequence, while vocal
    tract timbre comes exclusively from the 64-dim speaker embedding `spk_emb` via
    FiLM layers.
  - Because the public speakers in Stage 03 span `{en, es, fr, de}` over the same
    shared $K=1000$ unit embedding table, every unit centroid $u \in [0..999]$
    (including non-English phones like Spanish trilled `/r/`, French nasal vowels,
    and German `/y:/`, `/ç/`) is already anchored in the generator's pre-net and
    upsampling trunk.
  - Fine-tuning the multilingual vocoder with the user's English recordings (mixed
    50/50 with `{en, es, fr, de}` public-speaker batches and NANSY-style formant/pitch
    perturbation) adapts the `"me"` FiLM embedding while keeping the non-English
    phone representations intact!

## Tasks

- [ ] **Extract multilingual $K=1000$ units for the user's M2 recordings:**
  - [ ] Run `extract_units.py` with the Stage 03 `mhubert-vp` $K=1000$ teacher
        and causal student on the preprocessed `me_s*.wav` manifest from M2 Stage
        10 (including the 50% formant/pitch/EQ perturbed copies).
- [ ] **Cross-lingual fine-tune (`python_training/train_unit_vocoder.py`):**
  - [ ] Initialise the `"me"` row in Stage 03's multilingual vocoder from the
        nearest English speaker row (by ECAPA cosine).
  - [ ] Fine-tune on a **40% `"me"` (English) + 60% public `{en, es, fr, de}`**
        batch mix (15% `en`, 15% `es`, 15% `fr`, 15% `de`).
  - [ ] **Crucial regularization:** freeze the unit embedding table
        (`unit_emb.weight.requires_grad = False`) during the second half of the
        `"me"` fine-tune so unit centroids for Spanish/French/German phones rarely
        seen in the user's English audio cannot drift!
- [ ] **Cross-lingual evaluation (`eval_s2ut.py`):**
  - [ ] Run the full S2UT chain into `--voice me` across all 6 `test_*_200.tsv`
        directions (`es→en`, `fr→en`, `de→en`, `en→es`, `en→fr`, `en→de`).
  - [ ] Compare against translating into the default native public voices (`en`,
        `es`, `fr`, `de`):
    - **ASR-BLEU & ASR-chrF++:** confirms that using the `"me"` speaker embedding
      on `En→{Es,Fr,De}` does not degrade target-language intelligibility;
    - **ECAPA target speaker cosine (`tgt_cos`):** cosine similarity between the
      translated Spanish/French/German output in voice `"me"` and the user's
      held-out English reference clips;
    - **UTMOS** across all 6 directions.
- [ ] **Re-export bundle (Stage 06)** with `"me"` added to `voices.json`
      (`license: "private — do not distribute"`).
- [ ] **Live-mic multilingual source test set:**
  - [ ] Record 20 prompts per source language (`en`, `es`, `fr`, `de` — read by
        the user for languages they speak, or played from held-out clean speaker
        recordings through a reference monitor into the room mic for languages
        they don't) for Stage 11's live evaluation.

## Implementation notes

- **Why freezing `unit_emb` prevents cross-lingual accent collapse.** Because the
  user's `"me"` training set is 100% English, units corresponding specifically to
  German `/y:/` or French `/œ̃/` only appear in the 60% public-speaker batches.
  Freezing `unit_emb` (which is already well-trained in Stage 03) forces the
  `"me"` speaker row and FiLM layers to adapt only vocal-tract timbre and
  harmonic structure, preserving crisp target-language pronunciation in `es`,
  `fr`, and `de`.
- **Optional 5-minute target-language adaptation.** If the user also reads 5
  minutes of Spanish, French, or German prompts during recording, include them in
  the `"me"` manifest; the pipeline supports it seamlessly, though it is not
  required for the DoD.

## Files touched

- `python_training/train_unit_vocoder.py` (`--freeze-unit-emb` option for
  cross-lingual speaker fine-tune),
  `models/<bundle>/**` (updated `vocoder_s{1,2,4}.onnx` and `voices.json`,
  git-ignored),
  this file (cross-lingual evaluation table)

## Definition of done

- Across all 3 `En→{Es,Fr,De}` directions, translating with `--voice me` achieves:
  - ASR-BLEU within $-1.5$ BLEU of the native public target voice for that
    language;
  - ECAPA speaker cosine to the user's held-out English `"me"` reference clips
    $\ge$ the zero-shot baseline $+ 0.05$ and within $0.05$ of same-language
    (`X→En`) `"me"` cosine;
  - UTMOS $\ge 3.0$ in every output language.
- Public voices in `voices.json` regress by $\le 0.02$ ECAPA cosine and $\le 0.5$
  ASR-BLEU.
- Live check: `fluxvoice --s2ut --src-lang en --tgt-lang es --voice me` runs for
  10 minutes with 0 `misses` after warm-up.
