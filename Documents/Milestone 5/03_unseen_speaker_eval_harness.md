# Stage 03 — Unseen-speaker evaluation harness & baselines

> **Milestone 5** · [Master plan](00_master.md) · [← Stage 02](02_voice_profile_contract_store.md) · [Stage 04 →](04_speaker_encoder_vocoder_gate.md)

**Status:** ☐ not started
**Depends on:** [M2 / Stage 01](../Milestone%202/01_python_toolchain_eval_harness.md) (`eval_vc.py`, frozen lists, metric code).
Baseline numbers additionally need a trained M2 bundle
([M2 / Stage 06](../Milestone%202/06_stateful_onnx_export.md)), which does not exist yet.
**Owner persona:** `@ml-speech-trainer`

## Objective

Before training anything, fix how "the converted voice resembles the voice in
the user's files" is measured: which held-out speakers, how much enrolment
audio, how degraded, against which baseline and which ceiling. Stage 04's gate
and Stage 11's exit numbers are read from this harness.

## Context & verified starting point (2026-11-18)

- `fvml/evaluate.py` already scores WER and CER (Whisper large-v3), target
  cosine and source cosine (ECAPA), UTMOS, F0 correlation and RTF, on
  conversions produced by `fv_vc_file`.
- The existing lists (`eval_lists/test_clean_200.tsv`, `targets.tsv`) measure
  conversion into voices the vocoder was *trained* on. Nothing measures a
  voice the models have never seen, and nothing varies the amount or quality
  of reference audio.
- `nearest_speaker()` in `train_unit_vocoder.py` already picks the closest
  trained row by ECAPA centroid. Together with statistics computed from the
  user's audio this is the "zero-shot baseline" that
  [M2 / Stage 10](../Milestone%202/10_target_voice_finetune.md) compares
  against, and it needs no new model.
- LibriTTS-R and VCTK are registered in `fvml/datasets.py`
  (`build_libritts_r`, `build_vctk`); both are CC BY 4.0.

## Design

### Frozen lists (`python_training/eval_lists/enroll/`)

- **`enroll_speakers.tsv`** — 40 speakers that no model in the project is ever
  trained on: 20 from LibriTTS-R `test-clean` and 20 from VCTK, balanced by
  gender and spread across the VCTK accent labels. Selection is deterministic
  (sorted ids, evenly spaced picks), as in `make_eval_lists.py`.
- **Per speaker:**
  - nested enrolment sets of 10 s, 30 s, 60 s and 180 s of net speech, built
    from whole utterances in a fixed order;
  - a disjoint reference set of at least 20 s for scoring, from different
    utterances (different chapters or sessions where the corpus allows).
- **`enroll_sources.tsv`** — 50 source utterances taken from
  `test_clean_200.tsv`, none spoken by a target speaker.
- **Conditions:**
  - *main*: 40 speakers × 50 sources at 60 s → 2 000 conversions;
  - *duration sweep*: 10 speakers × 50 sources × {10, 30, 60, 180} s;
  - *degraded enrolment* (60 s, 10 speakers): additive noise at 20 dB and
    10 dB (MUSAN), reverberation (RIRS_NOISES), MP3 at 64 kbit/s round trip,
    3.4 kHz band limit, and a clipped copy. Only the enrolment audio is
    degraded; sources and references stay clean.
- **`excluded_speakers.txt`** — the 40 ids, consumed by every training script
  in Stage 04 through a new `--exclude-speakers` option. A hash of the three
  list files is written into each results file.

### Metrics

| Metric | Meaning | Code |
|---|---|---|
| `tgt_cos` | ECAPA cosine between each output and the target's reference centroid | existing |
| `src_cos` | ECAPA cosine between output and its own source (leakage) | existing |
| WER / CER, UTMOS, F0 PCC | As in M2 | existing |
| F0 register error | Median output F0 against the target's median, in semitones | new, `fvml/f0_ref.py` |
| Ceiling | `tgt_cos` of the target's *real* held-out speech against its own reference centroid | new |
| Floor | `tgt_cos` of the *unconverted* sources against the target centroid | new |
| Normalised similarity | `(tgt_cos − floor) / (ceiling − floor)` per speaker | new |
| Embedding stability | Cosine between embeddings from disjoint 30 s sets of one speaker, against other speakers; equal error rate | new; calibrates the Stage 10 live match |
| Enrolment time | Wall-clock on one core for 60 s and 180 s | new |

Every table reports the mean, the per-speaker spread, and the minimum over
subgroups (gender, corpus, accent group). A method that helps on average and
hurts one group is reported as such.

### Baselines

- **B0 — nearest voice.** The closest trained row by ECAPA centroid, with
  F0 and energy statistics computed from the enrolment audio. Available on
  any M2 bundle. This is what the application could offer today, and the
  number Stage 04 must beat.
- **B1 — fine-tune upper bound (optional, time-boxed to three days).** The
  M2 Stage 10 fine-tune for two speakers with more than 30 minutes of data.
  It shows how far Instant is from the expensive path.

### Rules

- **The evaluator is never the conditioning model.** ECAPA scores; it does
  not produce embeddings for the vocoder in any shipped configuration.
- **The shipped runtime is measured.** Conversions go through `fv_vc_file`
  and, once Stage 07 exists, enrolment through `fv_enroll`. Before that the
  Python ORT path is used and every result is labelled *provisional*.
- **Held-out means held out everywhere:** content encoder, vocoder, speaker
  encoder and any refinement basis.

## Tasks

- [ ] `make_enroll_eval_lists.py`: builds and hashes the lists; refuses to
      overwrite frozen lists without `--force`; unit-tested on a synthetic
      manifest for determinism, disjointness and balance.
- [ ] `fvml/enroll_eval.py`: enrolment-set assembly, the degradations (reusing
      `fvml/augment.py`; MP3 round trip through an optional local encoder,
      skipped with a recorded reason when absent), ceiling, floor, normalised
      similarity, subgroup tables, embedding-stability trials.
- [ ] `eval_enroll.py`: `--bundle`, `--method {nearest,instant,refined}`,
      `--condition {main,duration,degraded}`, `--enroll-seconds`, `--limit`,
      `--out`. Writes `per_utt.csv`, `per_speaker.csv`, `summary.json`,
      `summary.md`.
- [ ] `--exclude-speakers` in `extract_units.py`, `train_content_encoder.py`
      and `train_unit_vocoder.py`, with a test that an excluded speaker never
      reaches a batch.
- [ ] Tests with fake ASR, speaker and MOS models
      (`python_training/tests/test_enroll_eval.py`), registered in CTest.
- [ ] **B0 on the M2 gate bundle** for the three conditions. *Blocked until a
      trained bundle exists.*
- [ ] **B1** for two speakers, if the time box allows.
- [ ] Fill the results tables and state the two thresholds this stage hands
      on: the minimum net speech (knee of the duration curve) and the lowest
      SNR at which `tgt_cos` falls by no more than 0.02 from clean.

## Results *(fill in)*

| Method | Condition | Enrol (s) | WER | `tgt_cos` | Ceiling | Floor | Norm. sim. | Subgroup min | `src_cos` | UTMOS | F0 reg. err (st) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 nearest | main | 60 | | | | | | | | | |
| B0 nearest | duration | 10 / 30 / 60 / 180 | | | | | | | | | |
| B0 nearest | degraded | 60 | | | | | | | | | |
| B1 fine-tune | 2 speakers | — | | | | | | | | | |

Thresholds handed to Stage 07: minimum net speech = …; minimum SNR = ….

## Implementation notes

- **Why a ceiling and a floor.** A raw cosine of 0.45 means little on its own.
  Between "an unconverted stranger" and "the real person on another day" it
  has a position, and that position is comparable across speakers.
- **Why degrade only the enrolment audio.** The question is what happens when
  the user's files are poor. Degrading sources as well would measure the
  content encoder's robustness, which M2 Stage 04 owns.
- **Cost.** About 4 000 conversions per method plus Whisper decoding. Use
  `--limit` for iteration and run the full set only for gate rows.
- **Corpus licences.** Both corpora are CC BY 4.0; their audio stays under
  `$FLUXVOICE_DATA` and only ids and hashes are committed.

## Files touched

- `python_training/make_enroll_eval_lists.py`, `eval_enroll.py`,
  `fvml/enroll_eval.py` (new), `fvml/f0_ref.py`, `extract_units.py`,
  `train_content_encoder.py`, `train_unit_vocoder.py`,
  `python_training/eval_lists/enroll/**` (new),
  `python_training/tests/test_enroll_eval.py` (new), `CMakeLists.txt`,
  `python_training/DATASETS.md`

## Definition of done

- The three lists are frozen and hashed; rebuilding them reproduces the same
  files.
- `eval_enroll.py` runs end to end on the fixture bundle with fake scoring
  models in CTest.
- B0 has a filled row for each condition on a trained bundle, or the row is
  marked *blocked: no trained bundle* with the date.
- The two thresholds are stated with the curve they were read from.
