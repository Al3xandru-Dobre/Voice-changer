# Stage 04 — Speaker encoder & speaker-conditioned vocoder (decision gate)

> **Milestone 5** · [Master plan](00_master.md) · [← Stage 03](03_unseen_speaker_eval_harness.md) · [Stage 05 →](05_on_device_embedding_refinement.md)

**Status:** ☐ not started
**Depends on:** [Stage 03](03_unseen_speaker_eval_harness.md) (lists, baseline B0, `--exclude-speakers`),
[M2 / Stage 05](../Milestone%202/05_causal_vocoder_decision_gate.md) (vocoder trainer, gate winner checkpoint),
[M2 / Stage 04](../Milestone%202/04_content_encoder_candidates.md) (unit extraction)
**Owner persona:** `@ml-speech-trainer`

## Objective

Train a speaker encoder that turns reference audio into the 64-value
`spk_emb`, and a vocoder that follows that embedding for voices it has never
heard, without changing the vocoder's inputs, its real-time cost class or the
quality of the curated voices. Decide by a written rule whether the result is
good enough to ship as the Instant tier.

## Context & verified starting point (2026-11-18)

- The vocoder's timbre input is `spk_emb` `[B, 64]`, applied through FiLM
  (`gamma, beta = Linear(spk_emb)`) after each upsampling stage
  (`fvml/vocoder.py`, class `FiLM`). `UnitVocoder.speaker_embedding()` already
  accepts either a row index or a float vector, so a vector from an encoder
  needs no change to the generator.
- The vectors come from `nn.Embedding(n_speakers, 64)` with five rows
  (`DEFAULT_SPEAKERS`: LJSpeech and four Hi-Fi TTS readers). The generator is
  V2-class, about 1.0 M parameters (`upsample_initial_channel = 128`).
- Per-speaker F0 and energy statistics are computed by
  `compute_speaker_stats()` in `train_unit_vocoder.py` from the causal tracker
  in `fvml/features.py`.
- `MixedBatchSampler`, `EarlyStopper`, `--finetune-from` and
  `--exclude-targets` exist from M2 Stage 10.
- Corpora with many speakers and usable licences are registered or listed:
  LibriTTS-R (CC BY 4.0; `train-clean-100` registered), VCTK 0.92 (CC BY 4.0,
  optional group). The SpeechBrain ECAPA weights are trained on VoxCeleb,
  which [`DATASET/README.md`](../../DATASET/README.md) classes as
  research-only.

## Candidates

| Arm | Speaker conditioning | Why | Shippable by default |
|---|---|---|---|
| **S-A** (recommended) | Small encoder trained jointly with the vocoder: log-mel front-end → convolutional frame encoder → learned (attentive) pooling → `Linear` → 64 values, L2-normalised and scaled | The StreamVC design; the space is shaped by what the vocoder needs; no external weights | Yes |
| **S-C** | Train the vocoder with a large learned table first, then regress audio → row | Fallback if joint training destabilises the GAN | Yes |
| **S-B** | Frozen ECAPA (192-d) + a learned projection to 64 | Fast reference point for how much a strong verification model helps | No: VoxCeleb-derived weights; also the same model that scores `tgt_cos`. Research flag only, scored with a second verifier |

Each arm is tried at generator width 128 (today) and, if needed, 256.

## Design of the recommended arm

- **The encoder is offline.** It runs once per enrolment, so it may look at
  the whole window. No causality constraint, no state.
- **Front-end inside the module.** Log-mel (80 bins, 25 ms window, 10 ms hop)
  computed with fixed convolution kernels, so the exported graph needs no
  STFT operator and the C++ side needs no mel code.
- **Fixed 3-second windows.** The encoder sees 3 s of speech. Longer
  references are handled by the enrolment function (windows averaged with
  weights, Stage 06), and training uses that same function.
- **Reference ≠ target.** For each training item the embedding is computed
  from a *different* utterance of the same speaker. Otherwise the 64 values
  could carry content.
- **A smooth space.** L2-normalise, multiply by a fixed `embedding_scale`,
  add small Gaussian noise during training. Timbre blend
  ([M2 / Stage 09](../Milestone%202/09_voice_controls_presets.md)) and
  refinement (Stage 05) both move through the space between trained points.
- **Auxiliary speaker classification** on the embedding (weighted, optional
  arm). It makes the space discriminative, which the own-voice live match in
  Stage 10 relies on.
- **Reference-side augmentation only:** noise, reverberation, band limiting,
  lossy-codec round trips and level changes on the encoder's input. The
  target waveform stays clean. This follows the existing rule that noise and
  impulse responses are input-side only.
- **One source of embeddings.** After training, the curated voices'
  `voices.json` embeddings are produced by the same enrolment function over
  their own audio. Public voices and personal profiles then live in one
  space, and the table is no longer exported.

## Data

| Corpus | Speakers | Licence | Use |
|---|---|---|---|
| LibriTTS-R `train-clean-100` + `train-clean-360` | about 1 150 | CC BY 4.0 | Main speaker variety |
| VCTK 0.92 (mic1) | about 110 | CC BY 4.0 | Accents |
| LJSpeech, Hi-Fi TTS (the five curated voices) | 5 | Public domain, CC BY 4.0 | Keep the shipped voices; sampled in a fixed share of batches |
| MUSAN, RIRS_NOISES | — | per-file, Apache 2.0 | Reference-side augmentation only |

- The 40 Stage 03 speakers are excluded everywhere with `--exclude-speakers`.
- Corpora collected without the speakers' participation (scraped or
  "in the wild" sets, VoxCeleb) are not used for the shipped encoder
  ([Stage 01](01_privacy_consent_misuse_architecture.md), threat table).
- Attribution for every corpus goes into `MODEL_CARD.md` at export.

## Tasks

- [ ] `fvml/speaker_encoder.py`: `SpeakerEncoder` (front-end, frame encoder,
      pooling, projection), `count_params`, unit tests for shape, determinism
      and insensitivity to level.
- [ ] `fvml/enroll.py::enroll()`: the reference enrolment function (windows,
      weights, normalisation, statistics). Used by training, by the exporter
      for the curated voices, and mirrored in C++ in Stage 07.
- [ ] `train_unit_vocoder.py`: `--speaker-encoder {joint,table,frozen-ecapa}`,
      reference sampling (same speaker, different utterance), reference
      augmentation, balanced speaker batches, `--aux-speaker-cls`,
      `--exclude-speakers`. The table path stays the default so existing runs
      and tests are unchanged.
- [ ] Unit extraction and statistics for the new corpora
      (`extract_units.py`, `compute_speaker_stats`), resumable.
- [ ] **Timing first:** measure wall-clock per 10 000 steps and disk use on
      the 8 GB laptop GPU before committing to a schedule. Record it here.
- [ ] Train the arms. Start from the M2 gate winner checkpoint
      (`--finetune-from`); from scratch only if fine-tuning plateaus.
- [ ] Leakage check: enrol from utterance *A*, convert with units from
      utterance *B*; a content probe on the embedding must be at chance.
- [ ] Score every arm with `eval_enroll.py` (main condition) and with
      `eval_vc.py` on the curated voices. Measure vocoder cost per step with
      the M2 Stage 05 gate tooling (`gate_report.py`), in ONNX Runtime.
- [ ] Embedding-stability trials: equal error rate at 30 s against 10 s.
- [ ] Fill the gate table and record the decision in the
      [master plan](00_master.md) (open decisions 5, 6 and 10).

## Decision gate

Written before any arm is scored. Measured on the Stage 03 main condition
(60 s of enrolment), in the shipped runtime.

An arm is **eligible** when all of these hold:

- **Curated voices do not regress** against the M2 gate winner: WER within
  +1 point, target cosine within −0.02, UTMOS within −0.1.
- **Unseen speakers are intelligible:** WER ≤ 10 % absolute and at most
  2 points above the curated voices.
- **Unseen speakers sound acceptable:** UTMOS ≥ 3.0.
- **It beats the baseline:** mean `tgt_cos` ≥ B0 + 0.05, and no subgroup
  (gender, corpus, accent group) is below B0.
- **The real-time budget holds:** vocoder per-step p95 stays inside the M2
  compute line for the gated hop; no new lookahead; contract inputs unchanged.

**Choose** the eligible arm with the highest normalised similarity; break
ties on lower vocoder cost. **Stretch:** normalised similarity ≥ 0.6.

If nothing is eligible, try in this order and record each step:

1. Width 256, if its cost still fits the budget.
2. Arm S-C.
3. Make Stage 05 refinement part of the default path, if its own gate passes.
4. Ship only the baseline, labelled "closest built-in voice, with your pitch
   range", and record the deviation. It is not called a personal voice.

The **verified badge** (open decision 10) ships only if the chosen encoder's
equal error rate is ≤ 5 %; the threshold is then set for ≤ 1 % false accepts.

## Gate table *(fill in)*

| Arm | Width | Gen. params | Vocoder p95 ORT (ms) | Curated WER / cos / UTMOS | Unseen WER | Unseen `tgt_cos` | B0 | Norm. sim. | Subgroup min Δ | EER | Eligible |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S-A | 128 | | | | | | | | | | |
| S-A | 256 | | | | | | | | | | |
| S-C | 128 | | | | | | | | | | |
| S-B *(research)* | 128 | | | | | | | | | | — |

## Implementation notes

- **Why the generator is not changed.** Keeping `spk_emb` as the only timbre
  input means Mode A's graphs, contract and per-hop cost are those M2 already
  measured. The cost of this feature is paid once, at enrolment.
- **Capacity is the open question.** A 1 M-parameter causal generator may not
  carry a thousand timbres as well as it carries five. That is why width is a
  gate arm and why the baseline and the ceiling are both in the table.
- **Accent and style do not transfer.** The units and the prosody come from
  the live speaker. The interface text is written from this fact, not from
  the similarity score.
- **Decision gates measure the shipped runtime**
  ([BUG-015](../Bug%20Report/BUG-015_gate_decides_on_pytorch_clock.md)):
  ONNX Runtime timings, a relative tolerance on a non-silent output, and any
  exporter fallback reported.

## Files touched

- `python_training/fvml/speaker_encoder.py`, `fvml/enroll.py` (new),
  `fvml/vocoder.py`, `train_unit_vocoder.py`, `extract_units.py`,
  `gate_report.py`, `python_training/tests/test_speaker_encoder.py`,
  `python_training/tests/test_enroll.py` (new), `python_training/DATASETS.md`,
  `CMakeLists.txt`, this file (timing, gate table, decision),
  [master plan](00_master.md) (open decisions)

## Definition of done

- The gate table has a measured row for every trained arm, and the decision
  is recorded with the rule that produced it.
- The leakage check and the curated-voice regression check are recorded.
- A checkpoint pair (vocoder + speaker encoder) for the chosen arm exists
  with its `speaker_space`, ready for Stage 06, or the fallback taken is
  recorded as a dated deviation.
