# Stage 10 — Target-voice recording & fine-tune

> **Milestone 2** · [Master plan](00_master.md) · [← Stage 09](09_voice_controls_presets.md) · [Stage 11 →](11_verification_exit.md)

**Status:** ◐ tooling implemented and tested offline (2026-11-14); recording
session, fine-tune run, evaluation and live check pending
**Depends on:** Stage 05 (bootstrap vocoder + winner ids), Stage 06 (export path).
The recording session itself has no dependencies and can happen any time.
**Research:** §6 data & recording · §5 multi-speaker conditioning · §7 evaluation

## Objective

Add the user's own recorded voice as a selectable target ("Me", open decision 5)
without degrading the public voices the presets use. Steps:
1. record 1–3 h under a written protocol;
2. preprocess it reproducibly;
3. fine-tune the multi-speaker vocoder with a new speaker row, with perturbation
   augmentation;
4. export through Stage 06.

Also record a **live-mic source evaluation set**. It is the only test input that
matches the real microphone and room.

## Context & references

- **README §3 recipe:** "Record 1–3 h of clean, dry audio of the target speaker
  → pass through Stage 1 → train the vocoder". Here this becomes a fine-tune of
  the Stage 05 multi-speaker vocoder rather than a single-speaker model, which
  converges faster and keeps the other voices.
- **No transcripts needed.** The pipeline is textless.
- **Perturbation.** Units from other people still carry some speaker
  information. Formant/pitch/EQ perturbation before unit extraction makes the
  vocoder robust to it (research §6).

## Recording protocol

- **Equipment:** one microphone (USB or XLR + interface), pop filter, 15–30 cm
  fixed distance, the same position and gain in every session, headphones
  (never speakers).
- **Room:** quiet, dry (soft furnishings, closet, or a reflection filter). Record
  30 s of room tone at the start of each session.
- **Format:** 48 kHz / 24-bit WAV, mono. Peaks around −6 dBFS, never clipping.
- **Material** (about 1 h minimum, 3 h ideal), aiming at the pitch range the
  controls will use:
  - CMU ARCTIC prompts
  - Harvard sentences
  - read passages
  - questions and exclamations
  - emotional reads (happy, sad, emphatic)
  - 10–15 min of spontaneous speech
- **Sessions:** ≤ 45 min each, with breaks and water. Note the session id in the
  filenames (`me_s01_0001.wav`).
- **Takes:** keep individual takes to ≤ ~5 minutes. Held-out selection prefers
  *whole takes* so a held-out reference never shares a take with a training
  clip; one 45-min take per session would force the `take-tail` fallback
  (held-out tails + guard clips) instead.
- **Consent:** record only voices you have the right to use. For anyone other
  than the user, keep written consent with the dataset.
- **Live-mic source set:** separately, 10 min of natural speech through the
  *everyday* mic and room (the one used with fluxvoice). Used as conversion
  *input* in Stage 11.

## Tasks

- [x] `python_training/prepare_target_voice.py`:
  - [x] high-pass at 60–70 Hz
  - [x] loudness-normalise to −23 LUFS (`pyloudnorm`)
  - [x] VAD segmentation into 2–12 s clips
  - [x] reject clips containing any sample ≥ 0.999 FS (clipping) or with an
        estimated SNR below 30 dB (room-tone based)
  - [x] resample to the pipeline rate
  - [x] write a manifest with per-clip SNR and duration
  - [x] hold out 5 min for evaluation
- [ ] **Features.** Stage 03 tracker features + winner-encoder units for every
      clip.
  - [x] Additionally, for a fraction (≈ 50%) of training clips, extract units
        from a **perturbed** copy (formant shift ±10–15% and pitch ±2 st via a
        Praat-style "change gender" or WORLD resynthesis, plus random EQ).
        Conditioning features always come from the clean clip.
- [x] **Fine-tune.**
  - [x] Add a speaker row "me" to the embedding table, initialised from the
        nearest existing row (ECAPA on the recordings).
  - [x] Train on a 50/50 mix of own-voice and public-voice batches, so the other
        rows don't drift.
  - [x] Stop on held-out mel loss + UTMOS.
- [ ] **Evaluate:**
  - [ ] own-voice resynthesis UTMOS
  - [ ] VC from `test_clean_200` into "me" vs. into the nearest existing voice
        (zero-shot baseline): target cosine to held-out "me" clips, WER
  - [ ] regression check on the public voices
- [ ] **Export** through Stage 06 (new bundle version) and add the "me" entry to
      `voices.json` with `license: "private — do not distribute"`.
- [ ] **Live check:** `fluxvoice --vc --voice me` for 10 min.

## Implementation notes

- **Why a row rather than a separate model.** The user's voice then works with
  every control, including `timbre_blend` between "me" and a public voice, and
  the bundle stays one file per role.
- **Keep the private voice private.** The bundle containing "me" is personal
  data and is git-ignored like all of `models/`. The model card must say it is
  not for distribution.
- **Bad audio can't be fixed in training.** Re-record rather than denoise:
  denoisers leave artefacts the vocoder will faithfully learn.
- **Robustness to the live mic** is tested, not assumed. The live-mic source set
  shows whether noise and reverb augmentation in the student (Stage 04) was
  enough.
- **Per-take loudness normalisation** (`prepare_target_voice.py`). The whole
  take is normalised to −23 LUFS *before* VAD so one gain serves every clip of
  the take; the session's room tone gets the same HPF and the same take gain,
  keeping the SNR estimate consistent.
- **Clipping is checked on the raw samples** (`is_clipped`, ≥ 0.999 FS) before
  any gain, plus a second `clipped_after_norm` check on each segment after
  normalisation — a loudness-normalised clip that now peaks at full scale is
  rejected, never repaired.
- **Deterministic held-out selection** (`select_heldout`): whole *takes* are
  preferred — sorted by (session, take), evenly spaced picks grown until
  ≥ 5 min — so a held-out "me" reference never shares a take with a training
  clip. Whole-take holdout is accepted only when the total stays
  ≤ max(2 × target, target + 60 s) and ≥ 50 % of the audio remains for
  training; otherwise the `take-tail` fallback holds out contiguous segment
  tails from evenly spaced takes plus one *guard* clip before each tail
  (excluded from both train and held-out, reason `heldout_guard`), and a
  warning recommends shorter takes.
- **Perturbation lives in `extract_units.py`** (`--perturb-fraction` etc.), not
  in the dataset: the manifest `path` stays the clean clip, so the trainer's
  pitch/energy conditioning and the audio it copies are always clean; only the
  encoded waveform is perturbed. `should_perturb` is a sha256 uniform — the
  same clip is perturbed identically on every rerun. Praat's own RNG is seeded
  per utterance (`random_initializeWithSeedUnsafelyButPredictably`) so the
  "Change gender" resynthesis is bit-reproducible. The perturbation config is
  part of the units-directory identity, so a rerun with a different config
  raises `UnitsMismatch`.
- **Speaker-row init:** `--new-speaker-init first` (default, = the old row-0
  copy) keeps existing runs valid; the recipe uses `nearest` (ECAPA centroid
  cosine, `fvml.speaker`) or a named row. The chosen source is recorded in the
  checkpoint (`speaker_init`).
- **Batch-level 50/50 mix:** `MixedBatchSampler` draws each batch *entirely*
  from the mix group (own voice) with probability `--mix-ratio`, else entirely
  from the public voices — a 50/50 mix at batch granularity, not sample
  granularity, so the public rows keep seeing homogeneous updates.
- **Early stop** (`EarlyStopper`): a validation counts as improved when held-out
  mel loss improves, or (with `--val-utmos`) mean val UTMOS improves by > 0.01;
  stop after `--early-stop-patience` validations without improvement.
  `best.pt` stays chosen by mel alone.
- **Private licence flows** from `speaker_metadata.json` (written by
  `prepare_target_voice.py`, passed via `--speaker-metadata`) into the
  checkpoint, then into `voices.json` (`license: "private — do not distribute"`)
  and `MODEL_CARD.md` (top "PRIVATE — contains a personal voice" banner,
  `export_report.json` → `"private": true`) during Stage 06 export — no
  post-processing needed.

## Recipe (recording → live check)

```powershell
# 1. Record per the protocol above: me_sNN_TTTT.wav + me_sNN_roomtone.wav
#    (48 kHz / 24-bit mono) under data/raw_me/ (never commit — private data).
# 2. Clean + segment + held-out split.
python prepare_target_voice.py --input data/raw_me --speaker me `
    --display "Me (private)" --min-minutes 60 --heldout-s 300
# 3. Units with the Stage 05 winner encoder + 50% perturbation.
python extract_units.py --tokenizer student-L1 `
    --checkpoint checkpoints/<winner>/final.pt `
    --manifest <data>/private/me/train.jsonl --out-dir <data>/units/me_student `
    --perturb-fraction 0.5
# 4. Fine-tune: new "me" row from the nearest existing row, 50/50 batches,
#    early stop on val mel + UTMOS; held-out clips stay excluded.
python train_unit_vocoder.py `
    --train-manifest eval_lists/train_manifest.jsonl,<data>/private/me/train.jsonl `
    --train-units <data>/units/train_student,<data>/units/me_student `
    --exclude-targets eval_lists/targets.tsv,<data>/private/me/me_targets.tsv `
    --finetune-from checkpoints/<base_vocoder>/final.pt `
    --new-speaker-init nearest --mix-speakers me --mix-ratio 0.5 `
    --speaker-metadata <data>/private/me/speaker_metadata.json `
    --early-stop-patience 3 --val-utmos --steps 50000 --run-name voc_me
# 5. Export a NEW bundle (same Stage 06 command, new output dir).
python export_stateful_onnx.py --encoder-ckpt <enc> `
    --vocoder-ckpt checkpoints/voc_me/final.pt --out models/voice_me_v1
# 6. DoD evaluation on the shipped runtime.
python eval_target_voice.py --bundle-before models/<base_bundle> `
    --bundle-after models/voice_me_v1 --voice me `
    --checkpoint checkpoints/voc_me/final.pt `
    --me-targets <data>/private/me/me_targets.tsv `
    --heldout-manifest <data>/private/me/heldout.jsonl --out eval_out/stage10
# 7. Live check (10 min, 0 misses after warm-up).
fluxvoice --vc --voice me --stats 10
```

## Files touched

- `python_training/fvml/target_voice.py` (new), `python_training/fvml/vcfile.py`
  (new), `python_training/fvml/augment.py` (speaker perturbation),
  `python_training/prepare_target_voice.py` (new),
  `python_training/extract_units.py` (perturbation flags),
  `python_training/train_unit_vocoder.py` (fine-tune mode),
  `python_training/export_stateful_onnx.py` (private-voice banner/report flag),
  `python_training/eval_controls.py` (new, Stage 09 offline DoD),
  `python_training/eval_target_voice.py` (new, Stage 10 DoD driver),
  `python_training/tests/test_target_voice.py`,
  `python_training/tests/test_eval_controls.py`,
  `models/<bundle>/**` (new version, git-ignored),
  `Documents/Milestone 2/00_master.md` (open decision 5 confirmed)

## Definition of done

- ≥ 60 min of clean own-voice audio after preprocessing, 0 clipped clips, and a
  5-minute held-out set.
- Held-out target cosine for VC into "me" improves by ≥ 0.05 over the zero-shot
  baseline (nearest existing row).
- Public voices regress by ≤ 0.02 target cosine, and WER stays within +2 pts of
  their pre-fine-tune values.
- `fluxvoice --vc --voice me` runs live for 10 min with 0 `misses` after warm-up.
