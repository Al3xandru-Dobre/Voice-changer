# Stage 10 — Target-voice recording & fine-tune

> **Milestone 2** · [Master plan](00_master.md) · [← Stage 09](09_voice_controls_presets.md) · [Stage 11 →](11_verification_exit.md)

**Status:** ☐ not started
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
- **Consent:** record only voices you have the right to use. For anyone other
  than the user, keep written consent with the dataset.
- **Live-mic source set:** separately, 10 min of natural speech through the
  *everyday* mic and room (the one used with fluxvoice). Used as conversion
  *input* in Stage 11.

## Tasks

- [ ] `python_training/prepare_target_voice.py`:
  - [ ] high-pass at 60–70 Hz
  - [ ] loudness-normalise to −23 LUFS (`pyloudnorm`)
  - [ ] VAD segmentation into 2–12 s clips
  - [ ] reject clips containing any sample ≥ 0.999 FS (clipping) or with an
        estimated SNR below 30 dB (room-tone based)
  - [ ] resample to the pipeline rate
  - [ ] write a manifest with per-clip SNR and duration
  - [ ] hold out 5 min for evaluation
- [ ] **Features.** Stage 03 tracker features + winner-encoder units for every
      clip.
  - [ ] Additionally, for a fraction (≈ 50%) of training clips, extract units
        from a **perturbed** copy (formant shift ±10–15% and pitch ±2 st via a
        Praat-style "change gender" or WORLD resynthesis, plus random EQ).
        Conditioning features always come from the clean clip.
- [ ] **Fine-tune.**
  - [ ] Add a speaker row "me" to the embedding table, initialised from the
        nearest existing row (ECAPA on the recordings).
  - [ ] Train on a 50/50 mix of own-voice and public-voice batches, so the other
        rows don't drift.
  - [ ] Stop on held-out mel loss + UTMOS.
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

## Files touched

- `python_training/prepare_target_voice.py` (new),
  `python_training/train_unit_vocoder.py` (fine-tune mode, perturbation),
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
