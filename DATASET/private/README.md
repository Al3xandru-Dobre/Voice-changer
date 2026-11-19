# private — own-voice & consented recordings (never committed)

**Purpose:** document where private/consented voice data lives so it is never
accidentally published. **This directory is documentation only — no audio.**

**Consumers:** `prepare_target_voice.py` (own-voice enrolment, M2 Stage 10),
`eval_target_voice.py --voice me`, per-person bundle fine-tuning.

## Convention (from `python_training/DATASETS.md` § "Own voice")

- Raw takes named `<speaker>_sNN_TTTT.wav` + a per-session **room tone** file.
- Everything lives under `<data root>/private/<speaker>/` — that tree is
  git-ignored; raw files and prepared manifests **must never be committed**.
- Bundles/checkpoints containing a private voice carry
  `"license": "private — do not distribute"` in `voices.json` and a PRIVATE
  banner in `MODEL_CARD.md`.
- Consent: record only voices you have the right to use. For anyone other than
  yourself, keep written consent *with* the dataset (scanned consent form next
  to `consent.txt` under `data/private/<speaker>/`).
- Room tone is required — `fvml/target_voice.py` estimates the noise floor and
  rejects takes below the SNR threshold.

## Data hygiene checklist

- [ ] `git status` never shows `data/private/` or `DATASET/private/` audio
- [ ] private voice manifests stay under `data/private/` (not `manifests/`)
- [ ] eval lists for private voices live under `data/private/<speaker>/eval/`
- [ ] no speaker-identifying metadata in committed files
