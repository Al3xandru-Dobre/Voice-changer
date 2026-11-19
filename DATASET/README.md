# DATASET — Corpus Collection Plan & Catalog

> Master plan for broadening the FluxVoice-S2UT training/eval data coverage with
> **open, commercially usable licences** (CC0, public domain, CC BY, MIT/BSD/Apache).
> Machine-readable catalog: [`catalog.tsv`](catalog.tsv).

This directory is a **tracked catalog and plan** — it contains documentation only
(no audio; `DATASET/.gitignore` blocks audio/archive files). Actual audio keeps
flowing into `$FLUXVOICE_DATA` (default `<repo>/data/`, git-ignored) through the
existing tooling:

- `python_training/prepare_data.py` — Mode A (English VC) corpora + augmentation
- `python_training/prepare_s2ut_data.py` — Mode B `En↔{Es,Fr,De}` paired + vocoder corpora
- `python_training/prepare_target_voice.py` — private own-voice enrolment
- `python_training/DATASETS.md` — authoritative list of what is *currently* wired up

New corpora listed here are **proposals** until a `build_*` entry exists in
`fvml/datasets.py` / `fvml/s2ut_data.py` and a row is added to `DATASETS.md`.

---

## 1. Licence policy

| Tier | Licences | Rule |
|---|---|---|
| **Preferred** | CC0-1.0, public domain, MIT, BSD, Apache-2.0 | Use freely; keep attribution text for `MODEL_CARD.md` / `voices.json`. |
| **Acceptable** | CC BY 4.0 | Use; attribution required (per `DATASETS.md` convention). |
| **Flagged** | CC BY-SA, per-file mixed licences (FSD50K, MUSAN), gated-but-permissive | Legal review before training; keep per-file `LICENSE` metadata; noise/RIR **input-side only** (never ships in model output). |
| **Non-commercial** | CC BY-NC(*), research licences, gated user agreements | Opt-in only via `--include-noncommercial`; isolated under `data/nc/` so default manifests cannot read it (BUG-031 defence). Never in shipped bundles. |
| **Forbidden** | CC *-ND for synthesis targets, GPL audio pools (VoxForge), LDC/proprietary, "identify the speaker" breaches | Do not ingest. |

---

## 2. Directory layout (dataset types by purpose)

| Dir | Purpose | Feeds |
|---|---|---|
| [`core_speech/`](core_speech/) | Large multi-speaker English pools | M2 content encoder (HuBERT km100 distillation), causal Unit-HiFi-GAN |
| [`target_voices/`](target_voices/) | Studio-grade speaker pools for presets/fine-tune | M2 Stage 05 preset voices, Stage 10 fine-tune |
| [`eval/`](eval/) | Mode A eval corpora (WER, speaker sim, F0, MOS) | `eval_vc.py`, `eval_lists/` |
| [`s2ut_paired/`](s2ut_paired/) | Parallel speech-translation pairs | M3 Wait-*k* translator (all 6 directions) |
| [`multilingual_speech/`](multilingual_speech/) | Monolingual `{es,fr,de,+}` pools | M3 multilingual encoder + vocoder upgrade |
| [`augmentation_noise_rir/`](augmentation_noise_rir/) | Noise, music, babble, impulse responses | `fvml/augment.py` input-side augmentation |
| [`expressive/`](expressive/) | Emotional/expressive/style speech | Prosody research, future style presets |
| [`noncommercial/`](noncommercial/) | NC/research-only catalog (opt-in) | Research probes only — `data/nc/` |
| [`private/`](private/) | Own-voice consent recordings convention | `data/private/<speaker>/` (never committed) |

---

## 3. Gap analysis — what we have vs. what we lack

**Already registered** (see `python_training/DATASETS.md`):
LibriSpeech (100/360 + eval splits), LJSpeech, Hi-Fi TTS (4 readers), MUSAN,
RIRS_NOISES, VCTK (optional) · CVSS-C, CoVoST 2 `en_de`, Europarl-ST, CSS10,
CML-TTS · VoxPopuli (NC opt-in) · private own-voice.

**Gaps this plan addresses:**

1. **Read-speech bias.** All current English pools are audiobook read speech.
   No conversational, accented, or in-the-wild data → weak robustness on real
   microphone input. *Fix: Common Voice (CC0), CV Spontaneous Speech (CC0),
   People's Speech CC-BY subset, AMI (CC BY).*
2. **Vocoder quality ceiling.** 16 kHz LibriSpeech is adequate; 24 kHz
   LibriTTS-R (CC BY) and studio M-AILABS (BSD) / CMU Arctic add cleaner
   target-side material.
3. **Multilingual vocoder thin.** Only CSS10 + CML-TTS per language (~40 h).
   *Fix: MLS `es/fr/de` (CC BY), M-AILABS `es/fr/de` (BSD), Common Voice
   `es/fr/de` (CC0).*
4. **Augmentation narrowness.** Only MUSAN + one RIR set. *Fix: FSD50K
   (licence-filtered subset), DEMAND (CC BY-SA, flagged), OpenAIR RIRs
   (per-IR licence check).*
5. **Expressive speech absent.** Almost nothing commercial-clean exists —
   documented in `expressive/`; rely on param controls + augmentation.
6. **Eval coverage.** Add LibriSpeech `test-other`, FLEURS (CC BY, 102-lang),
   Common Voice test, CV Spontaneous test for accent/spontaneous WER.

---

## 4. Top proposals (verified licences, commercial-clean first)

| Corpus | Licence | Lang | Size | Why |
|---|---|---|---|---|
| Common Voice Scripted v27 | CC0-1.0 | 295 langs | ~42.6 kh | Speaker/accent diversity, real mics (Mozilla Data Collective) |
| Common Voice Spontaneous v5 | CC0-1.0 | 80 langs | ~0.3 kh validated | Only permissive *conversational* speech; rob. eval |
| LibriTTS-R (SLR 141) | CC BY 4.0 | en | 585 h | 24 kHz Miipher-restored LibriTTS → better vocoder |
| MLS (SLR 94) | CC BY 4.0 | en,es,fr,de,it,nl,pl,pt | 44.6 kh en + 0.7–5.7 kh/lang | Multilingual scale (use `*_opus` tarballs) |
| M-AILABS | BSD-3-style | en,de,fr,es,it,uk,ru,pl | ~1 kh | Studio readers; en_US 102 h, de 237 h, fr 190 h, es 108 h |
| CMU Arctic | Permissive (any purpose) | en | ~8 h | Small studio target voices (slt/bdl/rms/clb + accented) |
| LibriLight | MIT tooling + public-domain audio | en | 60 kh | Encoder pretrain pool (unlabelled ok for SSL distillation) |
| People's Speech `cc-by-clean` | CC BY 4.0 | en | ~30 kh (subset CC-BY only) | Conversational diversity |
| AMI meetings | CC BY 4.0 | en | ~100 h | Spontaneous multi-mic eval/train |
| FLEURS | CC BY 4.0 | 102 langs | ~12 h/lang | Multilingual eval sets `en/es/fr/de` |
| FSD50K | corpus CC BY; per-clip CC0/BY/NC/Sampling+ | — | 108 h | Noise events; **filter to CC0+CC-BY clips only** |
| DEMAND | CC BY-SA 3.0 *(flagged)* | — | 1.5 h ×16ch | Environmental noise, input-side only |
| Emilia-YODAS | CC BY 4.0 *(provenance caution)* | en,de,fr,zh,ja,ko | 114 kh | In-the-wild spontaneous; large scale-up option |
| OpenAIR | per-IR (mostly CC BY) — verify | — | ~200 IRs | More reverb diversity |

## 5. Non-commercial bucket (opt-in, `data/nc/` only)

Europarl-ST (see §6), Expresso (CC BY-NC), mTEDx & MuST-C (CC BY-NC-ND),
TED-LIUM 3 (NC-ND), ESD (research licence), L2-ARCTIC (NC), SPGISpeech
(Kensho non-commercial), RAVDESS (NC-SA), IEMOCAP (gated), SeamlessAlign (NC),
ESC-50 (NC), Emilia original (NC), VoxCeleb (research eval), Speech Accent
Archive (NC-SA). Details in [`noncommercial/README.md`](noncommercial/).

## 6. Discrepancies found during research — action required

- **`europarl-st-en_{es,fr,de}` licence is wrong in the registry.**
  `fvml/s2ut_data.py` and `python_training/DATASETS.md` list it as
  **"CC BY 4.0"**, but the MLLP README and ICASSP 2020 paper state
  **CC BY-NC 4.0** ("Attribution-NonCommercial"). Under the project's own
  licence guardrail it belongs in group `nc` behind `--include-noncommercial`,
  isolated under `data/nc/`. This affects the `en_{es,fr,de}` paired builders
  that currently synthesize Piper targets on top of it. Fix pending decision:
  either reclassify it NC (primary `en→{es,fr}` sources then shrink to
  reverse-CoVoST synthesis only — `en_de` still has CC0 CoVoST 2), or keep as
  clearly-marked research-only input. **Recommend filing a BUG-0xx report.**
- **`voxpopuli-s2ut` NC flag is conservative, possibly over-strict.** The
  repo's licence table marks *data* CC0 (code + pretrained models CC BY-NC),
  while the repo `LICENSE` file is CC BY-NC. Keep `nc/` isolation until the
  S2UT subset provenance is re-verified — current treatment is safe.
- **`whisper-large-v3` is listed as "Apache 2.0"** in `DATASETS.md`; OpenAI
  ships Whisper under **MIT**. Cosmetic; even more permissive.
- `css10-*` shows "Apache-2.0 / CC0-1.0" — correct, but pin the exact licence
  file in `datasets.lock.json` provenance notes.

## 7. Phased acquisition plan

| Phase                                  | Contents                                                                                                                   | Est. download | Disk (16 kHz FLAC) |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ------------- | ------------------ |
| P0 — already registered, pending fetch | `prepare_data.py --datasets eval` + `core`                                                                                 | ~85 GB        | ~120 GB            |
| P1 — cheap wins                        | LibriTTS-R `dev/test` + `train-clean-100`, CMU Arctic, CV Spontaneous (en/es/fr/de), FLEURS `en/es/fr/de`, FSD50K filtered | ~20 GB        | ~25 GB             |
| P2 — diversity                         | Common Voice `en,es,fr,de` scripted subsets, M-AILABS `en/de/fr/es`, MLS `es/fr/de` (opus), AMI                            | ~120 GB       | ~90 GB             |
| P3 — scale (optional)                  | LibriLight `unlab-600h`, People's Speech CC-BY subset, LibriTTS-R `train-clean-360`, MLS `en` opus                         | ~800 GB       | ~500 GB            |
| P4 — NC research (opt-in)              | Expresso, mTEDx `es/fr/de`, L2-ARCTIC, ESC-50                                                                              | ~60 GB        | `data/nc/` only    |

Check disk first: `prepare_data.py --list` prints free space; respect
`--delete-archives` to avoid double storage.

## 8. Standing rules for adding a corpus

1. Verify the licence on the **canonical source page** (not a mirror); record
   the licence string + URL in the dir README and `catalog.tsv`.
2. Mixed per-file licences (FSD50K, MUSAN): keep a per-clip licence column and
   filter to commercial-clean before manifests are built.
3. Add a `build_*` to `fvml/datasets.py` or `S2UT_DATASETS` (with
   `noncommercial=True` when needed), pin SHA256 in `datasets.lock.json`
   (TOFU), and register a `--self-test`-able path per AGENTS.md rule 5.
4. Update `python_training/DATASETS.md` (source, licence, attribution text,
   use) and this catalog's `status` column.
5. Never commit audio or private voices (see `private/README.md`).
