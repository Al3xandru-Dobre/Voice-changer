# Datasets, models and licences (Milestones 2 & 3)

Everything the training and evaluation stack downloads, what it is used for and
what its licence asks of us. `prepare_data.py` fetches Milestone 2 datasets and
`prepare_s2ut_data.py` prepares Milestone 3 bidirectional `En↔{Es,Fr,De}`
datasets; the evaluation models are fetched on first use by `eval_vc.py` /
`eval_s2ut.py`. All land under `$FLUXVOICE_DATA` (default `<repo>/data/`, which
ignores itself in git).

Copy the **Attribution** text into a bundle's `MODEL_CARD.md` and `voices.json`
(`attribution` field) when a model is trained on the dataset, and into the root
README credits (Stage 11).

## Milestone 2 — English VC & Augmentation Datasets (`prepare_data.py`)

| Key (`prepare_data.py`) | Source | Licence | Used for | Size (archive) |
|---|---|---|---|---|
| `librispeech-test-clean` | [OpenSLR 12](https://www.openslr.org/12/) | CC BY 4.0 | `eval_lists/test_clean_200.tsv` (conversion sources), baseline, `silence_hallucination_20.tsv` | 0.35 GB |
| `librispeech-dev-clean` | OpenSLR 12 | CC BY 4.0 | Streaming-parity streams (Stages 04-06), pitch accuracy (Stage 03) | 0.34 GB |
| `librispeech-train-clean-100` | OpenSLR 12 | CC BY 4.0 | Teacher k-means, student training, pitch/energy priors (Stages 03-04) | 6.39 GB |
| `librispeech-train-clean-360` | OpenSLR 12 | CC BY 4.0 | Student training (Stage 04) | 23.05 GB |
| `ljspeech` | [keithito.com](https://keithito.com/LJ-Speech-Dataset/) | Public domain | Bootstrap target voice `lj` (Stage 05) | 2.75 GB |
| `hifitts` | [OpenSLR 109](https://www.openslr.org/109/) | CC BY 4.0 | Target voices for presets and blending (Stage 05); readers 92, 8051, 6097, 9017 by default | 41.35 GB |
| `musan` | [OpenSLR 17](https://www.openslr.org/17/) | CC BY 4.0 overall; per-file licences (see below) | Noise/music/babble augmentation (Stage 04) + `silence_hallucination_20.tsv` | 11.09 GB |
| `rirs` | [OpenSLR 28](https://www.openslr.org/28/) (contains the [OpenSLR 26](https://www.openslr.org/26/) simulated RIRs) | Apache 2.0 | Reverberation augmentation (Stage 04) | 1.31 GB |
| `vctk` (optional) | [Edinburgh DataShare 10283/3443](https://datashare.ed.ac.uk/handle/10283/3443) | CC BY 4.0 | Accent evaluation; extra pitch-accuracy data (Stage 03) | 11.75 GB |

## Milestone 3 — Bidirectional S2UT & Multilingual Vocoder Datasets (`prepare_s2ut_data.py`)

| Key (`prepare_s2ut_data.py`) | Source | Licence | Used for | Notes |
|---|---|---|---|---|
| `cvss-c-{es,fr,de}_en` | [google-research-datasets/cvss](https://github.com/google-research-datasets/cvss) + [Mozilla Common Voice](https://commonvoice.mozilla.org/) / CoVoST 2 | CC BY 4.0 (CVSS-C) + CC0 1.0 (Common Voice / CoVoST 2) | Primary training + SimulEval benchmark for `{Es,Fr,De}→En` | Canonical English target speech joined with human `{es,fr,de}` source audio |
| `covost2-en_de` | [facebook/covost2](https://github.com/facebookresearch/covost) + Common Voice `en` | CC0 1.0 | Primary training + benchmark for `En→De` (~364 h) | Target German audio synthesized with canonical Piper VITS (`de_DE-thorsten-high`) and filtered by Whisper round-trip CER ≤ 5% |
| `europarl-st-en_{es,fr,de}` | [Europarl-ST v1.1](https://www.mllp.upv.es/europarl-st/) | CC BY 4.0 | Parallel `En→{Es,Fr,De}` training & reordering evaluation | Target `{es,fr,de}` audio synthesized with canonical Piper VITS (`es_ES-davefx-medium`, `fr_FR-siwis-medium`, `de_DE-thorsten-high`) with CER ≤ 5% gate |
| `css10-{es,fr,de}` | [kyubyong/css10](https://github.com/kyubyong/css10) | Apache 2.0 / CC0 1.0 (LibriVox) | Multilingual causal Unit-HiFi-GAN studio voices (`es` ~24 h, `fr` ~19 h, `de` ~16 h) | Single-speaker studio readings resampled to 16 kHz mono |
| `cml-tts-{es,fr,de}` | [OpenSLR 146 (`ylacombe/cml-tts`)](https://www.openslr.org/146/) | CC BY 4.0 | Multi-speaker diversity for the `{es,fr,de}` vocoder (M3 Stage 03) | Curated 2 male + 2 female speakers per language filtered by SNR ≥ 30 dB and duration 2–12 s |
| `voxpopuli-s2ut` *(opt-in `--include-noncommercial`)* | [facebookresearch/voxpopuli](https://github.com/facebookresearch/voxpopuli) | **CC BY-NC 4.0 (non-commercial)** | Research comparison only; **never** included in shipped commercial bundles | Isolated strictly under `$FLUXVOICE_DATA/nc/` so default training manifests cannot read it |

### Canonical TTS voices used during `En→{Es,Fr,De}` target-audio synthesis

| Language | Piper VITS checkpoint (`rhasspy/piper-voices`) | Licence | Speaker ID in manifest |
|---|---|---|---|
| Spanish (`es`) | `es_ES-davefx-medium` | CC BY 4.0 (Piper MIT runtime) | `piper_es_davefx` |
| French (`fr`) | `fr_FR-siwis-medium` (SIWIS database) | CC BY 4.0 (Piper MIT runtime) | `piper_fr_siwis` |
| German (`de`) | `de_DE-thorsten-high` (Thorsten-Voice) | CC0 1.0 (Piper MIT runtime) | `piper_de_thorsten` |

## Attribution

- **LibriSpeech:** "LibriSpeech ASR corpus" © 2014 Vassil Panayotov, prepared
  with the assistance of Daniel Povey, licensed CC BY 4.0. V. Panayotov,
  G. Chen, D. Povey, S. Khudanpur, "LibriSpeech: an ASR corpus based on public
  domain audio books", ICASSP 2015.
- **LJSpeech:** public domain; no attribution required. Courtesy citation:
  K. Ito and L. Johnson, "The LJ Speech Dataset", 2017.
- **Hi-Fi TTS:** "The Hi-Fi Multi-Speaker English TTS Dataset is made available
  by NVIDIA CORPORATION under a Creative Commons Attribution 4.0 International
  License." E. Bakhturina, V. Lavrukhin, B. Ginsburg, Y. Zhang, "Hi-Fi
  Multi-Speaker English TTS Dataset", arXiv:2104.01497, 2021. Name the reader
  (e.g. "Hi-Fi TTS reader 92, Cori Samuel / LibriVox") in each voice's
  `attribution`.
- **MUSAN:** D. Snyder, G. Chen, D. Povey, "MUSAN: A Music, Speech, and Noise
  Corpus", arXiv:1510.08484, 2015, CC BY 4.0. Per-file credits are in each
  subdirectory's `LICENSE` (the manifest's `licence_file` column).
- **RIRS_NOISES:** T. Ko, V. Peddinti, D. Povey, M. L. Seltzer, S. Khudanpur, "A
  Study on Data Augmentation of Reverberant Speech for Robust Speech
  Recognition", ICASSP 2017, Apache 2.0.
- **VCTK:** C. Veaux, J. Yamagishi, K. MacDonald, "CSTR VCTK Corpus: English
  Multi-speaker Corpus for CSTR Voice Cloning Toolkit (version 0.92)",
  University of Edinburgh, CSTR, 2019, CC BY 4.0.
- **CVSS-C:** Y. Jia, M. T. Ramanovich, Q. Wang, H. Zen, "CVSS Corpus and
  Massively Multilingual Speech-to-Speech Translation", LREC 2022, CC BY 4.0.
- **CoVoST 2:** C. Wang, A. Wu, J. Pino, "CoVoST 2 and Massively Multilingual
  Speech-to-Text Translation", INTERSPEECH 2021, CC0 1.0.
- **Europarl-ST:** J. Iranzo-Sánchez et al., "Europarl-ST: A Multilingual Corpus
  for Speech Translation of Parliamentary Debates", ICASSP 2020 (MLLP-UPV),
  CC BY 4.0.
- **CSS10:** K. Park and T. Mulc, "CSS10: A Collection of Single Speaker Speech
  Datasets for 10 Languages", INTERSPEECH 2019, Apache 2.0 / CC0 LibriVox.
- **CML-TTS:** A. C. Oliveira et al., "CML-TTS: A Multilingual Dataset for
  Speech Synthesis in Low-Resource Languages", TSD 2023 (OpenSLR 146),
  CC BY 4.0.
- **SIWIS French Speech Synthesis Database:** P.-E. Honnet et al., Idiap /
  University of Geneva, 2017, CC BY 4.0.
- **Thorsten-Voice:** T. Müller and D. Kreutz, CC0 1.0 Public Domain.

**MUSAN, per subset** (checked against arXiv:1510.08484 §2 at download time;
only content that allows commercial use was included in the corpus):

| Subset | Content | Licence |
|---|---|---|
| `speech` | LibriVox readings (~20 h); US government hearings and debates (~40 h) | Public domain (US government works believed public domain) |
| `music` | Jamendo, Free Music Archive, Incompetech, HD Classical Music (~42 h) | Creative Commons, varying per file (several are CC BY, which needs credit); see `musan/music/*/LICENSE` |
| `noise` | Free Sound (public domain portion) and Sound Bible (~6 h) | Public domain / Creative Commons per file; see `musan/noise/*/LICENSE` |

Augmentation noise is mixed into *training input only* and never ships in a
model's output, but keep the `LICENSE` files with any redistributed copy.

## Own voice (private — Milestone 2 Stage 10)

The user's own recordings (`<speaker>_sNN_TTTT.wav` + per-session room tone)
are **private data**: raw files and the prepared dataset live under
`<data root>/private/<speaker>/`, which is git-ignored; **they must never be
committed**. Bundles and checkpoints containing a private voice carry
`"license": "private — do not distribute"` in `voices.json` and a PRIVATE
banner in `MODEL_CARD.md`. Consent: record only voices you have the right to
use; for anyone other than the user, keep written consent with the dataset.

## Checksums

`prepare_data.py` and `prepare_s2ut_data.py` verify published MD5s (where
present) and pin SHA256 hashes in [`datasets.lock.json`](datasets.lock.json).
Archives without a prior pin record their SHA256 on first verified download
(trust on first use); commit the updated lock file after preparing them.

## Evaluation models

| Model | Source (pinned revision) | Licence | Used for |
|---|---|---|---|
| Whisper large-v3 | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) @ `06f233fe` | Apache 2.0 | WER/CER (`eval_vc.py`) and forced-`tgt_lang` ASR-BLEU / ASR-chrF++ (`eval_s2ut.py`) |
| ECAPA-TDNN | [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) @ `0f99f2d0` | Apache 2.0 | Target / source cosine, EER |
| UTMOS22 strong | [tarepan/SpeechMOS](https://github.com/tarepan/SpeechMOS) v1.2.0 code (lock) + v1.0.0 weights (SHA256 `38aa51ab…`), from [sarulab-speech/UTMOS22](https://github.com/sarulab-speech/UTMOS22) | MIT | Naturalness (`eval_vc.py` & `eval_s2ut.py`) |
| Praat pitch (parselmouth 0.4.7) | [YannickJadoul/Parselmouth](https://github.com/YannickJadoul/Parselmouth) | GPL-3.0-or-later | Offline reference F0 (eval only; never linked into the product) |

## Teacher & research models

| Model | Source | Licence | Stage & guardrail |
|---|---|---|---|
| HuBERT-Base | `facebook/hubert-base-ls960` | MIT (fairseq lineage) | M2 Stage 04 English km100 teacher |
| Mimi | `kyutai/mimi` | CC BY 4.0 | M2 Stage 04 kill test (80 ms frame rejected for Mode A) |
| **mHuBERT-vp** | [`mhubert_base_vp_en_es_fr_it3.pt`](https://dl.fbaipublicfiles.com/hubert/mhubert_base_vp_en_es_fr_it3.pt) (fairseq) | **MIT** | **M3 Stage 03 default commercial-clean multilingual km1000 teacher** |
| mHuBERT-147 *(opt-in)* | [`utter-project/mHuBERT-147`](https://huggingface.co/utter-project/mHuBERT-147) | **CC BY-NC-SA 4.0 (non-commercial)** | M3 Stage 03 research probe only (`--include-noncommercial`); never used to distill shipped commercial weights |
