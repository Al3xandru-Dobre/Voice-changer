# Datasets, models and licences (Milestone 2)

Everything the training and evaluation stack downloads, what it is used for and
what its licence asks of us. `prepare_data.py` fetches the datasets; the
evaluation models are fetched on first use by `eval_vc.py`. Both land under
`$FLUXVOICE_DATA` (default `<repo>/data/`, which ignores itself in git).

Copy the **Attribution** text into a bundle's `MODEL_CARD.md` and `voices.json`
(`attribution` field) when a model is trained on the dataset, and into the root
README credits (Stage 11).

## Datasets

| Key (`prepare_data.py`) | Source | Licence | Used for | Size (archive) |
|---|---|---|---|---|
| `librispeech-test-clean` | [OpenSLR 12](https://www.openslr.org/12/) | CC BY 4.0 | `eval_lists/test_clean_200.tsv` (conversion sources), baseline | 0.35 GB |
| `librispeech-dev-clean` | OpenSLR 12 | CC BY 4.0 | Streaming-parity streams (Stages 04-06), pitch accuracy (Stage 03) | 0.34 GB |
| `librispeech-train-clean-100` | OpenSLR 12 | CC BY 4.0 | Teacher k-means, student training, pitch/energy priors (Stages 03-04) | 6.39 GB |
| `librispeech-train-clean-360` | OpenSLR 12 | CC BY 4.0 | Student training (Stage 04) | 23.05 GB |
| `ljspeech` | [keithito.com](https://keithito.com/LJ-Speech-Dataset/) | Public domain | Bootstrap target voice `lj` (Stage 05) | 2.75 GB |
| `hifitts` | [OpenSLR 109](https://www.openslr.org/109/) | CC BY 4.0 | Target voices for presets and blending (Stage 05); readers 92, 8051, 6097, 9017 by default | 41.35 GB |
| `musan` | [OpenSLR 17](https://www.openslr.org/17/) | CC BY 4.0 overall; per-file licences (see below) | Noise/music/babble augmentation of student input (Stage 04) | 11.09 GB |
| `rirs` | [OpenSLR 28](https://www.openslr.org/28/) (contains the [OpenSLR 26](https://www.openslr.org/26/) simulated RIRs) | Apache 2.0 | Reverberation augmentation (Stage 04) | 1.31 GB |
| `vctk` (optional) | [Edinburgh DataShare 10283/3443](https://datashare.ed.ac.uk/handle/10283/3443) | CC BY 4.0 | Accent evaluation; extra pitch-accuracy data (Stage 03) | 11.75 GB |

**Attribution**

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
  Recognition", ICASSP 2017, Apache 2.0. The real RIRs inside come from RWCP,
  the REVERB 2014 challenge and AIR; see the corpus README.
- **VCTK:** C. Veaux, J. Yamagishi, K. MacDonald, "CSTR VCTK Corpus: English
  Multi-speaker Corpus for CSTR Voice Cloning Toolkit (version 0.92)",
  University of Edinburgh, CSTR, 2019, CC BY 4.0.

**MUSAN, per subset** (checked against arXiv:1510.08484 §2 at download time;
only content that allows commercial use was included in the corpus):

| Subset | Content | Licence |
|---|---|---|
| `speech` | LibriVox readings (~20 h); US government hearings and debates (~40 h) | Public domain (US government works believed public domain) |
| `music` | Jamendo, Free Music Archive, Incompetech, HD Classical Music (~42 h) | Creative Commons, varying per file (several are CC BY, which needs credit); see `musan/music/*/LICENSE` |
| `noise` | Free Sound (public domain portion) and Sound Bible (~6 h) | Public domain / Creative Commons per file; see `musan/noise/*/LICENSE` |

Augmentation noise is mixed into *training input only* and never ships in a
model's output, but keep the `LICENSE` files with any redistributed copy.

## Checksums

OpenSLR and DataShare publish MD5 only, and Hi-Fi TTS publishes no checksum.
`prepare_data.py` checks the published MD5 and the SHA256 in
[`datasets.lock.json`](datasets.lock.json). The SHA256 pins for LibriSpeech,
LJSpeech and VCTK come from torchaudio's dataset tables (v2.5.0) and were
confirmed by our own downloads where marked ✓. Archives without a pin get one
on their first verified download (trust on first use), so commit the lock after
preparing them.

| Archive | Published MD5 | SHA256 pin |
|---|---|---|
| test-clean.tar.gz | `32fa31d2…` ✓ | torchaudio ✓ |
| dev-clean.tar.gz | `42e2234b…` ✓ | torchaudio ✓ |
| train-clean-100.tar.gz | `2a93770f…` | torchaudio |
| train-clean-360.tar.gz | `c0e676e4…` | torchaudio |
| LJSpeech-1.1.tar.bz2 | — | torchaudio ✓ |
| hi_fi_tts_v0.tar.gz | — | first download |
| musan.tar.gz | `0c472d4f…` | first download |
| rirs_noises.zip | `e6f48e25…` | first download |
| VCTK-Corpus-0.92.zip | `8a6ba294…` | torchaudio |

## Evaluation models

| Model | Source (pinned revision) | Licence | Used for |
|---|---|---|---|
| Whisper large-v3 | [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) @ `06f233fe` | Apache 2.0 | WER/CER, with Whisper's English text normaliser |
| ECAPA-TDNN | [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) @ `0f99f2d0` | Apache 2.0 | Target / source cosine, EER |
| UTMOS22 strong | [tarepan/SpeechMOS](https://github.com/tarepan/SpeechMOS) v1.2.0 code (lock) + v1.0.0 weights (SHA256 `38aa51ab…`), from [sarulab-speech/UTMOS22](https://github.com/sarulab-speech/UTMOS22) | MIT | Naturalness |
| Praat pitch (parselmouth 0.4.7) | [YannickJadoul/Parselmouth](https://github.com/YannickJadoul/Parselmouth) | GPL-3.0-or-later | Offline reference F0 (eval only; never linked into the product) |

## Models for later stages (recorded here when first used)

| Model | Licence | Stage |
|---|---|---|
| HuBERT-Base `facebook/hubert-base-ls960` | MIT (fairseq lineage) | 04 teacher |
| Mimi (`kyutai/mimi`) | CC BY 4.0 | 04 kill test |
| mHuBERT-147 | CC BY-NC-SA 4.0 (non-commercial) | deferred to M3 |
