# core_speech — English multi-speaker training pools (Mode A)

**Purpose:** train the 50 Hz StreamVC-style causal content encoder (HuBERT km100
distillation, M2 Stage 04) and the causal Unit-HiFi-GAN vocoder (M2 Stage 05).
Priority: speaker diversity, microphone/channel variety, clean transcript.

**Consumers:** `prepare_data.py` → `fvml/datasets.py` builders → `train_content_encoder.py`,
`train_unit_vocoder.py`, `extract_units.py`.

## Already registered (fetch via `prepare_data.py --datasets …`)

| Key | Corpus | Licence | Size | Status |
|---|---|---|---|---|
| `librispeech-train-clean-100` | LibriSpeech (OpenSLR 12) | CC BY 4.0 | 6.4 GB | registered |
| `librispeech-train-clean-360` | LibriSpeech | CC BY 4.0 | 23 GB | registered |
| `ljspeech` | LJSpeech 1.1 | Public domain | 2.7 GB | registered |
| `hifitts` | Hi-Fi TTS (OpenSLR 109) | CC BY 4.0 | 41 GB (4 default readers) | registered |
| `vctk` | VCTK 0.92 mic1 | CC BY 4.0 | 11.7 GB | optional |
| `libritts-r-*` | LibriTTS-R (OpenSLR 141) | CC BY 4.0 | dev/test 4.6 GB core, train 8.1–75.8 GB | **registered** — 24 kHz restored; `train-{360,other-500}` optional |

## Proposed additions — verified licences

| Corpus | Licence | Hours | Source | Fit |
|---|---|---|---|---|
| **Common Voice Scripted v27 `en`** | CC0-1.0 | ~3.6 kh recorded | mozilladatacollective.com | Real mics, accents, ages — fixes read-speech bias |
| **CV Spontaneous Speech v5 `en`** | CC0-1.0 | ~18 h (validated 8.4 h) | Mozilla Data Collective | Unscripted answers → disfluency robustness |
| **M-AILABS `en_US` (+`en_UK`)** | BSD-3-style, commercial OK* | ~147 h | caito.de / mirrors | Studio LibriVox readers; vocoder-quality |
| **LibriLight `unlab-600h/6kh`** | MIT tooling + public-domain audio | 600–60,000 h | github.com/facebookresearch/libri-light | SSL encoder pretrain (no transcript needed) |
| **People's Speech `cc-by-clean`** | CC BY 4.0 | ~10 kh subset | mlcommons.org / HF `MLCommons/peoples_speech` | Conversational, archive.org-sourced; **use only `cc-by-*` configs**, not `cc-by-sa-*` |
| **AMI meeting corpus (IHM/SDM)** | CC BY 4.0 | ~100 h | groups.inf.ed.ac.uk/ami/corpus | Spontaneous multi-speaker, far-field eval |
| **MLS `english`** | CC BY 4.0 | ~44.5 kh | openslr.org/94 | Scale-up only; use `mls_english_opus` (651 GB) |

*M-AILABS note: licence permits commercial use but requires a separate licence
for German/Swiss/Austrian universities — irrelevant here, record in DATASETS.md.

## Non-commercial / rejected

- Emilia original 101 kh — CC BY-NC (use **Emilia-YODAS** subset, CC BY, with caution)
- GigaSpeech — gated application form (ESB lists Apache-2.0; verify before ingest)
- VoxForge — GPL, avoid training pools
- TED-LIUM 3 — CC BY-NC-ND; SPGISpeech — non-commercial agreement
- YODAS — murky YouTube provenance

## Ingest notes

- New corpora need a `build_*` in `fvml/datasets.py` writing the standard
  manifest row (`utt_id, path, duration, speaker, dataset, subset, licence`).
- Common Voice ships MP3 + TSV — builder joins TSV rows to `clips/`, resamples,
  keeps `client_id` as speaker and `accent`/`gender` columns where present.
- LibriLight is unlabelled: tag `subset=unlab-*`, `text=None`; never route into
  CTC-supervised heads.
