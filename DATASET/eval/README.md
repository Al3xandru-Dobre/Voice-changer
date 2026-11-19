# eval — Mode A evaluation corpora

**Purpose:** source utterances + reference clips for `eval_vc.py` (Whisper WER,
ECAPA cosine/EER, UTMOS, F0 PCC, PNMI, speaker-leakage probe). Eval data must be
disjoint from every training manifest.

**Consumers:** `eval_lists/test_clean_200.tsv`, `eval_lists/targets.tsv`,
`make_eval_lists.py`, `eval_vc.py`, `eval_pitch.py`, `eval_controls.py`.

## Already registered / in use

| Corpus | Licence | Use |
|---|---|---|
| LibriSpeech `test-clean` | CC BY 4.0 | `test_clean_200.tsv` conversion sources |
| LibriSpeech `dev-clean` | CC BY 4.0 | streaming-parity streams, pitch accuracy |
| LibriTTS-R `dev/test-{clean,other}` | CC BY 4.0 | 24 kHz reference eval (`libritts-r-*` keys, **registered**) |
| MUSAN `speech/babble` | CC BY 4.0 | `silence_hallucination_20.tsv` |

## Proposed additions — verified licences

| Corpus | Licence | Size | Why |
|---|---|---|---|
| **LibriSpeech `test-other`** | CC BY 4.0 | 0.33 GB | Noisy/accented WER robustness slice |
| **LibriTTS-R `test-clean`/`dev-other`** | CC BY 4.0 | ~2.2 GB | 24 kHz reference-quality eval |
| **FLEURS `en_us`** | CC BY 4.0 | ~1 GB | Clean parallel sentences; shares text with `es/fr/de` for cross-lingual probes |
| **Common Voice `en` test split** | CC0-1.0 | (part of corpus) | Accents + real mics; build a `cv_accent_200.tsv` |
| **CV Spontaneous ASR Shared Task test** | CC0-1.0 | 0.8 GB | Disfluent spontaneous speech eval |
| **VCTK `test` speakers** | CC BY 4.0 | (part of corpus) | Accent conversion eval (already optional download) |

## Non-commercial eval (opt-in)

- VoxCeleb1-O trial pairs — research licence; EER eval only, never training
- EARS anechoic (verify licence — likely research/NC)

## Rules

- Frozen eval lists are committed under `python_training/eval_lists/` and only
  rebuilt with `make_eval_lists.py --check` on purpose (see `eval_lists/README.md`).
- New lists must record their corpus licence in the TSV header comment and be
  excluded from all training manifests (`--exclude` filter in builders).
