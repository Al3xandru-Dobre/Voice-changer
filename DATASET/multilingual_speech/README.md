# multilingual_speech — monolingual `{es,fr,de,+}` pools (Mode B)

**Purpose:** per-language speech for the multilingual encoder (mHuBERT-vp km1000
teacher distillation, M3 Stage 03) and the multilingual causal Unit-HiFi-GAN.
Independent from `s2ut_paired/` (no translation targets needed).

**Consumers:** `prepare_s2ut_data.py --datasets vocoder` → monolingual
`manifests/*.jsonl`, `train_unit_vocoder.py --langs es,fr,de`, SPM training.

## Already registered

| Key | Corpus | Lang | Licence | Status |
|---|---|---|---|---|
| `css10-{es,fr,de}` | CSS10 | es ~24 h, fr ~19 h, de ~16 h | Apache-2.0 / CC0 (LibriVox) | registered — single-speaker studio voices |
| `cml-tts-{es,fr,de}` | CML-TTS (SLR 146) | es, fr, de | CC BY 4.0 | registered — curated 2M+2F, SNR≥30 dB, 2–12 s |

## Proposed additions — verified licences

| Corpus | Licence | Langs | Hours | Fit |
|---|---|---|---|---|
| **MLS `es`/`fr`/`de`** (`*_opus`) | CC BY 4.0 | es,fr,de (+it,nl,pl,pt) | es ~1.1 kh / fr ~1.3 kh / de ~5.7 kh | Main multilingual scale-up; opus 14/16/29 GB |
| **M-AILABS `es`/`fr`/`de`** | BSD-3-style | es 108 h, fr 190 h, de 237 h (+it 127 h) | ~0.7 kh | Studio readers → multilingual vocoder voices |
| **Common Voice `es`/`fr`/`de`** | CC0-1.0 | es,fr,de | kh each | Crowdsourced accents/mics → encoder robustness |
| **CML-TTS `it,nl,pl,pt,ru`** | CC BY 4.0 | expansion langs | ~30 h each | Future language expansion, same builder |
| **MLS `it,nl,pl,pt`** | CC BY 4.0 | expansion | 0.3–3.5 kh | Language-expansion reserve |
| **Emilia-YODAS `de`/`fr`** | CC BY 4.0 *(provenance caution — verify before adopt)* | de,fr | ~10 kh | In-the-wild spontaneous speech |

## Non-commercial / rejected

- VoxPopuli `{es,fr,de}` — flagged NC in registry (data may be CC0; verify)
- mTEDx `{es,fr,de}` monolingual packs — CC BY-NC-ND
- Emilia original — CC BY-NC
- BibleTTS — check licence before use (some releases CC BY-SA)

## Ingest notes

- MLS opus tarballs contain `.opus` + `transcript.txt` per book — builder must
  decode to 16 kHz FLAC; do not ship `.opus` paths in manifests.
- Keep `src_lang == tgt_lang` manifests in `manifests/` (not `manifests/s2ut/`).
- Curated-speaker vocoder subsets reuse the CML-TTS selection rule
  (`CML_MIN_SNR_DB`, 2M+2F) via a shared `build_cml_tts`-style filter.
