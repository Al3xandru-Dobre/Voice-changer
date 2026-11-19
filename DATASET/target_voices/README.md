# target_voices — studio speaker pools for presets & fine-tuning

**Purpose:** clean single-speaker corpora used to build preset target voices
(`voices.json`, M2 Stage 05) and to fine-tune the vocoder per voice
(M2 Stage 10). Requirements: studio-quiet recordings, stable timbre, permissive
licence that allows the *derived voice* to ship in commercial bundles.

**Consumers:** `prepare_target_voice.py`, `eval_target_voice.py`,
`eval_lists/targets.tsv`, `fvml/target_voice.py`.

## Already registered

| Corpus | Licence | Voices used |
|---|---|---|
| LJSpeech 1.1 | Public domain | `lj` bootstrap voice (chapter LJ050 held out for eval) |
| Hi-Fi TTS | CC BY 4.0 | readers `92` (F), `8051` (F), `6097` (M), `9017` (M); more via `--hifitts-speakers` |
| VCTK 0.92 | CC BY 4.0 | optional — 110 speakers, accent variety |
| own voice | private | `data/private/<speaker>/` — see `../private/` |

## Proposed additions — verified licences

| Corpus | Licence | Voices | Source | Notes |
|---|---|---|---|---|
| **M-AILABS** `en_US`, `en_UK` | BSD-3-style | ~6 readers (Judy Bieber 61 h, Elliot Miller 32 h, etc.) | caito.de | Studio-ish LibriVox reads; strong preset candidates |
| **CMU Arctic** | Permissive — "free for any purpose" | slt (F), bdl (M), rms (M), clb (F), awb (Scottish), jmk (Canadian), ksp (Indian) | festvox.org/cmu_arctic | ~1,150 phonetically balanced utts/voice; small but classic |
| **LibriTTS-R top speakers** | CC BY 4.0 | pick 4–8 high-duration readers | openslr.org/141 | Restored 24 kHz; verify per-speaker hours before adopting |
| **more Hi-Fi TTS readers** | CC BY 4.0 | 10 readers total in registry | OpenSLR 109 | Just widen `--hifitts-speakers` (e.g. add `6671`, `12787`) |
| **VCTK voices** | CC BY 4.0 | p225..p376 | Edinburgh DataShare | Promote to preset voices for UK accents |

## Non-commercial (never shipped — research only)

- Expresso (CC BY-NC 4.0) — 4 expressive studio voices, would be ideal but NC
- Blizzard/NIT-SONG challenge voices — research licences
- ESD emotional voices — research licence

## Standing rules

- Every shipped voice carries `attribution` in `voices.json` (see `DATASETS.md`
  convention: e.g. "Hi-Fi TTS reader 92, Cori Samuel / LibriVox").
- Held-out eval clips per voice go to `eval_lists/targets.tsv`; training must
  never see them (Stage 05 exclusion rule).
