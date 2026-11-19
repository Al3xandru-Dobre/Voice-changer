# expressive — emotional, stylistic & prosody-rich speech

**Purpose:** style/emotion research for expressive voice conversion and future
style-conditioned presets. **Reality check:** nearly all expressive corpora are
non-commercial — this directory documents what exists and the licence status;
commercial-clean expressive coverage is essentially empty and must be covered
by param controls + augmentation instead.

**Consumers:** research probes only; `eval_target_voice.py` style lists; future
`style=` preset params (M4 voice studio).

## Candidate corpora (none commercial-clean)

| Corpus | Licence | Size | Content |
|---|---|---|---|
| Expresso (Meta) | **CC BY-NC 4.0** | 40 h | 4 speakers, 8 read + 26 improvised styles incl. singing/whisper — best quality, NC |
| ESD (NUS/SUTD) | research licence | ~29 h | 10 en + 10 zh speakers × 5 emotions, parallel text |
| RAVDESS | CC BY-NC-SA 4.0 | ~1.5 h | 24 actors × 8 emotions, studio |
| EmoV-DB | unverified (free for research?) | ~10 h | 4 speakers × 5 emotions, en |
| CREMA-D | unverified (ODbL?) | ~5 h | 91 actors, crowd-rated |
| IEMOCAP | gated, non-commercial agreement | ~12 h | dyadic acted + improvised |
| TESS | unverified | ~1 h | 2 speakers × 7 emotions |
| L2-ARCTIC | CC BY-NC 4.0 | ~27 h | 24 non-native speakers (accent, not emotion) |
| EARS (2024) | unverified — likely research | ~100 h | 107 speakers anechoic, expressive reads |

## Commercial-clean substitutes (strategy)

- **Param-driven expression** (current plan): pitch shift/scale, energy scale,
  tempo, whisper/air mix via `VoiceParams` — no expressive corpus needed.
- **Synthesis-time augmentation**: parselmouth/Praat resynthesis perturbations
  (`fvml/augment.py::perturb_speaker`, seeded RNG per BUG rule 14) create
  pitch/formant/EQ variation without copyrighted emotion data.
- **CV Spontaneous Speech (CC0)** contains natural laughter/breath and
  expressive prosody — usable, unlabelled for emotion though.
- If expressive VC becomes a product feature, budget for a small in-house
  studio recording (see `private/` consent rules) — that is the only clean path.

## Rule

Any NC expressive corpus lives under `data/nc/expressive/` and never feeds a
shipped voice or preset.
