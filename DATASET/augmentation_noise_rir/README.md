# augmentation_noise_rir — noise, music, babble & impulse responses

**Purpose:** input-side robustness augmentation — `fvml/augment.py` mixes noise
(p 0.6, SNR 5–20 dB) and applies RIR reverb (p 0.4, cut at direct path, ≤1 s).
Augmentation audio is mixed into *training inputs only* and never ships inside a
model or its output — but per-file licences must still be tracked.

**Consumers:** `fvml/augment.py` (`noise_paths`, `rir_paths`), NANSY-style
speaker perturbation (`fvml/augment.py::perturb_speaker`), bug-repro corpora
(`silence_hallucination_20.tsv`).

## Already registered

| Corpus | Licence | Use |
|---|---|---|
| MUSAN (SLR 17) | corpus CC BY 4.0; per-file licences in each subset's `LICENSE` | `noise/` (~6 h), `music/` (~42 h, some CC-BY), `speech/` (~60 h babble) |
| RIRS_NOISES (SLR 28, incl. SLR 26 simulated) | Apache-2.0 | real + simulated RIRs, `rirs/` manifest |

## Proposed additions — verified licences

| Corpus | Licence | Size | Fit |
|---|---|---|---|
| **FSD50K** (Zenodo) | corpus CC BY 4.0; per-clip CC0 / CC-BY / CC-BY-NC / Sampling+ | ~50 GB, 108 h | Human-labeled sound events → richer noise; **builder must filter to CC0+CC-BY clips** using `dev/eval_clips_info_FSD50K.json` licence map |
| **DEMAND** (Zenodo 1227121) | CC BY-SA 3.0 *(flagged)* | 7.4 GB, 1.5 h ×16ch | 18 indoor/outdoor environments, stereo collapse of array channels; input-side only |
| **OpenAIR** (openairlib.net) | per-IR licence (mostly CC BY) — verify per room | ~1 GB | Real measured RIRs (churches, halls, stairwells) → reverb diversity |
| **Microsoft DNS noise set** | mixed provenance (AudioSet/Freesound) — verify | ~50 GB | Wide noise coverage; only after licence audit |
| **MS-SNSD** | verify (repo licence unclear) | ~20 GB | Noise + RIR convenience pack |

## Non-commercial / rejected

- ESC-50 — CC BY-NC 3.0 → `noncommercial/` if ever needed
- WHAM!/WHAMR! — WSJ-derived (LDC)
- NoiseX-92 — no clear licence
- BBC Sound Effects — non-commercial (RemArc licence)

## Ingest rules

1. Per-file licence column in every noise manifest (same as MUSAN's
   `licence_file` field); drop NC clips at build time unless
   `--include-noncommercial`.
2. RIR builder: copy only mono/stereo IRs ≤ 2 s, normalize to unit energy;
   keep `rir_id` traceable to source room.
3. Babble = speech corpora re-use (MUSAN speech, Common Voice) — don't ingest
   dedicated babble corpora.
