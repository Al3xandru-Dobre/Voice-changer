# s2ut_paired — parallel speech-translation corpora (Mode B)

**Purpose:** paired `(src_audio, src_text, tgt_audio, tgt_text)` utterances for
the Wait-*k* S2UT translator covering all 6 directions
`En↔{Es,Fr,De}` (M3 Stages 01/05). Target audio must be either real human
speech or canonical Piper TTS (CER ≤ 5% round-trip gate).

**Consumers:** `prepare_s2ut_data.py` → `fvml/s2ut_data.py` builders →
`manifests/s2ut/<key>.jsonl`, `train_s2ut.py`, `eval_s2ut.py`,
`eval_lists/s2ut/test_*_200.tsv`.

## Already registered

| Key | Pair | Licence | Status |
|---|---|---|---|
| `cvss-c-{es,fr,de}_en` | {es,fr,de}→en | CC BY 4.0 ✔ verified | registered — canonical En target speech + CoVoST 2 join |
| `covost2-en_de` | en→de | CC0-1.0 ✔ | registered — Piper `de_DE-thorsten` targets |
| `europarl-st-en_{es,fr,de}` | en→{es,fr,de} | ⚠ **CC BY-NC 4.0 — registry says "CC BY 4.0"** | registered but **mis-licensed**, see below |

### ⚠ Europarl-ST licence discrepancy

MLLP's README and the ICASSP 2020 paper state the corpus is **CC BY-NC 4.0**
(non-commercial). The registry (`s2ut_data.py`, `DATASETS.md`) claims
"CC BY 4.0". Until fixed it taints the default `paired` group for commercial
use. Options:
- reclassify `noncommercial=True` → isolated under `data/nc/` (safe default), or
- drop it as a training source and keep only for research eval.
Either way `en→{es,fr}` then relies on reverse-CVSS synthesis — consider
compensating with `cvss-t`/Common Voice en + Piper targets (still clean).

## Proposed additions

| Corpus | Licence | Directions | Notes |
|---|---|---|---|
| **CoVoST 2 `en` extra pairs** | CC0 | — (X→en only) | No en→X; targets must stay synthesized |
| **CVSS-T `{es,fr,de}_en`** | CC BY 4.0 | {es,fr,de}→en | Voice-matched English targets; optional, check if transferred-voice targets help voice preservation |
| **FLEURS n-way parallel text + audio** | CC BY 4.0 | en/es/fr/de eval | Build `fleurs_s2ut_eval` lists; text is n-way parallel |
| **Common Voice `es/fr/de` + Piper en** | CC0 + CC BY voices | {es,fr,de}→en augmentation | More source diversity beyond CVSS-C's Common Voice v4 base |

## Non-commercial (opt-in only — `data/nc/`)

| Corpus | Licence | Why interesting (research only) |
|---|---|---|
| Europarl-ST v1.1 | CC BY-NC 4.0 | Real parliamentary speech, 72 directions |
| mTEDx (SLR 100) | CC BY-NC-ND | es/fr/de + 5 more X→en pairs (ND: no resynthesis) |
| MuST-C | CC BY-NC-ND | TED-domain en→{es,fr,de} text targets |
| MaSS | unverified — CMU Wilderness (bible.is) provenance | 8-lang aligned utterances incl. en/es/fr/de |
| SeamlessAlign / SeamlessExpressive | CC BY-NC(-SA) | Large mined S2ST pairs for comparison |
| VoxPopuli S2UT | flagged NC (data table says CC0 — verify) | already isolated in `data/nc/` |

## Rules

- Every row carries `license` (see `PAIRED_REQUIRED_KEYS` in
  `fvml/s2ut_data.py`); the manifest-level licence must match the strictest
  source licence used.
- NC rows may only be written under `manifests/s2ut/nc/` via
  `--include-noncommercial` (BUG-031 defence).
