# noncommercial — NC / research-only catalog (opt-in)

**Purpose:** single place listing everything that must stay behind
`--include-noncommercial` and `data/nc/` (BUG-031 defence). Nothing here may
reach a shipped bundle, `voices.json`, `MODEL_CARD.md`, or a default manifest.

**Consumers:** `prepare_s2ut_data.py --include-noncommercial`,
`train_*.py --include-noncommercial` research flags.

## Currently registered NC

| Corpus | Licence | Notes |
|---|---|---|
| VoxPopuli S2UT | flagged CC BY-NC 4.0 | Registry isolates it; upstream licence table says data CC0 but repo LICENSE is NC — keep isolation pending verification |
| mHuBERT-147 *(model, not data)* | CC BY-NC-SA 4.0 | research teacher probe only |

## ⚠ Mis-licensed — pending fix

| Corpus | Claimed in registry | Actual |
|---|---|---|
| **Europarl-ST v1.1** | "CC BY 4.0" | **CC BY-NC 4.0** (MLLP README + ICASSP 2020 paper) — see `../README.md` §6 |

## Catalog of NC corpora (do not ingest by default)

| Corpus | Licence | Content | Potential use (research only) |
|---|---|---|---|
| Europarl-ST | CC BY-NC 4.0 | EP debates, 72 directions | en→{es,fr,de} paired |
| Expresso | CC BY-NC 4.0 | 40 h expressive studio | expressive VC probe |
| mTEDx | CC BY-NC-ND 4.0 | 8 langs incl. es/fr/de | X→en ST eval/train |
| MuST-C | CC BY-NC-ND | TED ST | en→{es,fr,de} |
| TED-LIUM 3 | CC BY-NC-ND 3.0 | 452 h en talks | ASR eval |
| SPGISpeech | Kensho non-commercial agreement | 5 kh financial | robust ASR eval |
| ESD | research licence | 29 h emotional | expressive VC probe |
| L2-ARCTIC | CC BY-NC 4.0 | 27 h accented en | accent eval |
| RAVDESS | CC BY-NC-SA 4.0 | emotion actors | emotion probe |
| IEMOCAP | gated NC agreement | 12 h dyadic | emotion probe |
| Emilia (original) | CC BY-NC 4.0 | 101 kh in-the-wild | scale research |
| SeamlessAlign / SeamlessExpressive | CC BY-NC(-SA) | mined S2ST pairs | S2UT comparison |
| ESC-50 | CC BY-NC 3.0 | 2 kh env sounds | noise aug (NC-only runs) |
| VoxCeleb 1/2 | research-only (Oxford) | speaker rec | EER eval lists |
| Speech Accent Archive | CC BY-NC-SA 2.0 | 2 k speakers | accent eval |
| EARS | likely research — verify | 100 h anechoic | clean eval |
| MaSS | unverified (bible.is provenance) | 8-lang aligned | parallel probes |

## Forbidden entirely (never ingest)

- VoxForge — GPL (copyleft ambiguity on weights)
- TIMIT, WSJ, Switchboard, Fisher — LDC paid licences
- AudioSet — YouTube IDs only, cannot redistribute audio
- CC *-ND audio intended for derivative synthesis (resynthesis = derivative)
