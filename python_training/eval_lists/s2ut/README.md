# Frozen S2UT evaluation lists (Milestone 3 / Stage 01)

Committed under `python_training/eval_lists/s2ut/`, built deterministically from
prepared manifests by `python_training/make_s2ut_eval_lists.py`, and scored by
`python_training/eval_s2ut.py`. Paths are relative to `$FLUXVOICE_DATA`.

| File | Contents |
|---|---|
| `test_{es_en,fr_en,de_en,en_es,en_fr,en_de}_200.tsv` | 200 held-out test utterances per direction (3–12 s, ≥ 20 source speakers, balanced gender, normalized `src_text` & `tgt_text`, duration ratio `gamma = T_tgt / T_src`). Built with `make_s2ut_eval_lists.py --only paired` once `prepare_s2ut_data.py --datasets paired` has prepared the 6 directions. |
| `reordering_torture_60.tsv` | 60 held-out utterances (10 per direction) ranked for cross-lingual syntactic reordering (German subordinate verb-final `weil/dass/obwohl/wenn` clauses, Spanish/French clitic object pronouns and post-nominal adjective chains). |
| `silence_hallucination_20.tsv` | 10 room-tone / noise-only clips (MUSAN `noise`) + 10 LibriSpeech `test-clean` utterances followed by 5.0 s of trailing digital silence (`16k/s2ut_eval/`), auditing `kFlush` termination and zero silence-hallucination babble (`0` Whisper words on noise clips). |

Rebuild / verify:

```powershell
uv run --project python_training python python_training/make_s2ut_eval_lists.py --only silence --check
uv run --project python_training python python_training/make_s2ut_eval_lists.py --only paired --check
```
