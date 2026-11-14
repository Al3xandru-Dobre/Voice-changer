# Frozen evaluation lists

Committed, and changed only on purpose (rebuild with `make_eval_lists.py` /
`make_s2ut_eval_lists.py`, check with `--check`). Paths are relative to
`$FLUXVOICE_DATA`. Column meanings are in `fvml/evallist.py`.

## Milestone 2 — Mode A Voice Conversion (`eval_lists/`)

| File | Contents |
|---|---|
| `test_clean_200.tsv` | 200 LibriSpeech test-clean utterances, 3–15 s: 5 from each of the 40 speakers (100 F / 100 M), with reference transcripts. The conversion sources for every M2 evaluation. |
| `targets.tsv` | 20 held-out reference clips (3–20 s) per target voice: `lj` (LJSpeech chapter LJ050) and `hifitts_{92,8051,6097,9017}` (each reader's official test split). |

**Training must not see `targets.tsv` clips.** Stage 05 excludes LJSpeech
chapter LJ050 and the Hi-Fi TTS test splits; Stage 10 keeps its own private
list for the "me" voice (`eval_vc.py --targets <private.tsv> --voice me`).

Scoring a Hi-Fi TTS voice needs its audio: `prepare_data.py --datasets hifitts`.

## Milestone 3 — Mode B Simultaneous S2UT (`eval_lists/s2ut/`)

See [`s2ut/README.md`](s2ut/README.md) for the 6-direction `test_*_200.tsv`
lists, `reordering_torture_60.tsv`, and `silence_hallucination_20.tsv` scored
by `eval_s2ut.py`.
