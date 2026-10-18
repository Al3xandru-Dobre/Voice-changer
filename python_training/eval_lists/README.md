# Frozen evaluation lists

Committed, and changed only on purpose (rebuild with `make_eval_lists.py`,
check with `make_eval_lists.py --check`). Paths are relative to
`$FLUXVOICE_DATA`. Column meanings are in `fvml/evallist.py`.

| File | Contents |
|---|---|
| `test_clean_200.tsv` | 200 LibriSpeech test-clean utterances, 3–15 s: 5 from each of the 40 speakers (100 F / 100 M), with reference transcripts. The conversion sources for every M2 evaluation. |
| `targets.tsv` | 20 held-out reference clips (3–20 s) per target voice: `lj` (LJSpeech chapter LJ050) and `hifitts_{92,8051,6097,9017}` (each reader's official test split). |

**Training must not see `targets.tsv` clips.** Stage 05 excludes LJSpeech
chapter LJ050 and the Hi-Fi TTS test splits; Stage 10 keeps its own private
list for the "me" voice (`eval_vc.py --targets <private.tsv> --voice me`).

Scoring a Hi-Fi TTS voice needs its audio: `prepare_data.py --datasets hifitts`.
