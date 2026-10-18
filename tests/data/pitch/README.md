# Pitch tracker golden vectors (Milestone 2 / Stage 03)

Written by `python_training/make_pitch_golden.py` from the Python reference
tracker (`python_training/fvml/features.py`). Checked by
`python_training/tests/test_features.py` (exact) and
`tests/test_pitch_tracker.cpp` (C++ port, Stage 03 tolerances). Regenerate,
and commit in the same change, whenever a tracker constant or step changes:

```powershell
cd python_training
.venv\Scripts\python.exe make_pitch_golden.py           # rewrite
.venv\Scripts\python.exe make_pitch_golden.py --check   # 0 = up to date
```

| File | Content |
|---|---|
| `header.json` | every tracker constant (frame, window, lag range, thresholds, voiced decision, high-pass coefficients, EMA constants, priors), the feature column order and the case list |
| `<input>.pcm16` | 16 kHz mono input, little-endian int16; both languages read it as `int16 / 32768` |
| `<input>_L<k>.f32` | expected features for lookahead `k` ∈ {0, 1, 2}: frames × 7 little-endian float32, row-major, columns as `header.json["features"]`; frame `t` covers samples `[320 t, 320 (t + 1))` |

Inputs:

- `synthetic` (2.7 s, generated): sawtooth glide 80 → 400 Hz (1.2 s),
  3 × (100 ms 150 Hz sawtooth + 100 ms white noise), white noise (0.3 s),
  digital silence (0.2 s), a 220 Hz sine clipped at full scale (0.4 s).
- `libri` (1.82 s): LibriSpeech test-clean utterance `61-70968-0058`
  ("WILL YOU FORGIVE ME NOW"), unmodified 16-bit samples.

## Attribution

`libri.pcm16` is from **LibriSpeech** (Vassil Panayotov, Guoguo Chen, Daniel
Povey and Sanjeev Khudanpur, "LibriSpeech: an ASR corpus based on public
domain audio books", ICASSP 2015), <https://www.openslr.org/12>, licensed
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). It is the
raw audio of one utterance, stored as headerless PCM.
