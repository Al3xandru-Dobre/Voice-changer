# BUG-030 — Non-terminating `kFlush` babble passes `gate_s2ut.py` and `eval_s2ut.py` while `silence_hallucination_20.tsv` carries English target text

> [Bug index](00_index.md) · Found by: [REVIEW-M3-S01-S05_2026-11-15](REVIEW-M3-S01-S05_2026-11-15.md)

| | |
|---|---|
| **Tag** | `BUG-030` |
| **Name** | *Blind tail audit* — babbling decoders score `flush_tail_chunks = 0` (`PASS`) and `hallucinated_words = 0`, while `sil_tail_*` has wrong-language `tgt_text` |
| **Reported** | 2026-11-15 |
| **Severity** | Medium–High (the Stage 05 decision gate cannot reject a translator that fails to predict `<EOS>` and hallucinates speech across trailing silence) |
| **Status** | ◐ fixed in tree 2026-11-18 ([fix plan](FIX-PLAN-BUG-029-032_2026-11-18.md); GCC/Clang + pytest green, native MSVC/AppleClang legs pending) |
| **Component** | `python_training/gate_s2ut.py:228-260, 351-358`, `python_training/fvml/s2ut_eval.py:304-336`, `python_training/make_s2ut_eval_lists.py:236-247`, `python_training/eval_lists/s2ut/silence_hallucination_20.tsv` |
| **Blocks** | [Milestone 3 / Stage 01](../Milestone%203/01_multilingual_data_simuleval_harness.md) flush/hallucination harness, [Stage 05](../Milestone%203/05_causal_s2ut_translator_policy_gate.md) Definition of Done |

## Summary

Stage 05's Definition of Done (`05_causal_s2ut_translator_policy_gate.md:314-315`) requires:

> *"`silence_hallucination_20.tsv` produces 0 hallucinated Whisper words and terminates `kFlush` within <= 3 chunks after the translated sentence ends."*

Four interacting bugs across `gate_s2ut.py`, `fvml/s2ut_eval.py`, and `make_s2ut_eval_lists.py` invert or disable this check:

1. **`flush_tail_chunks()` reports `0` (`PASS`) on a decoder that babbles regular units forever (`gate_s2ut.py:351-358`).**
   `flush_tail_chunks(steps)` finds the last step with `t_tgt_frames_emitted > 0` and sums `writes` *after* that step. When a broken decoder never predicts `<EOS>` and instead hallucinates/loops regular speech units (`0..vocab-1`) on every `kFlush` step until `WaitKPolicy`'s `max_flush_ratio` / `max_flush_steps` forces a reset (`25` flush steps = `2.0 s` of babble on a `3.28 s` clip), `t_tgt_frames_emitted` is `4` on the final step, so `steps[last + 1:]` is empty and **`flush_tail_chunks` returns `0` (`PASS`)**. Moreover, `dod_flush_tail_ok` is omitted from `check_row()` / `eligible` (`gate_s2ut.py:254, 963`), so even a nonzero value never blocks a candidate from winning the gate.
2. **`hallucinated_words` ignores `trailing_silence` rows, and `trailing_non_silent_ms` is never gated (`s2ut_eval.py:308-322`, `gate_s2ut.py:228-235`).**
   In `s2ut_eval.py:310`, `hallucinated_words` is counted only on `is_noise` rows (`kind == "noise_only" or not r["tgt_text"].strip()`). On quiet `noise_only` clips (`< -55 dBFS`), `WaitKPolicy` stays in `IDLE` (`source_wanted() == False`) and never calls the translator at all (`test_gate_s2ut.py:200`). On the 10 `trailing_silence` rows (`sil_tail_00..09`, 5 s of silence appended after speech specifically to test `kFlush` termination), `hallucinated_words` is skipped and `trailing_non_silent_ms` is stored on the row (`gate_s2ut.py:400`) without any threshold check in `check_row()`.
3. **`s2ut_eval.py`'s `DEFAULT_TAIL_ALLOWANCE_MS = 320 ms` penalizes valid Wait-*k* translations (`s2ut_eval.py:44, 328-330`).**
   For `trailing_silence` rows, `exp_end = speech_end_s + 0.320 s`. Yet `WaitKPolicy::kTailFlushMs` alone is `400 ms` (so `kFlush` does not even begin until `speech_end_s + 480 ms`), on top of initial Wait-*k* buffering (`(k - 1) * 80 ms`) and Romance/German target duration expansion (`gamma ~ 1.15–1.25`). A clean, zero-babble 1.2× `Wait-4` translation that emits `<EOS>` immediately after finishing its sentence still reports `trailing_non_silent_ms = 560 ms`.
4. **Committed `silence_hallucination_20.tsv` stores English `tgt_text` for `es`, `fr`, and `de` target languages (`make_s2ut_eval_lists.py:236-247`).**
   `sil_tail_00..09` set `src_lang = "en"` and cycle `tgt_lang` across `("es", "fr", "de")`, but copy the English LibriSpeech transcript into `tgt_text` (e.g. `sil_tail_00`: `en -> es`, `tgt_text = "stuff it into you his belly counselled him"`). Running `eval_s2ut.py` with default `--metrics` scores Spanish/French/German Whisper transcripts against English reference text.

## Environment

- Windows 11 x64, `python_training/.venv` (CPython 3.12.13, PyTorch 2.11.0), 2026-11-15.

## Steps to reproduce

```powershell
cd python_training
.\.venv\Scripts\python.exe -c "
import numpy as np, torch
from fvml import s2ut_eval, translator as tr, s2ut_stream as ss
from fvml.waitk_policy import WaitKConfig
import gate_s2ut as gate

cfg = tr.tiny_config(latent_dim=8, vocab=16)
model = tr.S2utTranslator(cfg).eval()
with torch.no_grad():
    model.decoder.unit_head.bias.view(4, cfg.n_classes)[:, cfg.vocab:] = -50.0
    model.decoder.eos_head.bias.fill_(-50.0)   # never predicts <EOS>; babbles forever

class Front:
    clock = 'ort'
    def reset(self): pass
    def latents(self, c): return np.zeros((4, 8), np.float32)
class LoudVoc:
    clock = 'ort'
    def reset(self): pass
    def synthesize(self, u):
        return (0.25 * np.sin(2 * np.pi * 200.0 * np.arange(1280, dtype=np.float32) / 16000.0))

st = ss.StreamingTranslator(ss.TorchRuntime(model, 'en', 'es'), WaitKConfig(k=3, max_flush_ratio=1.5))
wav_in = np.concatenate([0.25 * np.sin(2 * np.pi * 180.0 * np.arange(52480, dtype=np.float32) / 16000.0),
                         np.zeros(80000, dtype=np.float32)])
wav_out, steps, stats = gate.convert_utterance(wav_in, st, Front(), LoudVoc())
print('flush_steps:', sum(s['decision'] == 'FLUSH' for s in steps),
      'trailing_non_silent_ms:', s2ut_eval.trailing_non_silent_duration_ms(wav_out, 3.28 + 0.32),
      'gate flush_tail_chunks:', stats['flush_tail_chunks'])
"
```

## Expected

A model that never predicts `<EOS>` and babbles across 25 flush steps (`2,000 ms`) fails the flush/silence check in `gate_s2ut.py` (`eligible == False`). Conversely, a model that emits `<EOS>` within `<= 3` chunks after a normal translated sentence ends passes with `0` excess trailing babble.

## Actual

```text
flush_steps: 25 trailing_non_silent_ms: 2080.0 gate flush_tail_chunks: 0
```

`gate_s2ut.py` reports `flush tail PASS (max 0 chunk(s))` and marks the babbling candidate `eligible = True` (`test_gate_s2ut.py:88-90, 176-182` constructs this exact non-terminating model and asserts it passes the gate).

## Evidence

| File | Lines | Observation |
|---|---|---|
| `python_training/gate_s2ut.py` | `351-358` | `flush_tail_chunks` only counts silent `<PAD>` writes *after* the last regular unit, returning `0` when the model babbles regular units up to the hard ratio/step cap |
| `python_training/tests/test_gate_s2ut.py` | `88-90, 176-182` | Sets `eos_head.bias = -50.0` (never `<EOS>`), runs to flush ratio bound (~1.9× source), and asserts `eligible == True` and `flush_tail_chunks_max == 0` |
| `python_training/fvml/s2ut_eval.py` | `308-335` | `hallucinated_words` ignores `trailing_silence`; `exp_end = speech_end_s + 0.32` underestimates valid `Wait-k` + `400 ms` flush latency |
| `python_training/eval_lists/s2ut/silence_hallucination_20.tsv` | `12-21` | `src_lang=en`, `tgt_lang in {es,fr,de}`, `tgt_text` = English LibriSpeech sentence |

## Root cause

`flush_tail_chunks` conflated "decoder kept stepping after emitting `<PAD>`" with "decoder failed to terminate `kFlush` via `<EOS>`", while `s2ut_eval.py` checked Whisper babble only on `noise_only` clips and sized `DEFAULT_TAIL_ALLOWANCE_MS` (`320 ms`) below `kTailFlushMs` (`400 ms`).

## Suggested fix

1. Track whether each `trailing_silence` utterance in `convert_utterance` terminated via model `<EOS>` (`res.eos`) vs. hitting `WaitKPolicy`'s `max_flush_ratio` / `max_flush_steps` hard cap; fail the gate check when `trailing_silence` utterances hit the hard cap without `<EOS>` or exceed `trailing_non_silent_ms` beyond an onset- and `gamma`-aware bound (e.g. `wait_ms + max(speech_end_s * 1.35, speech_end_s + tail_flush_ms) + 3 * 80 ms`).
2. Gate `dod_flush_tail_ok` and `trailing_silence` termination inside `check_row()` so non-terminating models cannot be `eligible` (`BEST_PRACTICES.md` §4.7: gates must be able to fail).
3. In `s2ut_eval.py` + `silence_hallucination_20.tsv`, leave `tgt_text` empty on un-translated LibriSpeech `trailing_silence` rows (or build `sil_tail_*` from paired S2UT test clips that have real target-language references), and slice Whisper transcription of the post-allowance tail (`out_wave[exp_end:]`) to count post-utterance hallucinated words directly.

## Workaround

Inspect `per_utt.csv` (`trailing_non_silent_ms`, `duration_out / speech_end_s`) and `trace.jsonl` (`eos` events) manually on `silence_hallucination_20.tsv`, and always pass `--metrics flush` (never `bleu`) when running `eval_s2ut.py` on `silence_hallucination_20.tsv`.

## Runtime support (M3 Stage 08)

The C++ Mode B runtime reports the non-terminating case: `S2utTelemetry::cap_ends` (and the trace field `cap_end`, `[stats] s2ut_cap=`) count utterances whose flush ended on the `max_flush_ratio` bound or `max_flush_steps` without an `<EOS>`. `fv_s2ut_file --dump-trace` and `tools/analyze_soak.py --s2ut` can therefore surface this live; it does not replace the gate fix above.
