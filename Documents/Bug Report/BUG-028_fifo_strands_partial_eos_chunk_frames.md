# BUG-028 — `UnitProsodyFifo` + `s4` vocoder strands 1–3 tail frames at `<EOS>` and leaks them into the next utterance

> [Bug index](00_index.md) · Found by: [REVIEW-M3-S01-S05_2026-11-15](REVIEW-M3-S01-S05_2026-11-15.md)

| | |
|---|---|
| **Tag** | `BUG-028` |
| **Name** | *Stranded tail frames* — partial `<EOS>` chunks (`1..3` frames) never reach an `s2`/`s4` vocoder and corrupt the next utterance |
| **Reported** | 2026-11-15 |
| **Severity** | High (truncates the final 20–60 ms of ~75% of translated utterances in Mode B `s4` and prepends stale frames onto subsequent utterances) |
| **Status** | ☐ open |
| **Component** | `python_training/gate_s2ut.py:519-537`, `src/pipeline/UnitProsodyFifo.hpp:65-120`, `python_training/fvml/waitk_policy.py:350-388` |
| **Blocks** | [Milestone 3 / Stage 04](../Milestone%203/04_waitk_policy_streaming_state_machine.md), [Stage 05](../Milestone%203/05_causal_s2ut_translator_policy_gate.md) gate WAVs, [Stage 07](../Milestone%203/07_cpp_translator_runtime_offline_runner.md), [Stage 08](../Milestone%203/08_realtime_mode_b_pipeline.md) |

## Summary

Mode B runs the vocoder at `s4` (`frames_per_step = 4` = 80 ms per `Run`, per `00_master.md:228` and `gate_s2ut.py:711`), while `translator_dec` emits `stacked_4` chunks whose final step ends at an arbitrary frame boundary (`[u0, <EOS>, <PAD>, <PAD>]`, `[u0, u1, <EOS>, <PAD>]`, or `[u0, u1, u2, <EOS>]`, and similarly for `reduced_dur`).

`StreamingTranslator._decode()` (`s2ut_stream.py:321-322`) and `UnitProsodyFifo::push_chunk()` (`UnitProsodyFifo.hpp:69-73`, `waitk_policy.py:359-364`) strip `<EOS>`/`<PAD>` and push only the `1..3` valid regular frames into the FIFO. However, `UnitProsodyFifo::pop_hop(4)` returns `None` / `false` whenever `size() < 4`, and neither `UnitProsodyFifo` nor `gate_s2ut.py::convert_utterance` drains or pads partial tail frames when an utterance finishes (`flushes_completed()` increments):

1. **Single-utterance truncation (`gate_s2ut.py`).** In `convert_utterance()` (`gate_s2ut.py:521-537`), `fifo.pop_hop(cfg.chunk_frames)` returns `None`, emitting silence, and the loop exits because `not translator.policy.utterance_active() and fifo.size() < cfg.chunk_frames`. The final `1..3` frames (`20–60 ms`, typically a word-final consonant or syllable) are dropped from `<utt_id>.wav`, even though `trace.jsonl` (`line 531`) counted them in `t_tgt_frames_emitted`.
2. **Cross-utterance contamination (multi-phrase clips & live Mode B).** When a stream continues after a completed flush (or across a short pause), `StreamingTranslator` resets `enc_state` and `dec_state` (`s2ut_stream.py:381-384`), **but `fifo` still holds the `1..3` stranded frames from Utterance 1**. When Utterance 2 begins and pushes its first 4-frame chunk `[v0, v1, v2, v3]`, `pop_hop(4)` pops `[u_last.., v0, ..]`, synthesizing the severed tail of Utterance 1 at the onset of Utterance 2 and permanently shifting Utterance 2's frame-to-chunk phase.

## Environment

- Windows 11 x64, `python_training/.venv` (CPython 3.12.13, PyTorch 2.11.0), and C++20 `src/pipeline/UnitProsodyFifo.hpp`, 2026-11-15.

## Steps to reproduce

```powershell
cd python_training
.\.venv\Scripts\python.exe -c "
import numpy as np
from fvml import translator as tr, s2ut_stream as ss
from fvml.waitk_policy import WaitKConfig
import gate_s2ut as gate

cfg = tr.tiny_config(latent_dim=8, vocab=16)
model = tr.S2utTranslator(cfg).eval()
# 6-frame target utterance: step 0 -> [1,2,3,4], step 1 (in pause) -> [5,6,EOS,PAD]
def teacher(step):
    return np.array([1, 2, 3, 4]) if step == 0 else np.array([5, 6, cfg.eos, cfg.pad])
class Front:
    clock = 'ort'
    def reset(self): pass
    def latents(self, c): return np.zeros((4, 8), np.float32)
## likely the np.zeros((4, 8), np.float32) is causing the issue because it creates a 2 dimensional array instead of a 1 dimensional array
class Voc:
    clock = 'ort'
    def reset(self): self.hops = []
    def synthesize(self, u):
        self.hops.append(list(u))
        return np.full(1280, 0.2, np.float32)

voc = Voc()
st = ss.StreamingTranslator(ss.TorchRuntime(model, 'es', 'en'), WaitKConfig(k=2), teacher=teacher)
wav_in = np.concatenate([0.25 * np.ones(2 * 1280, np.float32), np.zeros(8 * 1280, np.float32)])
_, steps, _ = gate.convert_utterance(wav_in, st, Front(), voc)
print('trace frames:', sum(s['t_tgt_frames_emitted'] for s in steps), 'vocoded hops:', voc.hops)
"
```

## Expected

All 6 emitted target frames (`[1, 2, 3, 4, 5, 6]`) are synthesized before the utterance finishes, leaving `fifo.size() == 0` on flush completion so the next utterance starts cleanly.

## Actual

```text
trace frames: 6 vocoded hops: [[1, 2, 3, 4]]
```

Frames `[5, 6]` (`40 ms`) are recorded in `trace.jsonl` as emitted, never synthesized by `vocoder_s4`, and left stranded inside `fifo` (`size == 2`).

## Evidence

| File | Lines | Behavior |
|---|---|---|
| `src/pipeline/UnitProsodyFifo.hpp` | `92-120` | `pop_hop(n_frames, ...)` requires `n_frames <= size_`; no method exists to drain/pad a `< n_frames` remainder at `<EOS>` / flush completion |
| `python_training/fvml/waitk_policy.py` | `375-388` | Twin `pop_hop` returns `None` when `n_frames > self._size` |
| `python_training/gate_s2ut.py` | `519-537` | Pushes `res.frames`, calls `fifo.pop_hop(4)`, and breaks when `fifo.size() < 4` once `!utterance_active()` |
| `python_training/tests/test_gate_s2ut.py` | `257` | Asserts `len(emitted) - len(voc.units) < 4`, inadvertently tolerating up to 3 dropped tail frames |

## Root cause

`UnitProsodyFifo` was designed assuming steady-state rate bridging where future chunks always arrive to complete a 4-frame vocoder hop (`test_fifo_rate_bridging` pushes an exact multiple of 4 frames). At utterance end (`<EOS>` or `flushes_completed` transition), 3 out of 4 target frame lengths (`N % 4 in {1, 2, 3}`) leave a sub-hop remainder that `vocoder_s2`/`vocoder_s4` cannot consume without tail padding + post-vocoder sample trimming (or silent-frame padding).

## Suggested fix

1. Add `drain_tail_hop(std::size_t n_frames, ...)` (returning the count `valid_frames` in `1..n_frames - 1` and padding the remaining `n_frames - valid_frames` slots with the last unit + `voicing = 0.0f`, low `z_energy` so the vocoder renders silence after frame `valid_frames - 1`, or zeroing output samples `[valid_frames * frame_samples .. hop)`) to both `UnitProsodyFifo.hpp` and `fvml/waitk_policy.py`.
2. Call `drain_tail_hop` (and `fifo.clear()` as a safety invariant) whenever `res.eos` / `flushes_completed` ends an utterance and `0 < fifo.size() < vocoder_fps`.
3. Add a C++ + Python parity test in `test_waitk_policy.{cpp,py}` and `test_gate_s2ut.py` pushing `[10, 11, EOS, PAD]` followed by a second utterance `[20, 21, 22, 23]` with `hop = 4`, asserting all 6 frames are vocoded in their own utterances without cross-utterance bleed.

## Workaround

Running the vocoder at `s1` (`frames_per_step = 1`) drains 1 frame per hop without stranding frames, at higher per-frame ORT dispatch overhead (`gate_s2ut.py` currently hardcodes `vocoder_s4.onnx`).
