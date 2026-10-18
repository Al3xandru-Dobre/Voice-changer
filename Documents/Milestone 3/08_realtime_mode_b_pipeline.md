# Stage 08 — Real-time Mode B pipeline, READ/FLUSH gating & governor

> **Milestone 3** · [Master plan](00_master.md) · [← Stage 07](07_cpp_translator_runtime_offline_runner.md) · [Stage 09 →](09_s2ut_controls_language_switching.md)

**Status:** ☐ not started
**Depends on:** Stage 04 (`WaitKPolicy` + `UnitProsodyFifo`), Stage 07 (v2 `OrtModel` + `fv_s2ut_file`), [Milestone 2 / Stage 08](../Milestone%202/08_realtime_mode_a_pipeline.md) (`InferencePipeline`, `LatencyGovernor`)
**Research:** [M3 research](../Research_documents/M3_WAITK_S2UT_RESEARCH.md) §4 latency budget · §5.2 READ bypass & FLUSH gate override · §9.2 Mode B governor

## Objective

Wire the complete Mode B pipeline (`encoder → translator_enc → WaitKPolicy →
translator_dec → UnitProsodyFifo → vocoder`) into `InferencePipeline` and the
real-time worker thread so `fluxvoice --s2ut --src-lang <L> --tgt-lang <L>`
translates live microphone speech as a continuous, dropout-free stream:
- accumulate 50 Hz Stage 1 `latents` into 80 ms translation chunks ($C = 4$
  frames);
- skip `translator_dec` and `vocoder` during `Decision::kRead`, emitting clean
  ramped silence;
- keep the output noise gate open during `Decision::kFlush` while translated
  units drain, even though the live microphone is silent;
- coordinate `LatencyGovernor` with `WaitKPolicy` so burst chunks during
  `kFlush` or adaptive steps are not mistaken for queue drift;
- report Mode B step counters and lag telemetry in `[stats]`.

## Context & references

- `src/pipeline/InferencePipeline.cpp` (lines 10, 38–39, 47–48) already holds
  `translator_state_` and `waitk_{3}` placeholders.
- [M2 / Stage 08](../Milestone%202/08_realtime_mode_a_pipeline.md) defines the
  Mode A per-hop chain in `InferencePipeline`:
  `HPF → PitchTracker → delay(L) → encoder → F0/energy map → vocoder → [OLA] → gate → emit`.
- **What changes in Mode B per hop:**
  1. `HPF → PitchTracker` still runs on incoming mic audio to classify frame
     speech energy (`speech_detected` for `WaitKPolicy::on_chunk_energy`) and
     update the utterance-level `prosody_ctx = [mean_z_f0, mean_z_energy]`.
  2. `encoder.step()` emits `latents` `[1, frames_per_step, D]`, which are pushed
     into a 4-frame chunk accumulator.
  3. Whenever 4 frames (80 ms) of `latents` have accumulated:
     - feed chunk speech detection (`voiced_frames > 0 || energy_db >
       speech_thresh_db`) to `waitk_.on_chunk_energy(speech, 80)`;
     - if `speech` or `waitk_.t_src() > 0`: run `translator_enc.step()` and call
       `waitk_.notify_source_unit()`;
     - query `waitk_.decide()`:
       - **`kRead`:** do nothing further (`UnitProsodyFifo` remains empty → hop
         emits silence).
       - **`kWrite`:** run `translator_dec.step()`, push the 4 emitted `(units,
         prosody)` frames into `UnitProsodyFifo`, and call
         `waitk_.notify_target_unit(eos_prob >= 0.5f)`.
       - **`kFlush`:** run `translator_dec.step()` (up to 1 chunk per hop so
         per-hop compute stays bounded!), push emitted frames into
         `UnitProsodyFifo`, and call `waitk_.notify_target_unit(eos)`. When
         `WaitKPolicy` finishes the flush and resets, also call
         `translator_enc.reset()`, `translator_dec.reset()`, and reset the
         utterance `prosody_ctx` accumulator!
  4. **Vocoder consumption:** pop `frames_per_step` frames from `UnitProsodyFifo`:
     - if available: map the predicted `(z_f0, z_energy)` into the target voice's
       domain (`voices.json` + `VoiceParams`), run `vocoder.step()`, and mark the
       output gate **active**;
     - if the FIFO is empty (`kRead` or idle silence): skip `vocoder.step()` and
       fill the hop output with zeros (passing through the output envelope ramp so
       speech $\leftrightarrow$ `kRead` transitions never click).

## Tasks

- [ ] **Extend `src/pipeline/InferencePipeline.{hpp,cpp}` for Mode B:**
  - [ ] Add `PipelineMode { kVC, kS2UT }`.
  - [ ] Implement the 4-frame `latents` chunk accumulator, `prosody_ctx` running
        voiced accumulator, `translator_enc` / `translator_dec` orchestration,
        `UnitProsodyFifo` consumer, and automatic translator KV-state reset upon
        `WaitKPolicy` flush completion.
  - [ ] Switch `tools/fv_s2ut_file.cpp` (Stage 07) to drive `InferencePipeline`
        in `PipelineMode::kS2UT`.
- [ ] **Extend `src/pipeline/LatencyGovernor.hpp` for Mode B:**
  - [ ] Add a Mode B `burst_cushion_samples` (default `3 * chunk_samples` =
        240 ms) active while `waitk.t_src() > 0` or `!fifo.empty()`.
  - [ ] Suppress soft silence-dropping while `WaitKPolicy` is in `kWrite` or
        `kFlush` or `UnitProsodyFifo` is non-empty, so buffered translated frames
        are never discarded before being heard.
- [ ] **CLI options in `src/main.cpp`:**
  - [ ] `--s2ut` (selects Mode B)
  - [ ] `--src-lang <code>` and `--tgt-lang <code>` (required with `--s2ut`;
        validated against `langs.json` before opening any audio device, with
        `src_lang != tgt_lang` check)
  - [ ] `--wait-k <1..10>` (optional; defaults to `langs.json`'s pair default, or
        `3`)
  - [ ] `--list-langs` (prints supported language codes and default Wait-*k* per
        pair from `langs.json`)
  - [ ] Add CTest `cli_rejects_*` cases: `--s2ut` without `--src-lang`/`--tgt-lang`,
        unknown language code, `--src-lang en --tgt-lang en` (same language),
        `--wait-k 0`, `--wait-k 11`.
- [ ] **Mode B telemetry in `[stats]` and `tools/analyze_soak.py`:**
  - [ ] Append `s2ut_r=<reads> s2ut_w=<writes> s2ut_f=<flushes> lag_ms=<current
        (t_src - t_tgt) * 80>` to `[stats]` when running in Mode B.
  - [ ] Extend `tools/analyze_soak.py` with `--s2ut` mode (plus `--self-test`
        cases): verifies `misses == 0` after warm-up, `errors == 0`, `queued`
        within `target + burst_cushion`, and `lag_ms` returning to `0` after
        utterance flushes.
- [ ] **Fake-model unit tests (`tests/test_inference_pipeline.cpp` extended,
      ORT-free):**
  - [ ] Mode B with $k=3$: feeding 10 voiced chunks followed by 10 silent chunks
        produces 2 `kRead` silence chunks, 8 `kWrite` chunks, 2 `kFlush` chunks
        (10 translated chunks total), then resets `t_src == 0 && t_tgt == 0` and
        zeroes fake translator states.
  - [ ] Tail-flush audio is **not** attenuated by the output noise gate even
        though input audio during the flush is all zeros.
  - [ ] `vocoder_ fake` step count equals `writes + flushes` (never called during
        `kRead` or idle silence).
  - [ ] Zero heap allocations in steady state across full `kRead → kWrite →
        kFlush → reset` cycles (`operator new` counter).

## Implementation notes

- **Cap `translator_dec` to 1 chunk per hop during `kFlush`.** Draining all $k$
  remaining chunks in a single worker hop would multiply `translator_dec` +
  `vocoder` compute by $k\times$ in one 80 ms window and cause an audio callback
  underrun! Draining 1 chunk per 80 ms hop paces the tail flush at natural real-time
  speaking speed (`1x` RTF) and keeps per-hop compute flat.
- **What if the user starts a new sentence while `kFlush` is draining?** Stage 04's
  mid-flush resumption rule (`on_chunk_energy(true, 80)`) exits `kFlush` back into
  `kWrite`, while `translator_enc` continues appending the new sentence into the
  ring buffer without dropping a frame.

## Files touched

- `src/pipeline/InferencePipeline.hpp`, `src/pipeline/InferencePipeline.cpp`,
  `src/pipeline/LatencyGovernor.hpp`, `src/main.cpp`, `tools/fv_s2ut_file.cpp`,
  `tools/analyze_soak.py`, `tests/test_inference_pipeline.cpp`,
  `tests/test_latency_governor.cpp`, `CMakeLists.txt`

## Definition of done

- All ORT-free fake-model Mode B tests in `test_inference_pipeline` and
  `test_latency_governor` pass (exact `kRead`/`kWrite`/`kFlush` counts, vocoder
  bypass on `kRead`, un-muted `kFlush` tail, zero steady-state allocations).
- `analyze_soak.py --self-test` passes its new Mode B synthetic log checks.
- With the v2 fixture bundle via `fluxvoice --s2ut --src-lang es --tgt-lang en
  --wait-k 3`, measured onset $T - B$ equals Mode A identity $T - B + 3 \times
  80\text{ ms} \pm 5\text{ ms}$.
- 10-minute live Mode B run with the real bundle (`es→en` and `en→de`): 0
  `misses` after warm-up, `inf_p95_us` $\le 40,000\;\mu\text{s}$ (50% of 80 ms),
  and clean tail flushes at the end of every spoken sentence.
