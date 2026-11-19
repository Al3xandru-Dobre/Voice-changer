# Stage 08 — Real-time Mode B pipeline, READ/FLUSH gating & governor

> **Milestone 3** · [Master plan](00_master.md) · [← Stage 07](07_cpp_translator_runtime_offline_runner.md) · [Stage 09 →](09_s2ut_controls_language_switching.md)

**Status:** ◐ implemented and fixture-verified (2026-11-18); real-bundle checks pending Stage 06 / live rig
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

- [x] **Extend `src/pipeline/InferencePipeline.{hpp,cpp}` for Mode B:**
  - [x] Add `PipelineMode { kVC, kS2UT }`.
  - [x] Implement the 4-frame `latents` chunk accumulator, `prosody_ctx` running
        voiced accumulator, `translator_enc` / `translator_dec` orchestration,
        `UnitProsodyFifo` consumer, and automatic translator KV-state reset upon
        `WaitKPolicy` flush completion.
  - [x] Switch `tools/fv_s2ut_file.cpp` (Stage 07) to drive `InferencePipeline`
        in `PipelineMode::kS2UT`.
- [x] **Extend `src/pipeline/LatencyGovernor.hpp` for Mode B:**
  - [x] Add a Mode B `burst_cushion_samples` (default `3 * chunk_samples` =
        240 ms) active while `waitk.t_src() > 0` or `!fifo.empty()`.
  - [x] Suppress soft silence-dropping while `WaitKPolicy` is in `kWrite` or
        `kFlush` or `UnitProsodyFifo` is non-empty, so buffered translated frames
        are never discarded before being heard.
- [x] **CLI options in `src/main.cpp`:**
  - [x] `--s2ut` (selects Mode B)
  - [x] `--src-lang <code>` and `--tgt-lang <code>` (required with `--s2ut`;
        validated against `langs.json` before opening any audio device, with
        `src_lang != tgt_lang` check)
  - [x] `--wait-k <1..10>` (optional; defaults to `langs.json`'s pair default, or
        `3`)
  - [x] `--list-langs` (prints supported language codes and default Wait-*k* per
        pair from `langs.json`)
  - [x] Add CTest `cli_rejects_*` cases: `--s2ut` without `--src-lang`/`--tgt-lang`,
        unknown language code, `--src-lang en --tgt-lang en` (same language),
        `--wait-k 0`, `--wait-k 11`.
- [x] **Mode B telemetry in `[stats]` and `tools/analyze_soak.py`:**
  - [x] Append `s2ut_r=<reads> s2ut_w=<writes> s2ut_f=<flushes> lag_ms=<current
        (t_src - t_tgt) * 80>` to `[stats]` when running in Mode B.
  - [x] Extend `tools/analyze_soak.py` with `--s2ut` mode (plus `--self-test`
        cases): verifies `misses == 0` after warm-up, `errors == 0`, `queued`
        within `target + burst_cushion`, and `lag_ms` returning to `0` after
        utterance flushes.
- [x] **Fake-model unit tests (`tests/test_inference_pipeline.cpp` extended,
      ORT-free):**
  - [x] Mode B with $k=3$: feeding 10 voiced chunks followed by 10 silent chunks
        produces 2 `kRead` silence chunks, 8 `kWrite` chunks, 2 `kFlush` chunks
        (10 translated chunks total), then resets `t_src == 0 && t_tgt == 0` and
        zeroes fake translator states.
  - [x] Tail-flush audio is **not** attenuated by the output noise gate even
        though input audio during the flush is all zeros.
  - [x] `vocoder_ fake` step count equals `writes + flushes` (never called during
        `kRead` or idle silence).
  - [x] Zero heap allocations in steady state across full `kRead → kWrite →
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

## As implemented (2026-11-18)

**Status: ◐ implemented and fixture-verified; real-bundle checks pending Stage 06 / live rig.**

### Files and APIs

- `src/pipeline/InferencePipeline.{hpp,cpp}`: `PipelineMode { kVC, kS2UT }`; `ModeBOptions` (`policy`, `src_lang_id`, `tgt_lang_id`, `translate`, `speech_threshold_dbfs`, `voiced_threshold`, `prosody_ctx`, `eos_threshold`, `fifo_capacity_frames`, `burst_cushion_samples`); `S2utTelemetry` (lock-free relaxed counters: reads, writes, flushes, utterances, `cap_ends`, lag); `S2utObserver` / `S2utChunkEvent` for offline tools (`--dump-trace`); `s2ut_idle()`, `waitk()` (worker-only), `mode()` (relaxed atomic, readable from the UI thread), `set_params()`.
- `src/pipeline/LatencyGovernor.hpp`: `set_translation_state(cushion, hold)`, `cushion_active()`, `hold_active()`, `burst_cushion_samples()`.
- `src/pipeline/UnitProsodyFifo.hpp`: `pad_to_multiple()`, `Frame::pad`, back-pressure counters. `src/pipeline/WaitKPolicy.hpp`: `set_adaptive_mode()`.
- `src/main.cpp`: `--s2ut`, `--src-lang`, `--tgt-lang`, `--wait-k`, `--list-langs`; `[stats]` fields; CTest `cli_rejects_s2ut_*` (driven by `tests/check_s2ut_cli.cmake`), `cli_list_langs`, `cli_list_params_langs`.
- `tools/analyze_soak.py --s2ut` (with `--latency-target-ms`, `--hop-ms`) and `--self-test` cases.
- Tests: `tests/test_inference_pipeline.cpp` (Mode B), `tests/test_latency_governor.cpp` (`test_burst_cushion_and_hold`), `tests/test_waitk_policy.cpp`, `test_ort_model.cpp` `test_s2ut_bundle_pipeline`.

### Deviations from the stage text

| Stage text | As implemented | Reason |
|---|---|---|
| `<EOS>`: `eos_prob >= 0.5` | `<EOS>` unit **or** `eos_prob > 0.5` | Twin of `fvml/s2ut_stream.py` |
| Chunk speech: `voiced_frames > 0 \|\| energy_db > thresh` | Any 20 ms input frame with RMS above -55 dBFS on the raw input (twin of `gate_s2ut.py --vad-dbfs`); the `voiced_frames` clause was dropped | Matches the Stage 05 gate |
| `prosody_ctx` running accumulator | Mean (`z_f0`, `z_energy`) over voiced (tracker voicing >= 0.5) frames of the current utterance; reset on utterance end and on Stage 09 switches; optional off for parity runs | Parity with the Python driver, which feeds zeros |
| Readiness passed to the policy | `sigmoid(logit)` via one function `readiness_from_logit()` | BUG-029 is still open; change this function and the Python driver together |
| Gate follows `WaitKPolicy` active output + predicted target energy | The output gate opens on **every translated frame**, not on predicted energy; pad frames render as silence | The frame itself is the evidence; no energy threshold to mistune |
| FIFO drained by whole hops | BUG-028 fix: `UnitProsodyFifo::pad_to_multiple` after **every** push keeps the FIFO hop-aligned; pad samples render as silence. Back-pressure exists (a write is deferred, never dropped) in C++ and the golden twin only | A partial `<EOS>` chunk otherwise stranded 1-3 frames |
| One `decide()` per chunk | Extra writes in an adaptive double-write step re-check `decide()`; nothing is decoded past an `<EOS>` heard in a pause | Prevents decoding past `<EOS>` |
| Governor: cushion `3 x chunk` while `t_src > 0` or FIFO non-empty; suppress soft drops in `kWrite`/`kFlush` or FIFO non-empty | Cushion = 3 chunks (240 ms) **while translating**; soft drops are held while frames are queued or decoding; additionally **every hop whose samples touch translated audio is protected**: the final-frame hop, its ramp-down successor, and a Stage 09 switch hop | Edge hops carry translated samples but have no queued frame |
| Test expectation 2 `kRead` / 8 `kWrite` / 2 `kFlush` | Holds **only** with `tail_flush_ms = 0` and a decoder that reports `<EOS>` at catch-up. Under the Stage 04 defaults (400 ms tail) 10 speech + 10 silent chunks give **4 / 13 / 3**. Both are pinned in `test_inference_pipeline` | The stage example predates the Stage 04 tail rules |
| Mode B model shape | Requires a stateful vocoder (`overlap_samples == 0`) and conditioning frames == content frames (constructor throws `std::invalid_argument`) | Mode A OLA / 20 ms conditioning shapes do not fit 80 ms content frames |
| `[stats]` | Adds `s2ut_r`, `s2ut_w`, `s2ut_f`, `lag_ms` as specified, plus `s2ut_utt` (completed utterances) and `s2ut_cap` (BUG-030 visibility). `lag_ms` may be **negative** during a flush | Visibility; `t_tgt` can run ahead of `t_src` while draining |
| Output envelope | The `kRead` bypass emits silence through the normal output chain (no click) | As specified; recorded for completeness |
| ORT-less `--s2ut` diagnostic | Emitted **after** `langs.json` validation, so the `cli_rejects_s2ut_*` cases run in ORT-less CI | Rejection tests need no ORT |
| `--vc` on a translator bundle | Keeps the smallest hop variant (Mode A latency first) and loads the translators so live `mode s2ut` works; `--s2ut` defaults to `s4` | Mode A must not pay for Mode B |
| Cap on `kFlush` | `S2utTelemetry::cap_ends`, `[stats] s2ut_cap` and the trace `cap_end` count utterances whose flush ended on the ratio bound or `max_flush_steps` without `<EOS>` | BUG-030 visibility |

Open items: the 10-minute live run (0 `misses`, `inf_p95_us` <= 40,000) and the live T - B onset check are **not run** (no trained bundle, no rig). The `analyze_soak.py --s2ut` live thresholds therefore remain unexercised on a real log.

## Verification (AGENTS rule 7; recorded 2026-11-18)

Commands: `cmake -S . -B build-<leg> -DFLUXVOICE_STRICT_WARNINGS=ON -DFLUXVOICE_BUILD_TESTS=ON -DFLUXVOICE_WARNINGS_AS_ERRORS=ON`, `cmake --build build-<leg> --config Release` (all targets, including `fluxvoice_header_checks`), `ctest --test-dir build-<leg> -C Release --output-on-failure`; the venv leg adds `python_training_m3_stage07_golden`.

| Leg | Result |
|---|---|
| Linux GCC 13.3 + ORT 1.31.0 | PASS: warning-free `-Werror` build, hardware-free CTest (non-venv 104/104 at the lead's last gate; final total: see lead's final gate) |
| Linux Clang 18.1.3 + ORT 1.31.0 | PASS (same gate) |
| Linux GCC, no ORT | PASS (103/103 at the lead's last gate; final: see lead's final gate) |
| Linux Clang, no ORT | PASS (same gate) |
| Venv `python_training_m3_stage07_golden` | PASS: 27 tests incl. 9 C++ parity cases, bit-exact |
| Windows / MSVC, macOS / AppleClang | NOT RUN (CI pending) |
| Live T - B onset, 10-min live run | NOT RUN (no trained bundle / no rig) |
